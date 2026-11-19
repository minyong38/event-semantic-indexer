#!/usr/bin/env python3
"""두 인덱스(이벤트 기반 vs 프레임 기반)를 같은 질문·같은 정답으로 비교.

queries.json 형식:
  [{"q": "a person carrying a box", "gt": ["ch0-231", "ch2-205"]}, ...]
  gt는 make_sheet.py 시트에 적힌 번호 (ch번호-진짜트랙ID).

지표 (사람 단위):
  recall@k  상위 k개 결과 안에 정답 인물이 한 명이라도 있는 질문 비율
  고유인물@k 상위 k개 결과 중 서로 다른 인물 수 (중복이 많으면 작아짐)

  python evaluate.py --queries queries.json --frags ../camsim/out_raw \
      --index event=~/idx_raw --index frame=~/idx_frame --text ../model/clip_text.onnx
"""
import argparse
import json
from pathlib import Path

import hnswlib
import numpy as np
import onnxruntime as ort
import open_clip

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True)
ap.add_argument("--frags", required=True)
ap.add_argument("--index", action="append", required=True, help="이름=경로")
ap.add_argument("--text", required=True)
ap.add_argument("-k", type=int, nargs="+", default=[1, 5, 10])
ap.add_argument("--detail", action="store_true", help="질문별 정답 순위·인덱스 포함 여부")
ap.add_argument("--depth", type=int, default=50, help="질문별 순위를 찾을 깊이")
args = ap.parse_args()

# cam_id → 진짜 트랙 (채널별)
true_of = {}
for f in Path(args.frags).glob("fragments_ch*.jsonl"):
    for line in open(f):
        d = json.loads(line)
        true_of[(d["ch"], d["cam_id"])] = f"ch{d['ch']}-{d['true_id']}"

queries = json.load(open(args.queries))
tok = open_clip.get_tokenizer("ViT-B-32")
txt = ort.InferenceSession(args.text, providers=["CPUExecutionProvider"])
qv = txt.run(None, {"tokens": tok([q["q"] for q in queries]).numpy()})[0]
qv /= np.linalg.norm(qv, axis=1, keepdims=True)

kmax = max(max(args.k), args.depth if args.detail else 0)
ranks = {}  # 이름 → 질문별 (첫 정답 순위, 인덱스에 있는 정답 인물 수)
for spec in args.index:
    name, path = spec.split("=", 1)
    path = Path(path).expanduser()
    items = [json.loads(l) for l in open(path / "items.jsonl")]
    stats = json.load(open(path / "stats.json"))
    # 잇기로 합쳐진 그룹은 조각 여러 개(members) → 그중 하나라도 정답이면 적중
    persons = [[true_of.get((it["ch"], m), "?") for m in it.get("members", [it["cam_id"]])] for it in items]
    person = [ps[0] for ps in persons]  # 고유 인물 수는 대표 조각 기준
    idx = hnswlib.Index(space="ip", dim=stats["dim"])
    idx.load_index(str(path / "index.bin"), max_elements=len(items))
    idx.set_ef(max(200, kmax * 4))
    labels, _ = idx.knn_query(qv, k=min(kmax, len(items)))

    print(f"\n== {name}: 임베딩 {stats['embedded']}개, 평균 {stats['embed_ms_avg']:.1f} ms ==")
    for k in args.k:
        hits = uniq = 0
        for q, row in zip(queries, labels):
            got = [person[int(l)] for l in row[:k]]
            hits += any(p in q["gt"] for l in row[:k] for p in persons[int(l)])
            uniq += len(set(got))
        print(f"  recall@{k}: {hits / len(queries):.2f}   고유인물@{k}: {uniq / len(queries):.1f}")
    present = {p for ps in persons for p in ps}
    rows = []
    for q, row in zip(queries, labels):
        first = next((i + 1 for i, l in enumerate(row) if any(p in q["gt"] for p in persons[int(l)])), None)
        rows.append((first, sum(g in present for g in q["gt"])))
    ranks[name] = rows

if args.detail:
    names = list(ranks)
    print("\n== 질문별: 첫 정답 순위 (없으면 -) / 인덱스에 있는 정답 인물 수 ==")
    print("  " + " | ".join(f"{n:>14}" for n in names) + " | 질문")
    for i, q in enumerate(queries):
        cells = []
        for n in names:
            r, c = ranks[n][i]
            cells.append(f"{(str(r) if r else '-'):>5} ({c:>2}/{len(q['gt']):>2})")
        print("  " + " | ".join(f"{c:>14}" for c in cells) + " | " + q["q"])
