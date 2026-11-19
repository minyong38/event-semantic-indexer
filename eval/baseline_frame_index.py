#!/usr/bin/env python3
"""비교군 A: 프레임 기반 인덱스. 메타데이터 틱 중 초당 --hz번, 화면의 사람 크롭을 '전부' 임베딩.

이벤트 기반 인덱서와 같은 모델·같은 전처리(정사각형 패딩 → 224)·같은 크기 필터를 쓴다.
비교 목적의 측정 도구라 파이썬으로 둔다 (모델 실행은 같은 ONNX Runtime).

  python baseline_frame_index.py --meta ../camsim/out_raw --videos ../camsim/meva_raw \
      --model ../model/clip_image.onnx --out ~/idx_frame --hz 1
"""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import hnswlib
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
OBJ = re.compile(r'<tt:Object ObjectId="(\d+)">.*?<tt:BoundingBox left="(\d+)" top="(\d+)" right="(\d+)" bottom="(\d+)"')
UTC = re.compile(r'UtcTime="([^"]+)"')


def preprocess(bgr):  # C++ embedder.cpp와 동일
    side = max(bgr.shape[:2])
    sq = np.full((side, side, 3), (104, 117, 123), np.uint8)
    y, x = (side - bgr.shape[0]) // 2, (side - bgr.shape[1]) // 2
    sq[y:y + bgr.shape[0], x:x + bgr.shape[1]] = bgr
    rgb = cv2.cvtColor(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]


ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--videos", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hz", type=float, default=1.0)
ap.add_argument("--start", default="2026-11-19T09:00:00")
ap.add_argument("--min-shot-h", type=int, default=64)
args = ap.parse_args()

out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)
start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp()
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix in (".avi", ".mp4"))  # camsim과 같은 순서 = 채널 번호

vecs, items = [], []
n_embed, ms_total = 0, 0.0
for ch, vpath in enumerate(videos):
    meta = Path(args.meta) / f"meta_ch{ch}.xmls"
    if not meta.exists():
        continue
    # 샘플 틱: 초당 hz번 → {프레임시각: [(cam_id, box)]}
    ticks, next_t = {}, 0.0
    for line in open(meta):
        if "<tt:ImageRef>" in line:
            continue
        t = datetime.fromisoformat(UTC.search(line).group(1).replace("Z", "+00:00")).timestamp() - start
        if t + 1e-6 < next_t:
            continue
        next_t = t + 1.0 / args.hz
        ticks[round(t, 3)] = [(int(m[0]), tuple(map(int, m[1:]))) for m in OBJ.findall(line)]

    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = {round(t * fps): t for t in ticks}
    idx = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            t = want.pop(idx)
            for cam_id, (l, tp, r, b) in ticks[t]:
                crop = frame[max(tp, 0):b, max(l, 0):r]
                if crop.shape[0] < args.min_shot_h or crop.shape[1] < 8:
                    continue
                t0 = time.perf_counter()
                v = sess.run(None, {in_name: preprocess(crop)})[0][0]
                ms_total += (time.perf_counter() - t0) * 1000
                vecs.append(v / np.linalg.norm(v))
                items.append({"label": len(items), "ch": ch, "cam_id": cam_id, "t": t})
                n_embed += 1
        idx += 1
    print(f"[frame] ch{ch} {vpath.name}: 누적 임베딩 {n_embed}")

index = hnswlib.Index(space="ip", dim=len(vecs[0]))
index.init_index(max_elements=len(vecs), M=16, ef_construction=200)
index.add_items(np.stack(vecs), np.arange(len(vecs)))
index.save_index(str(out / "index.bin"))
with open(out / "items.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
json.dump({"embedded": n_embed, "embed_ms_avg": ms_total / max(n_embed, 1), "dim": len(vecs[0]), "hz": args.hz},
          open(out / "stats.json", "w"))
print(f"[frame] 임베딩 {n_embed}개, 평균 {ms_total / max(n_embed, 1):.2f} ms")
