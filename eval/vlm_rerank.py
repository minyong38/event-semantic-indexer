#!/usr/bin/env python3
"""VLM 1단계: 임베딩 검색 상위 N명을 VLM으로 재정렬하면 정확도가 얼마나 오르나.

후보는 C++ search_server(HTTP)에서 받는다 → 파이썬은 VLM 판정만 (실험용).
질문마다 상위 depth명 각각에 "Does this image show <질문>?" P(yes) → P(yes) 순으로 재정렬.
지표는 evaluate.py와 같다: recall@k = 상위 k개 안에 정답 인물이 한 명이라도 있는 질문 비율
(이벤트 결과는 묶인 조각 중 하나라도 정답이면 맞음).

  (다른 창에서 search_server 실행 중이어야 함)
  python vlm_rerank.py --frags ../camsim/out_raw --queries queries.json --depth 10 20
"""
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--frags", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--server", default="http://localhost:8765")
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--depth", type=int, nargs="+", default=[10, 20], help="VLM에 보낼 후보 수")
ap.add_argument("-k", type=int, nargs="+", default=[1, 5, 10])
args = ap.parse_args()

true_of = {}
for f in Path(args.frags).glob("fragments_ch*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        true_of[(d["ch"], d["cam_id"])] = f"ch{d['ch']}-{d['true_id']}"


def persons(h):
    return {true_of.get((h["ch"], m)) for m in h["members"]}


def get(path):
    with urllib.request.urlopen(args.server + path, timeout=60) as r:
        return r.read()


model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
proc = AutoProcessor.from_pretrained(args.model, min_pixels=224 * 224, max_pixels=448 * 448)
tok = proc.tokenizer
yes_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
no_ids = list({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})


def p_yes(img, q):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"Does this image show {q}? Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    return torch.sigmoid(torch.logsumexp(lg[yes_ids], 0) - torch.logsumexp(lg[no_ids], 0)).item()


queries = json.load(open(args.queries))
D = max(args.depth)
res = {}  # (index, 방식) → [질문별 정답 첫 순위 or None]
calls, vlm_ms = {"event": 0, "frame": 0}, {"event": 0.0, "frame": 0.0}

for qi, item in enumerate(queries, 1):
    q, gt = item["q"], set(item["gt"])
    for kind in ("event", "frame"):
        r = json.loads(get(f"/search?index={kind}&k={D}&q=" + urllib.parse.quote(q)))
        hits = r["hits"]
        ps = []
        for h in hits:
            img = Image.open(io.BytesIO(get(h["img"]))).convert("RGB")
            torch.cuda.synchronize()
            t = time.perf_counter()
            ps.append(p_yes(img, q))
            torch.cuda.synchronize()
            vlm_ms[kind] += (time.perf_counter() - t) * 1000
            calls[kind] += 1

        def first_hit(order):
            for rank, i in enumerate(order, 1):
                if persons(hits[i]) & gt:
                    return rank
            return None

        res.setdefault((kind, "임베딩만"), []).append(first_hit(range(len(hits))))
        for d in args.depth:
            top = list(range(min(d, len(hits))))
            order = sorted(top, key=lambda i: (-ps[i], i))  # P(yes) 높은 순, 같으면 원래 순위
            res.setdefault((kind, f"VLM 상위{d}"), []).append(first_hit(order))
    print(f"[{qi}/{len(queries)}] {q}", flush=True)

n = len(queries)
print(f"\n질문 {n}개, VLM {args.model.split('/')[-1]}")
print(f"{'인덱스':<8}{'방식':<12}" + "".join(f"{'@' + str(k):>8}" for k in args.k) + f"{'VLM 호출/질문':>14}")
for kind in ("event", "frame"):
    for name in ["임베딩만"] + [f"VLM 상위{d}" for d in args.depth]:
        firsts = res[(kind, name)]
        rec = [sum(1 for f in firsts if f and f <= k) / n for k in args.k]
        c = 0 if name == "임베딩만" else int(name.replace("VLM 상위", ""))
        print(f"{kind:<8}{name:<12}" + "".join(f"{v:>8.2f}" for v in rec) + f"{c:>14}")
for kind in ("event", "frame"):
    print(f"VLM 1장 평균 {vlm_ms[kind] / max(calls[kind], 1):.0f} ms ({kind})")
