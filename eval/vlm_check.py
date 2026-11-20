#!/usr/bin/env python3
"""VLM 0단계: RTX 3070에서 작은 VLM이 돌아가는가, 우리 베스트샷을 맞게 판정하는가.

질문 몇 개에 대해 정답 인물(검수한 GT) 베스트샷과 아닌 인물 베스트샷을 섞어 VLM에 묻는다:
  "Does this image show <질문>? Answer yes or no."
다음 토큰의 Yes/No 확률로 P(yes)를 계산 → 정답은 높고 오답은 낮아야 한다.
측정: 1장 지연(ms), GPU 메모리(GB), 정답/오답 P(yes) 평균, 0.5 기준 정확도.

  pip install "transformers>=4.45" accelerate
  python vlm_check.py --frags ../camsim/out_raw --queries queries.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--frags", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--n", type=int, default=3, help="질문당 정답·오답 각 몇 명")
ap.add_argument("--pick", default="a person carrying white plastic bags,a person in a neon green hoodie,"
                                   "an elderly woman with white hair sitting,a woman in a red top sitting,"
                                   "a person loading bags into a car")
args = ap.parse_args()

# 인물별 가장 큰 베스트샷 (make_sheet.py와 같은 기준)
src = Path(args.frags)
best = {}
for f in sorted(src.glob("fragments_ch*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        p = src.parent / d["bestshot"]
        try:
            h = Image.open(p).height
        except Exception:
            continue
        if h < 64:
            continue
        key = f"ch{d['ch']}-{d['true_id']}"
        if key not in best or h > best[key][0]:
            best[key] = (h, p)

queries = {q["q"]: q["gt"] for q in json.load(open(args.queries))}
picks = [q for q in args.pick.split(",") if q in queries]

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
proc = AutoProcessor.from_pretrained(args.model, min_pixels=224 * 224, max_pixels=448 * 448)
print(f"모델 로드 {time.perf_counter() - t0:.1f}s, 가중치 GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GB")

tok = proc.tokenizer
yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")}
no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")}


def p_yes(img, q):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"Does this image show {q}? Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1].float()
    ly = torch.logsumexp(logits[list(yes_ids)], 0)
    ln = torch.logsumexp(logits[list(no_ids)], 0)
    return torch.sigmoid(ly - ln).item()


rng = random.Random(0)
# 워밍업
p_yes(Image.open(next(iter(best.values()))[1]).convert("RGB"), "a person")

lat, pos, neg = [], [], []
for q in picks:
    gt = [g for g in queries[q] if g in best]
    others = [k for k in best if k not in queries[q] and k.split("-")[0] == (gt[0].split("-")[0] if gt else "ch0")]
    samples = [(g, True) for g in rng.sample(gt, min(args.n, len(gt)))] + \
              [(o, False) for o in rng.sample(others, min(args.n, len(others)))]
    print(f"\n질문: {q}")
    for key, is_gt in samples:
        img = Image.open(best[key][1]).convert("RGB")
        torch.cuda.synchronize()
        t = time.perf_counter()
        p = p_yes(img, q)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
        (pos if is_gt else neg).append(p)
        mark = "O" if (p >= 0.5) == is_gt else "X"
        print(f"  {'정답' if is_gt else '오답'} {key:<9} P(yes)={p:.3f}  {mark}  ({lat[-1]:.0f} ms, 높이 {img.height}px)")

acc = (sum(p >= 0.5 for p in pos) + sum(p < 0.5 for p in neg)) / max(len(pos) + len(neg), 1)
lat.sort()
print("\n== 요약 ==")
print(f"1장 지연: 중간값 {lat[len(lat) // 2]:.0f} ms, 최대 {lat[-1]:.0f} ms")
print(f"GPU 최대 메모리: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (CLIP 0.56 GB와 합쳐 8 GB 안인지)")
print(f"P(yes) 평균: 정답 {sum(pos) / len(pos):.3f} / 오답 {sum(neg) / len(neg):.3f}")
print(f"0.5 기준 정확도: {acc:.2f} ({len(pos) + len(neg)}장)")
print(f"→ 질문당 상위 10명 검증 시 약 {lat[len(lat) // 2] * 10 / 1000:.1f}초")
