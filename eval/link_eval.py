#!/usr/bin/env python3
"""조각 잇기 정확도 (정답을 아는 강제 끊김 실험용).

대역 카메라를 --break-rate로 돌리면 같은 진짜 트랙(true_id)이 여러 조각(cam_id)으로 쪼개진다.
인덱서(--stitch 1)가 만든 links.jsonl을 정답과 비교한다.
  맞게 이음   : 같은 true_id끼리 이은 링크
  잘못 이음   : 다른 true_id끼리 이은 링크  ← 치명적 (한 사람의 베스트샷이 사라질 수 있음)
  못 이음     : 이어야 했는데(같은 true_id의 연속 조각) 링크가 없는 쌍  ← 중복·비용

  python link_eval.py --frags ../camsim/out_brk --idx ~/idx_brk
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--frags", required=True)
ap.add_argument("--idx", required=True)
args = ap.parse_args()

true_of, chain = {}, defaultdict(list)
for f in Path(args.frags).glob("fragments_ch*.jsonl"):
    for line in open(f):
        d = json.loads(line)
        true_of[(d["ch"], d["cam_id"])] = d["true_id"]
        chain[(d["ch"], d["true_id"])].append((d["first_t"], d["cam_id"]))

should = set()  # 이어야 하는 (ch, 앞 조각, 뒤 조각)
for (ch, _), frs in chain.items():
    frs.sort()
    for (_, a), (_, b) in zip(frs, frs[1:]):
        should.add((ch, a, b))

links = [json.loads(l) for l in open(Path(args.idx).expanduser() / "links.jsonl")]
good = bad = 0
made = set()
for l in links:
    a, b = true_of.get((l["ch"], l["from"])), true_of.get((l["ch"], l["to"]))
    if a is None or b is None:
        continue
    made.add((l["ch"], l["from"], l["to"]))
    if a == b:
        good += 1
    else:
        bad += 1
missed = len(should - made)
total = good + bad
print(f"이어야 할 쌍 {len(should)}개 / 만든 링크 {total}개")
print(f"  맞게 이음 {good}  잘못 이음 {bad}  못 이음 {missed}")
if total:
    print(f"  정밀도(만든 링크 중 맞은 비율) {good / total:.2f}")
if should:
    print(f"  재현율(이어야 할 쌍 중 이은 비율) {len(should & made) / len(should):.2f}")
