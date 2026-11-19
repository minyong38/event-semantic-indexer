#!/usr/bin/env python3
"""정답 라벨링용 인물 목록 시트.

대역 카메라 결과(fragments_ch*.jsonl)에서 진짜 트랙(ch, true_id)마다 가장 큰 베스트샷 한 장을
골라 번호를 붙인 이미지 시트로 만든다. 시트를 보고 queries.json에 정답 번호를 적는다.

  python make_sheet.py ../camsim/out_raw sheets
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
MIN_H = 64  # 인덱서 필터와 같은 기준 — 너무 작은 사람은 라벨 대상에서 뺌

best = {}  # (ch, true_id) → (높이, 경로)
for f in sorted(src.glob("fragments_ch*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        shot = Path(d["bestshot"])
        if not shot.is_absolute():
            shot = src.parent / shot  # camsim 폴더 기준 상대경로
        d["bestshot"] = str(shot)
        img = cv2.imread(d["bestshot"])
        if img is None or img.shape[0] < MIN_H:
            continue
        key = (d["ch"], d["true_id"])
        if key not in best or img.shape[0] > best[key][0]:
            best[key] = (img.shape[0], d["bestshot"])

keys = sorted(best)
W, H, COLS, PER = 120, 240, 10, 40
for s in range(0, len(keys), PER):
    tiles = []
    for ch, tid in keys[s:s + PER]:
        img = cv2.resize(cv2.imread(best[(ch, tid)][1]), (W, H))
        cv2.rectangle(img, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(img, f"ch{ch}-{tid}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % COLS:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    out = dst / f"sheet_{s // PER:02d}.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)
print(f"[sheet] 라벨 대상 인물 {len(keys)}명")
