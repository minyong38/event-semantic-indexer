#!/usr/bin/env python3
"""정답 검수용: 질문마다 정답으로 적힌 인물 사진을 한 장에 모아 보여준다.

  python review_sheets.py queries.json ../camsim/out_raw review
  → review/q01.jpg ... (맨 위에 질문, 아래에 정답 인물 타일 + 번호)
틀린 번호는 queries.json에서 지우고, 빠진 사람은 sheets/에서 찾아 넣는다.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

qpath, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)

best = {}  # "chN-true_id" → (높이, 경로)
for f in sorted(src.glob("fragments_ch*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        shot = Path(d["bestshot"])
        if not shot.is_absolute():
            shot = src.parent / shot
        img = cv2.imread(str(shot))
        if img is None:
            continue
        key = f"ch{d['ch']}-{d['true_id']}"
        if key not in best or img.shape[0] > best[key][0]:
            best[key] = (img.shape[0], str(shot))

W, H, COLS = 110, 220, 10
for i, q in enumerate(json.load(open(qpath)), 1):
    tiles = []
    for key in q["gt"]:
        if key in best:
            img = cv2.resize(cv2.imread(best[key][1]), (W, H))
        else:
            img = np.zeros((H, W, 3), np.uint8)  # 사진 없음 = 번호가 틀렸을 수 있음
            cv2.putText(img, "no img", (10, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(img, (0, 0), (W, 20), (0, 0, 0), -1)
        cv2.putText(img, key, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % COLS:
        tiles.append(np.full((H, W, 3), 40, np.uint8))
    grid = np.vstack([np.hstack(tiles[j:j + COLS]) for j in range(0, len(tiles), COLS)])
    title = np.full((36, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"q{i:02d}  {q['q']}  (gt {len(q['gt'])})", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(dst / f"q{i:02d}.jpg"), np.vstack([title, grid]))
print(f"[review] {i}개 질문 시트 → {dst}")
