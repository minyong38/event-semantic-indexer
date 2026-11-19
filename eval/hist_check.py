#!/usr/bin/env python3
"""잇기 검증 실험: 이은 두 조각의 베스트샷 색 분포(HSV 히스토그램)가 비슷한가.
신경망 없이 몇 μs — 이벤트 기반 비용 구조를 안 깨는 외형 검사.
  python3 hist_check.py ../camsim/out_brk links_brk.jsonl
"""
import json, sys, glob
from pathlib import Path
import cv2, numpy as np

src, links = Path(sys.argv[1]), sys.argv[2]
F = {}
for f in glob.glob(str(src / "fragments_ch*.jsonl")):
    for l in open(f):
        d = json.loads(l); F[(d["ch"], d["cam_id"])] = d

def hist(path):
    img = cv2.imread(str(src.parent / path))
    if img is None: return None
    h = img.shape[0]
    out = []
    for a, b in ((0.15, 0.5), (0.55, 0.9)):  # 상의 / 하의
        p = cv2.cvtColor(img[int(h*a):int(h*b)], cv2.COLOR_BGR2HSV)
        hs = cv2.calcHist([p], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hs, hs, 1, 0, cv2.NORM_L1)
        out.append(hs)
    return out

rows = []
for l in map(json.loads, open(links)):
    a, b = F[(l["ch"], l["from"])], F[(l["ch"], l["to"])]
    ha, hb = hist(a["bestshot"]), hist(b["bestshot"])
    if ha is None or hb is None: continue
    d = max(cv2.compareHist(ha[i], hb[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(2))
    rows.append((d, a["true_id"] == b["true_id"], l))
good = [d for d, s, _ in rows if s]; bad = [d for d, s, _ in rows if not s]
print(f"링크 {len(rows)}  맞음 {len(good)} 중앙값 {np.median(good):.3f} / 틀림 {len(bad)} 중앙값 {np.median(bad):.3f}")
for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    g = sum(1 for d in good if d < th); x = sum(1 for d in bad if d < th)
    print(f"  임계 {th:.2f}: 남는 링크 {g+x}  맞음 {g} 틀림 {x}  정밀도 {g/(g+x):.3f}  재현율 {g/680:.3f}")
json.dump([dict(l, hist=d, same=s) for d, s, l in rows], open("hist_links.json", "w"))
