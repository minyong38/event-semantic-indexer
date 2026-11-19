#!/usr/bin/env python3
"""검증용 검색 도구 (제품 경로 아님 — 질문 경로는 C++로 옮길 예정).

C++ 인덱서가 만든 index.bin에 자연어 질문을 던져 상위 k명을 보여준다.
  python query_check.py ../indexer/build/idx_event "a person carrying a box" -k 5
결과 베스트샷을 <idx>/query_<번호>.jpg 한 장으로 이어 붙여 저장한다.
"""
import argparse
import json
from pathlib import Path

import cv2
import hnswlib
import numpy as np
import onnxruntime as ort
import open_clip

ap = argparse.ArgumentParser()
ap.add_argument("idx")
ap.add_argument("query")
ap.add_argument("-k", type=int, default=5)
args = ap.parse_args()

idx = Path(args.idx)
here = Path(__file__).parent
items = [json.loads(l) for l in open(idx / "items.jsonl")]
stats = json.load(open(idx / "stats.json"))

tok = open_clip.get_tokenizer("ViT-B-32")([args.query]).numpy()
q = ort.InferenceSession(str(here / "clip_text.onnx")).run(None, {"tokens": tok})[0][0]
q = q / np.linalg.norm(q)

index = hnswlib.Index(space="ip", dim=stats["dim"])
index.load_index(str(idx / "index.bin"), max_elements=len(items))
labels, dists = index.knn_query(q, k=min(args.k, len(items)))

tiles = []
print(f'질문: "{args.query}"')
for rank, (lab, dist) in enumerate(zip(labels[0], dists[0]), 1):
    it = items[int(lab)]
    print(f"  {rank}. 유사도 {1 - dist:.3f}  ch{it['ch']} id{it['cam_id']}  {it['top']}/{it['bottom']}  {it['shot']}")
    img = cv2.imread(it["shot"])
    if img is not None:
        tiles.append(cv2.resize(img, (96, 192)))
if tiles:
    n = len(list(idx.glob("query_*.jpg")))
    cv2.imwrite(str(idx / f"query_{n}.jpg"), np.hstack(tiles))
    print(f"  → {idx / f'query_{n}.jpg'}")
