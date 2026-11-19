#!/usr/bin/env python3
"""C++ 토크나이저 검증: queries.json 30개 + 까다로운 문장을 open_clip과 C++(search_server --tokens)로 비교.
  python tok_check.py --bin ~/build/indexer/search_server --vocab ../model/bpe_simple_vocab_16e6.txt
"""
import argparse, json, subprocess
import open_clip
ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--vocab", required=True)
a = ap.parse_args()
tok = open_clip.get_tokenizer("ViT-B-32")
qs = [q["q"] for q in json.load(open("queries.json"))]
qs += ["A Person's  BAG, 2 boxes!!", "don't they'll we've", "neon-yellow beanie (left)", "x-ray 3D t-shirt"]
bad = 0
for q in qs:
    py = [int(t) for t in tok([q])[0].tolist() if t not in (0,)][1:-1]
    cc = [int(x) for x in subprocess.run([a.bin, "--tokens", q, "--vocab", a.vocab], capture_output=True, text=True).stdout.split()]
    ok = py == cc
    bad += not ok
    if not ok:
        print(f"다름: {q!r}\n  py {py}\n  c++ {cc}")
print(f"{len(qs)}개 중 일치 {len(qs) - bad}개")
