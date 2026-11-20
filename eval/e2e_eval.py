#!/usr/bin/env python3
"""서버 경로 평가: C++ search_server의 /search?verify=N 을 그대로 호출해 recall@k를 잰다.
(vlm_rerank.py 실험과 같은 숫자가 나오면 → 실험 결과가 실제 서버 경로에서도 재현됨)

  python e2e_eval.py --frags ../camsim/out_raw --queries queries.json --verify 0 10 20
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--frags", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--server", default="http://localhost:8765")
ap.add_argument("--verify", type=int, nargs="+", default=[0, 10, 20])
ap.add_argument("--index", nargs="+", default=["event", "frame"])
ap.add_argument("-k", type=int, nargs="+", default=[1, 5, 10])
args = ap.parse_args()

true_of = {}
for f in Path(args.frags).glob("fragments_ch*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        true_of[(d["ch"], d["cam_id"])] = f"ch{d['ch']}-{d['true_id']}"

queries = json.load(open(args.queries))
K = max(args.k)
print(f"{'인덱스':<8}{'verify':>7}" + "".join(f"{'@' + str(k):>7}" for k in args.k) +
      f"{'응답 p50':>10}{'VLM p50':>9}{'오류':>6}")
for kind in args.index:
    for v in args.verify:
        firsts, total_ms, vlm_ms, errs = [], [], [], 0
        for item in queries:
            gt = set(item["gt"])
            url = f"{args.server}/search?index={kind}&k={K}&verify={v}&q=" + urllib.parse.quote(item["q"])
            t0 = time.perf_counter()
            r = json.loads(urllib.request.urlopen(url, timeout=120).read())
            total_ms.append((time.perf_counter() - t0) * 1000)
            if v and r.get("vlm"):
                if "error" in r["vlm"]:
                    errs += 1
                else:
                    vlm_ms.append(r["vlm"]["ms"])
            first = None
            for h in r["hits"]:
                if {true_of.get((h["ch"], m)) for m in h["members"]} & gt:
                    first = h["rank"]
                    break
            firsts.append(first)
        n = len(queries)
        rec = [sum(1 for f in firsts if f and f <= k) / n for k in args.k]
        p50 = sorted(total_ms)[n // 2]
        vp = f"{sorted(vlm_ms)[len(vlm_ms) // 2]:.0f}ms" if vlm_ms else "-"
        print(f"{kind:<8}{v:>7}" + "".join(f"{x:>7.2f}" for x in rec) + f"{p50:>8.0f}ms{vp:>9}{errs:>6}", flush=True)
