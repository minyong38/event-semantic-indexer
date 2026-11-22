#!/usr/bin/env python3
"""서버 경로 평가: C++ search_server의 /search 를 그대로 호출해 recall@k와 VLM 호출 수를 잰다.

--verify 항목 형식
  0            VLM 안 씀
  10           상위 10명 고정 검증
  10:3:0.7     최대 10명, 3명씩 검증하다 P(yes) ≥ 0.7이면 멈춤 (애매할 때만 더 부름)

  python e2e_eval.py --frags ../camsim/out_raw --queries queries.json --verify 0 10 10:3:0.7 20:3:0.7
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
ap.add_argument("--verify", nargs="+", default=["0", "10", "10:3:0.7", "20:3:0.7"])
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
print(f"{'인덱스':<8}{'검증 규칙':<12}" + "".join(f"{'@' + str(k):>7}" for k in args.k) +
      f"{'VLM 호출':>9}{'응답 p50':>10}{'오류':>6}")
for kind in args.index:
    for spec in args.verify:
        parts = spec.split(":")
        q_extra = f"&verify={parts[0]}"
        if len(parts) == 3:
            q_extra += f"&step={parts[1]}&stop={parts[2]}"
        firsts, total_ms, calls, errs = [], [], 0, 0
        for item in queries:
            gt = set(item["gt"])
            url = f"{args.server}/search?index={kind}&k={K}{q_extra}&parse=0&q=" + urllib.parse.quote(item["q"])
            t0 = time.perf_counter()
            r = json.loads(urllib.request.urlopen(url, timeout=180).read())
            total_ms.append((time.perf_counter() - t0) * 1000)
            v = r.get("vlm")
            if v:
                if "error" in v:
                    errs += 1
                calls += v.get("calls", 0)
            first = None
            for h in r["hits"]:
                if {true_of.get((h["ch"], m)) for m in h["members"]} & gt:
                    first = h["rank"]
                    break
            firsts.append(first)
        n = len(queries)
        rec = [sum(1 for f in firsts if f and f <= k) / n for k in args.k]
        p50 = sorted(total_ms)[n // 2]
        label = "없음" if spec == "0" else (f"고정 {parts[0]}" if len(parts) == 1 else f"{parts[1]}씩≥{parts[2]}/{parts[0]}")
        print(f"{kind:<8}{label:<12}" + "".join(f"{x:>7.2f}" for x in rec) +
              f"{calls / n:>9.1f}{p50:>8.0f}ms{errs:>6}", flush=True)
