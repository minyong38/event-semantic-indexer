#!/usr/bin/env python3
"""채널별 임베딩 부하 → 'CPU 1코어당 몇 채널' 환산.

인덱스 폴더마다 items.jsonl(임베딩 1건 = 1줄)과 stats.json(embed_ms_avg)을 읽는다.
  부하(코어 점유율) = 초당 임베딩 수 × 1장 시간
  1코어당 채널 수   = 1 / 부하
평균만 보면 몰릴 때를 놓치므로 10초 창 최댓값(피크)도 같이 본다.

  python capacity.py --dur 300 --index frame=~/idx_frame --index stB=~/idx_stB --index d0.5=~/idx_d05
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--index", action="append", required=True, help="이름=경로")
ap.add_argument("--dur", type=float, default=300.0, help="영상 길이(초)")
ap.add_argument("--frags", help="fragments 폴더: 이벤트 인덱스 시각을 조각의 last_t로 (인덱서 t_last 정밀도 버그 우회)")
ap.add_argument("--win", type=float, default=10.0, help="피크 창(초)")
args = ap.parse_args()

LAST = {}
if args.frags:
    for f in Path(args.frags).glob("fragments_ch*.jsonl"):
        for l in open(f):
            d = json.loads(l); LAST[(d["ch"], d["cam_id"])] = d["last_t"]

NAMES = {0: "ch0 체육관(붐빔)", 1: "ch1", 2: "ch2 실외"}

for spec in args.index:
    name, path = spec.split("=", 1)
    p = Path(path).expanduser()
    ms = json.load(open(p / "stats.json"))["embed_ms_avg"]
    per_ch, times = Counter(), defaultdict(list)
    t0 = None
    rows = [json.loads(l) for l in open(p / "items.jsonl")]
    for it in rows:
        if "members" in it and LAST:
            it["t"] = max(LAST.get((it["ch"], m), 0) for m in it["members"])
        t = it.get("t", it.get("t_last"))
        if isinstance(t, (int, float)):
            t0 = t if t0 is None else min(t0, t)
    for it in rows:
        ch = it["ch"]
        per_ch[ch] += 1
        t = it.get("t", it.get("t_last"))
        if isinstance(t, (int, float)):
            times[ch].append(t - t0)

    print(f"\n== {name}: 임베딩 {sum(per_ch.values())}개, 1장 {ms:.1f} ms ==")
    print(f"  {'채널':<16}{'임베딩':>7}{'초당':>8}{'코어 점유':>10}{'1코어당 채널':>13}{'피크 10초 점유':>15}{'피크 기준 채널':>14}")
    for ch in sorted(per_ch):
        n = per_ch[ch]
        rate = n / args.dur
        load = rate * ms / 1000
        peak = 0
        ts = sorted(times[ch])
        j = 0
        for i, t in enumerate(ts):
            while ts[j] < t - args.win:
                j += 1
            peak = max(peak, i - j + 1)
        pload = peak / args.win * ms / 1000
        print(f"  {NAMES.get(ch, f'ch{ch}'):<16}{n:>7}{rate:>8.2f}{load:>9.1%}"
              f"{(1 / load if load else float('inf')):>13.1f}{pload:>14.1%}{(1 / pload if pload else float('inf')):>14.1f}")
