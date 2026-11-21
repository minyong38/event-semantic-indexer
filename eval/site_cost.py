#!/usr/bin/env python3
"""장소·카메라별 임베딩 수: 이벤트(사람 1명 1번) vs 프레임(보이는 동안 1초마다).

  python site_cost.py --event ~/idx_multi --frame ~/idx_multi_frame --videos ../camsim/meva_multi
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--event", required=True)
ap.add_argument("--frame", required=True)
ap.add_argument("--videos", required=True)
args = ap.parse_args()

PAT = re.compile(r"\.(\d\d-\d\d)-\d\d\.\d\d-\d\d-\d\d\.([a-z]+)\.(G\d+)\.")
KO = {"school": "학교", "bus": "버스 정류장", "hospital": "병원", "admin": "사무동"}
videos = sorted(p.name for p in Path(args.videos).iterdir() if p.suffix in (".avi", ".mp4"))
chan = []
for v in videos:
    m = PAT.search(v)
    chan.append((m.group(2), m.group(3), m.group(1)) if m else ("?", "?", "?"))


def count(d):
    c = defaultdict(int)
    for l in open(Path(d).expanduser() / "items.jsonl"):
        if l.strip():
            c[json.loads(l)["ch"]] += 1
    return c


ev, fr = count(args.event), count(args.frame)


def ratio(e, f):
    return f"1/{f / e:.0f}" if e and f / e >= 1.5 else (f"{e / f:.1f}배" if f else "-")


print(f"{'채널':<5}{'장소':<12}{'카메라':<7}{'시작':<7}{'이벤트':>7}{'프레임':>8}{'비율':>8}")
for ch, (site, cam, t) in enumerate(chan):
    print(f"ch{ch:<3}{KO.get(site, site):<12}{cam:<7}{t:<7}{ev[ch]:>7}{fr[ch]:>8}{ratio(ev[ch], fr[ch]):>8}")

print(f"\n{'장소':<12}{'카메라·클립':>10}{'빈 클립':>8}{'이벤트':>8}{'프레임':>8}{'비율':>8}")
tot_e = tot_f = 0
for site in ("school", "bus", "hospital", "admin"):
    chs = [c for c, x in enumerate(chan) if x[0] == site]
    e, f = sum(ev[c] for c in chs), sum(fr[c] for c in chs)
    empty = sum(1 for c in chs if ev[c] == 0 and fr[c] == 0)
    tot_e, tot_f = tot_e + e, tot_f + f
    print(f"{KO[site]:<12}{len(chs):>10}{empty:>8}{e:>8}{f:>8}{ratio(e, f):>8}")
print(f"{'전체':<12}{len(chan):>10}{'':>8}{tot_e:>8}{tot_f:>8}{ratio(tot_e, tot_f):>8}")
