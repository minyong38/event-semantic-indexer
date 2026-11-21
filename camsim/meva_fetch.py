#!/usr/bin/env python3
"""MEVA 공개 데이터(AWS S3, 계정 불필요) 목록 보기 / 받기.

  목록: 그 날짜·시간대에 어떤 카메라 클립이 있는지 (카메라별 클립 수·용량)
    python meva_fetch.py list --date 2018-03-07 --hour 10
    python meva_fetch.py list --date 2018-03-07            (그 날 시간대 목록만)
  받기: 고른 카메라들의 연속 클립 N개씩
    python meva_fetch.py get --date 2018-03-07 --hour 10 --cams G299,G330 --clips 4 --out meva_multi
"""
import argparse
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
DROP = "drops-123-r13"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
# 예: 2018-03-07.10-00-00.10-05-00.school.G330.r13.avi
PAT = re.compile(r"(\d{4}-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.([a-z]+)\.(G\d+)\.r13\.avi$")


def list_keys(prefix, delimiter=None):
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60).read())
        keys += [(c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)) for c in root.findall("s:Contents", NS)]
        prefixes += [p.find("s:Prefix", NS).text for p in root.findall("s:CommonPrefixes", NS)]
        if root.find("s:IsTruncated", NS).text != "true":
            return keys, prefixes
        token = root.find("s:NextContinuationToken", NS).text


def clips(date, hour):
    keys, _ = list_keys(f"{DROP}/{date}/{hour}/")
    by_cam = defaultdict(list)
    for k, size in keys:
        m = PAT.search(k)
        if m:
            by_cam[(m.group(4), m.group(5))].append((m.group(2), k, size))
    for v in by_cam.values():
        v.sort()
    return by_cam


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
a = sub.add_parser("list")
a.add_argument("--date", required=True)
a.add_argument("--hour")
b = sub.add_parser("get")
b.add_argument("--date", required=True)
b.add_argument("--hour", required=True)
b.add_argument("--cams", required=True, help="쉼표로: G299,G330")
b.add_argument("--clips", type=int, default=4, help="카메라당 연속 클립 수 (클립 1개 = 5분)")
b.add_argument("--out", default="meva_multi")
args = ap.parse_args()

if args.cmd == "list" and not args.hour:
    _, prefixes = list_keys(f"{DROP}/{args.date}/", delimiter="/")
    print("시간대:", " ".join(p.rstrip("/").split("/")[-1] for p in prefixes))
elif args.cmd == "list":
    by_cam = clips(args.date, args.hour)
    print(f"{args.date} {args.hour}시: 카메라 {len(by_cam)}대")
    for (site, cam), v in sorted(by_cam.items()):
        print(f"  {cam:<6}{site:<10}클립 {len(v):>2}개  {sum(s for _, _, s in v) / 2**20:>7.0f} MB  "
              f"시작 {v[0][0]} ~ {v[-1][0]}")
else:
    by_cam = clips(args.date, args.hour)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = args.cams.split(",")
    for (site, cam), v in sorted(by_cam.items()):
        if cam not in want:
            continue
        for _, key, size in v[: args.clips]:
            dst = out / key.split("/")[-1]
            if dst.exists() and dst.stat().st_size == size:
                print(f"  있음 {dst.name}")
                continue
            print(f"  받는 중 {dst.name} ({size / 2**20:.0f} MB)", flush=True)
            urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", dst)
    print(f"→ {out}")
