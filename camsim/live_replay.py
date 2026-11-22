#!/usr/bin/env python3
"""실시간 재생기: 대역 카메라가 미리 만든 메타데이터(meta_chN.xmls + 베스트샷)를 '지금' 시각으로 옮겨
원래 간격 그대로 다시 흘려보낸다. 검색 서버 실시간 모드(--live)가 이 폴더를 따라 읽는다.

  - 시각(UtcTime)은 '재생 시작 = 지금'으로 옮긴다. 줄 사이 간격은 원본 그대로라,
    카메라가 사람이 안 보인 지 3.6초 뒤에 요약 프레임을 보내는 것도 그대로 재현된다.
  - 요약 프레임을 쓰기 직전에 베스트샷 파일을 복사한다 (이미지가 준비된 뒤 알림이 오는 순서).
  - 원본 영상이 끝날 때 대역 카메라가 몰아서 낸 요약은 실제 카메라처럼 바꾼다:
      마지막 장면까지 보이던 사람 → 뺀다 (아직 안 나감 = 실시간이면 요약이 아직 없음)
      끝나기 직전에 사라진 사람 → 마지막으로 보인 뒤 3.6초(--id-gap)에 보낸다 (카메라가 판단하는 시간)
  - 베스트샷은 시작 전에 미리 복사해 둔다 (윈도우 드라이브 복사가 느려서 재생이 밀리지 않게).
  - 예정보다 늦게 쓴 시간(재생 지연)을 재서 끝에 알려 준다 — 서버 반영 지연과 구분하려고.
  - 여러 채널을 동시에 재생한다. 채널 번호는 원본 그대로 → 검색 서버가 영상 파일 이름으로 장소를 안다.
  - 시작할 때 dst 폴더의 이전 재생 결과(meta_ch*.xmls, bestshots/, start.txt)를 지운다.
    검색 서버는 이 재생기보다 먼저 켜 두되, 재생을 다시 할 때는 서버도 다시 켠다.

  python live_replay.py --src out_multi --dst ~/live_out [--channels all|16-23|2,4,19] [--seconds 300]
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="out_multi", help="대역 카메라 출력 폴더")
ap.add_argument("--dst", required=True, help="검색 서버 --live 폴더 (WSL 홈 아래 권장: ~/live_out)")
ap.add_argument("--channels", default="all")
ap.add_argument("--seconds", type=float, default=1e9, help="이만큼만 재생 (초)")
ap.add_argument("--orig-start", default="2026-11-19T09:00:00", help="대역 카메라를 돌릴 때의 --start (UTC)")
ap.add_argument("--lead", type=float, default=1.0, help="시작 전 대기 (초)")
ap.add_argument("--id-gap", type=float, default=3.6, help="카메라가 사라졌다고 판단하는 시간 (대역 카메라와 같게)")
args = ap.parse_args()

UTC = re.compile(r'UtcTime="([^"]+)"')
REF = re.compile(r"<tt:ImageRef>([^<]+)</tt:ImageRef>")


def epoch(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def utc_str(e):
    d = datetime.fromtimestamp(e, tz=timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


src, dst = Path(args.src), Path(args.dst).expanduser()
avail = sorted(int(p.stem[7:]) for p in src.glob("meta_ch*.xmls"))
if args.channels == "all":
    chans = avail
elif "-" in args.channels:
    a, b = map(int, args.channels.split("-"))
    chans = [c for c in avail if a <= c <= b]
else:
    chans = [int(c) for c in args.channels.split(",")]
base = datetime.fromisoformat(args.orig_start).replace(tzinfo=timezone.utc).timestamp()

# ── 재생 일정 만들기: (재생 시각, 채널, 찍을 상대 시각, 줄, 베스트샷 이름) ──
events, still_on, late_left = [], 0, 0
for ch in chans:
    lines = (src / f"meta_ch{ch}.xmls").read_text(encoding="utf-8").splitlines()
    rel = [epoch(m.group(1)) - base if (m := UTC.search(l)) else None for l in lines]
    is_sum = ["<tt:ImageRef>" in l for l in lines]
    last_frame = max((i for i in range(len(lines)) if not is_sum[i] and rel[i] is not None), default=-1)
    last_frame_t = rel[last_frame] if last_frame >= 0 else 0
    prev = 0.0
    for i, l in enumerate(lines):
        if rel[i] is None:
            continue
        stamp = rel[i]
        if is_sum[i] and i > last_frame:  # 영상이 끝나서 대역 카메라가 몰아서 낸 요약
            if rel[i] >= last_frame_t - 0.1:
                still_on += 1  # 마지막 장면까지 보이던 사람 = 아직 안 나감 → 실시간이면 요약 없음
                continue
            stamp = rel[i] + args.id_gap  # 막 사라진 사람 → 카메라라면 3.6초 뒤에 알림
            late_left += 1
        when = max(stamp, prev)  # 파일 순서 = 카메라가 낸 순서 (늦게 판단한 요약은 그때 나감)
        prev = when
        shot = REF.search(l).group(1).rsplit("/", 1)[-1] if is_sum[i] else None
        events.append((when, ch, stamp, l, shot))
events.sort(key=lambda e: (e[0], e[1]))
events = [e for e in events if e[0] <= args.seconds]

# ── 출력 폴더 준비 ──
dst.mkdir(parents=True, exist_ok=True)
for f in dst.glob("meta_ch*.xmls"):
    f.unlink()
shutil.rmtree(dst / "bestshots", ignore_errors=True)
files = {}
for ch in chans:
    (dst / "bestshots" / f"ch{ch}").mkdir(parents=True, exist_ok=True)
shots = [(ch, shot) for _, ch, _, _, shot in events if shot]
t_copy = time.time()
with ThreadPoolExecutor(16) as pool:  # 베스트샷 미리 복사 (요약 프레임보다 먼저 있어야 하고, 재생 중 복사는 느림)
    list(pool.map(lambda x: shutil.copyfile(src / "bestshots" / f"ch{x[0]}" / x[1],
                                            dst / "bestshots" / f"ch{x[0]}" / x[1]), shots))
print(f"[replay] 베스트샷 {len(shots)}장 미리 복사 ({time.time() - t_copy:.1f}초)", flush=True)
for ch in chans:
    files[ch] = open(dst / f"meta_ch{ch}.xmls", "w", encoding="utf-8", buffering=1)  # 줄마다 바로 씀

start = time.time() + args.lead
(dst / "start.txt").write_text(f"{start:.3f}\n")
n_sum = sum(1 for e in events if e[4])
print(f"[replay] 채널 {len(chans)}개, 줄 {len(events)}개 (사람 퇴장 {n_sum}개) — 영상 끝까지 보이던 {still_on}명은 뺌, "
      f"끝나기 직전 사라진 {late_left}명은 {args.id_gap}초 뒤에 보냄, 약 {events[-1][0]:.0f}초 재생 → {dst}", flush=True)

written = summaries = 0
lags = []
next_report = 10.0
for when, ch, r, line, shot in events:
    delay = start + when - time.time()
    if delay > 0:
        time.sleep(delay)
    lags.append(max(0.0, -delay))  # 예정보다 늦게 쓴 시간
    if shot:
        summaries += 1
    files[ch].write(UTC.sub(f'UtcTime="{utc_str(start + r)}"', line, count=1) + "\n")
    written += 1
    if when >= next_report:
        print(f"[replay] {when:4.0f}초: 줄 {written}개, 사람 퇴장 {summaries}개", flush=True)
        next_report += 10.0

for f in files.values():
    f.close()
lags.sort()
print(f"[replay] 끝: 줄 {written}개, 사람 퇴장 {summaries}개, 재생 지연 p50 {lags[len(lags) // 2] * 1000:.0f} ms · "
      f"p95 {lags[int(len(lags) * 0.95)] * 1000:.0f} ms · 최대 {lags[-1] * 1000:.0f} ms", flush=True)
