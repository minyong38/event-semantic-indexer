#!/usr/bin/env python3
"""실시간 모드 측정: 재생기가 흘려보내는 동안 검색을 계속 날리고, 끝나면 반영 지연을 정리한다.

  1) 검색 서버를 실시간 모드로 켠다 (--live ~/live_out)
  2) 재생기를 켠다 (camsim/live_replay.py --dst ~/live_out)
  3) 바로 이걸 켠다:  python live_eval.py --seconds 300

재는 것
  - 검색 응답 시간: 사람이 계속 추가되는 중에도 검색이 느려지지 않는가 (1초마다 검색)
  - 반영 지연: 사람이 마지막으로 보인 뒤 검색되기까지 (서버 /live) = 카메라 요약 프레임 + 확정 + 임베딩·추가
  - 실제로 검색되는가: 마지막에 새로 들어온 사람들이 검색 결과에 실제로 나오는지 확인
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://localhost:8765")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--every", type=float, default=1.0)
args = ap.parse_args()

QUERIES = ["a person walking", "a person carrying a bag", "a person in dark clothes", "a person standing"]


def get(path, timeout=30):
    return json.loads(urllib.request.urlopen(args.server + path, timeout=timeout).read())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else float("nan")


get("/live?reset=1")  # 서버의 지연 기록을 비우고 지금부터 잰다
lat, errors, growth = [], 0, []
t0 = time.time()
i = 0
while time.time() - t0 < args.seconds:
    q = QUERIES[i % len(QUERIES)]
    i += 1
    s = time.perf_counter()
    try:
        r = get(f"/search?index=event&k=10&q={urllib.parse.quote(q)}")
        lat.append((time.perf_counter() - s) * 1000)
        n = r.get("embedded", 0)
    except Exception as e:  # noqa: BLE001
        errors += 1
        n = -1
        print(f"  검색 실패: {e}", flush=True)
    el = time.time() - t0
    if not growth or el - growth[-1][0] >= 15:
        growth.append((el, n))
        live = get("/live")
        tot = (live.get("latency_sec") or {}).get("total") or {}
        print(f"[{el:5.0f}초] 검색 가능한 사람 {n}명, 반영 지연 p50 {tot.get('p50', '-')}초, "
              f"검색 {lat[-1] if lat else 0:.0f} ms", flush=True)
    time.sleep(max(0.0, args.every - (time.perf_counter() - s)))

live = get("/live")
L = live.get("latency_sec", {})
print("\n== 실시간 모드 결과 ==")
print(f"채널 {live.get('channels')}개, 받은 줄 {live.get('lines')}개, 사람 퇴장 {live.get('summaries')}개 → "
      f"확정 {live.get('groups')}명 → 임베딩 {live.get('kept')}명 (짧음 {live.get('drop_short')}, 작음 {live.get('drop_small')})")
print("반영 지연 (초): 사람이 마지막으로 보인 뒤 → 검색 가능")
for k, name in (("camera", "카메라 판단(퇴장 알림)"), ("deliver", "알림이 서버에 읽히기까지"),
                ("stitch", "확정(잇기 대기)"), ("embed_add", "임베딩·인덱스 추가"),
                ("after_camera", "카메라 알림 → 검색 가능"), ("total", "전체")):
    v = L.get(k) or {}
    print(f"  {name:<20} p50 {v.get('p50', '-')}  p95 {v.get('p95', '-')}  최대 {v.get('max', '-')}")
print(f"검색 응답 (사람이 추가되는 중): p50 {pct(lat, 0.5):.0f} ms, p95 {pct(lat, 0.95):.0f} ms, "
      f"최대 {max(lat) if lat else 0:.0f} ms, 실패 {errors}회 / {len(lat) + errors}회")

# 마지막에 들어온 사람들이 실제로 검색 결과에 나오는가 (전체를 k로 요청해서 확인)
recent = live.get("recent", [])
n = live.get("embedded", 0)
if recent and n:
    r = get(f"/search?index=event&k={n}&q={urllib.parse.quote('a person')}", timeout=120)
    found = {h["label"] for h in r["hits"]}
    ok = sum(1 for x in recent if x["label"] in found)
    print(f"방금 들어온 {len(recent)}명 중 검색 결과에 나온 사람: {ok}명")
    for x in recent[-4:]:
        print(f"  {x['channel']}  {x['left_at']}에 나감 → {x['searchable_after_sec']}초 뒤 검색 가능")
