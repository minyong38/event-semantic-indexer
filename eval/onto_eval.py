#!/usr/bin/env python3
"""온톨로지(질문 해석 + 하드 필터) 효과 측정 — 정답 라벨 없이 자동 채점.

질문마다 기대하는 장소·시각을 적어 두고 두 방식으로 검색한다.
  원문: 질문 문장을 그대로 임베딩 (장소·시각도 CLIP이 알아서 해석해야 함)
  LLM : VLM 서비스(Qwen2-VL-2B)가 질문을 온톨로지 칸으로 전부 쪼갬 (parse=2)
  규칙: 장소·시각·색은 온톨로지 동의어 사전·정규식, 남은 한국어 묘사만 LLM 번역 (parse=1)
  → 장소·시각은 하드 필터, 나머지 묘사만 임베딩
지표
  해석 정확도   : LLM이 뽑은 장소·시각이 기대값과 같은 질문 비율
  장소 틀림     : 상위 10개 중 질문의 장소가 아닌 결과 수 (평균)
  시각 틀림     : 시각 조건이 있는 질문에서, 상위 10개 중 조건 밖 결과 수 (평균)
  응답 시간

  데이터(24클립): 학교 09:55~10:05, 버스·병원 10:55~11:00, 사무동 09:55~11:00 → 시각 질문은 학교·사무동으로
  (search_server를 --vlm 과 함께, 카메라 여러 장소 데이터로 띄운 상태에서)
  python onto_eval.py --index event
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://localhost:8765")
ap.add_argument("--index", default="event")
ap.add_argument("-k", type=int, default=10)
args = ap.parse_args()

# (질문, 기대 장소, 이후, 이전)
Q = [
    ("a person walking at the bus stop", "bus", None, None),
    ("a person carrying a bag at the bus stop", "bus", None, None),
    ("a person in dark clothes at the hospital", "hospital", None, None),
    ("a person standing in the hospital hallway", "hospital", None, None),
    ("a person walking in the office building", "admin", None, None),
    ("a person sitting at the school", "school", None, None),
    ("a person carrying a backpack at the school", "school", None, None),
    ("a person walking at the school after 10:00", "school", "10:00", None),
    ("a person standing at the school before 10:00", "school", None, "10:00"),
    ("a person in the office building after 10:30", "admin", "10:30", None),
    ("버스 정류장에서 가방 든 사람", "bus", None, None),
    ("병원에서 걷고 있는 사람", "hospital", None, None),
    ("학교에서 앉아 있는 사람", "school", None, None),
    ("사무동에서 걷는 사람", "admin", None, None),
    ("10시 이전에 학교에서 걷던 사람", "school", None, "10:00"),
    ("10시 반 이후 사무동에 있던 사람", "admin", "10:30", None),
]


def sec(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 3600 + int(m) * 60


def clock_sec(c):
    h, m, s = map(int, c.split(":"))
    return h * 3600 + m * 60 + s


def run(q, parse):
    url = f"{args.server}/search?index={args.index}&k={args.k}&parse={parse}&q=" + urllib.parse.quote(q)
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(url, timeout=120).read())
    return r, (time.perf_counter() - t0) * 1000


MODES = (("원문", 0), ("LLM", 2), ("규칙", 1))
rows = []
print(f"{'질문':<40}{'규칙 해석 → 검색어':<46}{'장소틀림 원문/LLM/규칙':>22}")
for q, site, after, before in Q:
    row = {"q": q, "korean": any("\uac00" <= ch <= "\ud7a3" for ch in q)}

    def wrong_site(r):
        return sum(1 for h in r["hits"] if h.get("site") != site)

    def wrong_time(r):
        if not after and not before:
            return None
        bad = 0
        for h in r["hits"]:
            t = clock_sec(h["clock"])  # 처음 보인 시각
            if after and t + (h["t_end"] - h["t"]) < sec(after):
                bad += 1
            elif before and t > sec(before):
                bad += 1
        return bad

    for name, mode in MODES:
        r, ms = run(q, mode)
        p = r.get("parsed") or {}
        row[name] = {"ok": p.get("site") == site and p.get("after") == after and p.get("before") == before,
                     "ws": wrong_site(r), "wt": wrong_time(r), "ms": ms, "text": r.get("searched_text", ""),
                     "p": p}
    rp = row["규칙"]["p"]
    desc = f"{'✓' if row['규칙']['ok'] else '✗'} {rp.get('site')}/{rp.get('after')}/{rp.get('before')} → {row['규칙']['text']}"
    llm_mark = "✓" if row["LLM"]["ok"] else "✗"
    print(f"{q[:38]:<40}{desc[:44]:<46}{row['원문']['ws']:>8} / {row['LLM']['ws']}{llm_mark} / {row['규칙']['ws']}", flush=True)
    rows.append(row)


def avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def p50(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


for title, sel in (("영어 질문", [r for r in rows if not r["korean"]]), ("한국어 질문", [r for r in rows if r["korean"]]),
                   ("전체", rows)):
    wt = [r for r in sel if r["원문"]["wt"] is not None]
    print(f"\n[{title} {len(sel)}개]        {'원문':>8}{'LLM':>8}{'규칙':>8}")
    print(f"  해석 정확도             {'-':>8}{sum(r['LLM']['ok'] for r in sel):>5}/{len(sel)}"
          f"{sum(r['규칙']['ok'] for r in sel):>5}/{len(sel)}")
    print(f"  장소 틀림(상위{args.k})     " + "".join(f"{avg([r[m]['ws'] for r in sel]):>8.1f}" for m, _ in MODES))
    if wt:
        print(f"  시각 틀림({len(wt)}개 질문)  " + "".join(f"{avg([r[m]['wt'] for r in wt]):>8.1f}" for m, _ in MODES))
    print(f"  응답 p50(ms)           " + "".join(f"{p50([r[m]['ms'] for r in sel]):>8.0f}" for m, _ in MODES))
