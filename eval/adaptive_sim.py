#!/usr/bin/env python3
"""VLM을 '애매할 때만' 부르는 규칙 비교 (시뮬레이션).

1) 수집: 질문마다 C++ search_server에 verify=20을 한 번 보내, 임베딩 상위 20명의
   [임베딩 순위, 임베딩 점수, VLM P(yes), 정답 여부]를 cache 파일에 저장 (VLM 서비스 필요, 한 번만)
2) 비교: 저장한 점수로 여러 규칙을 다시 돌려 질문당 평균 VLM 호출 수와 recall@k를 잰다 (VLM 다시 안 돌림)

규칙
  fixed N        : 상위 N명을 항상 검증 (지금 서버 = fixed 10)
  early b/τ/cap  : 임베딩 순서대로 b명씩 검증, 지금까지 P(yes) 최고가 τ 이상이면 멈춤, 최대 cap명
  skip m + ...   : 임베딩 1위-2위 점수 차가 m 이상이면 VLM 생략, 아니면 뒤 규칙
재정렬: 검증한 사람은 P(yes) 높은 순(같으면 임베딩 순), 그 뒤에 검증 안 한 사람을 임베딩 순으로.

  python adaptive_sim.py --frags ../camsim/out_raw --queries queries.json            (수집 + 비교)
  python adaptive_sim.py --frags ../camsim/out_raw --queries queries.json --no-fetch (비교만)
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--frags", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--server", default="http://localhost:8765")
ap.add_argument("--index", default="event")
ap.add_argument("--cache", default="adaptive_cache.json")
ap.add_argument("--no-fetch", action="store_true")
args = ap.parse_args()

true_of = {}
for f in Path(args.frags).glob("fragments_ch*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        true_of[(d["ch"], d["cam_id"])] = f"ch{d['ch']}-{d['true_id']}"

queries = json.load(open(args.queries))
cache = Path(args.cache)
if not args.no_fetch or not cache.exists():
    rows = []
    for i, item in enumerate(queries, 1):
        gt = set(item["gt"])
        url = f"{args.server}/search?index={args.index}&k=20&verify=20&parse=0&q=" + urllib.parse.quote(item["q"])
        r = json.loads(urllib.request.urlopen(url, timeout=180).read())
        if r.get("vlm") and "error" in r["vlm"]:
            raise SystemExit(f"VLM 오류: {r['vlm']['error']} — vlm_service와 --vlm 옵션 확인")
        cand = sorted(({"clip_rank": h["clip_rank"], "clip": h["score"], "vlm": h.get("vlm", -1.0),
                        "gt": bool({true_of.get((h["ch"], m)) for m in h["members"]} & gt)} for h in r["hits"]),
                      key=lambda c: c["clip_rank"])
        rows.append({"q": item["q"], "cand": cand})
        print(f"[{i}/{len(queries)}] {item['q']}", flush=True)
    json.dump(rows, open(cache, "w"), ensure_ascii=False)
rows = json.load(open(cache))
n = len(rows)


def rerank(cand, k):  # 앞 k명만 검증했다고 보고 순서 만들기
    ver = sorted(cand[:k], key=lambda c: (-c["vlm"], c["clip_rank"]))
    return ver + cand[k:]


def fixed(N):
    return lambda cand: N


def early(b, tau, cap):
    def f(cand):
        k = 0
        while k < min(cap, len(cand)):
            k = min(k + b, cap, len(cand))
            if max(c["vlm"] for c in cand[:k]) >= tau:
                break
        return k
    return f


def skip(m, then):
    return lambda cand: 0 if len(cand) > 1 and cand[0]["clip"] - cand[1]["clip"] >= m else then(cand)


def evaluate(policy):
    calls, firsts = 0, []
    for r in rows:
        k = policy(r["cand"])
        calls += k
        order = rerank(r["cand"], k)
        firsts.append(next((i for i, c in enumerate(order, 1) if c["gt"]), None))
    rec = [sum(1 for f in firsts if f and f <= kk) / n for kk in (1, 5, 10)]
    return calls / n, rec


policies = [(f"고정 {N}명", fixed(N)) for N in (0, 3, 5, 10, 20)]
for b in (2, 3, 5):
    for tau in (0.6, 0.7, 0.8, 0.9):
        policies.append((f"{b}명씩, yes≥{tau} 멈춤, 최대10", early(b, tau, 10)))
for tau in (0.7, 0.8):
    policies.append((f"3명씩, yes≥{tau} 멈춤, 최대20", early(3, tau, 20)))
for m in (0.01, 0.02, 0.03):
    policies.append((f"1·2위 차≥{m} 생략 + 3명씩 yes≥0.8", skip(m, early(3, 0.8, 10))))

print(f"\n질문 {n}개 ({args.index} 인덱스), 같은 VLM 점수로 규칙만 바꿔 비교")
print(f"{'규칙':<34}{'VLM 호출/질문':>12}{'@1':>7}{'@5':>7}{'@10':>7}")
for name, p in policies:
    c, (r1, r5, r10) = evaluate(p)
    print(f"{name:<34}{c:>12.1f}{r1:>7.2f}{r5:>7.2f}{r10:>7.2f}")
gaps = sorted(r["cand"][0]["clip"] - r["cand"][1]["clip"] for r in rows)
print(f"\n참고: 임베딩 1·2위 점수 차 중간값 {gaps[n // 2]:.3f}, 최대 {gaps[-1]:.3f}")
