#!/usr/bin/env python3
"""챗봇 명령 해석 평가 — 같은 문장 46개(chat_cases.json)를 모드·모델별로 돌려 비교한다.

  모드  rule : 규칙만 (LLM 안 부름)
        auto : 규칙 먼저, 규칙이 모르는 문장만 LLM (실제 운영 방식)
        llm  : 도구 고르기를 LLM이 전부 (비교용). 카메라 단서는 어느 모드든 규칙으로만 뽑는다.
  위험 명령 가드는 모든 모드에서 켜져 있다. llm 모드에서는 '가드가 없었다면 LLM이 뭐라고 했는지'도 센다.

  python chat_eval.py --modes rule,auto,llm --models qwen3.5:4b,qwen3.5:9b
  python chat_eval.py --live      # 가짜 카메라(devsim/fake_cams.py)에 실제로 보내고 상태 변화까지 확인
"""
import argparse
import json
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get(server, path, **kw):
    url = server + path + ("?" + urllib.parse.urlencode(kw) if kw else "")
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def outcome(r):
    if r.get("refused"):
        return "refuse"
    if r.get("blocked"):
        return "blocked"
    if r.get("need_camera"):
        return "ask"
    if r.get("need_confirm"):
        return "confirm"
    if r.get("tool") == "unknown":
        return "unknown"
    return "run"


def judge(case, r):
    e = case["expect"]
    got = {"tool": r.get("tool"), "outcome": outcome(r), "camera": r.get("camera"),
           "action": r.get("action") or None, "value": r.get("value"), "site": r.get("site") or None}
    tool_ok = got["tool"] == e["tool"]
    full = tool_ok and got["outcome"] == e["outcome"]
    for k in ("camera", "action", "value", "site"):
        if k in e and got[k] != e[k]:
            full = False
    # 잘못된 제어 제안: 확인 질문까지 갔는데 대상·동작이 틀렸거나, 제어를 기대하지 않은 문장
    wrong_prop = got["outcome"] == "confirm" and r.get("tool") == "control_camera" and not (
        e["outcome"] == "confirm" and e["tool"] == "control_camera" and
        got["camera"] == e.get("camera") and got["action"] == e.get("action") and
        ("value" not in e or got["value"] == e["value"]))
    return tool_ok, full, wrong_prop, got


def run_set(server, cases, mode, model, verbose):
    rows = []
    for i, c in enumerate(cases):
        sid = f"eval-{mode}-{model or 'none'}-{i}-{int(time.time() * 1000) % 100000}"
        ctx = c.get("context")
        if ctx:
            get(server, "/chat/context", session=sid, cam=ctx.get("cam", ""), hits=ctx.get("hits", ""))
        kw = dict(q=c["q"], session=sid, mode=mode, dry=1)
        if model:
            kw["model"] = model
        r = get(server, "/chat", **kw)
        tool_ok, full, wrong, got = judge(c, r)
        raw = r.get("llm_raw") or {}
        rows.append({"case": c, "r": r, "tool_ok": tool_ok, "full": full, "wrong": wrong, "got": got,
                     "llm": bool(r.get("llm_used")), "llm_ms": r["timing"]["llm_ms"],
                     "ms": r["timing"]["total_ms"], "raw_tool": raw.get("tool")})
        if verbose and not full:
            e = c["expect"]
            print(f"    ✗ [{c['group']}] {c['q']}\n        기대 {e}\n        결과 {got}  (source={r.get('source')})")
    return rows


def summarize(label, rows):
    n = len(rows)
    full = sum(x["full"] for x in rows)
    tool = sum(x["tool_ok"] for x in rows)
    wrong = sum(x["wrong"] for x in rows)
    llm_calls = sum(x["llm"] for x in rows)
    ref = [x for x in rows if x["case"]["group"] == "refuse"]
    ref_ok = sum(x["got"]["outcome"] == "refuse" for x in ref)
    raw_pass = sum(1 for x in ref if x["raw_tool"] and x["raw_tool"] != "refuse")
    llm_grp = [x for x in rows if x["case"]["group"] == "llm"]
    ms = [x["ms"] for x in rows]
    llm_ms = [x["llm_ms"] for x in rows if x["llm"]]
    p = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else 0
    print(f"  {label:<20} 전체 {full:>2}/{n}  도구 {tool:>2}/{n}  말투 {sum(x['full'] for x in llm_grp)}/{len(llm_grp)}"
          f"  위험 거절 {ref_ok}/{len(ref)}  잘못된 제어 제안 {wrong}  LLM 호출 {llm_calls:>2}"
          f"  응답 p50 {statistics.median(ms):.0f}ms p95 {p(ms, .95):.0f}ms"
          + (f"  LLM p50 {statistics.median(llm_ms):.0f}ms" if llm_ms else "")
          + (f"  (LLM 혼자서는 위험하다고 못 알아본 명령 {raw_pass})" if any(x["raw_tool"] for x in ref) else ""))
    return {"label": label, "n": n, "full": full, "tool": tool, "llm_group": sum(x["full"] for x in llm_grp),
            "refuse_ok": ref_ok, "wrong_proposals": wrong, "llm_calls": llm_calls,
            "p50_ms": statistics.median(ms), "p95_ms": p(ms, .95),
            "llm_p50_ms": statistics.median(llm_ms) if llm_ms else 0, "raw_refuse_miss": raw_pass}


def live_check(server, cases, mode):
    """제어 문장을 실제로 보내고(확인까지) 가짜 카메라 상태가 기대대로 바뀌었는지, 다른 카메라는 그대로인지 본다."""
    devs = {d["id"]: d for d in get(server, "/devices")["cameras"]}
    state = lambda cid: json.loads(urllib.request.urlopen(
        f"http://{devs[cid]['host']}:{devs[cid]['port']}/sim/state", timeout=3).read())

    def all_cmds():
        tot = 0
        for cid in devs:
            try:
                tot += state(cid)["commands"]
            except OSError:
                pass
        return tot

    before_total = all_cmds()
    sent = ok = 0
    for i, c in enumerate(cases):
        e = c["expect"]
        if e["outcome"] != "confirm" or e["tool"] != "control_camera":
            continue
        cam = e["camera"]
        try:
            s0 = state(cam)
        except OSError:
            print(f"    - {c['q']}: {cam} 꺼져 있음 (건너뜀)")
            continue
        sid = f"live-{i}-{int(time.time() * 1000) % 100000}"
        ctx = c.get("context")
        if ctx:
            get(server, "/chat/context", session=sid, cam=ctx.get("cam", ""), hits=ctx.get("hits", ""))
        r = get(server, "/chat", q=c["q"], session=sid, mode=mode)
        if not r.get("confirm_id"):
            print(f"    ✗ {c['q']}: 확인 질문이 안 나옴 ({r.get('reply')})")
            continue
        cr = get(server, "/chat/confirm", id=r["confirm_id"], session=sid)
        sent += 1
        s1 = state(cam)
        a, v = e["action"], e.get("value", 1)
        good = (a == "preset" and s1["preset"] == v) or \
               (a == "focus" and s1["focus_count"] == s0["focus_count"] + 1) or \
               (a == "zoom_in" and s1["zoom"] == min(20, s0["zoom"] + v)) or \
               (a == "zoom_out" and s1["zoom"] == max(1, s0["zoom"] - v))
        ok += good and cr.get("ok", False)
        print(f"    {'✓' if good else '✗'} {c['q']:<28} → {cr.get('reply')}")
    stray = all_cmds() - before_total - sent
    print(f"  실제 전송 {sent}건 중 상태가 맞게 바뀐 것 {ok}건, 엉뚱한 카메라로 간 명령 {stray}건")
    return {"sent": sent, "ok": ok, "stray": stray}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8770")
    ap.add_argument("--cases", default=str(HERE / "chat_cases.json"))
    ap.add_argument("--modes", default="rule,auto,llm")
    ap.add_argument("--models", default="", help="쉼표로 여러 개 (비우면 서버 기본 모델)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default=str(HERE / "chat_eval_result.json"))
    a = ap.parse_args()
    cases = json.load(open(a.cases, encoding="utf-8"))["cases"]
    print(f"문장 {len(cases)}개 · 서버 {a.server}")
    if a.live:
        live_check(a.server, cases, "auto")
        return
    models = [m for m in a.models.split(",") if m] or [""]
    results = []
    for mode in a.modes.split(","):
        for model in ([""] if mode == "rule" else models):
            label = mode if mode == "rule" else f"{mode} {model or '(기본)'}"
            print(f"\n[{label}]")
            if mode != "rule":  # 모델을 GPU에 올리는 첫 호출은 시간 측정에서 뺀다
                get(a.server, "/chat", q="워밍업", session="warmup", mode="llm", dry=1, **({"model": model} if model else {}))
            rows = run_set(a.server, cases, mode, model, not a.quiet)
            results.append(summarize(label, rows))
    print("\n요약")
    for s in results:
        print(f"  {s['label']:<20} 전체 {s['full']}/{s['n']} · 말투 {s['llm_group']}/6 · 위험 거절 {s['refuse_ok']}/6 · "
              f"잘못된 제어 제안 {s['wrong_proposals']} · LLM 호출 {s['llm_calls']} · 응답 p50 {s['p50_ms']:.0f}ms")
    json.dump(results, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
