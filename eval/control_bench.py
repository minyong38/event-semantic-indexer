#!/usr/bin/env python3
"""관제 영향 측정: 인덱싱 부하가 같은 장비의 실시간 영상 처리(관제 화면)를 방해하는가.

  화면 대역(probe): 체육관 영상을 원래 fps로 실시간 디코딩 → 실제 fps, 프레임 간격 p99, 늦은 프레임 비율
  인덱싱 부하(load): 측정된 임베딩 발생 시각(체육관 채널)을 N채널로 복제(채널마다 랜덤 시작점)해
                     실제 시간대로 재생, 실제 베스트샷을 ONNX로 임베딩 → 검색 반영 지연(예정 시각 → 완료)
  두 방식 모두 같은 ONNX Runtime 설정(세션당 1스레드)·같은 워커 수.

  python control_bench.py --frags ../camsim/out_raw --video ../camsim/meva_raw/<G330>.avi \
      --model ../model/clip_image.onnx --event ~/idx_stB --frame ~/idx_frame --dur 60
"""
import argparse
import json
import multiprocessing as mp
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def preprocess(bgr):  # C++ embedder.cpp와 동일
    import cv2
    side = max(bgr.shape[:2])
    sq = np.full((side, side, 3), (104, 117, 123), np.uint8)
    y, x = (side - bgr.shape[0]) // 2, (side - bgr.shape[1]) // 2
    sq[y:y + bgr.shape[0], x:x + bgr.shape[1]] = bgr
    rgb = cv2.cvtColor(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]


# ── 화면 대역 ──────────────────────────────────────────────
def probe(video, dur, ready, out):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    period = 1.0 / fps
    ready.wait()
    t_start = time.perf_counter()
    shown, late, gaps, last = 0, 0, [], None
    k = 0
    while time.perf_counter() - t_start < dur:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.resize(frame, (960, 540))  # 화면 표시용 스케일 (관제 클라이언트 흉내)
        now = time.perf_counter()
        if last is not None:
            gaps.append(now - last)
        last = now
        shown += 1
        due = t_start + (k + 1) * period
        if now > due + period:  # 한 프레임 이상 밀림
            late += 1
        k += 1
        sl = due - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
    el = time.perf_counter() - t_start
    g = np.array(gaps) * 1000
    out.put({"fps_target": fps, "fps": shown / el, "gap_p50": float(np.percentile(g, 50)),
             "gap_p99": float(np.percentile(g, 99)), "gap_max": float(g.max()), "late": late / max(shown, 1)})


# ── 인덱싱 부하 ────────────────────────────────────────────
def trace_times(idx, frags, kind):
    last = {}
    for f in Path(frags).glob("fragments_ch*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            last[(d["ch"], d["cam_id"])] = d["last_t"]
    ts = []
    for l in open(Path(idx).expanduser() / "items.jsonl"):
        it = json.loads(l)
        if it["ch"] != 0:  # 붐비는 체육관 채널만
            continue
        if kind == "event":  # 조각 끝 + 잇기 대기(1초) 뒤에 확정·임베딩
            ts.append(max(last.get((0, m), 0) for m in it["members"]) + 1.0)
        else:
            ts.append(float(it["t"]))
    t0 = min(ts)
    return sorted(t - t0 for t in ts)


def load(model, inputs, schedule, workers, dur, ready, out):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]) for _ in range(workers)]
    name = sessions[0].get_inputs()[0].name
    q = queue.Queue()
    lags, done = [], [0]
    lock = threading.Lock()
    stop = threading.Event()

    def work(sess):
        k = 0
        while not stop.is_set():
            try:
                due = q.get(timeout=0.1)
            except queue.Empty:
                continue
            sess.run(None, {name: inputs[k % len(inputs)]})
            k += 1
            with lock:
                lags.append(time.perf_counter() - due)
                done[0] += 1

    ths = [threading.Thread(target=work, args=(s,), daemon=True) for s in sessions]
    for th in ths:
        th.start()
    sessions[0].run(None, {name: inputs[0]})  # 워밍업
    ready.wait()
    t_start = time.perf_counter()
    for t in schedule:
        due = t_start + t
        sl = due - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
        q.put(due)
    while time.perf_counter() - t_start < dur:
        time.sleep(0.05)
    backlog = q.qsize()
    stop.set()
    l = np.array(lags) if lags else np.zeros(1)
    out.put({"scheduled": len(schedule), "done": done[0], "backlog": backlog,
             "lag_p50": float(np.percentile(l, 50)), "lag_p95": float(np.percentile(l, 95)),
             "lag_max": float(l.max())})


def make_schedule(trace, channels, dur, seed=0):
    rng = random.Random(seed)
    period = trace[-1] + 1.0
    sched = []
    for _ in range(channels):
        off = rng.uniform(0, period)
        n = 0
        while True:  # 트레이스를 이어 붙여 dur만큼
            base = n * period - off
            if base > dur:
                break
            sched += [base + t for t in trace if 0 <= base + t < dur]
            n += 1
    return sorted(sched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frags", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--dur", type=float, default=60)
    ap.add_argument("--workers", type=int, default=4, help="인덱싱에 주는 코어 수")
    ap.add_argument("--cases", default="none,event:16,frame:16,event:64")
    args = ap.parse_args()

    import cv2
    shots = sorted((Path(args.frags) / "bestshots" / "ch0").glob("*.jpg"))[:64]
    inputs = [preprocess(cv2.imread(str(p))) for p in shots]
    traces = {"event": trace_times(args.event, args.frags, "event"),
              "frame": trace_times(args.frame, args.frags, "frame")}

    rows = []
    for case in args.cases.split(","):
        kind, ch = (case.split(":") + ["0"])[:2]
        ch = int(ch)
        ready, out_p, out_l = mp.Event(), mp.Queue(), mp.Queue()
        procs = [mp.Process(target=probe, args=(args.video, args.dur, ready, out_p))]
        sched = []
        if kind != "none":
            sched = make_schedule(traces[kind], ch, args.dur)
            procs.append(mp.Process(target=load, args=(args.model, inputs, sched, args.workers,
                                                       args.dur, ready, out_l)))
        for p in procs:
            p.start()
        time.sleep(3)  # 세션 로딩 대기
        ready.set()
        rp = out_p.get()
        rl = out_l.get() if kind != "none" else None
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        label = "부하 없음" if kind == "none" else f"{'이벤트' if kind == 'event' else '프레임'} {ch}ch"
        rows.append((label, len(sched) / args.dur, rp, rl))
        print(f"[{label}] 끝", flush=True)

    print(f"\n화면: 목표 {rows[0][2]['fps_target']:.1f}fps, 인덱싱 워커 {args.workers}개, 각 {args.dur:.0f}초")
    print(f"{'조건':<12}{'임베딩/초':>9}{'화면fps':>9}{'간격p99ms':>10}{'최대ms':>9}{'늦은프레임':>10}"
          f"{'처리/예정':>12}{'밀린수':>7}{'반영지연p50':>12}{'p95':>8}")
    for label, rate, rp, rl in rows:
        s = (f"{label:<12}{rate:>9.1f}{rp['fps']:>9.1f}{rp['gap_p99']:>10.1f}{rp['gap_max']:>9.0f}{rp['late']:>10.1%}")
        if rl:
            s += f"{rl['done']:>7}/{rl['scheduled']:<5}{rl['backlog']:>6}{rl['lag_p50']:>11.2f}s{rl['lag_p95']:>7.2f}s"
        print(s)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
