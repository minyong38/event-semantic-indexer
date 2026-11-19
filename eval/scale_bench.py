#!/usr/bin/env python3
"""워커 수를 늘리면 임베딩 처리량이 그만큼 느는가 (스레드 방식 vs 프로세스 방식).

  python scale_bench.py --model ../model/clip_image.onnx --sec 10
"""
import argparse
import multiprocessing as mp
import os
import threading
import time

import numpy as np


def run_worker(model, sec, out, intra=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    name = s.get_inputs()[0].name
    s.run(None, {name: x})
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < sec:
        s.run(None, {name: x})
        n += 1
    out.put(n / (time.perf_counter() - t0))


def threads(model, w, sec):
    import queue
    q = queue.Queue()
    ths = [threading.Thread(target=run_worker, args=(model, sec, q)) for _ in range(w)]
    for t in ths: t.start()
    for t in ths: t.join()
    return sum(q.get() for _ in range(w))


def procs(model, w, sec):
    q = mp.Queue()
    ps = [mp.Process(target=run_worker, args=(model, sec, q)) for _ in range(w)]
    for p in ps: p.start()
    r = sum(q.get() for _ in range(w))
    for p in ps: p.join()
    return r


if __name__ == "__main__":
    mp.set_start_method("spawn")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sec", type=float, default=10)
    a = ap.parse_args()
    print(f"코어(os.cpu_count) = {os.cpu_count()}")
    print(f"{'워커':>4}{'스레드 장/초':>12}{'프로세스 장/초':>14}{'1장 ms(프로세스)':>18}")
    for w in (1, 2, 4, 8):
        if w > (os.cpu_count() or 1):
            break
        t = threads(a.model, w, a.sec)
        p = procs(a.model, w, a.sec)
        print(f"{w:>4}{t:>12.1f}{p:>14.1f}{1000 * w / p:>18.1f}")
