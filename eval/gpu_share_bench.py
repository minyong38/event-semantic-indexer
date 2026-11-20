#!/usr/bin/env python3
"""같은 GPU에서 인덱싱 부하 + VLM 검증을 동시에 돌리면 VLM 응답이 얼마나 느려지나.

실제 장비처럼 인덱서와 검색(VLM 검증)을 별도 프로세스로 같은 GPU에 올린다.
  인덱싱 부하: CLIP 이미지 인코더(FP16)가 64채널 분량을 일정 속도로 임베딩
      프레임 1Hz = 1,240장/초 (19.37/채널, control_bench 실측 × 64), 0.05초마다 모아서 배치
      프레임 5Hz = 6,198장/초
      이벤트     = 39.5장/초 (실측 트레이스 64채널), 0.5초 마이크로배치
  VLM 검증: Qwen2-VL-2B가 베스트샷 10장에 yes/no를 묻는 '질문 1개'를 계속 반복 → 질문당 지연
도착은 균일 속도로 근사(실제 이벤트는 몰림이 있음).

  python gpu_share_bench.py --frags ../camsim/out_raw --dur 40
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

CASES = [("부하 없음", 0, 0), ("이벤트 64ch", 39.5, 0.5), ("프레임 64ch 1Hz", 1239.7, 0.05),
         ("프레임 64ch 5Hz", 6198.4, 0.05)]


def index_load(rate, tick, dur, ready, out):
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    vis = model.visual.eval().cuda().half()
    buf = torch.randn(64, 3, 224, 224, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        vis(buf)
        torch.cuda.synchronize()
    ready.wait()
    t0 = time.perf_counter()
    done, lags = 0, []
    while (now := time.perf_counter()) - t0 < dur:
        due = int(rate * (now - t0)) - done  # 지금까지 도착했어야 할 장 수 - 처리한 수
        while due > 0:
            n = min(due, 64)
            with torch.no_grad():
                vis(buf[:n])
            torch.cuda.synchronize()
            done += n
            due -= n
        lags.append(int(rate * (time.perf_counter() - t0)) - done)  # 밀린 장 수
        nxt = t0 + (int((time.perf_counter() - t0) / tick) + 1) * tick
        time.sleep(max(0.0, nxt - time.perf_counter()))
    el = time.perf_counter() - t0
    out.put({"target": rate * el, "done": done, "backlog": max(lags) if lags else 0, "final": lags[-1] if lags else 0})


def vlm(frags, dur, loaded, ready, out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    name = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(name, min_pixels=224 * 224, max_pixels=448 * 448)
    shots = sorted((Path(frags) / "bestshots" / "ch0").glob("*.jpg"), key=lambda p: -Image.open(p).height)[:10]
    imgs = [Image.open(p).convert("RGB") for p in shots]
    q = "a person carrying white plastic bags"
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"Does this image show {q}? Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = [proc(text=[text], images=[im], return_tensors="pt").to("cuda") for im in imgs]

    def one_query():
        for inp in inputs:
            with torch.no_grad():
                model(**inp).logits[0, -1].float().cpu()

    one_query()  # 워밍업
    loaded.set()
    ready.wait()
    time.sleep(1.0)  # 인덱싱 부하가 먼저 자리 잡게
    t0, lat = time.perf_counter(), []
    while time.perf_counter() - t0 < dur - 1.5:
        t = time.perf_counter()
        one_query()
        lat.append(time.perf_counter() - t)
    lat.sort()
    out.put({"n": len(lat), "p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1] if len(lat) > 1 else lat[0],
             "max": lat[-1], "mem": torch.cuda.max_memory_allocated() / 2**30})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frags", required=True)
    ap.add_argument("--dur", type=float, default=40)
    args = ap.parse_args()

    rows = []
    for label, rate, tick in CASES:
        loaded, ready, ov, ol = mp.Event(), mp.Event(), mp.Queue(), mp.Queue()
        pv = mp.Process(target=vlm, args=(args.frags, args.dur, loaded, ready, ov))
        pv.start()
        loaded.wait()
        pl = None
        if rate > 0:
            pl = mp.Process(target=index_load, args=(rate, tick, args.dur, ready, ol))
            pl.start()
            time.sleep(8)  # CLIP 로드 대기
        ready.set()
        rv = ov.get()
        rl = ol.get() if pl else None
        for p in (pv, pl):
            if p:
                p.join(timeout=30)
        rows.append((label, rate, rv, rl))
        print(f"[{label}] 끝", flush=True)

    base = rows[0][2]["p50"]
    print(f"\nVLM 질문 1개 = 베스트샷 10장 검증, 각 {args.dur:.0f}초")
    print(f"{'조건':<16}{'인덱싱 목표':>10}{'처리율':>8}{'최대 밀림':>9}{'VLM p50':>9}{'p95':>8}{'느려짐':>8}{'VLM 질문수':>10}")
    for label, rate, rv, rl in rows:
        s = f"{label:<16}"
        if rl:
            s += f"{rate:>8.0f}/s{rl['done'] / rl['target']:>8.0%}{rl['backlog']:>9}"
        else:
            s += f"{'-':>10}{'-':>8}{'-':>9}"
        s += f"{rv['p50']:>8.2f}s{rv['p95']:>7.2f}s{rv['p50'] / base:>7.1f}x{rv['n']:>10}"
        print(s)
    print(f"VLM 프로세스 GPU 메모리 {rows[0][2]['mem']:.2f} GB")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
