#!/usr/bin/env python3
"""CLIP 이미지 인코더를 GPU로 돌렸을 때 처리량 → 64채널 인덱싱이 GPU를 몇 % 쓰는가.

실제 어플라이언스는 임베딩을 GPU(가속기)에서 돌린다. 하드웨어가 바뀌어도 일의 양(프레임 vs 이벤트)은 그대로이므로,
GPU 처리량으로 두 방식의 GPU 점유율을 환산한다. 부하 수치는 control_bench 실측(체육관 채널 기준):
  프레임 방식 64ch = 19.37장/초/채널 × 64 = 1,240장/초 (초당 1번 샘플 기준, 5번이면 ×5)
  이벤트 방식 64ch = 39.5장/초 (실측 트레이스 64채널 복제)

  python gpu_clip_bench.py --sec 5
"""
import argparse
import time

import open_clip
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--sec", type=float, default=5.0, help="배치 크기별 측정 시간(초)")
args = ap.parse_args()

assert torch.cuda.is_available(), "CUDA 안 보임"
dev = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
visual = model.visual.eval().to(dev)

# 정밀도 확인: FP16 결과가 FP32와 같은 방향인가
x = torch.randn(8, 3, 224, 224, device=dev)
with torch.no_grad():
    a = torch.nn.functional.normalize(visual(x).float(), dim=-1)
    with torch.autocast("cuda", dtype=torch.float16):
        b = torch.nn.functional.normalize(visual(x).float(), dim=-1)
print(f"FP16 vs FP32 코사인 유사도 (8장 평균): {(a * b).sum(-1).mean().item():.5f}")


def bench(bs, fp16):
    x = torch.randn(bs, 3, 224, 224, device=dev)
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else torch.autocast("cuda", enabled=False)
    with torch.no_grad(), ctx:
        for _ in range(5):  # 워밍업
            visual(x)
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < args.sec:
            visual(x)
            n += bs
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
    return n / el, el / (n / bs) * 1000  # 장/초, 배치 1회 ms


torch.cuda.reset_peak_memory_stats()
rows = {}
print(f"\n{'정밀도':<6}{'배치':>5}{'장/초':>10}{'배치 1회 ms':>13}")
for fp16 in (False, True):
    for bs in (1, 8, 32, 64):
        ips, ms = bench(bs, fp16)
        rows[(fp16, bs)] = ips
        print(f"{'FP16' if fp16 else 'FP32':<6}{bs:>5}{ips:>10.0f}{ms:>13.2f}")
mem = torch.cuda.max_memory_allocated() / 2**30
print(f"\nCLIP 이미지 인코더 최대 GPU 메모리: {mem:.2f} GB")

# 64채널 GPU 점유율 환산 (FP16 기준)
FRAME_64, EVENT_64 = 19.37 * 64, 39.5
best = max(rows[(True, b)] for b in (32, 64))  # 프레임 방식: 한 화면에 여러 명 → 큰 배치 가능
single = rows[(True, 1)]                        # 이벤트 방식: 사람이 떠날 때 1장씩 (보수적으로 배치 1)
print("\n64채널 인덱싱의 GPU 점유율 (FP16)")
print(f"  프레임 방식 초당 1번: {FRAME_64:,.0f}장/초 ÷ {best:,.0f} (큰 배치) = {FRAME_64 / best:6.1%}")
print(f"  프레임 방식 초당 5번: {FRAME_64 * 5:,.0f}장/초 ÷ {best:,.0f} (큰 배치) = {FRAME_64 * 5 / best:6.1%}")
print(f"  이벤트 방식         : {EVENT_64:,.1f}장/초 ÷ {single:,.0f} (배치 1)  = {EVENT_64 / single:6.1%}")
