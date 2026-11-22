#!/usr/bin/env python3
"""대역 카메라: 영상 → WiseAI 형식 메타데이터 + 베스트샷.

실제 카메라(PNM-C16083RVQ) 실측 명세(docs/wiseai-메타데이터-명세.md)를 재현한다.
  - 객체 프레임: 약 5Hz, ObjectId로 추적, BoundingBox/CenterOfGravity는 센서 픽셀값,
    tt:Transformation(Translate -1,1 / Scale 2/W, -2/H)으로 정규화 계수 제공
  - 일정 시간(기본 3.6초) 안 보이면 같은 사람도 새 ObjectId
  - 객체가 사라질 때 요약 프레임: bbox 0 + 옷 색상 + 베스트샷 경로

⚠️ 요약 프레임 안의 옷 색상/베스트샷 태그 구조는 실측 원본이 남아 있지 않아
   단순화한 형식이다(<tt:HumanBody>, <tt:ImageRef>). 위치·이름은 가정.

품질 손잡이(실험 변수):
  --miss-prob   메타데이터 틱마다 객체를 놓칠 확률 (검출 누락)
  --break-rate  보이는 1분당 강제 ID 끊김 횟수 (추적 끊김)
  --id-gap      이 시간 이상 안 보이면 새 ID (실측 3.6초)
  --shot-scale  베스트샷 해상도 배율 (1.0 = 원본, 0.5 = 절반)
  --realtime    실시간: 시각을 지금으로 바꾸고 영상 속도대로 내보냄 (줄마다 바로 씀 → 검색 서버 --live가 따라 읽음)

출력 (out_dir):
  meta_ch{N}.xmls       한 줄에 MetadataStream 문서 1개 (파서에 그대로 넣을 수 있음)
  bestshots/ch{N}/      베스트샷 JPEG
  fragments_ch{N}.jsonl 조각별 정답: 진짜 트랙 ID, 시작/끝, 베스트샷 경로 (평가용)

사용 예:
  python fake_wiseai_cam.py video.mp4 --ch 0 --out out/ --miss-prob 0.1 --break-rate 1
"""
import argparse
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

# ── 옷 색상 (HSV 중앙값 → 이름) ───────────────────────────────────
def color_name(bgr_patch: np.ndarray) -> str:
    if bgr_patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = np.median(hsv, axis=0)
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 190 else "gray"
    h2 = h * 2  # OpenCV H는 0~179
    if h2 < 15 or h2 >= 345: return "red"
    if h2 < 40: return "orange" if v > 120 else "brown"
    if h2 < 70: return "yellow"
    if h2 < 165: return "green"
    if h2 < 255: return "blue"
    if h2 < 290: return "purple"
    return "pink"


def clothing_colors(crop: np.ndarray) -> tuple[str, str]:
    h = crop.shape[0]
    top = crop[int(h * 0.15):int(h * 0.50)]
    bottom = crop[int(h * 0.55):int(h * 0.90)]
    return color_name(top), color_name(bottom)


# ── XML 생성 (실측 형식) ───────────────────────────────────────────
def utc_str(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def frame_doc(utc: datetime, w: int, h: int, objects_xml: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<tt:MetadataStream xmlns:tt="http://www.onvif.org/ver10/schema">'
        f'<tt:VideoAnalytics><tt:Frame UtcTime="{utc_str(utc)}">'
        '<tt:Transformation><tt:Translate x="-1.0" y="1.0"/>'
        f'<tt:Scale x="{2.0 / w:.6f}" y="{-2.0 / h:.6f}"/></tt:Transformation>'
        f"{objects_xml}"
        "</tt:Frame></tt:VideoAnalytics></tt:MetadataStream>"
    )


def object_xml(oid: int, box, conf: float) -> str:
    l, t, r, b = (int(round(v)) for v in box)
    return (
        f'<tt:Object ObjectId="{oid}"><tt:Appearance><tt:Shape>'
        f'<tt:BoundingBox left="{l}" top="{t}" right="{r}" bottom="{b}"/>'
        f'<tt:CenterOfGravity x="{(l + r) / 2:.1f}" y="{(t + b) / 2:.1f}"/>'
        "</tt:Shape><tt:Class>"
        f"<tt:ClassCandidate><tt:Type>Human</tt:Type><tt:Likelihood>{conf:.2f}</tt:Likelihood></tt:ClassCandidate>"
        f'<tt:Type Likelihood="{conf:.2f}">Human</tt:Type>'
        "</tt:Class></tt:Appearance></tt:Object>"
    )


def summary_xml(oid: int, top: str, bottom: str, image_ref: str) -> str:
    # 실측: 사라질 때 bbox가 전부 0인 마지막 프레임에 옷 색상 + 베스트샷 경로 (형식은 단순화)
    return (
        f'<tt:Object ObjectId="{oid}"><tt:Appearance><tt:Shape>'
        '<tt:BoundingBox left="0" top="0" right="0" bottom="0"/>'
        '<tt:CenterOfGravity x="0.0" y="0.0"/></tt:Shape>'
        '<tt:Class><tt:Type Likelihood="1.00">Human</tt:Type></tt:Class>'
        f'<tt:HumanBody><tt:Tops Color="{top}"/><tt:Bottoms Color="{bottom}"/></tt:HumanBody>'
        f"<tt:ImageRef>{image_ref}</tt:ImageRef>"
        "</tt:Appearance></tt:Object>"
    )


# ── 조각(카메라가 보는 트랙 하나) ──────────────────────────────────
@dataclass
class Fragment:
    cam_id: int
    true_id: int
    first_t: float
    last_t: float
    best_score: float = -1.0
    best_crop: np.ndarray | None = field(default=None, repr=False)

    def offer(self, frame, box, conf, w, h):
        l, t, r, b = (int(v) for v in box)
        l, t, r, b = max(l, 0), max(t, 0), min(r, w), min(b, h)
        if r - l < 8 or b - t < 16:
            return
        area = (r - l) * (b - t)
        edge = 0.5 if (l <= 2 or t <= 2 or r >= w - 2 or b >= h - 2) else 1.0
        score = conf * np.sqrt(area) * edge  # 크고, 확실하고, 안 잘린 것
        if score > self.best_score:
            self.best_score = score
            self.best_crop = frame[t:b, l:r].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ch", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="검출 입력 크기 (멀리 있는 작은 사람용)")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--start", default="2026-11-19T09:00:00", help="UTC 시작 시각")
    ap.add_argument("--id-gap", type=float, default=3.6)
    ap.add_argument("--miss-prob", type=float, default=0.0)
    ap.add_argument("--break-rate", type=float, default=0.0, help="보이는 1분당 강제 끊김")
    ap.add_argument("--shot-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realtime", action="store_true", help="실시간 모드 (검색 서버 --live용)")
    args = ap.parse_args()

    from ultralytics import YOLO  # 무거워서 여기서 import

    rng = random.Random(args.seed)
    out = Path(args.out)
    shot_dir = out / "bestshots" / f"ch{args.ch}"
    shot_dir.mkdir(parents=True, exist_ok=True)
    buffering = 1 if args.realtime else -1  # 실시간이면 줄 단위로 바로 씀
    meta_f = open(out / f"meta_ch{args.ch}.xmls", "w", encoding="utf-8", buffering=buffering)
    frag_f = open(out / f"fragments_ch{args.ch}.jsonl", "w", encoding="utf-8", buffering=buffering)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / args.hz))
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    if args.realtime:  # 영상 0초 = 지금
        start = datetime.now(timezone.utc)
        (out / "start.txt").write_text(f"{start.timestamp():.3f}\n")
    wall0 = time.time()
    p_break = args.break_rate / 60.0 / args.hz  # 틱당 강제 끊김 확률

    model = YOLO(args.model)
    active: dict[int, Fragment] = {}  # true_id → 현재 조각
    next_cam_id = 200
    stats = {"ticks": 0, "fragments": 0}

    def retire(frag: Fragment, t: float):
        """조각 종료 → 베스트샷 저장 + 요약 프레임 + 정답 기록."""
        nonlocal stats
        if frag.best_crop is None:
            return
        crop = frag.best_crop
        if args.shot_scale != 1.0:
            small = cv2.resize(crop, None, fx=args.shot_scale, fy=args.shot_scale)
            crop = cv2.resize(small, (crop.shape[1], crop.shape[0]))  # 정보만 깎고 크기 유지
        ts_ms = int(t * 1000)
        name = f"objectid_{frag.cam_id}_{ts_ms}.jpg"
        cv2.imwrite(str(shot_dir / name), crop)
        top, bottom = clothing_colors(frag.best_crop)
        ref = f"/download/ch{args.ch}/{name}"
        meta_f.write(frame_doc(start + timedelta(seconds=t), w, h,
                               summary_xml(frag.cam_id, top, bottom, ref)) + "\n")
        frag_f.write(json.dumps({
            "ch": args.ch, "cam_id": frag.cam_id, "true_id": frag.true_id,
            "first_t": round(frag.first_t, 3), "last_t": round(frag.last_t, 3),
            "top": top, "bottom": bottom, "bestshot": str(shot_dir / name),
        }) + "\n")
        stats["fragments"] += 1

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 추적은 매 프레임 (ID 일관성), 메타데이터는 hz 주기로만 내보냄
        res = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml",
                          imgsz=args.imgsz, verbose=False)[0]
        t = idx / fps
        idx += 1
        if (idx - 1) % step:
            continue
        stats["ticks"] += 1
        if args.realtime:  # 영상 속 시각이 될 때까지 기다렸다가 내보냄
            delay = wall0 + t - time.time()
            if delay > 0:
                time.sleep(delay)

        objs = []
        seen = set()
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                      res.boxes.id.int().cpu().tolist(),
                                      res.boxes.conf.cpu().tolist()):
                if rng.random() < args.miss_prob:
                    continue  # 검출 누락 주입
                frag = active.get(tid)
                if frag is not None and (t - frag.last_t > args.id_gap or rng.random() < p_break):
                    retire(frag, frag.last_t)  # 오래 안 보였거나 강제 끊김 → 새 ID
                    frag = None
                if frag is None:
                    frag = Fragment(next_cam_id, tid, t, t)
                    next_cam_id += 1
                    active[tid] = frag
                frag.last_t = t
                frag.offer(frame, box, conf, w, h)
                objs.append(object_xml(frag.cam_id, box, conf))
                seen.add(tid)

        # 오래 안 보인 조각은 퇴장 처리
        for tid in [k for k, f in active.items() if k not in seen and t - f.last_t > args.id_gap]:
            retire(active.pop(tid), t)

        if objs:  # 실측: 화면에 객체가 없으면 객체 프레임은 안 옴
            meta_f.write(frame_doc(start + timedelta(seconds=t), w, h, "".join(objs)) + "\n")

    if not args.realtime:  # 실시간이면 영상이 끝나도 화면에 남은 사람은 아직 안 나간 것 → 요약 안 보냄
        for frag in active.values():
            retire(frag, frag.last_t)
    meta_f.close()
    frag_f.close()
    print(f"[camsim] ch{args.ch}: {stats['ticks']}틱, 조각 {stats['fragments']}개 → {out}")


if __name__ == "__main__":
    main()
