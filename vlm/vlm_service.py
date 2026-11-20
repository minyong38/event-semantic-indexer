#!/usr/bin/env python3
"""VLM 검증 서비스 — 검색 서버(C++)가 상위 N명만 보내 "질문에 맞나?"를 묻는다.

  POST /verify  {"query": "<영어 문장>", "images": ["<jpg base64>", ...]}
             → {"scores": [P(yes), ...], "ms": 모델 시간}
  GET  /health

모델은 GPU에 한 번만 올려두고(약 4.2GB), 요청은 한 번에 하나씩 처리한다(GPU 1장).
P(yes): 다음 토큰이 Yes일 확률 vs No일 확률 → 0~1 (문장 생성 없이 1회 추론).

  python vlm_service.py --port 8766
"""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
args = ap.parse_args()

model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda").eval()
proc = AutoProcessor.from_pretrained(args.model, min_pixels=224 * 224, max_pixels=448 * 448)
tok = proc.tokenizer
YES = list({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
NO = list({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})
gpu = threading.Lock()


def p_yes(img, q):
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": f"Does this image show {q}? Answer yes or no."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        lg = model(**inp).logits[0, -1].float()
    return torch.sigmoid(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)).item()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"ok": True, "model": args.model})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/verify":
            return self.reply(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            q = req["query"]
            imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req["images"]]
            with gpu:
                t0 = time.perf_counter()
                scores = [p_yes(im, q) for im in imgs]
                ms = (time.perf_counter() - t0) * 1000
            print(f"[vlm] \"{q}\" {len(imgs)}장 {ms:.0f} ms", flush=True)
            self.reply(200, {"scores": scores, "ms": ms})
        except Exception as e:
            self.reply(500, {"error": str(e)})

    def log_message(self, *a):
        pass


print(f"[vlm] {args.model} 로드 완료, GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GB, "
      f"http://localhost:{args.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
