#!/usr/bin/env python3
"""VLM 서비스 — 검색 서버(C++)가 HTTP로 부른다. 모델(Qwen2-VL-2B)은 GPU에 한 번만 올린다(약 4.2GB).

  POST /verify  {"query": "<영어 문장>", "images": ["<jpg base64>", ...]}
             → {"scores": [P(yes), ...], "ms": 모델 시간}
     P(yes): 다음 토큰이 Yes일 확률 vs No일 확률 → 0~1 (문장 생성 없이 1회 추론)
  POST /parse   {"query": "<한국어나 영어 질문>"}
             → {"parsed": {"site", "after", "before", "top_color", "bottom_color", "description"}, "ms"}
     온톨로지(config/ontology.json)의 칸으로 질문을 쪼갠다. 칸 밖의 값은 버린다(null).
  POST /translate {"text": "<한국어 묘사>"} → {"text": "<영어 묘사>", "ms"}
     검색 서버가 장소·시각을 규칙으로 떼어낸 나머지만 번역. 예시를 넣지 않아 베낄 문장이 없다.
  GET  /health

요청은 한 번에 하나씩 처리한다(GPU 1장).

  python vlm_service.py --port 8766 [--ontology ../config/ontology.json]
"""
import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
ap.add_argument("--ontology", default=str(Path(__file__).resolve().parent.parent / "config" / "ontology.json"))
args = ap.parse_args()

onto = json.load(open(args.ontology))
SITES, COLORS = list(onto["sites"]), onto["colors"]

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


PARSE_PROMPT = (
    "Convert a CCTV person-search request into JSON.\n"
    f"Sites: school, bus (bus stop), hospital, admin (office building).\n"
    f"Colors: {', '.join(COLORS)}.\n"
    "Keys:\n"
    "  site: one site name above, or null if no place is mentioned\n"
    "  after: HH:MM (24h) if the request says after / from a time, else null\n"
    "  before: HH:MM (24h) if the request says before / until a time, else null\n"
    "  top_color, bottom_color: a color above if the clothing color is stated, else null\n"
    "  description: short English description of the person's look or action, WITHOUT place or time\n"
    "Output only the JSON object.\n"
    "Example: \"a man in a red jacket at the bus stop after 10 am\" -> "
    "{\"site\": \"bus\", \"after\": \"10:00\", \"before\": null, \"top_color\": \"red\", \"bottom_color\": null, "
    "\"description\": \"a man in a red jacket\"}\n"
    "Request: ")
HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse(q):
    msgs = [{"role": "user", "content": [{"type": "text", "text": PARSE_PROMPT + q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=96, do_sample=False)
    raw = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    # 온톨로지 밖의 값은 버린다 — LLM이 무엇을 내든 서버가 받는 건 정해진 칸과 값뿐
    clean = {
        "site": d.get("site") if d.get("site") in SITES else None,
        "after": d.get("after") if isinstance(d.get("after"), str) and HHMM.match(d["after"]) else None,
        "before": d.get("before") if isinstance(d.get("before"), str) and HHMM.match(d["before"]) else None,
        "top_color": d.get("top_color") if d.get("top_color") in COLORS else None,
        "bottom_color": d.get("bottom_color") if d.get("bottom_color") in COLORS else None,
        "description": d.get("description").strip() if isinstance(d.get("description"), str) and d["description"].strip() else None,
    }
    return clean, raw


TR_PROMPT = ("Translate this Korean description of a person in a CCTV video into a short English phrase. "
             "Keep only what is written; do not add clothing, colors or details. Output only the phrase.\n")


def translate(t):
    msgs = [{"role": "user", "content": [{"type": "text", "text": TR_PROMPT + t}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"').strip(".")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"ok": True, "model": args.model, "ontology": onto.get("version")})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/translate":
                with gpu:
                    t0 = time.perf_counter()
                    en = translate(req["text"])
                    ms = (time.perf_counter() - t0) * 1000
                print(f"[translate] \"{req['text']}\" → \"{en}\" {ms:.0f} ms", flush=True)
                return self.reply(200, {"text": en, "ms": ms})
            q = req["query"]
            if self.path == "/verify":
                imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req["images"]]
                with gpu:
                    t0 = time.perf_counter()
                    scores = [p_yes(im, q) for im in imgs]
                    ms = (time.perf_counter() - t0) * 1000
                print(f"[verify] \"{q}\" {len(imgs)}장 {ms:.0f} ms", flush=True)
                return self.reply(200, {"scores": scores, "ms": ms})
            if self.path == "/parse":
                with gpu:
                    t0 = time.perf_counter()
                    parsed, raw = parse(q)
                    ms = (time.perf_counter() - t0) * 1000
                print(f"[parse] \"{q}\" → {json.dumps(parsed, ensure_ascii=False)} {ms:.0f} ms", flush=True)
                return self.reply(200, {"parsed": parsed, "raw": raw, "ms": ms})
            self.reply(404, {"error": "not found"})
        except Exception as e:
            self.reply(500, {"error": str(e)})

    def log_message(self, *a):
        pass


print(f"[vlm] {args.model} 로드 완료, GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f} GB, "
      f"온톨로지 v{onto.get('version')} (장소 {len(SITES)}, 색 {len(COLORS)}), http://localhost:{args.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
