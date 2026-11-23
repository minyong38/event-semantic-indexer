#!/usr/bin/env python3
"""가짜 카메라 여러 대 — 챗봇 제어·탐색 시험용 (실제 카메라 없이).

카메라마다 HTTP 포트 하나를 열고 SUNAPI 형식 명령을 받는다. 받은 명령은 상태(줌·프리셋·초점 횟수)를
바꾸고 commands.jsonl에 남긴다. 네트워크 탐색(ONVIF WS-Discovery, UDP 3702)에도 응답해서
'등록 안 된 카메라 찾기'를 시험할 수 있다.

  초점 : GET /stw-cgi/image.cgi?msubmenu=focus&action=control&Channel=0&Mode=SimpleFocus
         ← 다보이조(sunapi_focus.cpp)가 실제 한화비전 카메라에 보냈던 명령과 같은 형식
  프리셋: GET /stw-cgi/ptzcontrol.cgi?msubmenu=preset&action=control&Channel=0&Preset=N
  줌   : GET /stw-cgi/ptzcontrol.cgi?msubmenu=relative&action=control&Channel=0&Zoom=±N
         ← 프리셋·줌은 SUNAPI 형식을 흉내 낸 것 (실제 카메라에서는 검증 안 함)
  정보 : GET /stw-cgi/system.cgi?msubmenu=deviceinfo&action=view
  상태 : GET /sim/state   (시뮬레이터 전용 — 평가 스크립트가 '정말 바뀌었나' 확인용)

  python fake_cams.py                      # 등록 카메라 12대 + 미등록 2대
  python fake_cams.py --offline G330       # G330은 꺼진 카메라(연결 거부)
"""
import argparse
import json
import socket
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent

# 24클립에 나오는 카메라 12대 (장소는 클립 파일 이름 기준) + 아직 등록 안 된 새 카메라 2대
CAMERAS = [
    # id,    site,       ptz,   model
    ("G299", "school",   False, "SIM-FIXED"),
    ("G328", "school",   False, "SIM-FIXED"),
    ("G330", "school",   False, "SIM-FIXED"),
    ("G340", "bus",      False, "SIM-FIXED"),
    ("G505", "bus",      True,  "SIM-PTZ"),
    ("G506", "bus",      True,  "SIM-PTZ"),
    ("G509", "bus",      False, "SIM-FIXED"),
    ("G301", "hospital", False, "SIM-FIXED"),
    ("G341", "hospital", True,  "SIM-PTZ"),
    ("G436", "hospital", False, "SIM-FIXED"),
    ("G326", "admin",    False, "SIM-FIXED"),
    ("G329", "admin",    False, "SIM-FIXED"),
    ("NEW-1", "bus",     True,  "SIM-PTZ"),     # 등록 안 됨 (config/devices.json에 없음)
    ("NEW-2", "school",  False, "SIM-FIXED"),   # 등록 안 됨
]
PRESETS = 4          # PTZ 카메라 프리셋 1~4
ZOOM_MIN, ZOOM_MAX = 1, 20

WSD_ADDR, WSD_PORT = "239.255.255.250", 3702

log_lock = threading.Lock()


class Cam:
    def __init__(self, cid, site, ptz, model, port):
        self.id, self.site, self.ptz, self.model, self.port = cid, site, ptz, model, port
        self.uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sim-{cid}"))
        self.zoom = 1
        self.preset = 1 if ptz else None
        self.focus_count = 0
        self.commands = 0
        self.lock = threading.Lock()

    def state(self):
        with self.lock:
            return {"id": self.id, "site": self.site, "ptz": self.ptz, "model": self.model,
                    "zoom": self.zoom, "preset": self.preset, "focus_count": self.focus_count,
                    "commands": self.commands}


def ok():
    return 200, "OK\r\n"


def ng(code, detail):
    # SUNAPI 오류 응답 흉내 — 본문에 Error가 들어가 다보이조 판정 로직(본문에 Error면 실패)과도 맞는다
    return 200, f"NG\r\nError Code: {code}\r\nError Details: {detail}\r\n"


def handle(cam, path, q):
    """SUNAPI 형식 요청 1개 처리 → (HTTP 코드, 본문)"""
    g = lambda k, d="": q.get(k, [d])[0]
    if path == "/stw-cgi/system.cgi" and g("msubmenu") == "deviceinfo" and g("action") == "view":
        return 200, (f"Model={cam.model}\r\nSerialNumber=SIM-{cam.id}\r\nFirmwareVersion=sim-1.0\r\n"
                     f"DeviceName={cam.id}\r\n")
    if path == "/stw-cgi/image.cgi" and g("msubmenu") == "focus" and g("action") == "control":
        if g("Mode") != "SimpleFocus":
            return ng(600, "Invalid Mode")
        with cam.lock:
            cam.focus_count += 1
        return ok()
    if path == "/stw-cgi/ptzcontrol.cgi" and g("action") == "control":
        sub = g("msubmenu")
        if sub == "preset":
            if not cam.ptz:
                return ng(608, "Not Supported (fixed camera)")
            try:
                n = int(g("Preset"))
            except ValueError:
                return ng(600, "Invalid Preset")
            if not 1 <= n <= PRESETS:
                return ng(602, f"Preset out of range (1~{PRESETS})")
            with cam.lock:
                cam.preset = n
            return ok()
        if sub == "relative":
            try:
                dz = int(g("Zoom"))
            except ValueError:
                return ng(600, "Invalid Zoom")
            with cam.lock:
                cam.zoom = max(ZOOM_MIN, min(ZOOM_MAX, cam.zoom + dz))
            return ok()
    return 404, "NG\r\nError Code: 404\r\nError Details: Unknown command\r\n"


def make_handler(cam, log_path):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/sim/state":
                body = json.dumps(cam.state(), ensure_ascii=False)
                self._send(200, body, "application/json; charset=utf-8")
                return
            code, body = handle(cam, u.path, q)
            if u.path.startswith("/stw-cgi/") and "view" not in q.get("action", [""]):
                with cam.lock:
                    cam.commands += 1
                rec = {"t": round(time.time(), 3), "cam": cam.id, "path": u.path,
                       "query": {k: v[0] for k, v in q.items()}, "result": body.split("\r\n")[0]}
                with log_lock, open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{cam.id}] {u.path}?{u.query} → {rec['result']}", flush=True)
            self._send(code, body, "text/plain; charset=utf-8")

        def _send(self, code, body, ctype):
            b = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass
    return H


def probe_match(cam, host, relates_to):
    scopes = " ".join([
        "onvif://www.onvif.org/type/video_encoder",
        f"onvif://www.onvif.org/name/{cam.id}",
        f"onvif://www.onvif.org/hardware/{cam.model}",
        f"onvif://www.onvif.org/location/{cam.site}",
    ])
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"'
        ' xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
        ' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
        ' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        "<e:Header>"
        f"<w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>"
        f"<w:RelatesTo>{relates_to}</w:RelatesTo>"
        "<w:To>http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous</w:To>"
        "<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</w:Action>"
        "</e:Header><e:Body><d:ProbeMatches><d:ProbeMatch>"
        f"<w:EndpointReference><w:Address>urn:uuid:{cam.uuid}</w:Address></w:EndpointReference>"
        "<d:Types>dn:NetworkVideoTransmitter</d:Types>"
        f"<d:Scopes>{scopes}</d:Scopes>"
        f"<d:XAddrs>http://{host}:{cam.port}/onvif/device_service</d:XAddrs>"
        "<d:MetadataVersion>1</d:MetadataVersion>"
        "</d:ProbeMatch></d:ProbeMatches></e:Body></e:Envelope>"
    ).encode("utf-8")


def wsd_responder(cams, host):
    """WS-Discovery: Probe를 받으면 켜져 있는 카메라마다 ProbeMatch 1개씩 돌려준다.
    멀티캐스트(239.255.255.250:3702)와 같은 포트로 오는 유니캐스트 둘 다 받는다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError:
            pass
    s.bind(("", WSD_PORT))
    try:
        mreq = struct.pack("4s4s", socket.inet_aton(WSD_ADDR), socket.inet_aton("0.0.0.0"))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        print(f"[wsd] 멀티캐스트 {WSD_ADDR}:{WSD_PORT} 대기", flush=True)
    except OSError as e:
        print(f"[wsd] 멀티캐스트 가입 실패({e}) — 유니캐스트 {WSD_PORT}만 응답", flush=True)
    while True:
        data, addr = s.recvfrom(65535)
        text = data.decode("utf-8", "replace")
        if "Probe" not in text or "ProbeMatches" in text:
            continue
        mid = ""
        a = text.find("MessageID>")
        if a >= 0:
            b = text.find("<", a)
            mid = text[a + len("MessageID>"):b].strip()
        for cam in cams:
            s.sendto(probe_match(cam, host, mid), addr)
        print(f"[wsd] Probe from {addr[0]}:{addr[1]} → {len(cams)}대 응답", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="XAddrs에 적을 주소")
    ap.add_argument("--base-port", type=int, default=9101)
    ap.add_argument("--offline", default="", help="꺼진 카메라 id (쉼표로 여러 개)")
    ap.add_argument("--log", default=str(ROOT / "out" / "commands.jsonl"))
    a = ap.parse_args()

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    offline = {x.strip() for x in a.offline.split(",") if x.strip()}
    online = []
    for i, (cid, site, ptz, model) in enumerate(CAMERAS):
        cam = Cam(cid, site, ptz, model, a.base_port + i)
        if cid in offline:
            print(f"[{cid}] 꺼짐 (포트 {cam.port} 안 염)", flush=True)
            continue
        srv = ThreadingHTTPServer(("0.0.0.0", cam.port), make_handler(cam, a.log))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        online.append(cam)
        print(f"[{cid}] {site:<8} {'PTZ ' if ptz else '고정'} http://{a.host}:{cam.port}", flush=True)
    threading.Thread(target=wsd_responder, args=(online, a.host), daemon=True).start()
    print(f"켜진 카메라 {len(online)}대 · 명령 기록 {a.log}  (Ctrl+C 종료)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
