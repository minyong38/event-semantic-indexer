#include "sunapi.hpp"

#include "http_client.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace {

std::atomic<int> g_seq{1};

std::string firstLine(const std::string& s) {
    const size_t e = s.find_first_of("\r\n");
    return e == std::string::npos ? s : s.substr(0, e);
}

// 다보이조와 같은 판정: HTTP 200이고 본문에 오류 표시가 없으면 성공
SunapiResult call(const std::string& host, int port, const std::string& path, int timeout_ms) {
    SunapiResult r;
    const HttpResult h = httpGet(host, port, path, timeout_ms);
    if (h.code == 0) {
        r.detail = h.error.empty() ? "카메라 응답 없음" : h.error;
        return r;
    }
    if (h.code != 200) {
        r.detail = "HTTP " + std::to_string(h.code) + ": " + firstLine(h.body);
        return r;
    }
    const std::string& b = h.body;
    if (b.rfind("NG", 0) == 0 || b.find("Error") != std::string::npos || b.find("error") != std::string::npos ||
        b.find("Fail") != std::string::npos || b.find("fail") != std::string::npos) {
        const size_t d = b.find("Error Details:");
        r.detail = d != std::string::npos ? firstLine(b.substr(d + 14)) : firstLine(b);
        while (!r.detail.empty() && r.detail[0] == ' ') r.detail.erase(0, 1);
        return r;
    }
    r.ok = true;
    r.detail = firstLine(b);
    return r;
}

std::string seq() { return "&SunapiSeqId=" + std::to_string(g_seq.fetch_add(1)); }

}  // namespace

SunapiResult sunapiFocus(const std::string& host, int port, int channel) {
    return call(host, port,
                "/stw-cgi/image.cgi?msubmenu=focus&action=control&Channel=" + std::to_string(channel) +
                    "&Mode=SimpleFocus" + seq(),
                3000);
}

SunapiResult sunapiPreset(const std::string& host, int port, int channel, int preset) {
    return call(host, port,
                "/stw-cgi/ptzcontrol.cgi?msubmenu=preset&action=control&Channel=" + std::to_string(channel) +
                    "&Preset=" + std::to_string(preset) + seq(),
                3000);
}

SunapiResult sunapiZoom(const std::string& host, int port, int channel, int steps) {
    return call(host, port,
                "/stw-cgi/ptzcontrol.cgi?msubmenu=relative&action=control&Channel=" + std::to_string(channel) +
                    "&Zoom=" + std::to_string(steps) + seq(),
                3000);
}

SunapiResult sunapiImage(const std::string& host, int port, int channel, const std::string& param, int value,
                         bool relative, int* before, int* after) {
    const std::string ch = std::to_string(channel);
    SunapiResult r;
    const HttpResult h = httpGet(host, port, "/stw-cgi/image.cgi?msubmenu=imageenhancements&action=view&Channel=" + ch, 3000);
    if (h.code != 200) {
        r.detail = h.code == 0 ? (h.error.empty() ? "카메라 응답 없음" : h.error) : "HTTP " + std::to_string(h.code);
        return r;
    }
    const std::string key = param + "=";
    const size_t p = h.body.find(key);
    if (p == std::string::npos) {
        r.detail = "지금 " + param + " 값을 읽지 못함";
        return r;
    }
    const int cur = std::atoi(h.body.c_str() + p + key.size());
    const int next = std::clamp(relative ? cur + value : value, 0, 100);
    r = call(host, port,
             "/stw-cgi/image.cgi?msubmenu=imageenhancements&action=set&Channel=" + ch + "&" + param + "=" +
                 std::to_string(next) + seq(),
             3000);
    if (before) *before = cur;
    if (after) *after = next;
    return r;
}

SunapiResult sunapiDeviceInfo(const std::string& host, int port, int timeout_ms, std::string* model) {
    const HttpResult h = httpGet(host, port, "/stw-cgi/system.cgi?msubmenu=deviceinfo&action=view", timeout_ms);
    SunapiResult r;
    if (h.code != 200) {
        r.detail = h.code == 0 ? (h.error.empty() ? "응답 없음" : h.error) : "HTTP " + std::to_string(h.code);
        return r;
    }
    r.ok = true;
    const size_t m = h.body.find("Model=");
    if (model && m != std::string::npos) *model = firstLine(h.body.substr(m + 6));
    return r;
}
