#pragma once

#include <string>

// 한화 SUNAPI(HTTP GET /stw-cgi/...)로 카메라를 제어한다.
//  - 초점: 다보이조(sunapi_focus.cpp)가 실제 카메라에 보냈던 SimpleFocus 요청과 같은 형식
//  - 프리셋·줌: SUNAPI 형식을 따른 명령 (devsim/fake_cams.py에서만 검증)
// 인증은 생략했다(시뮬레이터). 실제 카메라는 HTTP Digest가 필요하다 — 다보이조는 libcurl CURLAUTH_DIGEST.
struct SunapiResult {
    bool ok = false;
    std::string detail;  // 실패 사유 또는 카메라 응답 첫 줄
};

SunapiResult sunapiFocus(const std::string& host, int port, int channel);
SunapiResult sunapiPreset(const std::string& host, int port, int channel, int preset);
SunapiResult sunapiZoom(const std::string& host, int port, int channel, int steps);  // +확대 / -축소
// 화질(밝기·대비·채도): imageenhancements view로 지금 값을 읽고 set으로 바꾼다 (가짜 카메라에서만 검증).
// param = Brightness | Contrast | Saturation, relative면 value는 변화량. 결과는 0~100으로 자른다.
SunapiResult sunapiImage(const std::string& host, int port, int channel, const std::string& param, int value,
                         bool relative, int* before, int* after);
SunapiResult sunapiDeviceInfo(const std::string& host, int port, int timeout_ms, std::string* model);
