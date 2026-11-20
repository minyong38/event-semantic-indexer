#pragma once

#include <string>

// 초경량 HTTP/1.0 POST 클라이언트 — 검색 서버가 VLM 서비스를 부를 때 쓴다.
// 요청마다 연결 1개(Connection: close), 응답은 연결이 닫힐 때까지 읽는다.
struct HttpResult {
    int code = 0;          // HTTP 상태 코드 (0 = 연결·전송 실패)
    std::string body;
    std::string error;     // 실패 사유
};

// url 예: "http://localhost:8766" — 스킴 뒤 host:port만 쓴다
bool parseHostPort(const std::string& url, std::string& host, int& port);

HttpResult httpPost(const std::string& host, int port, const std::string& path, const std::string& body,
                    const std::string& content_type, int timeout_ms);
