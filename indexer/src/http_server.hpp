#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

// 초경량 HTTP/1.0 서버 — 다보이조 ClipHttpServer(정적 파일 전용)를 라우팅 가능하게 일반화.
// GET만 처리, 요청마다 스레드 1개(detach) + 처리 중 개수를 세어 stop()이 기다림.
// 인증 없음, 로컬 데모 전용.
struct HttpRequest {
    std::string method, path;
    std::map<std::string, std::string> query;  // URL 디코딩된 쿼리 파라미터
};

struct HttpResponse {
    int code = 200;
    std::string content_type = "application/json; charset=utf-8";
    std::string body;
};

class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;
    HttpServer(int port, Handler handler);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    bool start();
    void stop();

private:
    void acceptLoop();
    void handleClient(int fd);

    const int port_;
    Handler handler_;
    std::atomic<int> listen_fd_{-1};
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    int inflight_ = 0;
    std::mutex inflight_mutex_;
    std::condition_variable inflight_cv_;
};
