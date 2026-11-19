#include "http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <sstream>

namespace {

std::string urlDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '+') {
            out += ' ';
        } else if (s[i] == '%' && i + 2 < s.size()) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

const char* reason(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        default: return "Internal Server Error";
    }
}

bool sendAll(int fd, const char* p, size_t n) {
    while (n > 0) {
        const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

}  // namespace

HttpServer::HttpServer(int port, Handler handler) : port_(port), handler_(std::move(handler)) {}

HttpServer::~HttpServer() { stop(); }

bool HttpServer::start() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { std::perror("[http] socket"); return false; }
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("[http] bind");
        ::close(fd);
        return false;
    }
    if (::listen(fd, 16) < 0) {
        std::perror("[http] listen");
        ::close(fd);
        return false;
    }
    listen_fd_.store(fd);
    running_ = true;
    accept_thread_ = std::thread(&HttpServer::acceptLoop, this);
    return true;
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    const int fd = listen_fd_.exchange(-1);
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);  // accept() 블로킹 해제
        ::close(fd);
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    std::unique_lock<std::mutex> lk(inflight_mutex_);
    inflight_cv_.wait(lk, [this] { return inflight_ == 0; });
}

void HttpServer::acceptLoop() {
    while (running_) {
        const int lfd = listen_fd_.load();
        if (lfd < 0) break;
        sockaddr_in peer{};
        socklen_t len = sizeof(peer);
        const int fd = ::accept(lfd, reinterpret_cast<sockaddr*>(&peer), &len);
        if (fd < 0) {
            if (running_) std::perror("[http] accept");
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(inflight_mutex_);
            ++inflight_;
        }
        std::thread([this, fd] {
            handleClient(fd);
            ::close(fd);
            std::lock_guard<std::mutex> lk(inflight_mutex_);
            if (--inflight_ == 0) inflight_cv_.notify_all();
        }).detach();
    }
}

void HttpServer::handleClient(int fd) {
    // 헤더 끝(\r\n\r\n)까지 읽기 — GET만 받으므로 본문 없음
    std::string req;
    char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos && req.size() < 16384) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return;
        req.append(buf, static_cast<size_t>(n));
    }
    HttpRequest r;
    std::string target;
    std::istringstream line(req.substr(0, req.find("\r\n")));
    line >> r.method >> target;
    const size_t q = target.find('?');
    r.path = urlDecode(target.substr(0, q));
    if (q != std::string::npos) {
        std::istringstream qs(target.substr(q + 1));
        std::string kv;
        while (std::getline(qs, kv, '&')) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) r.query[urlDecode(kv)] = "";
            else r.query[urlDecode(kv.substr(0, eq))] = urlDecode(kv.substr(eq + 1));
        }
    }

    HttpResponse res;
    if (r.method != "GET") {
        res.code = 400;
        res.body = R"({"error":"GET only"})";
    } else {
        try {
            res = handler_(r);
        } catch (const std::exception& e) {
            res.code = 500;
            res.body = std::string(R"({"error":")") + e.what() + "\"}";
        }
    }
    std::ostringstream head;
    head << "HTTP/1.0 " << res.code << " " << reason(res.code) << "\r\n"
         << "Content-Type: " << res.content_type << "\r\n"
         << "Content-Length: " << res.body.size() << "\r\n"
         << "Access-Control-Allow-Origin: *\r\n"
         << "Connection: close\r\n\r\n";
    const std::string h = head.str();
    if (sendAll(fd, h.data(), h.size())) sendAll(fd, res.body.data(), res.body.size());
}
