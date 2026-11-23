#include "http_client.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <sstream>

bool parseHostPort(const std::string& url, std::string& host, int& port) {
    std::string s = url;
    const size_t scheme = s.find("://");
    if (scheme != std::string::npos) s = s.substr(scheme + 3);
    const size_t slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos) {
        host = s;
        port = 80;
    } else {
        host = s.substr(0, colon);
        port = std::stoi(s.substr(colon + 1));
    }
    return !host.empty() && port > 0;
}

namespace {

HttpResult httpRequest(const char* method, const std::string& host, int port, const std::string& path,
                       const std::string& body, const std::string& content_type, int timeout_ms) {
    HttpResult r;
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        r.error = "주소 해석 실패: " + host;
        return r;
    }
    const int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        ::freeaddrinfo(res);
        r.error = "socket 실패";
        return r;
    }
    timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        ::freeaddrinfo(res);
        ::close(fd);
        r.error = "연결 실패: " + host + ":" + std::to_string(port);
        return r;
    }
    ::freeaddrinfo(res);

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.0\r\n"
        << "Host: " << host << "\r\n";
    if (!content_type.empty()) req << "Content-Type: " << content_type << "\r\n";
    if (!body.empty() || std::strcmp(method, "POST") == 0) req << "Content-Length: " << body.size() << "\r\n";
    req << "Connection: close\r\n\r\n";
    const std::string head = req.str();
    for (const std::string* part : {&head, &body}) {
        const char* p = part->data();
        size_t n = part->size();
        while (n > 0) {
            const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
            if (w <= 0) {
                ::close(fd);
                r.error = "전송 실패";
                return r;
            }
            p += w;
            n -= static_cast<size_t>(w);
        }
    }

    std::string resp;
    char buf[8192];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n == 0) break;
        if (n < 0) {
            ::close(fd);
            r.error = "응답 대기 시간 초과";
            return r;
        }
        resp.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);

    const size_t sp = resp.find(' ');
    const size_t hdr_end = resp.find("\r\n\r\n");
    if (sp == std::string::npos || hdr_end == std::string::npos) {
        r.error = "응답 형식 오류";
        return r;
    }
    r.code = std::atoi(resp.c_str() + sp + 1);
    r.body = resp.substr(hdr_end + 4);
    return r;
}

}  // namespace

HttpResult httpPost(const std::string& host, int port, const std::string& path, const std::string& body,
                    const std::string& content_type, int timeout_ms) {
    return httpRequest("POST", host, port, path, body, content_type, timeout_ms);
}

HttpResult httpGet(const std::string& host, int port, const std::string& path, int timeout_ms) {
    return httpRequest("GET", host, port, path, "", "", timeout_ms);
}

std::string urlEncode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}
