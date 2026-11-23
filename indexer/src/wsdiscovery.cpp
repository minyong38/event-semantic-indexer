#include "wsdiscovery.hpp"

#include "http_client.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <sstream>

namespace {

constexpr const char* kMcastAddr = "239.255.255.250";
constexpr int kWsdPort = 3702;

std::string newUuid() {
    std::random_device rd;
    std::mt19937_64 g(rd());
    std::uniform_int_distribution<int> d(0, 15);
    const char* hex = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) s += '-';
        s += hex[d(g)];
    }
    return s;
}

// 다보이조와 같은 Probe: Types를 비워 모든 ONVIF 장비가 응답하게 한다
std::string buildProbe() {
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
           "<e:Envelope xmlns:e=\"http://www.w3.org/2003/05/soap-envelope\""
           " xmlns:w=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\""
           " xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\""
           " xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">"
           "<e:Header>"
           "<w:MessageID>uuid:" + newUuid() + "</w:MessageID>"
           "<w:To e:mustUnderstand=\"true\">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
           "<w:Action e:mustUnderstand=\"true\">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>"
           "</e:Header>"
           "<e:Body><d:Probe/></e:Body></e:Envelope>";
}

// <prefix:Local>값</prefix:Local> 의 '값' (네임스페이스 접두어는 무시, 닫는 태그는 건너뜀)
std::string tagText(const std::string& x, const std::string& local) {
    const std::string key = local + ">";
    size_t p = 0;
    while ((p = x.find(key, p)) != std::string::npos) {
        const size_t lt = x.rfind('<', p);
        const bool opening = lt != std::string::npos && lt + 1 < x.size() && x[lt + 1] != '/' &&
                             x.find('>', lt) >= p;  // '<'와 key 사이에 다른 태그가 끼지 않음
        const bool exact = lt != std::string::npos &&
                           (lt + 1 == p || x[p - 1] == ':');  // <Local> 또는 <pfx:Local>
        if (opening && exact) {
            const size_t a = p + key.size();
            const size_t b = x.find('<', a);
            return b == std::string::npos ? "" : x.substr(a, b - a);
        }
        p += key.size();
    }
    return "";
}

std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string pctDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

bool parseMatch(const std::string& x, WsdDevice* d) {
    const std::string xaddrs = trim(tagText(x, "XAddrs"));
    const std::string scopes = tagText(x, "Scopes");
    const std::string types = tagText(x, "Types");
    // ONVIF 장비만 (다보이조와 같은 기준: 스코프·주소·타입 어디든 onvif 흔적)
    const bool onvif = scopes.find("onvif") != std::string::npos || xaddrs.find("/onvif/") != std::string::npos ||
                       types.find("NetworkVideoTransmitter") != std::string::npos;
    if (!onvif || xaddrs.empty()) return false;
    std::istringstream xs(xaddrs);
    xs >> d->xaddr;
    if (!parseHostPort(d->xaddr, d->host, d->port)) return false;
    d->uuid = trim(tagText(x, "Address"));
    std::istringstream ss(scopes);
    for (std::string tok; ss >> tok;) {
        size_t i;
        if ((i = tok.find("/name/")) != std::string::npos) d->name = pctDecode(tok.substr(i + 6));
        else if ((i = tok.find("/hardware/")) != std::string::npos) d->hardware = pctDecode(tok.substr(i + 10));
        else if ((i = tok.find("/location/")) != std::string::npos) d->location = pctDecode(tok.substr(i + 10));
    }
    return true;
}

}  // namespace

std::vector<WsdDevice> wsdProbe(int wait_ms, const std::vector<std::string>& unicast_hosts, std::string* err) {
    std::vector<WsdDevice> out;
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        if (err) *err = "UDP 소켓 생성 실패";
        return out;
    }
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    local.sin_port = 0;
    ::bind(fd, reinterpret_cast<sockaddr*>(&local), sizeof(local));
    unsigned char ttl = 1, loop = 1;
    ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
    ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));

    const std::string probe = buildProbe();
    int sent = 0;
    auto sendAll = [&] {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(kWsdPort);
        ::inet_pton(AF_INET, kMcastAddr, &to.sin_addr);
        if (::sendto(fd, probe.data(), probe.size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof(to)) > 0) ++sent;
        for (const std::string& h : unicast_hosts) {
            if (::inet_pton(AF_INET, h.c_str(), &to.sin_addr) == 1 &&
                ::sendto(fd, probe.data(), probe.size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof(to)) > 0)
                ++sent;
        }
    };

    // UDP라 유실될 수 있어 두 번 보낸다 (다보이조는 0·0.7·1.6·3초에 네 번)
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::milliseconds(wait_ms);
    const auto second = start + std::chrono::milliseconds(wait_ms / 3);
    bool second_sent = false;
    sendAll();

    std::set<std::string> seen;
    char buf[65536];
    for (;;) {
        const auto now = Clock::now();
        if (now >= deadline) break;
        if (!second_sent && now >= second) {
            sendAll();
            second_sent = true;
        }
        const auto until = second_sent ? deadline : second;
        const int ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(until - now).count());
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, ms > 0 ? ms : 1) <= 0) continue;
        const ssize_t n = ::recvfrom(fd, buf, sizeof(buf), 0, nullptr, nullptr);
        if (n <= 0) continue;
        const std::string x(buf, static_cast<size_t>(n));
        if (x.find("ProbeMatch") == std::string::npos) continue;
        WsdDevice d;
        if (!parseMatch(x, &d)) continue;
        const std::string key = d.uuid.empty() ? d.xaddr : d.uuid;
        if (seen.insert(key).second) out.push_back(d);
    }
    ::close(fd);
    if (sent == 0 && err) *err = "Probe 전송 실패 (네트워크 확인)";
    return out;
}
