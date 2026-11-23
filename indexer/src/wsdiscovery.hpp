#pragma once

#include <string>
#include <vector>

// ONVIF WS-Discovery로 네트워크의 카메라를 찾는다.
// 다보이조 Qt 클라이언트(devicebackend.cpp)의 탐색을 Qt 없이 소켓으로 옮긴 것:
//   Probe SOAP을 멀티캐스트 239.255.255.250:3702로 보내고 → 카메라마다 오는 ProbeMatch(UDP)를 모은다.
//   멀티캐스트가 막힌 망(또는 같은 PC의 시뮬레이터)을 위해 유니캐스트 대상도 같이 보낼 수 있다.
struct WsdDevice {
    std::string uuid;      // EndpointReference — 중복 제거 키
    std::string xaddr;     // 장치 서비스 주소 (첫 번째 XAddrs)
    std::string host;
    int port = 80;
    std::string name, hardware, location;  // Scopes에서 뽑은 값 (없으면 빈칸)
};

// wait_ms 동안 응답을 모은다. ONVIF 흔적이 없는 응답(PC·프린터 WSD)은 버린다.
std::vector<WsdDevice> wsdProbe(int wait_ms, const std::vector<std::string>& unicast_hosts, std::string* err);
