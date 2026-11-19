#pragma once

#include <map>
#include <string>
#include <vector>

#include "detection.hpp"

// 트랙 조각 잇기 (한 채널 전용).
// 다보이조 IdentityTracker의 "인계"(5초 / 화면 폭 18%)를 일반 현장용으로 넓힌 것:
//   - 침대 같은 고정 앵커 없이, 사라진 자리에서 속도로 예측한 위치 근처에 새 ID가 뜨면 같은 그룹
//   - 그룹 단위로 임베딩을 미뤘다가(지연 확정) 이어지는 조각이 없으면 그때 한 번만 확정
struct StitchParams {
    bool enabled = true;
    double link_sec = 5.0;     // 사라진 뒤 이 시간 안에 새로 떠야 후보
    float link_dist = 0.18f;   // 예측 위치와 새 위치의 거리 상한 (정규화 좌표)
    float size_ratio = 2.0f;   // 키(bbox 높이) 비율 상한 — 너무 다르면 다른 사람
};

struct Group {
    int id = 0;
    std::vector<int> members;   // ObjectId 목록 (조각 순서)
    double first = -1, last = -1;
    double seen_sec = 0;        // 조각들이 실제로 보인 시간 합
    std::string shot;           // 그룹 대표 베스트샷 (가장 큰 것)
    int shot_h = 0;
    std::string top, bottom;    // 대표 베스트샷의 옷 색상
    int ended = 0;              // 요약 프레임이 온 조각 수
    bool committed = false;
};

class Stitcher {
public:
    explicit Stitcher(StitchParams p) : p_(p) {}

    // 객체 프레임마다 (t = 프레임 시각, 초)
    void observe(double t, const std::vector<Detection>& dets);
    // 요약 프레임 (조각 하나 퇴장). shot_h는 베스트샷 세로 픽셀.
    void summary(int object_id, const std::string& shot, int shot_h,
                 const std::string& top, const std::string& bottom);
    // t 시점에 더 기다릴 필요 없는 그룹을 확정해서 꺼낸다 (flush=true면 전부)
    std::vector<Group> takeReady(double t, bool flush);

    struct Link { int from, to; float dist; double gap; };
    const std::vector<Link>& links() const { return links_; }

private:
    struct Obj {
        double first = -1, last = -1;
        float x = 0, y = 0, vx = 0, vy = 0, h = 0;  // 발끝 위치·속도(초당)·키
        bool has_v = false;
        int group = 0;
        bool ended = false;
        bool has_next = false;  // 이미 다른 조각이 이어 붙었는가
    };
    int newGroup(int oid, double t);

    StitchParams p_;
    std::map<int, Obj> objs_;
    std::map<int, Group> groups_;
    std::vector<Link> links_;
    int next_group_ = 1;
};
