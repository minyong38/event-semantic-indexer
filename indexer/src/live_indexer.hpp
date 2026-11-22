#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "stitcher.hpp"

// 실시간 인덱싱: 카메라 메타데이터 폴더(meta_chN.xmls, bestshots/chN/)를 계속 따라 읽으면서
//   객체 프레임 → 조각 잇기(Stitcher)에 위치 전달
//   요약 프레임(사람 퇴장) → 그 조각 끝 표시
//   그룹 확정(모든 조각이 끝났고, 이어 붙을 시간이 지남) → 짧음·작음 필터 → 베스트샷 임베딩 → sink로 넘김
// 오프라인 indexer와 같은 규칙(잇기 화면 3%·1초, 0.5초 미만·세로 64px 미만 제외)을 쓴다.
// 사람이 없으면 카메라가 프레임을 안 보내므로, 확정 판단은 "마지막 프레임 시각 + 그 뒤로 흐른 벽시계"로 한다.
// 스트림 교체: 같은 이름으로 새 파일이 생기거나(재생 다시 시작) 파일이 잘리면(카메라 재시작) 그 채널을 처음부터
// 다시 읽는다(tail -F). 서버가 켜질 때 이미 멈춰 있던 파일(지난 스트림)은 이전 내용을 건너뛴다.
struct LivePerson {
    int ch = 0;
    std::vector<int> members;                         // ObjectId (조각 순서)
    double first_utc = 0, last_utc = 0, seen_sec = 0; // 스트림 시각(epoch 초): 처음·마지막으로 보인 때
    std::string shot, top, bottom;                    // 대표 베스트샷 경로, 옷 색(기록만)
    std::vector<float> vec;                           // CLIP 이미지 임베딩 (L2 정규화)
    double summary_utc = 0;                           // 카메라가 요약 프레임에 찍은 시각 = 퇴장을 알린 때 (스트림 시각)
    double summary_wall = 0;                          // 마지막 조각의 요약 프레임을 읽은 벽시계 (epoch 초)
    double ready_wall = 0;                            // 그룹을 확정한 벽시계
    double embed_ms = 0;
};

struct LiveParams {
    std::string dir;       // 대역 카메라(또는 재생기) 출력 폴더
    std::string model;     // CLIP 이미지 인코더 onnx
    int threads = 4;       // ONNX Runtime 스레드
    double min_dur = 0.5;  // 보인 시간이 이보다 짧으면 임베딩 안 함 (초)
    int min_shot_h = 64;   // 베스트샷 세로가 이보다 작으면 임베딩 안 함 (픽셀)
    StitchParams stitch;
    int poll_ms = 100;     // 파일을 다시 읽는 주기
    double stale_sec = 10; // 서버가 켜질 때 이만큼 안 바뀐 파일은 지난 스트림 → 이전 내용 건너뜀
};

class LiveIndexer {
public:
    using Sink = std::function<void(LivePerson&&)>;
    struct Stats {
        long lines = 0, summaries = 0, groups = 0, kept = 0, drop_short = 0, drop_small = 0;
        int channels = 0;
    };

    LiveIndexer(LiveParams p, Sink sink);
    ~LiveIndexer();
    void start();
    void stop();
    Stats stats() const;
    void resetStats();  // 측정 구간을 새로 시작할 때

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
