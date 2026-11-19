#pragma once

#include <string>

// 객체가 사라질 때 오는 요약 프레임 (실측: bbox 0 + 옷 색상 + 베스트샷 경로).
// 대역 카메라 형식: <tt:HumanBody><tt:Tops Color=".."/><tt:Bottoms Color=".."/></tt:HumanBody>
//                  <tt:ImageRef>/download/chN/objectid_X_T.jpg</tt:ImageRef>
// (실제 WiseAI의 태그 위치·이름은 원본 캡처가 없어 단순화한 형식)
struct Summary {
    int object_id = 0;
    std::string utc;        // tt:Frame UtcTime
    std::string top_color;
    std::string bottom_color;
    std::string image_ref;  // 카메라 기준 경로
};

// 요약 프레임이면 true + out 채움. 일반 객체 프레임이면 false.
bool parseSummary(const std::string& xml, Summary& out);

// tt:Frame UtcTime → epoch 초(소수). 실패 시 -1.
double parseUtc(const std::string& xml);
