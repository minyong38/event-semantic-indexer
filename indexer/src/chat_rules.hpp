#pragma once

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

// 관제에 등록된 카메라 1대 (config/devices.json)
struct Camera {
    int no = 0;              // 관제 번호 ("3번 카메라")
    std::string id, site, host;
    int port = 0;
    bool ptz = false;
    int presets = 0;
};

// 문장 하나를 해석한 결과: 어떤 도구를 어떤 값으로 부를지 + 문장에 나온 카메라 단서
struct Intent {
    std::string tool = "unknown";   // search_people | find_cameras | list_cameras | control_camera
                                    // | register_camera | refuse | unknown
    std::string action;             // zoom_in | zoom_out | focus | preset | brightness | contrast | saturation
    int value = 0;                  // 프리셋 번호 / 줌 단계 / 화질 값
    bool relative = false;          // 화질: true면 value는 변화량(±), false면 목표값(0~100)
    std::vector<std::string> cams;  // 문장에 직접 나온 카메라 (G506, NEW-1, 없는 번호면 "#7")
    std::string site;               // 장소 키 (ontology sites)
    bool ref_context = false;       // "그 카메라", "거기"
    int ordinal = -1;               // "두 번째" → 1
    std::string query;              // 사람 검색에 넘길 문장 (카메라 단서는 뺀 것)
    std::string reason;             // 거절·모름 사유
    std::string source = "rule";    // rule | llm
};

// 규칙 해석기 — 온톨로지(장소 동의어)와 정규식으로 확실한 것만 판단한다. 못 하면 tool=unknown.
class RuleParser {
public:
    explicit RuleParser(const nlohmann::json& ontology);

    Intent parse(const std::string& text, const std::vector<Camera>& cams) const;

    // 카메라 단서·장소·맥락 참조만 뽑는다. LLM이 도구를 정한 경우에도 카메라는 이것으로만 정한다.
    void extractRefs(const std::string& text, const std::vector<Camera>& cams, Intent* in) const;

    // 하면 안 되는 명령인가 (재부팅·전원·삭제·설정 변경…) — 어떤 모드에서도 먼저 막는 가드
    bool forbidden(const std::string& text, std::string* word) const;

    // 제어 동작 단어 (확대·축소·초점·프리셋·밝기·대비·채도)
    bool action(const std::string& text, Intent* in) const;

    // 화질 값: "70으로"=목표값, "올려/어둡게"=±10(또는 "20 올려"=±20), "원래대로"=50. 방향을 모르면 value 0(되묻기)
    void imageValue(const std::string& text, Intent* in) const;

private:
    std::vector<std::pair<std::string, std::regex>> sites_;
    std::regex refuse_, register_, discover_, list_, preset_, home_, zin_, zout_, focus_, steps_;
    std::regex bright_, contrast_, satur_, up_, down_, toNum_, amount_, defaultImg_;
    std::regex camId_, newId_, camNo_, bareNo_, context_, person_;
    std::vector<std::regex> ordinals_;
};
