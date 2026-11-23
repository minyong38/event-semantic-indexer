#include "chat_rules.hpp"

#include <algorithm>
#include <cctype>

using nlohmann::json;

namespace {

const auto kFlags = std::regex::ECMAScript | std::regex::icase;

std::string escapeRe(const std::string& s) {
    static const std::string special = R"(\^$.|?*+()[]{}/)";
    std::string out;
    for (char c : s) {
        if (special.find(c) != std::string::npos) out += '\\';
        out += c;
    }
    return out;
}

std::string squeeze(const std::string& s) {  // 연속 공백 하나로, 앞뒤 공백 제거
    std::string out;
    bool sp = false;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n') {
            sp = !out.empty();
        } else {
            if (sp) out += ' ';
            sp = false;
            out += c;
        }
    }
    return out;
}

}  // namespace

RuleParser::RuleParser(const json& onto) {
    // 장소 동의어 (긴 것부터 맞추도록 정렬)
    if (onto.contains("synonyms") && onto["synonyms"].is_object()) {
        for (const auto& [site, words] : onto["synonyms"].items()) {
            std::vector<std::string> ws;
            for (const auto& w : words) ws.push_back(w.get<std::string>());
            std::sort(ws.begin(), ws.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
            std::string alt;
            for (const auto& w : ws) alt += (alt.empty() ? "" : "|") + escapeRe(w);
            sites_.emplace_back(site, std::regex(alt, kFlags));
        }
    }

    // 하면 안 되는 명령: 챗봇에서는 아예 다루지 않는다 (관리자 화면에서 직접)
    refuse_ = std::regex(
        "재부팅|리부팅|재시작|reboot|restart|"
        "(전원|카메라|녹화|영상|화면)\\s*(을|를|도)?\\s*(꺼|끄|끊|중지|멈춰|정지)|"
        "삭제|지워|지우|초기화(?!면)|리셋|reset|포맷|format|"
        "비밀번호|패스워드|password|계정|"
        "펌웨어|firmware|업데이트|"
        "설정\\s*(을|를)?\\s*(바꿔|변경|바꾸)",
        kFlags);
    register_ = std::regex("(등록|추가|연결)\\s*(해|시켜|하자|할래|부탁)|register|add\\s*(the\\s*)?camera", kFlags);
    discover_ = std::regex(
        "(연결|등록)\\s*(이\\s*)?(안\\s*된|안된|안\\s*돼\\s*있는|안\\s*되어\\s*있는|되지\\s*않은)|미등록|"
        "새\\s*(로\\s*)?(설치된\\s*|들어온\\s*|생긴\\s*)?카메라|새로\\s*(연결|설치|들어온|생긴)|"
        "탐색|스캔|scan|discover|못\\s*찾은|unregistered|new\\s*camera",
        kFlags);
    list_ = std::regex(
        "카메라\\s*(목록|리스트|상태|현황|몇\\s*대|뭐\\s*있|어떤|보여|알려)|몇\\s*대|어떤\\s*카메라|등록된\\s*카메라|"
        "list\\s*(the\\s*)?cameras|camera\\s*list|show\\s*(me\\s*)?(the\\s*)?cameras",
        kFlags);

    // 제어 동작
    preset_ = std::regex("(\\d+)\\s*번\\s*(프리셋|위치|자리|포지션)|프리셋\\s*(\\d+)|preset\\s*(\\d+)", kFlags);
    home_ = std::regex("(원래|기본|처음|홈)\\s*(위치|자리|포지션)|홈\\s*(으로|위치)|home\\s*position|go\\s*home", kFlags);
    zin_ = std::regex("줌\\s*인|zoom\\s*in|확대|크게|당겨|가까이|키워", kFlags);
    zout_ = std::regex("줌\\s*아웃|zoom\\s*out|축소|작게|넓게|멀리", kFlags);
    focus_ = std::regex("초점|포커스|focus|흐려|흐릿|뿌옇|선명", kFlags);
    steps_ = std::regex("(\\d+)\\s*(단계|칸|스텝|steps?)", kFlags);

    // 카메라 단서
    camId_ = std::regex("(^|[^A-Za-z0-9])G(\\d{3})(?![0-9])", kFlags);
    newId_ = std::regex("(^|[^A-Za-z0-9])NEW-?(\\d+)", kFlags);
    camNo_ = std::regex("(\\d+)\\s*번\\s*(카메라|캠|채널)|카메라\\s*(\\d+)\\s*번?|cam(era)?\\s*(\\d+)", kFlags);
    bareNo_ = std::regex("(\\d+)\\s*번(?!\\s*째)");
    context_ = std::regex(
        "그\\s*(카메라|캠|채널|화면|곳|사람)|거기|저기|방금|이\\s*카메라|해당\\s*카메라|같은\\s*카메라|"
        "그거|그걸|그것|that\\s*camera|this\\s*camera|same\\s*camera",
        kFlags);
    ordinals_ = {std::regex("(첫|1|한)\\s*번\\s*째|first", kFlags), std::regex("(두|2)\\s*번\\s*째|second", kFlags),
                 std::regex("(세|3)\\s*번\\s*째|third", kFlags), std::regex("(네|4)\\s*번\\s*째|fourth", kFlags),
                 std::regex("(다섯|5)\\s*번\\s*째|fifth", kFlags)};

    // 사람 검색 신호: 사람·행동 단어 + 온톨로지의 색·하의 단어
    std::string person =
        "사람|남자|여자|남성|여성|어린이|학생|노인|할머니|할아버지|아저씨|아주머니|아줌마|커플|"
        "person|people|man|woman|men|women|guy|lady|kid|child|"
        "든\\s|들고|입은|쓴\\s|메고|멘\\s|끌고|타고|걷는|뛰는|앉은|앉아|서\\s*있는|가방|모자|우산";
    for (const char* key : {"color_words"}) {
        if (onto.contains(key) && onto[key].is_object())
            for (const auto& [_, words] : onto[key].items())
                for (const auto& w : words) person += "|" + escapeRe(w.get<std::string>());
    }
    if (onto.contains("bottom_words") && onto["bottom_words"].is_array())
        for (const auto& w : onto["bottom_words"]) person += "|" + escapeRe(w.get<std::string>());
    person_ = std::regex(person, kFlags);
}

bool RuleParser::forbidden(const std::string& text, std::string* word) const {
    // "프리셋" 안의 "리셋"이 걸리지 않게 먼저 지운다
    static const std::regex preset_word("프리셋|preset", kFlags);
    const std::string t = std::regex_replace(text, preset_word, " ");
    std::smatch m;
    if (!std::regex_search(t, m, refuse_)) return false;
    if (word) *word = m.str(0);
    return true;
}

bool RuleParser::action(const std::string& text, Intent* in) const {
    std::smatch m;
    if (std::regex_search(text, m, preset_)) {
        in->action = "preset";
        for (int g : {1, 3, 4})
            if (m[g].matched) in->value = std::stoi(m.str(g));
        return true;
    }
    if (std::regex_search(text, home_)) {
        in->action = "preset";
        in->value = 1;  // 기본 위치 = 1번 프리셋
        return true;
    }
    const bool zi = std::regex_search(text, zin_), zo = std::regex_search(text, zout_);
    if (zi || zo) {
        in->action = zi && !zo ? "zoom_in" : (zo && !zi ? "zoom_out" : "zoom_in");
        in->value = 1;
        if (std::regex_search(text, m, steps_)) in->value = std::clamp(std::stoi(m.str(1)), 1, 5);
        return true;
    }
    if (std::regex_search(text, focus_)) {
        in->action = "focus";
        return true;
    }
    return false;
}

void RuleParser::extractRefs(const std::string& text, const std::vector<Camera>& cams, Intent* in) const {
    std::string rest = text;
    auto addCam = [&](const std::string& id) {
        if (std::find(in->cams.begin(), in->cams.end(), id) == in->cams.end()) in->cams.push_back(id);
    };
    // 1) G### / NEW-n
    for (std::sregex_iterator it(text.begin(), text.end(), camId_), end; it != end; ++it) addCam("G" + it->str(2));
    for (std::sregex_iterator it(text.begin(), text.end(), newId_), end; it != end; ++it) addCam("NEW-" + it->str(2));
    rest = std::regex_replace(rest, camId_, "$1");
    rest = std::regex_replace(rest, newId_, "$1");

    // 2) 관제 번호 ("3번 카메라") — 프리셋·단계 숫자와 헷갈리지 않게 먼저 지운 문장에서 찾는다
    std::string scratch = std::regex_replace(std::regex_replace(rest, preset_, " "), steps_, " ");
    auto byNo = [&](int no) {
        for (const auto& c : cams)
            if (c.no == no) return c.id;
        return "#" + std::to_string(no);  // 없는 번호 → 실행 단계에서 "없는 카메라"로 답한다
    };
    for (std::sregex_iterator it(scratch.begin(), scratch.end(), camNo_), end; it != end; ++it) {
        const auto& m = *it;
        const std::string n = m[1].matched ? m.str(1) : (m[3].matched ? m.str(3) : m.str(5));
        if (!n.empty()) addCam(byNo(std::stoi(n)));
    }
    rest = std::regex_replace(rest, camNo_, " ");

    // 3) 장소
    for (const auto& [site, re] : sites_)
        if (std::regex_search(text, re)) {
            in->site = site;
            break;
        }

    // 4) 맥락 참조
    in->ref_context = std::regex_search(text, context_);
    for (size_t i = 0; i < ordinals_.size(); ++i)
        if (std::regex_search(text, ordinals_[i])) {
            in->ordinal = static_cast<int>(i);
            break;
        }

    // 5) 동작이 있고 카메라 단서가 전혀 없으면 맨 숫자("3번 확대")를 관제 번호로 본다
    Intent tmp;
    if (in->cams.empty() && !in->ref_context && in->ordinal < 0 && action(text, &tmp)) {
        std::smatch m;
        if (std::regex_search(scratch, m, bareNo_)) addCam(byNo(std::stoi(m.str(1))));
    }
    in->query = squeeze(rest);
}

Intent RuleParser::parse(const std::string& text, const std::vector<Camera>& cams) const {
    Intent in;
    extractRefs(text, cams, &in);
    std::string word;
    if (forbidden(text, &word)) {
        in.tool = "refuse";
        in.reason = word;
        return in;
    }
    static const std::regex negated("안\\s*(된|돼|되)");
    if (std::regex_search(text, register_) && !std::regex_search(text, negated)) {
        in.tool = "register_camera";
        return in;
    }
    if (std::regex_search(text, discover_)) {
        in.tool = "find_cameras";
        return in;
    }
    Intent act;
    const bool has_action = action(text, &act);
    const bool has_ref = !in.cams.empty() || in.ref_context || in.ordinal >= 0;
    if (std::regex_search(text, list_) && !has_action) {
        in.tool = "list_cameras";
        return in;
    }
    const bool person = std::regex_search(text + " ", person_);
    if (has_action && (has_ref || !person)) {
        in.tool = "control_camera";
        in.action = act.action;
        in.value = act.value;
        return in;
    }
    if (person) {
        in.tool = "search_people";
        return in;
    }
    in.tool = "unknown";
    in.reason = "규칙으로 판단할 단서 없음";
    return in;
}
