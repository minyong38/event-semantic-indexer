// 챗봇 서버 — 관제 요원이 말 한 문장을 보내면 도구 하나를 골라 실행한다.
//
//   해석: 규칙(온톨로지 동의어 + 정규식)이 확실한 건 바로 처리하고, 못 하면 로컬 LLM(Ollama)에 묻는다.
//         LLM 출력은 JSON 스키마로 선택지를 고정하고(도구·동작·프리셋 번호만), 카메라는 고르지 못하게 한다.
//         재부팅·전원·삭제·설정 변경은 어떤 모드에서도 먼저 막는다.
//   실행: 찾기(사람 검색·카메라 탐색·목록)는 바로, 바꾸기(카메라 제어·등록)는 확인을 받은 뒤에.
//   도구: 사람 검색 → search_server /search, 카메라 탐색 → ONVIF WS-Discovery, 제어 → SUNAPI
//
//   ./chat_server --devices ../config/devices.json --ontology ../config/ontology.json
//       [--port 8770] [--search http://localhost:8765] [--llm http://localhost:11434] [--model qwen3.5:4b]
//       [--wsd-unicast 127.0.0.1] [--log chat_log.jsonl]
//   GET /chat?q=문장&session=s1[&mode=auto|rule|llm][&dry=1][&model=qwen3.5:9b]
//   GET /chat/confirm?id=c1&session=s1    GET /chat/cancel?id=c1&session=s1
//   GET /devices    GET /health    GET /chat/context?session=s1&cam=G506&hits=G506,G340 (평가용)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "chat_rules.hpp"
#include "http_client.hpp"
#include "http_server.hpp"
#include "nlohmann/json.hpp"
#include "sunapi.hpp"
#include "wsdiscovery.hpp"

using nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
double wallNow() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
double r1(double v) { return std::round(v * 10) / 10; }

std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) out.push_back(t);
    return out;
}

struct Args {
    int port = 8770, k = 6, wsd_wait = 1500;
    std::string devices = "../config/devices.json", ontology = "../config/ontology.json";
    std::string search = "http://localhost:8765";
    std::string llm = "http://localhost:11434", model = "qwen3.5:4b";
    std::vector<std::string> wsd_unicast;
    std::string log = "chat_log.jsonl";
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--port") a.port = std::stoi(v);
        else if (k == "--devices") a.devices = v;
        else if (k == "--ontology") a.ontology = v;
        else if (k == "--search") a.search = v;
        else if (k == "--llm") a.llm = v;
        else if (k == "--model") a.model = v;
        else if (k == "--wsd-unicast") a.wsd_unicast = splitComma(v);
        else if (k == "--wsd-wait") a.wsd_wait = std::stoi(v);
        else if (k == "--log") a.log = v;
        else if (k == "--k") a.k = std::stoi(v);
        else std::fprintf(stderr, "모르는 옵션: %s\n", k.c_str());
    }
    return a;
}

// LLM에게 주는 설명. 예시 문장은 넣지 않는다 — 작은 모델은 예시를 베낀다(온톨로지 실험에서 확인).
const char* kSystemPrompt =
    "너는 CCTV 관제 시스템의 명령 해석기다. 사용자 문장 하나를 읽고 JSON으로만 답한다.\n"
    "tool 고르는 법:\n"
    "- search_people: 영상 속 사람을 찾는 요청 (옷·색·소지품·행동·장소·시간 묘사)\n"
    "- find_cameras: 네트워크에서 아직 등록 안 된 새 카메라를 찾는 요청\n"
    "- list_cameras: 등록된 카메라 목록이나 상태를 묻는 요청\n"
    "- control_camera: 카메라 화면을 바꾸는 요청. action은 zoom_in(확대), zoom_out(축소), focus(초점), "
    "preset(저장해 둔 위치로 이동, preset에 번호)\n"
    "- register_camera: 찾은 카메라를 관제에 등록하는 요청\n"
    "- refuse: 재부팅·전원 끄기·삭제·초기화·비밀번호나 설정 변경처럼 위험한 요청\n"
    "- unknown: 어디에도 맞지 않음\n"
    "카메라 번호와 이름은 시스템이 따로 처리하니 고르지 않는다. 해당 없으면 action은 none, preset은 0.";

struct Pending {
    std::string id, kind;  // control | register
    Intent in;
    Camera cam;
    double expires = 0;
};

struct Session {
    std::string last_cam;
    double last_cam_t = 0;
    std::vector<std::string> hit_cams;   // 마지막 사람 검색 결과의 카메라 (순서대로)
    std::vector<WsdDevice> unregistered; // 마지막 탐색에서 찾은 미등록 카메라
    std::map<std::string, Pending> pending;
};

constexpr double kContextSec = 600;  // "그 카메라"가 가리키는 대상은 10분 동안 유지
constexpr double kConfirmSec = 60;   // 확인 대기는 60초

class ChatService {
public:
    explicit ChatService(const Args& a) : a_(a), rules_(loadJson(a.ontology)) {
        onto_ = loadJson(a.ontology);
        const json d = loadJson(a.devices);
        for (const auto& c : d.value("cameras", json::array())) {
            Camera cam;
            cam.no = c.value("no", 0);
            cam.id = c.value("id", "");
            cam.site = c.value("site", "");
            cam.host = c.value("host", "127.0.0.1");
            cam.port = c.value("port", 80);
            cam.ptz = c.value("ptz", false);
            cam.presets = c.value("presets", 0);
            cams_.push_back(cam);
        }
        if (!a.search.empty()) parseHostPort(a.search, search_host_, search_port_);
        if (!a.llm.empty()) parseHostPort(a.llm, llm_host_, llm_port_);
    }

    size_t cameraCount() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cams_.size();
    }

    json chat(const std::string& text, const std::string& sid, const std::string& mode, bool dry,
              const std::string& model) {
        const auto t0 = Clock::now();
        const std::vector<Camera> cams = snapshot();
        Intent in;
        double llm_ms = 0;
        json llm_raw;
        std::string word;
        const bool guard = rules_.forbidden(text, &word);
        if (mode == "llm") {
            // 비교용: 도구는 LLM이 정한다 (카메라 단서는 그래도 규칙으로만)
            rules_.extractRefs(text, cams, &in);
            if (llmOn()) in = llmIntent(text, in, model, &llm_ms, &llm_raw);
        } else {
            in = rules_.parse(text, cams);
            if (in.tool == "unknown" && mode != "rule" && llmOn())
                in = llmIntent(text, in, model, &llm_ms, &llm_raw);
        }
        if (guard && in.tool != "refuse") {  // 가드는 해석 결과와 상관없이 먼저 막는다
            in.tool = "refuse";
            in.reason = word;
            in.source += "+guard";
        }
        const double parse_ms = msSince(t0) - llm_ms;

        const auto t1 = Clock::now();
        json out = run(in, text, sid, dry);
        out["text"] = text;
        out["mode"] = mode;
        out["llm_used"] = llm_ms > 0;
        if (llm_ms > 0) out["model"] = model.empty() ? a_.model : model;
        if (!llm_raw.is_null()) out["llm_raw"] = llm_raw;
        out["timing"] = {{"parse_ms", r1(parse_ms)}, {"llm_ms", r1(llm_ms)}, {"exec_ms", r1(msSince(t1))},
                         {"total_ms", r1(msSince(t0))}};
        log({{"t", r1(wallNow())}, {"session", sid}, {"text", text}, {"mode", mode}, {"tool", out["tool"]},
             {"source", out["source"]}, {"camera", out.value("camera", "")}, {"action", out.value("action", "")},
             {"value", out.value("value", 0)}, {"need_confirm", out.value("need_confirm", false)},
             {"reply", out.value("reply", "")}, {"llm_raw", llm_raw}, {"timing", out["timing"]}});
        return out;
    }

    json confirm(const std::string& id, const std::string& sid) {
        Pending p;
        {
            std::lock_guard<std::mutex> lk(mu_);
            Session& s = sessions_[sid];
            auto it = s.pending.find(id);
            if (it == s.pending.end()) return {{"ok", false}, {"reply", "확인할 명령이 없어요 (이미 처리됐거나 취소됨)."}};
            p = it->second;
            s.pending.erase(it);
        }
        if (wallNow() > p.expires) return {{"ok", false}, {"reply", "확인 시간이 지나서 취소했어요. 다시 말해 주세요."}};

        json out = {{"confirm_id", id}, {"kind", p.kind}, {"camera", p.cam.id}};
        if (p.kind == "control") {
            SunapiResult r;
            if (p.in.action == "focus") r = sunapiFocus(p.cam.host, p.cam.port, 0);
            else if (p.in.action == "preset") r = sunapiPreset(p.cam.host, p.cam.port, 0, p.in.value);
            else if (p.in.action == "zoom_in") r = sunapiZoom(p.cam.host, p.cam.port, 0, p.in.value);
            else if (p.in.action == "zoom_out") r = sunapiZoom(p.cam.host, p.cam.port, 0, -p.in.value);
            out["ok"] = r.ok;
            out["detail"] = r.detail;
            out["reply"] = r.ok ? label(p.cam) + " 카메라 " + done(p.in) : label(p.cam) + " 제어 실패: " + r.detail;
        } else {
            std::lock_guard<std::mutex> lk(mu_);
            const bool dup = std::any_of(cams_.begin(), cams_.end(), [&](const Camera& c) {
                return c.host == p.cam.host && c.port == p.cam.port;
            });
            if (!dup) cams_.push_back(p.cam);
            out["ok"] = !dup;
            out["reply"] = dup ? p.cam.id + ": 이미 등록돼 있어요."
                               : p.cam.id + " 카메라를 " + std::to_string(p.cam.no) + "번으로 등록했어요.";
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            Session& s = sessions_[sid];
            s.last_cam = p.cam.id;
            s.last_cam_t = wallNow();
        }
        log({{"t", r1(wallNow())}, {"session", sid}, {"confirm", id}, {"kind", p.kind}, {"camera", p.cam.id},
             {"action", p.in.action}, {"value", p.in.value}, {"ok", out["ok"]}, {"detail", out.value("detail", "")}});
        return out;
    }

    json cancel(const std::string& id, const std::string& sid) {
        std::lock_guard<std::mutex> lk(mu_);
        const bool erased = sessions_[sid].pending.erase(id) > 0;
        return {{"ok", erased}, {"reply", erased ? "취소했어요." : "취소할 명령이 없어요."}};
    }

    json devices() {
        json arr = json::array();
        for (const Camera& c : snapshot())
            arr.push_back({{"no", c.no}, {"id", c.id}, {"site", c.site}, {"site_ko", siteKo(c.site)}, {"host", c.host},
                           {"port", c.port}, {"ptz", c.ptz}, {"presets", c.presets}});
        return {{"cameras", arr}};
    }

    // 평가용: 대화 맥락을 직접 넣는다 ("그 카메라", "두 번째 사람" 해석 시험)
    json setContext(const std::string& sid, const std::string& cam, const std::string& hits) {
        std::lock_guard<std::mutex> lk(mu_);
        Session& s = sessions_[sid];
        s.last_cam = cam;
        s.last_cam_t = cam.empty() ? 0 : wallNow();
        s.hit_cams = splitComma(hits);
        return {{"ok", true}};
    }

private:
    static json loadJson(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            std::fprintf(stderr, "[chat] 파일을 못 엶: %s\n", path.c_str());
            return json::object();
        }
        return json::parse(f, nullptr, false);
    }

    bool llmOn() const { return !llm_host_.empty(); }

    std::vector<Camera> snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cams_;
    }

    std::string siteKo(const std::string& site) const {
        if (onto_.contains("sites") && onto_["sites"].contains(site)) return onto_["sites"][site].get<std::string>();
        return site;
    }

    std::string label(const Camera& c) const {
        return c.id + "(" + std::to_string(c.no) + "번, " + siteKo(c.site) + ")";
    }

    static std::string ask(const Intent& in) {
        if (in.action == "zoom_in") return std::to_string(in.value) + "단계 확대할까요?";
        if (in.action == "zoom_out") return std::to_string(in.value) + "단계 축소할까요?";
        if (in.action == "focus") return "초점을 맞출까요?";
        if (in.action == "preset") return std::to_string(in.value) + "번 프리셋으로 돌릴까요?";
        return in.action + "?";
    }

    static std::string done(const Intent& in) {
        if (in.action == "zoom_in") return std::to_string(in.value) + "단계 확대했어요.";
        if (in.action == "zoom_out") return std::to_string(in.value) + "단계 축소했어요.";
        if (in.action == "focus") return "초점을 맞췄어요.";
        if (in.action == "preset") return std::to_string(in.value) + "번 프리셋으로 돌렸어요.";
        return in.action;
    }

    // LLM: 도구·동작·프리셋 번호만 고른다. 출력은 JSON 스키마로 고정(정해진 값 밖은 나올 수 없음).
    Intent llmIntent(const std::string& text, Intent base, const std::string& model, double* ms, json* raw) {
        const auto t = Clock::now();
        const json schema = {
            {"type", "object"},
            {"properties",
             {{"tool",
               {{"type", "string"},
                {"enum",
                 {"search_people", "find_cameras", "list_cameras", "control_camera", "register_camera", "refuse",
                  "unknown"}}}},
              {"action", {{"type", "string"}, {"enum", {"zoom_in", "zoom_out", "focus", "preset", "none"}}}},
              {"preset", {{"type", "integer"}, {"minimum", 0}, {"maximum", 9}}}}},
            {"required", {"tool", "action", "preset"}}};
        const json req = {{"model", model.empty() ? a_.model : model},
                          {"stream", false},
                          {"think", false},
                          {"format", schema},
                          {"keep_alive", "30m"},
                          {"options", {{"temperature", 0}, {"num_predict", 64}}},
                          {"messages",
                           json::array({{{"role", "system"}, {"content", kSystemPrompt}},
                                        {{"role", "user"}, {"content", text}}})}};
        const HttpResult h = httpPost(llm_host_, llm_port_, "/api/chat", req.dump(), "application/json", 60000);
        *ms = msSince(t);
        if (h.code != 200) {
            base.tool = "unknown";
            base.reason = "LLM 호출 실패: " + (h.code ? "HTTP " + std::to_string(h.code) : h.error);
            *raw = {{"error", base.reason}};
            return base;
        }
        const json resp = json::parse(h.body, nullptr, false);
        const std::string content = resp.is_object() && resp.contains("message")
                                        ? resp["message"].value("content", "")
                                        : "";
        const json j = json::parse(content, nullptr, false);
        if (!j.is_object()) {
            base.tool = "unknown";
            base.reason = "LLM 응답 해석 실패";
            *raw = {{"error", base.reason}, {"content", content}};
            return base;
        }
        *raw = j;
        base.source = "llm";
        base.tool = j.value("tool", "unknown");
        const std::string act = j.value("action", "none");
        base.action = act == "none" ? "" : act;
        // 숫자도 카메라처럼 문장에 적힌 값은 규칙이 뽑는다 ("3단계", "기본 위치"=1번). 규칙이 못 뽑을 때만 LLM 값.
        base.value = 0;
        Intent ruled;
        const bool has = rules_.action(text, &ruled);
        if (base.action == "preset") {
            base.value = has && ruled.action == "preset" ? ruled.value : j.value("preset", 0);
        } else if (base.action == "zoom_in" || base.action == "zoom_out") {
            static const std::regex steps("(\\d+)\\s*(단계|칸|스텝|steps?)", std::regex::icase);
            std::smatch m;
            base.value = std::regex_search(text, m, steps) ? std::clamp(std::stoi(m.str(1)), 1, 5) : 1;
        }
        if (base.tool == "unknown") base.reason = "LLM도 판단하지 못함";
        return base;
    }

    json run(const Intent& in, const std::string& text, const std::string& sid, bool dry) {
        json out = {{"tool", in.tool},     {"source", in.source}, {"action", in.action},
                    {"value", in.value},   {"site", in.site},     {"need_confirm", false}};
        if (in.tool == "refuse") {
            out["refused"] = true;
            out["reply"] = (in.reason.empty() ? std::string("그런 명령은") : "'" + in.reason + "' 같은 명령은") +
                           " 챗봇에서 할 수 없어요. 관리자 화면에서 직접 해 주세요.";
            return out;
        }
        if (in.tool == "unknown") {
            out["reply"] =
                "무슨 말인지 잘 모르겠어요. 예: '버스 정류장에서 가방 든 사람 찾아줘', '등록 안 된 카메라 찾아줘', "
                "'G506 2번 프리셋으로'";
            out["reason"] = in.reason;
            return out;
        }
        if (in.tool == "control_camera") return proposeControl(in, sid, dry, out);
        if (in.tool == "register_camera") return proposeRegister(in, sid, dry, out);
        if (dry) {
            out["reply"] = "(dry) " + in.tool;
            if (in.tool == "search_people") out["query"] = in.query.empty() ? text : in.query;
            return out;
        }
        if (in.tool == "search_people") return searchPeople(in, text, sid, out);
        if (in.tool == "find_cameras") return discover(sid, out);
        if (in.tool == "list_cameras") return listCameras(in, out);
        out["reply"] = "아직 못 하는 일이에요.";
        return out;
    }

    // 카메라 고르기: 문장에 나온 것 → 검색 결과 순번 → 대화 맥락 → 장소. 애매하면 묻는다(추측하지 않음).
    struct Resolved {
        std::string cam, how, ask;
        std::vector<std::string> candidates;
    };

    Resolved resolveCam(const Intent& in, const Session& s, const std::vector<Camera>& cams) const {
        Resolved r;
        auto registered = [&](const std::string& id) {
            return std::any_of(cams.begin(), cams.end(), [&](const Camera& c) { return c.id == id; });
        };
        const bool fresh = !s.last_cam.empty() && wallNow() - s.last_cam_t < kContextSec;
        if (!in.cams.empty()) {
            if (in.cams.size() > 1) {
                r.ask = "한 번에 카메라 한 대만 제어할게요. 어느 카메라요?";
                r.candidates = in.cams;
                return r;
            }
            const std::string& id = in.cams[0];
            if (id[0] == '#') {
                r.ask = id.substr(1) + "번 카메라는 등록돼 있지 않아요.";
                return r;
            }
            if (!registered(id)) {
                r.ask = id.rfind("NEW", 0) == 0 ? id + ": 아직 등록 안 된 카메라라 제어할 수 없어요. 먼저 등록해 주세요."
                                                : id + ": 등록된 카메라가 아니에요.";
                return r;
            }
            r.cam = id;
            r.how = "named";
            return r;
        }
        if (in.ordinal >= 0) {
            if (in.ordinal < static_cast<int>(s.hit_cams.size())) {
                r.cam = s.hit_cams[in.ordinal];
                r.how = "result";
            } else {
                r.ask = "그 순번의 검색 결과가 없어요. 먼저 사람을 검색해 주세요.";
            }
            return r;
        }
        if (!in.site.empty()) {
            if (fresh) {
                for (const auto& c : cams)
                    if (c.id == s.last_cam && c.site == in.site) {
                        r.cam = c.id;
                        r.how = "context";
                        return r;
                    }
            }
            for (const auto& c : cams)
                if (c.site == in.site) r.candidates.push_back(c.id);
            if (r.candidates.size() == 1) {
                r.cam = r.candidates[0];
                r.how = "site";
                r.candidates.clear();
            } else if (!r.candidates.empty()) {
                std::string list;
                for (const auto& id : r.candidates) list += (list.empty() ? "" : ", ") + id;
                r.ask = siteKo(in.site) + " 카메라가 " + std::to_string(r.candidates.size()) + "대예요: " + list +
                        ". 어느 카메라요?";
            } else {
                r.ask = siteKo(in.site) + "에는 등록된 카메라가 없어요.";
            }
            return r;
        }
        if (fresh) {
            r.cam = s.last_cam;
            r.how = "context";
            return r;
        }
        r.ask = "어느 카메라요? 예: 'G506', '6번 카메라'";
        return r;
    }

    json proposeControl(const Intent& in, const std::string& sid, bool dry, json out) {
        const std::vector<Camera> cams = snapshot();
        std::lock_guard<std::mutex> lk(mu_);
        Session& s = sessions_[sid];
        const Resolved r = resolveCam(in, s, cams);
        if (!r.ask.empty()) {
            out["reply"] = r.ask;
            out["need_camera"] = true;
            out["candidates"] = r.candidates;
            return out;
        }
        const Camera c = *std::find_if(cams.begin(), cams.end(), [&](const Camera& x) { return x.id == r.cam; });
        out["camera"] = c.id;
        out["camera_how"] = r.how;
        if (in.action.empty()) {
            out["reply"] = "어떤 동작을 할까요? (확대·축소·초점·프리셋)";
            return out;
        }
        if (in.action == "preset" && !c.ptz) {
            out["blocked"] = "not_ptz";
            out["reply"] = label(c) + ": 고정 카메라라 프리셋이 없어요. 확대·축소·초점만 돼요.";
            return out;
        }
        if (in.action == "preset" && (in.value < 1 || in.value > c.presets)) {
            out["blocked"] = "range";
            out["reply"] = label(c) + ": 프리셋은 1~" + std::to_string(c.presets) + "번까지 있어요.";
            return out;
        }
        s.last_cam = c.id;
        s.last_cam_t = wallNow();
        out["need_confirm"] = true;
        out["reply"] = label(c) + " 카메라 " + ask(in);
        if (dry) return out;
        Pending p;
        p.id = "c" + std::to_string(next_id_++);
        p.kind = "control";
        p.in = in;
        p.cam = c;
        p.expires = wallNow() + kConfirmSec;
        s.pending[p.id] = p;
        out["confirm_id"] = p.id;
        return out;
    }

    json proposeRegister(const Intent& in, const std::string& sid, bool dry, json out) {
        std::lock_guard<std::mutex> lk(mu_);
        Session& s = sessions_[sid];
        if (s.unregistered.empty()) {
            out["reply"] = "등록할 카메라를 아직 못 찾았어요. 먼저 '등록 안 된 카메라 찾아줘'라고 해 주세요.";
            return out;
        }
        // 탐색 목록(s.unregistered)은 사용자가 본 순서 그대로 둔다("첫 번째"가 가리키는 대상이 바뀌지 않게).
        // 그 사이 등록된 카메라는 고를 때 빼고, 이름·순번으로 다시 가리키면 "이미 등록"이라고 답한다.
        auto isReg = [&](const WsdDevice& u) {
            return std::any_of(cams_.begin(), cams_.end(),
                               [&](const Camera& c) { return c.host == u.host && c.port == u.port; });
        };
        const WsdDevice* d = nullptr;
        for (const auto& name : in.cams)
            for (const auto& u : s.unregistered)
                if (u.name == name) d = &u;
        if (!d && !in.cams.empty()) {  // 이름을 댔는데 탐색 목록에 없으면 다른 카메라로 추측하지 않는다
            out["reply"] = in.cams[0] + ": 등록 안 된 카메라 목록에 없어요. 이미 등록됐거나 아직 못 찾은 카메라예요.";
            return out;
        }
        if (!d && in.ordinal >= 0 && in.ordinal < static_cast<int>(s.unregistered.size()))
            d = &s.unregistered[in.ordinal];
        if (!d) {
            std::vector<const WsdDevice*> remain;
            for (const auto& u : s.unregistered)
                if (!isReg(u)) remain.push_back(&u);
            if (remain.empty()) {
                out["reply"] = "찾은 카메라는 이제 모두 등록돼 있어요.";
                return out;
            }
            if (remain.size() == 1) {
                d = remain[0];
            } else {
                std::string list;
                json names = json::array();
                for (const WsdDevice* u : remain) {
                    list += (list.empty() ? "" : ", ") + u->name;
                    names.push_back(u->name);
                }
                out["need_camera"] = true;
                out["candidates"] = names;
                out["reply"] = "찾은 카메라가 여러 대예요: " + list + ". 어느 카메라를 등록할까요?";
                return out;
            }
        }
        if (isReg(*d)) {
            out["camera"] = d->name;
            out["reply"] = d->name + ": 이미 등록돼 있어요.";
            return out;
        }
        int no = 0;
        for (const auto& c : cams_) no = std::max(no, c.no);
        Camera c;
        c.no = no + 1;
        c.id = d->name.empty() ? "CAM-" + std::to_string(c.no) : d->name;
        c.site = d->location;
        c.host = d->host;
        c.port = d->port;
        c.ptz = d->hardware.find("PTZ") != std::string::npos;
        c.presets = c.ptz ? 4 : 0;
        out["camera"] = c.id;
        out["need_confirm"] = true;
        out["reply"] = c.id + "(" + siteKo(c.site) + ", " + d->hardware + ", " + d->host + ":" +
                       std::to_string(d->port) + ") 카메라를 " + std::to_string(c.no) + "번으로 등록할까요?";
        if (dry) return out;
        Pending p;
        p.id = "c" + std::to_string(next_id_++);
        p.kind = "register";
        p.in = in;
        p.cam = c;
        p.expires = wallNow() + kConfirmSec;
        s.pending[p.id] = p;
        out["confirm_id"] = p.id;
        return out;
    }

    json searchPeople(const Intent& in, const std::string& text, const std::string& sid, json out) {
        if (search_host_.empty()) {
            out["reply"] = "검색 서버가 설정돼 있지 않아요.";
            return out;
        }
        const std::string q = in.query.empty() ? text : in.query;
        const bool filter = !in.cams.empty();
        const int k = filter ? 30 : a_.k;
        const std::string path =
            "/search?q=" + urlEncode(q) + "&k=" + std::to_string(k) + "&verify=10&step=3&stop=0.7";
        const HttpResult h = httpGet(search_host_, search_port_, path, 30000);
        if (h.code != 200) {
            out["reply"] = "검색 서버에 연결하지 못했어요.";
            out["error"] = h.code ? "HTTP " + std::to_string(h.code) : h.error;
            return out;
        }
        const json r = json::parse(h.body, nullptr, false);
        static const std::regex gid("G\\d{3}");
        json results = json::array();
        std::vector<std::string> hit_cams;
        for (const auto& hit : r.value("hits", json::array())) {
            std::smatch m;
            const std::string ch = hit.value("channel", "");
            const std::string cam = std::regex_search(ch, m, gid) ? m.str(0) : "";
            if (filter && std::find(in.cams.begin(), in.cams.end(), cam) == in.cams.end()) continue;
            json x = {{"rank", static_cast<int>(results.size()) + 1},
                      {"camera", cam},
                      {"channel", ch},
                      {"site", hit.value("site", "")},
                      {"clock", hit.value("clock", "")},
                      {"score", hit.value("score", 0.0)},
                      {"img", a_.search + hit.value("img", "")}};
            if (hit.contains("vlm")) x["vlm"] = hit["vlm"];
            // 화면(QML 데모)이 결과를 누르면 그 시각 영상을 틀 수 있게 검색 서버 값을 그대로 넘긴다
            for (const char* f : {"video", "t", "t_end", "members", "top", "bottom"})
                if (hit.contains(f)) x[f] = hit[f];
            results.push_back(x);
            hit_cams.push_back(cam);
            if (static_cast<int>(results.size()) >= a_.k) break;
        }
        out["query"] = q;
        out["parsed"] = r.value("parsed", json());
        out["results"] = results;
        {
            std::lock_guard<std::mutex> lk(mu_);
            Session& s = sessions_[sid];
            s.hit_cams = hit_cams;
            if (!hit_cams.empty() && !hit_cams[0].empty()) {
                s.last_cam = hit_cams[0];
                s.last_cam_t = wallNow();
            }
        }
        if (results.empty()) {
            out["reply"] = "조건에 맞는 사람을 못 찾았어요.";
            return out;
        }
        const json& top = results[0];
        std::string reply = "가장 비슷한 사람: " + siteKo(top.value("site", "")) + " " + top.value("camera", "") + ", " +
                            top.value("clock", "");
        if (top.contains("vlm")) {
            char b[32];
            std::snprintf(b, sizeof(b), " (VLM 확인 %.2f)", top["vlm"].get<double>());
            reply += b;
        }
        if (results.size() > 1) reply += ". 외 " + std::to_string(results.size() - 1) + "명";
        reply += ". '그 카메라 확대해줘'처럼 이어서 말할 수 있어요.";
        out["reply"] = reply;
        return out;
    }

    json discover(const std::string& sid, json out) {
        std::string err;
        const std::vector<WsdDevice> found = wsdProbe(a_.wsd_wait, a_.wsd_unicast, &err);
        const std::vector<Camera> cams = snapshot();
        std::vector<WsdDevice> unreg;
        json arr = json::array();
        for (const auto& d : found) {
            const bool reg = std::any_of(cams.begin(), cams.end(), [&](const Camera& c) {
                return c.host == d.host && c.port == d.port;
            });
            arr.push_back({{"name", d.name}, {"model", d.hardware}, {"location", d.location}, {"host", d.host},
                           {"port", d.port}, {"registered", reg}});
            if (!reg) unreg.push_back(d);
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            sessions_[sid].unregistered = unreg;
        }
        out["found"] = arr;
        if (!err.empty()) out["error"] = err;
        std::string reply = "네트워크에서 카메라 " + std::to_string(found.size()) + "대를 찾았어요. ";
        if (unreg.empty()) {
            reply += "모두 등록돼 있어요.";
        } else {
            reply += "그중 " + std::to_string(unreg.size()) + "대가 등록 안 돼 있어요: ";
            for (size_t i = 0; i < unreg.size(); ++i)
                reply += (i ? ", " : "") + unreg[i].name + "(" + siteKo(unreg[i].location) + ", " + unreg[i].hardware + ")";
            reply += ". '첫 번째 등록해줘'처럼 말하면 등록할게요.";
        }
        out["reply"] = reply;
        return out;
    }

    json listCameras(const Intent& in, json out) {
        std::vector<Camera> cams;
        for (const Camera& c : snapshot()) {
            const bool named = std::find(in.cams.begin(), in.cams.end(), c.id) != in.cams.end();
            if (!in.cams.empty() ? named : (in.site.empty() || c.site == in.site)) cams.push_back(c);
        }
        json arr = json::array();
        int online = 0;
        for (const Camera& c : cams) {
            std::string model;
            const bool ok = sunapiDeviceInfo(c.host, c.port, 300, &model).ok;
            online += ok;
            arr.push_back({{"no", c.no}, {"id", c.id}, {"site", c.site}, {"ptz", c.ptz}, {"online", ok}, {"model", model}});
        }
        out["cameras"] = arr;
        std::string reply = (in.site.empty() ? "등록된 카메라 " : siteKo(in.site) + " 카메라 ") +
                            std::to_string(cams.size()) + "대 중 " + std::to_string(online) + "대가 응답해요";
        std::string off;
        for (const auto& c : arr)
            if (!c["online"].get<bool>()) off += (off.empty() ? "" : ", ") + c["id"].get<std::string>();
        reply += off.empty() ? "." : " (응답 없음: " + off + ").";
        out["reply"] = reply;
        return out;
    }

    void log(const json& j) {
        if (a_.log.empty()) return;
        std::lock_guard<std::mutex> lk(log_mu_);
        std::ofstream f(a_.log, std::ios::app);
        f << j.dump() << "\n";
    }

    Args a_;
    RuleParser rules_;
    json onto_;
    std::string search_host_, llm_host_;
    int search_port_ = 0, llm_port_ = 0;
    mutable std::mutex mu_;
    std::mutex log_mu_;
    std::vector<Camera> cams_;
    std::map<std::string, Session> sessions_;
    int next_id_ = 1;
};

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

}  // namespace

int main(int argc, char** argv) {
    const Args a = parseArgs(argc, argv);
    ChatService svc(a);

    HttpServer server(a.port, [&](const HttpRequest& r) {
        HttpResponse res;
        auto q = [&](const char* k, const std::string& d = "") {
            auto it = r.query.find(k);
            return it == r.query.end() ? d : it->second;
        };
        json out;
        if (r.path == "/chat") {
            const std::string text = q("q");
            if (text.empty()) {
                res.code = 400;
                out = {{"error", "q가 비었어요"}};
            } else {
                out = svc.chat(text, q("session", "default"), q("mode", "auto"), q("dry") == "1", q("model"));
            }
        } else if (r.path == "/chat/confirm") {
            out = svc.confirm(q("id"), q("session", "default"));
        } else if (r.path == "/chat/cancel") {
            out = svc.cancel(q("id"), q("session", "default"));
        } else if (r.path == "/chat/context") {
            out = svc.setContext(q("session", "default"), q("cam"), q("hits"));
        } else if (r.path == "/devices") {
            out = svc.devices();
        } else if (r.path == "/health") {
            out = {{"ok", true}};
        } else {
            res.code = 404;
            out = {{"error", "없는 경로"}};
        }
        res.body = out.dump();
        return res;
    });
    if (!server.start()) return 1;
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);
    std::fprintf(stderr, "[chat] http://localhost:%d  카메라 %zu대 · 검색 %s · LLM %s (%s)\n", a.port,
                 svc.cameraCount(), a.search.empty() ? "끔" : a.search.c_str(), a.llm.empty() ? "끔" : a.llm.c_str(),
                 a.model.c_str());
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    server.stop();
    return 0;
}
