// 검색 서버 — 인덱서가 만든 인덱스(이벤트·프레임)를 읽어 자연어 검색을 HTTP로 제공.
//
//   GET /search?q=<영어 문장>&index=event|frame&k=12[&verify=10]
//       verify=N: 임베딩 상위 N명 사진을 VLM 서비스에 보내 "질문에 맞나?" 점수(P(yes))로 다시 줄 세움
//       parse=1: 질문을 온톨로지(config/ontology.json) 칸으로 쪼갬 — 장소·시각·색은 동의어 사전과 정규식(규칙),
//                남은 묘사가 한국어면 VLM 서비스 /translate로 영어 번역만. parse=2: LLM(/parse)이 전부 (비교용)
//                {장소, 이후·이전 시각, 상의·하의 색, 설명} → 장소·시각은 검색 전에 후보를 거르는 하드 필터,
//                설명(장소·시각을 뺀 모습)만 임베딩·VLM 검증에 씀. 색은 어두운 곳에서 부정확해 기록만.
//       step=S&stop=T: N명을 한 번에 보내지 않고 S명씩 보내다가, 지금까지 최고 P(yes)가 T 이상이면 멈춤
//                      (애매할 때만 VLM을 더 부름 — 질문 30개 시뮬레이션: 3명씩·0.7에서 호출 −39%, 정확도 유지)
//   GET /img/event/<label>   그룹 대표 베스트샷 (파일 그대로)
//   GET /img/frame/<label>   그 시각 영상 프레임에서 사람 크롭 (메타데이터 박스)
//   GET /info                인덱스별 임베딩 수
//
//   search_server --event ~/idx_stB --frame ~/idx_frame --frags ../camsim/out_raw \
//                 --videos ../camsim/meva_raw --text ../model/clip_text.onnx \
//                 --vocab ../model/bpe_simple_vocab_16e6.txt [--port 8765] [--vlm http://localhost:8766]
//   검증용: --tokens "문장"  (토큰 ID 출력 후 종료), --query "문장" [--index event]  (상위 결과 출력 후 종료)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <hnswlib/hnswlib.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "http_client.hpp"
#include "http_server.hpp"
#include "metadata_parser.hpp"
#include "summary.hpp"
#include "text_embedder.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Args {
    std::string event, frame, frags, videos, text, vocab;
    std::string start = "2026-11-19T09:00:00";
    std::string tokens, query, index = "event";
    std::string vlm;  // VLM 서비스 주소 (비우면 검증·질문 해석 안 함)
    std::string ontology = "../config/ontology.json";
    int parse = 0;
    int port = 8765, k = 12, threads = 1, verify = 0, step = 0;
    double stop = 2.0;  // 2.0 = 멈추지 않음 (고정 N명)
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--event") a.event = v;
        else if (k == "--frame") a.frame = v;
        else if (k == "--frags") a.frags = v;
        else if (k == "--videos") a.videos = v;
        else if (k == "--text") a.text = v;
        else if (k == "--vocab") a.vocab = v;
        else if (k == "--start") a.start = v;
        else if (k == "--port") a.port = std::stoi(v);
        else if (k == "--threads") a.threads = std::stoi(v);
        else if (k == "--tokens") a.tokens = v;
        else if (k == "--query") a.query = v;
        else if (k == "--index") a.index = v;
        else if (k == "-k") a.k = std::stoi(v);
        else if (k == "--vlm") a.vlm = v;
        else if (k == "--verify") a.verify = std::stoi(v);
        else if (k == "--step") a.step = std::stoi(v);
        else if (k == "--stop") a.stop = std::stod(v);
        else if (k == "--ontology") a.ontology = v;
        else if (k == "--parse") a.parse = std::stoi(v);
        else std::fprintf(stderr, "알 수 없는 인자 %s\n", k.c_str());
    }
    return a;
}

std::string base64(const std::string& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = (uint8_t(in[i]) << 16) | (uint8_t(in[i + 1]) << 8) | uint8_t(in[i + 2]);
        out += t[v >> 18]; out += t[(v >> 12) & 63]; out += t[(v >> 6) & 63]; out += t[v & 63];
    }
    if (i < in.size()) {
        unsigned v = uint8_t(in[i]) << 16;
        if (i + 1 < in.size()) v |= uint8_t(in[i + 1]) << 8;
        out += t[v >> 18]; out += t[(v >> 12) & 63];
        out += i + 1 < in.size() ? t[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

std::string expand(const std::string& p) {
    if (!p.empty() && p[0] == '~') return std::string(std::getenv("HOME")) + p.substr(1);
    return p;
}

std::string readFile(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// /mnt/c/Users/.. → file:///C:/Users/..  (Windows의 QML이 원본 영상을 직접 연다)
std::string fileUrl(const fs::path& p) {
    std::string s = fs::absolute(p).lexically_normal().string();
    if (s.rfind("/mnt/", 0) == 0 && s.size() > 6 && s[6] == '/')
        s = std::string(1, static_cast<char>(std::toupper(s[5]))) + ":" + s.substr(6);
    return "file:///" + s;
}

long long key(int ch, long long v) { return (static_cast<long long>(ch) << 40) ^ v; }

struct Frag { double first = 0, last = 0; int true_id = -1; };

struct Index {
    std::vector<json> items;
    std::vector<int> ch;          // 라벨 → 채널
    std::vector<double> t0, t1;   // 라벨 → 보인 시각 (하루 중 초, 영상 파일 이름의 시작 시각 + 상대 시각)
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
};

Index loadIndex(const std::string& dir) {
    Index ix;
    std::ifstream f(fs::path(dir) / "items.jsonl");
    for (std::string l; std::getline(f, l);)
        if (!l.empty()) ix.items.push_back(json::parse(l));
    const json st = json::parse(readFile((fs::path(dir) / "stats.json").string()));
    ix.space = std::make_unique<hnswlib::InnerProductSpace>(st["dim"].get<int>());
    ix.hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(ix.space.get(), (fs::path(dir) / "index.bin").string(),
                                                                false, ix.items.size());
    ix.hnsw->setEf(200);
    return ix;
}

// 하드 필터: 허용된 라벨만 hnswlib 탐색 결과에 들어간다
struct AllowFilter : hnswlib::BaseFilterFunctor {
    const std::vector<char>* ok = nullptr;
    bool operator()(hnswlib::labeltype id) override { return id < ok->size() && (*ok)[id]; }
};

int hhmm(const std::string& s) { return std::stoi(s.substr(0, 2)) * 3600 + std::stoi(s.substr(3, 2)) * 60; }
std::string hhmmss(double sec) {
    const int t = static_cast<int>(sec);
    char b[16];
    std::snprintf(b, sizeof b, "%02d:%02d:%02d", t / 3600 % 24, t / 60 % 60, t % 60);
    return b;
}

class SearchService {
public:
    explicit SearchService(const Args& a) : text_(a.text, a.vocab, a.threads) {
        if (!a.vlm.empty() && parseHostPort(a.vlm, vlm_host_, vlm_port_))
            std::fprintf(stderr, "[search] VLM 서비스 %s:%d\n", vlm_host_.c_str(), vlm_port_);
        idx_["event"] = loadIndex(expand(a.event));
        idx_["frame"] = loadIndex(expand(a.frame));

        for (const auto& e : fs::directory_iterator(a.videos)) {
            const auto ext = e.path().extension();
            if (ext == ".avi" || ext == ".mp4") videos_.push_back(e.path());
        }
        std::sort(videos_.begin(), videos_.end());  // 대역 카메라와 같은 순서 = 채널 번호

        // 온톨로지: 장소 이름표. 없으면 질문 해석(parse) 끔
        if (fs::exists(a.ontology)) {
            onto_ = json::parse(readFile(a.ontology));
            std::fprintf(stderr, "[search] 온톨로지 v%d: 장소 %zu곳\n", onto_.value("version", 0), onto_["sites"].size());
            auto alt = [](std::vector<std::string> ws) {  // 긴 단어 먼저, 정규식 특수문자 이스케이프
                std::sort(ws.begin(), ws.end(), [](auto& x, auto& y) { return x.size() > y.size(); });
                std::string r;
                for (const auto& w : ws) {
                    if (!r.empty()) r += "|";
                    r += std::regex_replace(w, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)");
                }
                return "(?:" + r + ")";
            };
            const json syn = onto_.value("synonyms", json::object()), cw = onto_.value("color_words", json::object());
            for (const auto& [site, ws] : syn.items())
                siteRe_.emplace_back(site, std::regex("(?:\\b(?:at|in|near|inside|around)\\s+)?(?:the\\s+)?" +
                                                          alt(ws) + "(?:에서|에|의)?", std::regex::icase));
            for (const auto& [color, ws] : cw.items())
                colorRe_.emplace_back(color, std::regex(alt(ws), std::regex::icase));
            if (onto_.contains("bottom_words")) bottomRe_ = std::regex(alt(onto_["bottom_words"]), std::regex::icase);
        }
        // MEVA 파일 이름: 2018-03-07.09-55-00.10-00-00.bus.G340.r13.avi → 시작 시각·장소·카메라
        static const std::regex re(R"((\d\d)-(\d\d)-(\d\d)\.\d\d-\d\d-\d\d\.([a-z]+)\.(G\d+)\.)");
        for (size_t ch = 0; ch < videos_.size(); ++ch) {
            std::smatch m;
            const std::string fn = videos_[ch].filename().string();
            std::string site, cam;
            double start = 0;
            if (std::regex_search(fn, m, re)) {
                start = std::stoi(m[1]) * 3600 + std::stoi(m[2]) * 60 + std::stoi(m[3]);
                site = m[4];
                cam = m[5];
            }
            site_.push_back(site);
            start_.push_back(start);
            const std::string ko = onto_.contains("sites") && onto_["sites"].contains(site)
                                       ? onto_["sites"][site].get<std::string>() : site;
            chName_.push_back("CH" + std::to_string(ch + 1) + (ko.empty() ? "" : " " + ko) + (cam.empty() ? "" : " " + cam));
        }

        for (const auto& e : fs::directory_iterator(a.frags)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("fragments_ch", 0) != 0) continue;
            std::ifstream f(e.path());
            for (std::string l; std::getline(f, l);) {
                const json d = json::parse(l);
                frags_[key(d["ch"].get<int>(), d["cam_id"].get<int>())] =
                    {d["first_t"].get<double>(), d["last_t"].get<double>(), d["true_id"].get<int>()};
            }
        }

        // 프레임 인덱스 크롭용 박스: (채널, 시각 ms) → 사람 목록
        std::tm tm{};
        std::istringstream ss(a.start);
        ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
        const double t0 = static_cast<double>(timegm(&tm));
        for (size_t ch = 0; ch < videos_.size(); ++ch) {
            std::ifstream f(fs::path(a.frags) / ("meta_ch" + std::to_string(ch) + ".xmls"));
            for (std::string l; std::getline(f, l);) {
                if (l.find("<tt:ImageRef>") != std::string::npos) continue;
                const double t = parseUtc(l) - t0;
                boxes_[key(static_cast<int>(ch), std::llround(t * 1000))] = MetadataParser::parse(l);
            }
            caps_.emplace_back();
        }
        for (auto& [kind, ix] : idx_) {  // 라벨별 채널·보인 시각 (필터용)
            for (const json& it : ix.items) {
                const int ch = it["ch"].get<int>();
                double tf = 1e18, tl = -1;
                if (kind == "event") {
                    for (int m : it["members"].get<std::vector<int>>()) {
                        auto f = frags_.find(key(ch, m));
                        if (f == frags_.end()) continue;
                        tf = std::min(tf, f->second.first);
                        tl = std::max(tl, f->second.last);
                    }
                } else {
                    tf = tl = it["t"].get<double>();
                }
                if (tl < 0) tf = tl = 0;
                const double base = ch < static_cast<int>(start_.size()) ? start_[ch] : 0;
                ix.ch.push_back(ch);
                ix.t0.push_back(base + tf);
                ix.t1.push_back(base + tl);
            }
        }
        std::fprintf(stderr, "[search] event %zu개 / frame %zu개 / 채널 %zu개 로드\n", idx_["event"].items.size(),
                     idx_["frame"].items.size(), videos_.size());
    }

    const ClipTokenizer& tokenizer() const { return text_.tokenizer(); }

    // 질문 처리 단계: ① 임베딩 검색으로 후보 → ② (verify>0) 상위 N명만 VLM 검증·재정렬 → ③ 상위 k개 응답
    json search(const std::string& q, const std::string& kind, int k, int verify = 0, int step = 0, double stop = 2.0,
                int parse = 0) {
        const auto t0 = std::chrono::steady_clock::now();
        Index& ix = idx_.at(kind);

        // ⓪ 질문 해석: 온톨로지 칸으로 쪼개 하드 필터 + 검색할 설명
        json parsed = nullptr;
        std::string text = q;
        std::vector<char> ok;
        size_t allowed = ix.items.size();
        if (parse) {
            parsed = parse == 2 ? parseQuery(q) : ruleParse(q);
            if (parsed.contains("description") && parsed["description"].is_string())
                text = parsed["description"].get<std::string>();
            const std::string site = parsed.value("site", json()).is_string() ? parsed["site"].get<std::string>() : "";
            const int after = parsed.value("after", json()).is_string() ? hhmm(parsed["after"]) : -1;
            const int before = parsed.value("before", json()).is_string() ? hhmm(parsed["before"]) : -1;
            if (!site.empty() || after >= 0 || before >= 0) {
                ok.assign(ix.items.size(), 0);
                allowed = 0;
                for (size_t i = 0; i < ix.items.size(); ++i) {
                    const int ch = ix.ch[i];
                    bool pass = site.empty() || (ch < static_cast<int>(site_.size()) && site_[ch] == site);
                    if (after >= 0 && ix.t1[i] < after) pass = false;    // 그 시각 전에 이미 떠남
                    if (before >= 0 && ix.t0[i] > before) pass = false;  // 그 시각 뒤에 나타남
                    ok[i] = pass;
                    allowed += pass;
                }
            }
        }

        std::vector<float> v;
        {
            std::lock_guard<std::mutex> lk(text_mutex_);
            v = text_.embed(text);
        }
        const size_t want = std::min<size_t>(std::max(k, verify), allowed);
        AllowFilter filt;
        filt.ok = &ok;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> pq;
        if (want > 0) pq = ix.hnsw->searchKnn(v.data(), want, ok.empty() ? nullptr : &filt);
        std::vector<std::pair<float, size_t>> res;
        while (!pq.empty()) { res.push_back({pq.top().first, pq.top().second}); pq.pop(); }
        std::reverse(res.begin(), res.end());  // 가까운 순
        const double search_ms = ms(t0);

        std::vector<json> hits;
        int rank = 0;
        for (const auto& [dist, label] : res) {
            const json& it = ix.items[label];
            const int ch = it["ch"].get<int>();
            json h;
            h["clip_rank"] = ++rank;
            h["score"] = std::round((1.0 - dist) * 10000) / 10000;
            h["label"] = label;
            h["ch"] = ch;
            h["channel"] = ch < static_cast<int>(chName_.size()) ? chName_[ch] : ("CH" + std::to_string(ch + 1));
            h["site"] = ch < static_cast<int>(site_.size()) ? site_[ch] : "";
            h["clock"] = hhmmss(ix.t0[label]);
            h["img"] = "/img/" + kind + "/" + std::to_string(label);
            h["video"] = fileUrl(videos_[ch]);
            std::vector<int> members;
            if (kind == "event") {
                members = it["members"].get<std::vector<int>>();
                h["top"] = it.value("top", "");
                h["bottom"] = it.value("bottom", "");
            } else {
                members = {it["cam_id"].get<int>()};
                h["top"] = h["bottom"] = "";
            }
            double tf = 1e18, tl = -1;
            for (int m : members) {
                auto f = frags_.find(key(ch, m));
                if (f == frags_.end()) continue;
                tf = std::min(tf, f->second.first);
                tl = std::max(tl, f->second.last);
            }
            if (kind == "frame") tf = tl = it["t"].get<double>();
            h["t"] = tf;
            h["t_end"] = tl;
            h["members"] = members;
            auto f0 = frags_.find(key(ch, members.front()));
            h["person"] = "ch" + std::to_string(ch) + "-" +
                          (f0 != frags_.end() ? std::to_string(f0->second.true_id) : "?");
            hits.push_back(std::move(h));
        }

        json vlm_info = nullptr;
        if (verify > 0 && !hits.empty())
            vlm_info = verifyTop(text, kind, hits, std::min<size_t>(verify, hits.size()), step, stop);

        if (hits.size() > static_cast<size_t>(k)) hits.resize(k);
        json arr = json::array();
        std::set<std::string> persons;
        rank = 0;
        for (json& h : hits) {
            h["rank"] = ++rank;
            persons.insert(h["person"].get<std::string>());
            arr.push_back(h);
        }
        json out;
        out["query"] = q;
        out["searched_text"] = text;
        out["parsed"] = parsed;
        out["allowed"] = allowed;
        out["index"] = kind;
        out["embedded"] = ix.items.size();
        out["unique"] = persons.size();
        out["search_ms"] = search_ms;
        out["vlm"] = vlm_info;
        out["ms"] = ms(t0);
        out["hits"] = arr;
        return out;
    }

    // 검증 단계: 임베딩 순서대로 step명씩 VLM 서비스에 보내 P(yes)를 받는다.
    //   지금까지 최고 P(yes)가 stop 이상이면 "확실한 후보를 찾았다"로 보고 멈춤, 아니면 다음 묶음 (최대 n명)
    //   step=0이면 n명을 한 번에 (고정 N명). 검증한 사람만 P(yes) 순으로 다시 줄 세우고, 나머지는 임베딩 순서 그대로.
    // 서비스가 없거나 실패하면 그때까지 받은 점수만 쓰고, 하나도 없으면 임베딩 순위를 그대로 둔다
    // (검색은 VLM 없이도 동작해야 함).
    json verifyTop(const std::string& q, const std::string& kind, std::vector<json>& hits, size_t n, int step,
                   double stop) {
        json info;
        info["requested"] = n;
        if (vlm_host_.empty()) {
            info["error"] = "VLM 서비스 미설정 (--vlm)";
            return info;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const size_t batch = step > 0 ? static_cast<size_t>(step) : n;
        size_t done = 0;
        int rounds = 0;
        double best = -1.0, model_ms = 0;
        while (done < n) {
            const size_t m = std::min(batch, n - done);
            json req;
            req["query"] = q;
            req["images"] = json::array();
            for (size_t i = done; i < done + m; ++i)
                req["images"].push_back(base64(image(kind, hits[i]["label"].get<size_t>())));
            const HttpResult r = httpPost(vlm_host_, vlm_port_, "/verify", req.dump(), "application/json", 60000);
            if (r.code != 200) {
                info["error"] = r.code ? "VLM 서비스 응답 " + std::to_string(r.code) : r.error;
                break;
            }
            const json resp = json::parse(r.body);
            const auto scores = resp["scores"].get<std::vector<double>>();
            for (size_t i = 0; i < m; ++i) {
                const double sc = i < scores.size() ? scores[i] : -1.0;
                hits[done + i]["vlm"] = std::round(sc * 1000) / 1000;
                best = std::max(best, sc);
            }
            model_ms += resp.value("ms", 0.0);
            done += m;
            ++rounds;
            if (best >= stop) break;  // 확실한 후보를 찾음 → 더 안 부름
        }
        std::stable_sort(hits.begin(), hits.begin() + static_cast<long>(done), [](const json& a, const json& b) {
            return a["vlm"].get<double>() > b["vlm"].get<double>();  // 같으면 원래(임베딩) 순서 유지
        });
        info["calls"] = done;
        info["rounds"] = rounds;
        info["stopped_early"] = done < n;
        info["best"] = std::round(best * 1000) / 1000;
        info["ms"] = ms(t0);
        info["model_ms"] = model_ms;
        return info;
    }

    // 결과 이미지 (jpg 바이트). 없으면 빈 문자열.
    std::string image(const std::string& kind, size_t label) {
        Index& ix = idx_.at(kind);
        if (label >= ix.items.size()) return "";
        const json& it = ix.items[label];
        if (kind == "event") return readFile(it["shot"].get<std::string>());

        {
            std::lock_guard<std::mutex> lk(cache_mutex_);
            auto c = crop_cache_.find(label);
            if (c != crop_cache_.end()) return c->second;
        }
        const int ch = it["ch"].get<int>();
        const double t = it["t"].get<double>();
        const int cam = it["cam_id"].get<int>();
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lk(video_mutex_);
            cv::VideoCapture& cap = caps_[ch];
            if (!cap.isOpened()) cap.open(videos_[ch].string());
            const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;
            cap.set(cv::CAP_PROP_POS_FRAMES, std::round(t * fps));
            if (!cap.read(frame)) return "";
        }
        auto b = boxes_.find(key(ch, std::llround(t * 1000)));
        if (b != boxes_.end()) {
            for (const Detection& d : b->second) {
                if (d.object_id != cam) continue;
                cv::Rect r(cv::Point(int(d.left * frame.cols), int(d.top * frame.rows)),
                           cv::Point(int(d.right * frame.cols), int(d.bottom * frame.rows)));
                r &= cv::Rect(0, 0, frame.cols, frame.rows);
                if (r.area() > 0) frame = frame(r).clone();
                break;
            }
        }
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 90});
        std::string s(buf.begin(), buf.end());
        std::lock_guard<std::mutex> lk(cache_mutex_);
        crop_cache_[label] = s;
        return s;
    }

    // 규칙 해석: 장소 동의어 → site, "10시 반 이후"·"after 10:30"·"오후 3시까지" → after/before, 색 단어 → 색(기록만).
    //   뗀 나머지가 묘사. 한국어가 남으면 VLM 서비스에 번역만 맡김 (실패하면 원문 그대로).
    //   LLM이 칸을 채우게 했더니 프롬프트 예시를 베끼고 없는 시각·색을 지어내서 (onto_eval 9/16), 정해진 단어는 규칙으로.
    json ruleParse(const std::string& q) {
        json p = {{"how", "rule"}, {"site", nullptr}, {"after", nullptr}, {"before", nullptr},
                  {"top_color", nullptr}, {"bottom_color", nullptr}};
        std::string rest = " " + q + " ";
        auto put = [](int h, int m, const std::string& ampm) {
            if ((ampm == "오후" || ampm[0] == 'p' || ampm[0] == 'P') && h < 12) h += 12;
            char b[8];
            std::snprintf(b, sizeof b, "%02d:%02d", h % 24, m % 60);
            return std::string(b);
        };
        static const std::regex ko(
            R"((오전|오후)?\s*(\d{1,2})\s*시\s*(?:(\d{1,2})\s*분|(반))?\s*(이후|부터|넘어서|지나서|이전|전|까지)(?:에서|에)?)");
        static const std::regex en(
            R"(\b(after|since|from|before|until|by)\s+(\d{1,2})(?::(\d\d))?\s*(am|pm|a\.m\.|p\.m\.)?)",
            std::regex::icase);
        for (std::smatch m; std::regex_search(rest, m, ko);) {
            const std::string w = m[5];
            const int min = m[3].matched ? std::stoi(m[3]) : (m[4].matched ? 30 : 0);
            p[(w == "이전" || w == "전" || w == "까지") ? "before" : "after"] = put(std::stoi(m[2]), min, m[1]);
            rest = m.prefix().str() + " " + m.suffix().str();
        }
        for (std::smatch m; std::regex_search(rest, m, en);) {
            std::string w = m[1];
            for (auto& c : w) c = static_cast<char>(std::tolower(c));
            const int min = m[3].matched ? std::stoi(m[3]) : 0;
            p[(w == "before" || w == "until" || w == "by") ? "before" : "after"] = put(std::stoi(m[2]), min, m[4]);
            rest = m.prefix().str() + " " + m.suffix().str();
        }
        for (const auto& [site, re] : siteRe_) {
            std::smatch m;
            if (std::regex_search(rest, m, re)) {
                p["site"] = site;
                rest = m.prefix().str() + " " + m.suffix().str();
                break;
            }
        }
        for (const auto& [color, re] : colorRe_) {  // 색은 묘사에서 떼지 않는다 (임베딩에도 도움)
            std::smatch m;
            if (!std::regex_search(rest, m, re)) continue;
            const bool bottom = std::regex_search(m.suffix().str().substr(0, 10), bottomRe_);
            if (p[bottom ? "bottom_color" : "top_color"].is_null()) p[bottom ? "bottom_color" : "top_color"] = color;
        }
        rest = std::regex_replace(rest, std::regex(R"(\s+)"), " ");
        rest = std::regex_replace(rest, std::regex(R"(^[\s,.]+|[\s,.?!]+$)"), "");
        if (rest.empty() || rest == "사람" || rest == "a person") rest = "a person";
        const bool hangul = std::any_of(rest.begin(), rest.end(), [](char c) { return (c & 0x80) != 0; });
        if (hangul && !vlm_host_.empty()) {
            json req;
            req["text"] = rest;
            const HttpResult r = httpPost(vlm_host_, vlm_port_, "/translate", req.dump(), "application/json", 30000);
            if (r.code == 200) {
                const json resp = json::parse(r.body);
                p["translated_from"] = rest;
                p["ms"] = resp.value("ms", 0.0);
                rest = resp.value("text", rest);
            } else {
                p["error"] = r.code ? "translate 응답 " + std::to_string(r.code) : r.error;
            }
        }
        p["description"] = rest;
        return p;
    }

    // 질문 → 온톨로지 칸 (VLM 서비스 /parse). 실패하면 빈 객체 → 필터 없이 원문으로 검색
    json parseQuery(const std::string& q) {
        if (vlm_host_.empty()) return json{{"error", "VLM 서비스 미설정 (--vlm)"}};
        json req;
        req["query"] = q;
        const HttpResult r = httpPost(vlm_host_, vlm_port_, "/parse", req.dump(), "application/json", 30000);
        if (r.code != 200) return json{{"error", r.code ? "parse 응답 " + std::to_string(r.code) : r.error}};
        json resp = json::parse(r.body);
        json p = resp.value("parsed", json::object());
        p["ms"] = resp.value("ms", 0.0);
        return p;
    }

    json info() {
        json j;
        for (auto& [k, v] : idx_) j[k]["embedded"] = v.items.size();
        return j;
    }

private:
    static double ms(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    TextEmbedder text_;
    std::string vlm_host_;
    int vlm_port_ = 0;
    std::mutex text_mutex_, video_mutex_, cache_mutex_;
    std::map<std::string, Index> idx_;
    std::vector<fs::path> videos_;
    std::vector<std::string> site_, chName_;  // 채널 → 장소, 표시 이름
    std::vector<double> start_;               // 채널 → 영상 시작 시각 (하루 중 초)
    json onto_ = json::object();
    std::vector<std::pair<std::string, std::regex>> siteRe_, colorRe_;  // 온톨로지 동의어 → 정규식
    std::regex bottomRe_{"$^"};
    std::unordered_map<long long, Frag> frags_;
    std::unordered_map<long long, std::vector<Detection>> boxes_;
    std::vector<cv::VideoCapture> caps_;
    std::unordered_map<size_t, std::string> crop_cache_;
};

std::atomic<bool> g_stop{false};

}  // namespace

int main(int argc, char** argv) {
    const Args a = parseArgs(argc, argv);

    if (!a.tokens.empty()) {  // 토크나이저 검증: 파이썬 open_clip 결과와 비교용
        ClipTokenizer tok(a.vocab);
        for (int id : tok.bpeIds(a.tokens)) std::printf("%d ", id);
        std::printf("\n");
        return 0;
    }

    SearchService svc(a);

    if (!a.query.empty()) {  // 명령줄 검색
        const json r = svc.search(a.query, a.index, a.k, a.verify, a.step, a.stop, a.parse);
        if (!r["parsed"].is_null()) std::printf("해석: %s → 검색어 \"%s\", 후보 %zu명\n", r["parsed"].dump().c_str(),
                                                r["searched_text"].get<std::string>().c_str(), r["allowed"].get<size_t>());
        std::printf("\"%s\" [%s] %.1f ms, 서로 다른 인물 %d/%zu\n", a.query.c_str(), a.index.c_str(),
                    r["ms"].get<double>(), r["unique"].get<int>(), r["hits"].size());
        for (const json& h : r["hits"])
            std::printf("  %2d. %.3f  %s  t=%.1fs  %s%s\n", h["rank"].get<int>(), h["score"].get<double>(),
                        h["channel"].get<std::string>().c_str(), h["t"].get<double>(),
                        h["person"].get<std::string>().c_str(),
                        h.contains("vlm") ? ("  VLM " + std::to_string(h["vlm"].get<double>()).substr(0, 5) +
                                             " (임베딩 " + std::to_string(h["clip_rank"].get<int>()) + "위)").c_str()
                                          : "");
        if (!r["vlm"].is_null()) std::printf("VLM: %s\n", r["vlm"].dump().c_str());
        return 0;
    }

    HttpServer server(a.port, [&](const HttpRequest& r) {
        HttpResponse res;
        if (r.path == "/search") {
            auto q = r.query.find("q");
            const std::string kind = r.query.count("index") ? r.query.at("index") : "event";
            const int k = r.query.count("k") ? std::stoi(r.query.at("k")) : 12;
            const int verify = r.query.count("verify") ? std::stoi(r.query.at("verify")) : 0;
            const int step = r.query.count("step") ? std::stoi(r.query.at("step")) : 0;
            const double stop = r.query.count("stop") ? std::stod(r.query.at("stop")) : 2.0;
            const int parse = r.query.count("parse") ? std::stoi(r.query.at("parse")) : 0;
            if (q == r.query.end() || q->second.empty() || (kind != "event" && kind != "frame")) {
                res.code = 400;
                res.body = R"({"error":"q, index=event|frame"})";
                return res;
            }
            const json out = svc.search(q->second, kind, k, verify, step, stop, parse);
            std::fprintf(stderr, "[search] %s \"%s\" %.1f ms%s\n", kind.c_str(), q->second.c_str(),
                         out["ms"].get<double>(), verify ? (" (VLM " + std::to_string(verify) + "명)").c_str() : "");
            res.body = out.dump();
        } else if (r.path.rfind("/img/", 0) == 0) {
            const std::string rest = r.path.substr(5);
            const size_t slash = rest.find('/');
            const std::string kind = rest.substr(0, slash);
            if (slash == std::string::npos || (kind != "event" && kind != "frame")) { res.code = 404; return res; }
            res.body = svc.image(kind, std::stoul(rest.substr(slash + 1)));
            res.content_type = "image/jpeg";
            if (res.body.empty()) res.code = 404;
        } else if (r.path == "/info") {
            res.body = svc.info().dump();
        } else {
            res.code = 404;
        }
        return res;
    });
    if (!server.start()) return 1;
    std::fprintf(stderr, "[search] http://localhost:%d  (Ctrl+C 종료)\n", a.port);
    std::signal(SIGINT, [](int) { g_stop = true; });
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    server.stop();
    return 0;
}
