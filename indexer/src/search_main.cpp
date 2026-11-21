// 검색 서버 — 인덱서가 만든 인덱스(이벤트·프레임)를 읽어 자연어 검색을 HTTP로 제공.
//
//   GET /search?q=<영어 문장>&index=event|frame&k=12[&verify=10]
//       verify=N: 임베딩 상위 N명 사진을 VLM 서비스에 보내 "질문에 맞나?" 점수(P(yes))로 다시 줄 세움
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
    std::string vlm;  // VLM 서비스 주소 (비우면 검증 안 함)
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

const char* kChName[] = {"CH1 체육관", "CH2", "CH3 실외"};

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
        std::fprintf(stderr, "[search] event %zu개 / frame %zu개 / 채널 %zu개 로드\n", idx_["event"].items.size(),
                     idx_["frame"].items.size(), videos_.size());
    }

    const ClipTokenizer& tokenizer() const { return text_.tokenizer(); }

    // 질문 처리 단계: ① 임베딩 검색으로 후보 → ② (verify>0) 상위 N명만 VLM 검증·재정렬 → ③ 상위 k개 응답
    json search(const std::string& q, const std::string& kind, int k, int verify = 0, int step = 0, double stop = 2.0) {
        const auto t0 = std::chrono::steady_clock::now();
        Index& ix = idx_.at(kind);
        std::vector<float> v;
        {
            std::lock_guard<std::mutex> lk(text_mutex_);
            v = text_.embed(q);
        }
        const size_t want = std::min<size_t>(std::max(k, verify), ix.items.size());
        auto pq = ix.hnsw->searchKnn(v.data(), want);
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
            h["channel"] = ch < 3 ? kChName[ch] : ("CH" + std::to_string(ch + 1));
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
            vlm_info = verifyTop(q, kind, hits, std::min<size_t>(verify, hits.size()), step, stop);

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
        const json r = svc.search(a.query, a.index, a.k, a.verify, a.step, a.stop);
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
            if (q == r.query.end() || q->second.empty() || (kind != "event" && kind != "frame")) {
                res.code = 400;
                res.body = R"({"error":"q, index=event|frame"})";
                return res;
            }
            const json out = svc.search(q->second, kind, k, verify, step, stop);
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
