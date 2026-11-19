// 검색 서버 — 인덱서가 만든 인덱스(이벤트·프레임)를 읽어 자연어 검색을 HTTP로 제공.
//
//   GET /search?q=<영어 문장>&index=event|frame&k=12
//   GET /img/event/<label>   그룹 대표 베스트샷 (파일 그대로)
//   GET /img/frame/<label>   그 시각 영상 프레임에서 사람 크롭 (메타데이터 박스)
//   GET /info                인덱스별 임베딩 수
//
//   search_server --event ~/idx_stB --frame ~/idx_frame --frags ../camsim/out_raw \
//                 --videos ../camsim/meva_raw --text ../model/clip_text.onnx \
//                 --vocab ../model/bpe_simple_vocab_16e6.txt [--port 8765]
//   검증용: --tokens "문장"  (토큰 ID 출력 후 종료), --query "문장" [--index event]  (상위 결과 출력 후 종료)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
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
    int port = 8765, k = 12, threads = 1;
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
        else std::fprintf(stderr, "알 수 없는 인자 %s\n", k.c_str());
    }
    return a;
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

    json search(const std::string& q, const std::string& kind, int k) {
        const auto t0 = std::chrono::steady_clock::now();
        Index& ix = idx_.at(kind);
        std::vector<float> v;
        {
            std::lock_guard<std::mutex> lk(text_mutex_);
            v = text_.embed(q);
        }
        auto pq = ix.hnsw->searchKnn(v.data(), std::min<size_t>(k, ix.items.size()));
        std::vector<std::pair<float, size_t>> res;
        while (!pq.empty()) { res.push_back({pq.top().first, pq.top().second}); pq.pop(); }
        std::reverse(res.begin(), res.end());  // 가까운 순

        json hits = json::array();
        std::set<std::string> persons;
        int rank = 0;
        for (const auto& [dist, label] : res) {
            const json& it = ix.items[label];
            const int ch = it["ch"].get<int>();
            json h;
            h["rank"] = ++rank;
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
            const std::string person = "ch" + std::to_string(ch) + "-" +
                                       (f0 != frags_.end() ? std::to_string(f0->second.true_id) : "?");
            h["person"] = person;
            persons.insert(person);
            hits.push_back(h);
        }
        json out;
        out["query"] = q;
        out["index"] = kind;
        out["embedded"] = ix.items.size();
        out["unique"] = persons.size();
        out["ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        out["hits"] = hits;
        return out;
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
    TextEmbedder text_;
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
        const json r = svc.search(a.query, a.index, a.k);
        std::printf("\"%s\" [%s] %.1f ms, 서로 다른 인물 %d/%zu\n", a.query.c_str(), a.index.c_str(),
                    r["ms"].get<double>(), r["unique"].get<int>(), r["hits"].size());
        for (const json& h : r["hits"])
            std::printf("  %2d. %.3f  %s  t=%.1fs  %s\n", h["rank"].get<int>(), h["score"].get<double>(),
                        h["channel"].get<std::string>().c_str(), h["t"].get<double>(),
                        h["person"].get<std::string>().c_str());
        return 0;
    }

    HttpServer server(a.port, [&](const HttpRequest& r) {
        HttpResponse res;
        if (r.path == "/search") {
            auto q = r.query.find("q");
            const std::string kind = r.query.count("index") ? r.query.at("index") : "event";
            const int k = r.query.count("k") ? std::stoi(r.query.at("k")) : 12;
            if (q == r.query.end() || q->second.empty() || (kind != "event" && kind != "frame")) {
                res.code = 400;
                res.body = R"({"error":"q, index=event|frame"})";
                return res;
            }
            const json out = svc.search(q->second, kind, k);
            std::fprintf(stderr, "[search] %s \"%s\" %.1f ms\n", kind.c_str(), q->second.c_str(),
                         out["ms"].get<double>());
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
