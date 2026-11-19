// 이벤트 기반 인덱서: 카메라 메타데이터의 요약 프레임(사람 퇴장)을 신호로
// 베스트샷 한 장만 임베딩해서 벡터 인덱스에 쌓는다. (--stitch 1이면 끊긴 조각을 이어서 그룹 단위로)
//
//   ./indexer --meta out_raw --model clip_image.onnx --out idx [--min-dur 0.5] [--stitch 1]
//
// 입력: <meta>/meta_ch*.xmls (한 줄에 MetadataStream 문서 1개), <meta>/bestshots/chN/*.jpg
// 출력: <out>/index.bin (hnswlib), items.jsonl (벡터 번호 → 사람 정보), links.jsonl, stats.json
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include <hnswlib/hnswlib.h>
#include <opencv2/imgcodecs.hpp>

#include "embedder.hpp"
#include "metadata_parser.hpp"
#include "stitcher.hpp"
#include "summary.hpp"

namespace fs = std::filesystem;

namespace {
struct Args {
    std::string meta = "out_raw";
    std::string model = "clip_image.onnx";
    std::string out = "idx_event";
    double min_dur = 1.0;  // 그룹이 실제로 보인 시간이 이보다 짧으면 임베딩 안 함 (초)
    int min_shot_h = 64;   // 대표 베스트샷 세로가 이보다 작으면 임베딩 안 함 (픽셀)
    bool cuda = false;
    int threads = 4;
    StitchParams stitch;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    a.stitch.enabled = false;  // 기본은 잇기 없음 (기존 결과와 비교하려고)
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--cuda") { a.cuda = true; continue; }
        if (i + 1 >= argc) break;
        const std::string v = argv[++i];
        if (k == "--meta") a.meta = v;
        else if (k == "--model") a.model = v;
        else if (k == "--out") a.out = v;
        else if (k == "--min-dur") a.min_dur = std::stod(v);
        else if (k == "--min-shot-h") a.min_shot_h = std::stoi(v);
        else if (k == "--threads") a.threads = std::stoi(v);
        else if (k == "--stitch") a.stitch.enabled = (v == "1");
        else if (k == "--link-sec") a.stitch.link_sec = std::stod(v);
        else if (k == "--link-dist") a.stitch.link_dist = std::stof(v);
    }
    return a;
}

int channelOf(const fs::path& p) {  // meta_ch12.xmls → 12
    const std::string s = p.stem().string();
    return std::stoi(s.substr(s.find("ch") + 2));
}
}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    fs::create_directories(args.out);

    ImageEmbedder embedder(args.model, args.cuda, args.threads);
    hnswlib::InnerProductSpace space(embedder.dim());
    size_t capacity = 100000;
    hnswlib::HierarchicalNSW<float> index(&space, capacity, 16, 200);

    std::ofstream items(fs::path(args.out) / "items.jsonl");
    std::ofstream links(fs::path(args.out) / "links.jsonl");
    long summaries = 0, groups = 0, kept = 0, drop_short = 0, drop_small = 0, n_links = 0;
    double embed_ms_total = 0;

    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(args.meta))
        if (e.path().extension() == ".xmls") files.push_back(e.path());
    std::sort(files.begin(), files.end(),
              [](const fs::path& a, const fs::path& b) { return channelOf(a) < channelOf(b); });

    for (const auto& file : files) {
        const int ch = channelOf(file);
        Stitcher stitcher(args.stitch);

        // 그룹 확정 → 필터 → 임베딩
        auto commit = [&](const Group& g) {
            ++groups;
            if (g.seen_sec < args.min_dur) { ++drop_short; return; }
            if (g.shot_h < args.min_shot_h) { ++drop_small; return; }
            cv::Mat img = cv::imread(g.shot);
            if (img.empty()) { ++drop_small; return; }

            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> vec = embedder.embed(img);
            embed_ms_total +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (static_cast<size_t>(kept) >= capacity) {
                capacity *= 2;
                index.resizeIndex(capacity);
            }
            index.addPoint(vec.data(), static_cast<hnswlib::labeltype>(kept));

            items << "{\"label\":" << kept << ",\"ch\":" << ch << ",\"cam_id\":" << g.members.front()
                  << ",\"members\":[";
            for (size_t i = 0; i < g.members.size(); ++i) items << (i ? "," : "") << g.members[i];
            items << "],\"t_first\":" << std::fixed << std::setprecision(3) << g.first << ",\"t_last\":" << g.last << ",\"seen\":" << g.seen_sec
                  << ",\"top\":\"" << g.top << "\",\"bottom\":\"" << g.bottom << "\",\"shot\":\"" << g.shot
                  << "\"}\n" << std::defaultfloat;
            ++kept;
        };

        std::ifstream in(file);
        std::string line;
        double now = 0;
        while (std::getline(in, line)) {
            Summary s;
            if (parseSummary(line, s)) {
                ++summaries;
                const fs::path shot = fs::absolute(fs::path(args.meta) / "bestshots" /
                                                   ("ch" + std::to_string(ch)) /
                                                   fs::path(s.image_ref).filename());
                const cv::Mat img = cv::imread(shot.string());
                stitcher.summary(s.object_id, shot.string(), img.empty() ? 0 : img.rows, s.top_color,
                                 s.bottom_color);
                continue;
            }
            now = parseUtc(line);
            stitcher.observe(now, MetadataParser::parse(line));
            for (const Group& g : stitcher.takeReady(now, false)) commit(g);
        }
        for (const Group& g : stitcher.takeReady(std::numeric_limits<double>::max(), true)) commit(g);

        for (const auto& l : stitcher.links()) {
            links << "{\"ch\":" << ch << ",\"from\":" << l.from << ",\"to\":" << l.to << ",\"dist\":" << l.dist
                  << ",\"gap\":" << l.gap << "}\n";
            ++n_links;
        }
    }

    index.saveIndex((fs::path(args.out) / "index.bin").string());
    const double avg = kept ? embed_ms_total / kept : 0;
    std::ofstream st(fs::path(args.out) / "stats.json");
    st << "{\"summaries\":" << summaries << ",\"groups\":" << groups << ",\"links\":" << n_links
       << ",\"embedded\":" << kept << ",\"drop_short\":" << drop_short << ",\"drop_small\":" << drop_small
       << ",\"embed_ms_avg\":" << avg << ",\"dim\":" << embedder.dim() << "}\n";
    std::printf("[indexer] 요약 %ld개 → 그룹 %ld개 (이음 %ld) → 임베딩 %ld개 (짧음 %ld, 작음 %ld), 임베딩 평균 %.2f ms\n",
                summaries, groups, n_links, kept, drop_short, drop_small, avg);
    return 0;
}
