#include "live_indexer.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "embedder.hpp"
#include "metadata_parser.hpp"
#include "summary.hpp"

namespace fs = std::filesystem;

namespace {
double wallNow() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int channelOf(const fs::path& p) {  // meta_ch12.xmls → 12
    const std::string s = p.stem().string();
    return std::stoi(s.substr(s.find("ch") + 2));
}
}  // namespace

struct LiveIndexer::Impl {
    // 채널 하나 = 메타데이터 파일 하나를 끝에서부터 따라 읽는 상태
    struct Chan {
        int ch = 0;
        int fd = -1;
        ino_t ino = 0;                         // 연 파일의 inode (같은 이름의 새 파일인지 알아보려고)
        off_t pos = 0;                         // 읽은 위치 (파일이 잘렸는지 알아보려고)
        std::string pending;                   // 아직 줄바꿈이 안 온 마지막 조각
        Stitcher stitcher;
        double last_utc = -1, last_wall = 0;   // 마지막으로 읽은 객체 프레임의 스트림 시각, 그때의 벽시계
        std::map<int, double> summary_wall;    // ObjectId → 요약 프레임을 읽은 벽시계
        std::map<int, double> summary_utc;     // ObjectId → 카메라가 요약 프레임에 찍은 시각
        Chan(int c, int f, StitchParams sp) : ch(c), fd(f), stitcher(sp) {}
        ~Chan() { if (fd >= 0) ::close(fd); }
    };

    LiveParams p;
    Sink sink;
    ImageEmbedder embedder;
    std::map<int, std::unique_ptr<Chan>> chans;
    std::thread th;
    std::atomic<bool> running{false};
    mutable std::mutex stats_mutex;
    Stats st;

    Impl(LiveParams pp, Sink s) : p(std::move(pp)), sink(std::move(s)), embedder(p.model, false, p.threads) {}

    void count(long Stats::*field) {
        std::lock_guard<std::mutex> lk(stats_mutex);
        ++(st.*field);
    }

    // 채널 파일 찾기: 새 채널(카메라가 늦게 붙어도 됨), 같은 이름의 새 파일·잘린 파일(새 스트림)
    void discover() {
        const double now = wallNow();
        for (const auto& e : fs::directory_iterator(p.dir)) {
            const fs::path& f = e.path();
            if (f.extension() != ".xmls" || f.stem().string().rfind("meta_ch", 0) != 0) continue;
            const int ch = channelOf(f);
            struct stat fst {};
            if (::stat(f.c_str(), &fst) != 0) continue;
            auto it = chans.find(ch);
            const bool known = it != chans.end();
            if (known) {
                const Chan& old = *it->second;
                if (fst.st_ino == old.ino && fst.st_size >= old.pos) continue;  // 같은 스트림
                std::fprintf(stderr, "[live] ch%d: 새 스트림 → 처음부터 다시 읽음\n", ch);
            }
            const int fd = ::open(f.c_str(), O_RDONLY);
            if (fd < 0) continue;
            auto c = std::make_unique<Chan>(ch, fd, p.stitch);
            c->ino = fst.st_ino;
            if (!known && now - static_cast<double>(fst.st_mtime) > p.stale_sec) {
                // 서버가 켜지기 전부터 멈춰 있던 파일 = 지난 스트림 → 이전 내용은 건너뛰고 새 줄부터
                c->pos = ::lseek(fd, 0, SEEK_END);
                std::fprintf(stderr, "[live] ch%d: %.0f초 동안 안 바뀐 파일 → 이전 내용 건너뜀\n", ch,
                             now - static_cast<double>(fst.st_mtime));
            } else if (!known) {
                std::fprintf(stderr, "[live] ch%d 연결: %s\n", ch, f.filename().c_str());
            }
            chans[ch] = std::move(c);  // 새 스트림이면 이전 상태(잇기 중이던 조각)는 버린다
            std::lock_guard<std::mutex> lk(stats_mutex);
            st.channels = static_cast<int>(chans.size());
        }
    }

    void handleLine(Chan& c, const std::string& line) {
        count(&Stats::lines);
        Summary s;
        if (parseSummary(line, s)) {  // 요약 프레임: 사람 한 조각이 화면에서 사라짐
            count(&Stats::summaries);
            const fs::path shot = fs::absolute(fs::path(p.dir) / "bestshots" / ("ch" + std::to_string(c.ch)) /
                                               fs::path(s.image_ref).filename());
            const cv::Mat img = cv::imread(shot.string());
            c.stitcher.summary(s.object_id, shot.string(), img.empty() ? 0 : img.rows, s.top_color, s.bottom_color);
            c.summary_wall[s.object_id] = wallNow();
            c.summary_utc[s.object_id] = parseUtc(line);
            return;
        }
        const double t = parseUtc(line);  // 객체 프레임: 위치로 조각 잇기
        if (t < 0) return;
        c.stitcher.observe(t, MetadataParser::parse(line));
        c.last_utc = std::max(c.last_utc, t);
        c.last_wall = wallNow();
    }

    // 파일 끝에 새로 붙은 줄만 읽는다 (tail -f)
    void readNew(Chan& c) {
        char buf[1 << 16];
        ssize_t n;
        while ((n = ::read(c.fd, buf, sizeof buf)) > 0) {
            c.pos += n;
            c.pending.append(buf, static_cast<size_t>(n));
            size_t from = 0, nl;
            while ((nl = c.pending.find('\n', from)) != std::string::npos) {
                if (nl > from) handleLine(c, c.pending.substr(from, nl - from));
                from = nl + 1;
            }
            c.pending.erase(0, from);
        }
    }

    void commit(Chan& c, const Group& g) {
        const double ready = wallNow();
        count(&Stats::groups);
        if (g.seen_sec < p.min_dur) { count(&Stats::drop_short); return; }
        if (g.shot_h < p.min_shot_h) { count(&Stats::drop_small); return; }
        const cv::Mat img = cv::imread(g.shot);
        if (img.empty()) { count(&Stats::drop_small); return; }

        LivePerson lp;
        const auto t0 = std::chrono::steady_clock::now();
        lp.vec = embedder.embed(img);
        lp.embed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        lp.ch = c.ch;
        lp.members = g.members;
        lp.first_utc = g.first;
        lp.last_utc = g.last;
        lp.seen_sec = g.seen_sec;
        lp.shot = g.shot;
        lp.top = g.top;
        lp.bottom = g.bottom;
        lp.ready_wall = ready;
        for (int m : g.members) {
            auto it = c.summary_wall.find(m);
            if (it != c.summary_wall.end()) lp.summary_wall = std::max(lp.summary_wall, it->second);
            auto u = c.summary_utc.find(m);
            if (u != c.summary_utc.end()) lp.summary_utc = std::max(lp.summary_utc, u->second);
        }
        count(&Stats::kept);
        sink(std::move(lp));
    }

    void tick() {
        discover();
        for (auto& [ch, cp] : chans) {
            Chan& c = *cp;
            readNew(c);
            if (c.last_utc < 0) continue;
            // 사람이 없으면 프레임이 안 오므로, 마지막 프레임 시각에 흐른 벽시계를 더해 지금 스트림 시각을 추정
            const double now = c.last_utc + (wallNow() - c.last_wall);
            for (const Group& g : c.stitcher.takeReady(now, false)) commit(c, g);
        }
    }

    void loop() {
        while (running) {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                tick();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[live] 오류: %s\n", e.what());
            }
            std::this_thread::sleep_until(t0 + std::chrono::milliseconds(p.poll_ms));
        }
    }
};

LiveIndexer::LiveIndexer(LiveParams p, Sink sink) : impl_(std::make_unique<Impl>(std::move(p), std::move(sink))) {}

LiveIndexer::~LiveIndexer() { stop(); }

void LiveIndexer::start() {
    fs::create_directories(impl_->p.dir);
    impl_->running = true;
    impl_->th = std::thread([this] { impl_->loop(); });
    std::fprintf(stderr, "[live] %s 따라 읽는 중 (%dms마다)\n", impl_->p.dir.c_str(), impl_->p.poll_ms);
}

void LiveIndexer::stop() {
    if (!impl_ || !impl_->running) return;
    impl_->running = false;
    if (impl_->th.joinable()) impl_->th.join();
}

LiveIndexer::Stats LiveIndexer::stats() const {
    std::lock_guard<std::mutex> lk(impl_->stats_mutex);
    return impl_->st;
}

void LiveIndexer::resetStats() {
    std::lock_guard<std::mutex> lk(impl_->stats_mutex);
    const int ch = impl_->st.channels;
    impl_->st = Stats{};
    impl_->st.channels = ch;
}
