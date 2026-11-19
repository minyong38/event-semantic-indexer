#include "stitcher.hpp"

#include <algorithm>
#include <cmath>

int Stitcher::newGroup(int oid, double t) {
    Group g;
    g.id = next_group_++;
    g.members.push_back(oid);
    g.first = g.last = t;
    groups_[g.id] = g;
    return g.id;
}

void Stitcher::observe(double t, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        if (!d.isHuman() || d.width() <= 0) continue;
        const float fx = (d.left + d.right) / 2, fy = d.bottom;  // 발끝 (다보이조와 같은 기준)
        auto it = objs_.find(d.object_id);

        if (it == objs_.end()) {
            // ── 새 ID 등장: 최근에 사라진 조각 중 이어 붙일 후보 찾기 ──
            Obj o;
            o.first = o.last = t;
            o.x = fx; o.y = fy; o.h = d.height();
            int best = -1;
            float best_d = p_.link_dist;
            double best_gap = 0;
            if (p_.enabled) {
                for (auto& [oid, prev] : objs_) {
                    if (prev.has_next || prev.group == 0) continue;
                    const double gap = t - prev.last;
                    if (gap <= 0 || gap > p_.link_sec) continue;  // 아직 보이는 중이거나 너무 오래됨
                    const float ratio = std::max(prev.h, o.h) / std::max(1e-3f, std::min(prev.h, o.h));
                    if (ratio > p_.size_ratio) continue;
                    // 사라진 자리에서 속도로 예측 (예측 폭은 1초분으로 제한 — 외삽 폭주 방지)
                    const float dt = static_cast<float>(std::min(gap, 1.0));
                    const float px = prev.x + (prev.has_v ? prev.vx * dt : 0);
                    const float py = prev.y + (prev.has_v ? prev.vy * dt : 0);
                    const float dist = std::hypot(px - fx, py - fy);
                    if (dist < best_d) { best_d = dist; best = oid; best_gap = gap; }
                }
            }
            if (best >= 0) {
                Obj& prev = objs_[best];
                prev.has_next = true;
                o.group = prev.group;
                groups_[o.group].members.push_back(d.object_id);
                links_.push_back({best, d.object_id, best_d, best_gap});
            } else {
                o.group = newGroup(d.object_id, t);
            }
            objs_[d.object_id] = o;
            continue;
        }

        // ── 이미 보던 ID: 위치·속도 갱신 ──
        Obj& o = it->second;
        const double dt = t - o.last;
        if (dt > 1e-3) {
            o.vx = static_cast<float>((fx - o.x) / dt);
            o.vy = static_cast<float>((fy - o.y) / dt);
            o.has_v = true;
        }
        Group& g = groups_[o.group];
        g.seen_sec += std::min(dt, 1.0);  // 끊긴 구간은 보인 시간에 안 넣음
        g.last = std::max(g.last, t);
        o.x = fx; o.y = fy; o.h = d.height(); o.last = t;
    }
}

void Stitcher::summary(int object_id, const std::string& shot, int shot_h,
                       const std::string& top, const std::string& bottom) {
    auto it = objs_.find(object_id);
    if (it == objs_.end()) {  // 객체 프레임 없이 요약만 온 조각 — 혼자 그룹
        Obj o;
        o.group = newGroup(object_id, -1);
        it = objs_.emplace(object_id, o).first;
    }
    it->second.ended = true;
    Group& g = groups_[it->second.group];
    ++g.ended;
    if (shot_h > g.shot_h) {  // 그룹 안에서 가장 큰 베스트샷을 대표로
        g.shot = shot; g.shot_h = shot_h; g.top = top; g.bottom = bottom;
    }
}

std::vector<Group> Stitcher::takeReady(double t, bool flush) {
    std::vector<Group> out;
    for (auto& [gid, g] : groups_) {
        if (g.committed || g.ended == 0) continue;
        if (!flush) {
            // 모든 조각이 끝났고, 마지막 조각 이후 이어 붙일 시간이 지나야 확정
            if (g.ended < static_cast<int>(g.members.size())) continue;
            const Obj& tail = objs_[g.members.back()];
            if (t - tail.last <= p_.link_sec) continue;
        }
        g.committed = true;
        out.push_back(g);
    }
    return out;
}
