#include "summary.hpp"

#include <cstdio>
#include <ctime>

namespace {
std::string attrAfter(const std::string& s, size_t from, const std::string& name) {
    const std::string key = name + "=\"";
    size_t p = s.find(key, from);
    if (p == std::string::npos) return "";
    p += key.size();
    size_t q = s.find('"', p);
    return q == std::string::npos ? "" : s.substr(p, q - p);
}
}  // namespace

bool parseSummary(const std::string& xml, Summary& out) {
    size_t ref = xml.find("<tt:ImageRef>");
    if (ref == std::string::npos) return false;
    size_t ref_end = xml.find("</tt:ImageRef>", ref);
    if (ref_end == std::string::npos) return false;

    // ImageRef가 속한 tt:Object 시작을 뒤로 찾는다
    size_t obj = xml.rfind("<tt:Object ", ref);
    if (obj == std::string::npos) return false;

    out.object_id = std::stoi(attrAfter(xml, obj, "ObjectId"));
    out.utc = attrAfter(xml, 0, "UtcTime");
    size_t hb = xml.find("<tt:HumanBody>", obj);
    if (hb != std::string::npos && hb < ref) {
        size_t tops = xml.find("<tt:Tops", hb);
        size_t bottoms = xml.find("<tt:Bottoms", hb);
        if (tops != std::string::npos) out.top_color = attrAfter(xml, tops, "Color");
        if (bottoms != std::string::npos) out.bottom_color = attrAfter(xml, bottoms, "Color");
    }
    const size_t start = ref + std::string("<tt:ImageRef>").size();
    out.image_ref = xml.substr(start, ref_end - start);
    return true;
}

double parseUtc(const std::string& xml) {
    // "2026-11-19T09:00:12.400Z"
    const std::string u = attrAfter(xml, 0, "UtcTime");
    if (u.size() < 19) return -1;
    std::tm tm{};
    int ms = 0;
    if (std::sscanf(u.c_str(), "%d-%d-%dT%d:%d:%d.%dZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday,
                    &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &ms) < 6)
        return -1;
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return static_cast<double>(timegm(&tm)) + ms / 1000.0;
}
