#include "clip_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// UTF-8 문자열을 글자(코드포인트) 단위로 자른다
std::vector<std::string> utf8Chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

bool isLetter(unsigned char c) { return std::isalpha(c) || c >= 0x80; }  // 비ASCII는 글자로 취급
bool isDigit(unsigned char c) { return std::isdigit(c); }
bool isSpace(unsigned char c) { return std::isspace(c); }

// 정규식  's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+  를 손으로 스캔
std::vector<std::string> splitPieces(const std::string& t) {
    static const char* kContr[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<std::string> out;
    size_t i = 0;
    while (i < t.size()) {
        const unsigned char c = t[i];
        if (isSpace(c)) { ++i; continue; }
        if (c == '\'') {
            bool hit = false;
            for (const char* k : kContr) {
                const size_t n = std::char_traits<char>::length(k);
                if (t.compare(i, n, k) == 0) { out.push_back(k); i += n; hit = true; break; }
            }
            if (hit) continue;
        }
        size_t j = i;
        if (isLetter(c)) {
            while (j < t.size() && isLetter(t[j])) ++j;
        } else if (isDigit(c)) {
            j = i + 1;
        } else {
            while (j < t.size() && !isSpace(t[j]) && !isLetter(t[j]) && !isDigit(t[j])) ++j;
        }
        out.push_back(t.substr(i, j - i));
        i = j;
    }
    return out;
}

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string& vocab_path) {
    // bytes_to_unicode(): 보이는 바이트는 그대로, 나머지는 256+n 글자로
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    std::vector<std::string> vocab;
    for (size_t i = 0; i < bs.size(); ++i) {
        byte_char_[bs[i]] = utf8(cs[i]);
        vocab.push_back(utf8(cs[i]));
    }
    const size_t base = vocab.size();
    for (size_t i = 0; i < base; ++i) vocab.push_back(vocab[i] + "</w>");

    std::ifstream f(vocab_path);
    if (!f) throw std::runtime_error("BPE 어휘 파일 열기 실패: " + vocab_path);
    std::string line;
    std::getline(f, line);  // 첫 줄은 버전 주석
    const int kMerges = 49152 - 256 - 2;
    for (int r = 0; r < kMerges && std::getline(f, line); ++r) {
        std::istringstream ss(line);
        std::string a, b;
        ss >> a >> b;
        ranks_[{a, b}] = r;
        vocab.push_back(a + b);
    }
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");
    for (size_t i = 0; i < vocab.size(); ++i) encoder_[vocab[i]] = static_cast<int>(i);
    sot_ = encoder_.at("<|startoftext|>");
    eot_ = encoder_.at("<|endoftext|>");
}

std::vector<std::string> ClipTokenizer::bpe(const std::string& token) const {
    std::vector<std::string> word = utf8Chars(token);
    word.back() += "</w>";
    while (word.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = ranks_.find({word[i], word[i + 1]});
            if (it != ranks_.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT_MAX) break;
        // 가장 우선인 쌍을 왼쪽부터 전부 합친다 (원본과 같은 순서)
        const std::string a = word[at], b = word[at + 1];
        std::vector<std::string> next;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                next.push_back(a + b);
                i += 2;
            } else {
                next.push_back(word[i++]);
            }
        }
        word.swap(next);
    }
    return word;
}

std::vector<int> ClipTokenizer::bpeIds(const std::string& text) const {
    // 공백 정리 + 소문자
    std::string t;
    bool sp = false;
    for (unsigned char c : text) {
        if (isSpace(c)) { sp = true; continue; }
        if (sp && !t.empty()) t += ' ';
        sp = false;
        t += static_cast<char>(c < 0x80 ? std::tolower(c) : c);
    }
    std::vector<int> ids;
    for (const std::string& piece : splitPieces(t)) {
        std::string enc;
        for (unsigned char b : piece) enc += byte_char_[b];
        for (const std::string& sub : bpe(enc)) ids.push_back(encoder_.at(sub));
    }
    return ids;
}

std::vector<int64_t> ClipTokenizer::encode(const std::string& text, int context) const {
    std::vector<int> ids = bpeIds(text);
    std::vector<int64_t> out(context, 0);
    out[0] = sot_;
    const int n = std::min(static_cast<int>(ids.size()), context - 2);
    for (int i = 0; i < n; ++i) out[1 + i] = ids[i];
    out[1 + n] = eot_;
    return out;
}
