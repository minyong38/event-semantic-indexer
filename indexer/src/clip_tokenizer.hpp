#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// CLIP 텍스트 토크나이저 (open_clip SimpleTokenizer의 C++ 포팅).
// 바이트 수준 BPE: 문장 → 소문자·공백 정리 → 정규식 조각 → 바이트를 유니코드 글자로 → BPE 병합 → 토큰 ID.
// 출력은 [시작] + 토큰 + [끝], 77개로 0 패딩 (모델 입력 그대로).
// 원본과 다른 점: ftfy(깨진 유니코드 복원)·html unescape는 생략 — 데모 질문은 평범한 영어 문장.
class ClipTokenizer {
public:
    // vocab_path: bpe_simple_vocab_16e6.txt (gz 푼 것)
    explicit ClipTokenizer(const std::string& vocab_path);

    std::vector<int64_t> encode(const std::string& text, int context = 77) const;
    std::vector<int> bpeIds(const std::string& text) const;  // 시작·끝·패딩 없이 (검증용)

private:
    std::vector<std::string> bpe(const std::string& token) const;  // 바이트 인코딩된 조각 → BPE 조각들

    std::string byte_char_[256];                                 // 바이트 → 유니코드 글자(UTF-8)
    std::unordered_map<std::string, int> encoder_;               // BPE 조각 → ID
    std::map<std::pair<std::string, std::string>, int> ranks_;   // 병합 쌍 → 우선순위
    int sot_ = 0, eot_ = 0;
};
