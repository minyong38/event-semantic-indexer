#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clip_tokenizer.hpp"

// CLIP 텍스트 인코더(ONNX) 래퍼: 문장 → L2 정규화 벡터 (이미지 벡터와 내적 = 코사인 유사도)
class TextEmbedder {
public:
    TextEmbedder(const std::string& onnx_path, const std::string& vocab_path, int threads);
    ~TextEmbedder();

    std::vector<float> embed(const std::string& text);
    const ClipTokenizer& tokenizer() const { return tok_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ClipTokenizer tok_;
    int dim_ = 0;
};
