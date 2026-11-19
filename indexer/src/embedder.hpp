#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

// CLIP 이미지 인코더(ONNX) 래퍼. 결과는 L2 정규화된 벡터 (내적 = 코사인 유사도).
// ONNX Runtime 의존성은 .cpp 안에만 둔다 (pimpl).
class ImageEmbedder {
public:
    ImageEmbedder(const std::string& onnx_path, bool use_cuda, int threads);
    ~ImageEmbedder();

    std::vector<float> embed(const cv::Mat& bgr);  // 사람 크롭 1장
    int dim() const { return dim_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int dim_ = 0;
};
