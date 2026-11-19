#include "embedder.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace {
constexpr int kSize = 224;
// CLIP 학습 때 쓴 정규화 값 (open_clip / OpenAI CLIP 공통)
constexpr float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float kStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};
}  // namespace

struct ImageEmbedder::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "indexer"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

ImageEmbedder::ImageEmbedder(const std::string& onnx_path, bool use_cuda, int threads)
    : impl_(std::make_unique<Impl>()) {
    impl_->opts.SetIntraOpNumThreads(threads);
    impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
        // GPU 빌드 ORT일 때만 성공. CPU 빌드면 예외 → CPU로 진행
        try {
            OrtCUDAProviderOptions cuda{};
            impl_->opts.AppendExecutionProvider_CUDA(cuda);
            std::fprintf(stderr, "[embed] CUDA 실행 백엔드\n");
        } catch (const Ort::Exception& e) {
            std::fprintf(stderr, "[embed] CUDA 불가 → CPU (%s)\n", e.what());
        }
    }
    impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_path.c_str(), impl_->opts);

    Ort::AllocatorWithDefaultOptions alloc;
    impl_->in_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    impl_->out_name = impl_->session->GetOutputNameAllocated(0, alloc).get();
    auto shape = impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    dim_ = static_cast<int>(shape.back());
}

ImageEmbedder::~ImageEmbedder() = default;

std::vector<float> ImageEmbedder::embed(const cv::Mat& bgr) {
    // 사람 크롭은 세로로 길다. CLIP 기본(가운데 자르기)을 쓰면 머리·발이 잘리므로
    // 정사각형으로 여백을 채운 뒤(letterbox) 224로 줄인다 — 사람 전체를 보존.
    const int side = std::max(bgr.cols, bgr.rows);
    cv::Mat square(side, side, CV_8UC3, cv::Scalar(104, 117, 123));  // 평균색 여백 (BGR)
    bgr.copyTo(square(cv::Rect((side - bgr.cols) / 2, (side - bgr.rows) / 2, bgr.cols, bgr.rows)));
    cv::Mat resized, rgb;
    cv::resize(square, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input(3 * kSize * kSize);
    for (int y = 0; y < kSize; ++y) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kSize; ++x)
            for (int c = 0; c < 3; ++c)
                input[c * kSize * kSize + y * kSize + x] = (row[x][c] / 255.0f - kMean[c]) / kStd[c];
    }
    const int64_t in_shape[4] = {1, 3, kSize, kSize};
    Ort::Value in = Ort::Value::CreateTensor<float>(impl_->mem, input.data(), input.size(), in_shape, 4);

    const char* in_names[] = {impl_->in_name.c_str()};
    const char* out_names[] = {impl_->out_name.c_str()};
    auto outs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

    const float* p = outs[0].GetTensorData<float>();
    std::vector<float> v(p, p + dim_);
    double norm = 0;
    for (float f : v) norm += double(f) * f;
    const float inv = norm > 0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.f;
    for (float& f : v) f *= inv;
    return v;
}
