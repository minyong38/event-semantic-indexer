#include "text_embedder.hpp"

#include <cmath>

#include <onnxruntime_cxx_api.h>

struct TextEmbedder::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "search"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    std::string in_name, out_name;
    bool int64_in = true;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

TextEmbedder::TextEmbedder(const std::string& onnx_path, const std::string& vocab_path, int threads)
    : impl_(std::make_unique<Impl>()), tok_(vocab_path) {
    impl_->opts.SetIntraOpNumThreads(threads);
    impl_->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_path.c_str(), impl_->opts);
    Ort::AllocatorWithDefaultOptions alloc;
    impl_->in_name = impl_->session->GetInputNameAllocated(0, alloc).get();
    impl_->out_name = impl_->session->GetOutputNameAllocated(0, alloc).get();
    impl_->int64_in = impl_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() ==
                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    dim_ = static_cast<int>(impl_->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().back());
}

TextEmbedder::~TextEmbedder() = default;

std::vector<float> TextEmbedder::embed(const std::string& text) {
    std::vector<int64_t> ids = tok_.encode(text);
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    const int64_t shape[2] = {1, static_cast<int64_t>(ids.size())};
    Ort::Value in = impl_->int64_in
        ? Ort::Value::CreateTensor<int64_t>(impl_->mem, ids.data(), ids.size(), shape, 2)
        : Ort::Value::CreateTensor<int32_t>(impl_->mem, ids32.data(), ids32.size(), shape, 2);
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
