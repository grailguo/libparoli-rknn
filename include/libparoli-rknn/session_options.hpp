#pragma once

#include <cstddef>
#include <string>

namespace libparoli_rknn {

enum class BackendKind {
    Auto,
    Null,
    OnnxRuntime,
    Rknn,
};

struct SessionOptions {
    BackendKind backend = BackendKind::Auto;
    std::string encoder_path;
    std::string decoder_path;
    std::string config_path;
    std::string espeak_data_path;
    int speaker_id = 0;
    float length_scale = 1.0f;
    float noise_scale = 0.667f;
    float noise_w = 0.8f;
    std::size_t frames_per_chunk = 2048;
    bool enable_alignment = true;
    bool prefer_rknn_for_decoder = true;
};

}  // namespace libparoli_rknn
