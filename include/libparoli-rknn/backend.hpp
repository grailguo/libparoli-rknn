#pragma once

#include "libparoli-rknn/audio_chunk.hpp"
#include "libparoli-rknn/model_config.hpp"
#include "libparoli-rknn/session_options.hpp"

#include <memory>
#include <string>
#include <vector>

namespace libparoli_rknn {

struct SynthesisRequest {
    std::string text;
    std::vector<int> phoneme_ids;
    std::vector<char32_t> phonemes;
    int speaker_id = 0;
    float length_scale = 1.0f;
    float noise_scale = 0.667f;
    float noise_w = 0.8f;
};

class IStreamingBackend {
public:
    virtual ~IStreamingBackend() = default;

    virtual void load(const SessionOptions& options, const ModelConfig& config) = 0;
    virtual void start(const SynthesisRequest& request) = 0;
    virtual bool next(AudioChunk& chunk) = 0;
    virtual void reset() = 0;
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

std::unique_ptr<IStreamingBackend> create_backend(const SessionOptions& options);

}  // namespace libparoli_rknn
