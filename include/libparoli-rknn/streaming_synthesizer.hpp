#pragma once

#include "libparoli-rknn/audio_chunk.hpp"
#include "libparoli-rknn/audio_sink.hpp"
#include "libparoli-rknn/backend.hpp"
#include "libparoli-rknn/model_config.hpp"
#include "libparoli-rknn/phonemizer.hpp"
#include "libparoli-rknn/session_options.hpp"

#include <memory>
#include <optional>
#include <string>

namespace libparoli_rknn {

class StreamingSynthesizer {
public:
    explicit StreamingSynthesizer(SessionOptions options);

    void load();
    void start(const std::string& text);
    std::optional<AudioChunk> next();
    void synthesize_to_sink(const std::string& text);
    void reset();
    void set_sink(std::shared_ptr<IAudioChunkSink> sink);

    [[nodiscard]] const SessionOptions& options() const noexcept { return options_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::string backend_name() const;

private:
    SessionOptions options_;
    ModelConfig config_;
    std::unique_ptr<Phonemizer> phonemizer_;
    std::unique_ptr<IStreamingBackend> backend_;
    std::shared_ptr<IAudioChunkSink> sink_;
    bool loaded_ = false;
    bool running_ = false;
};

}  // namespace libparoli_rknn
