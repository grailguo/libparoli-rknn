#include "libparoli-rknn/streaming_synthesizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace libparoli_rknn {

namespace {
std::string read_all(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config: " + path);
    }
    std::ostringstream ss;
    ss << input.rdbuf();
    return ss.str();
}
}  // namespace

StreamingSynthesizer::StreamingSynthesizer(SessionOptions options)
    : options_(std::move(options)),
      phonemizer_(std::make_unique<PiperPhonemizer>(options_.espeak_data_path)),
      backend_(create_backend(options_)) {}

void StreamingSynthesizer::load() {
    if (!options_.config_path.empty()) {
        config_ = ModelConfig::from_json_text(read_all(options_.config_path));
    }
    if (options_.length_scale == 1.0f) {
        options_.length_scale = config_.default_length_scale;
    }
    if (options_.noise_scale == 0.667f) {
        options_.noise_scale = config_.default_noise_scale;
    }
    if (options_.noise_w == 0.8f) {
        options_.noise_w = config_.default_noise_w;
    }
    backend_->load(options_, config_);
    loaded_ = true;
}

void StreamingSynthesizer::start(const std::string& text) {
    if (!loaded_) {
        throw std::runtime_error("StreamingSynthesizer::load() must be called first");
    }
    SynthesisRequest request;
    request.text = text;
    request.phoneme_ids = phonemizer_->phonemize_ids(text);
    request.phonemes = phonemizer_->phonemize_codepoints(text);
    request.speaker_id = options_.speaker_id;
    request.length_scale = options_.length_scale;
    request.noise_scale = options_.noise_scale;
    request.noise_w = options_.noise_w;
    backend_->start(request);
    running_ = true;
}

std::optional<AudioChunk> StreamingSynthesizer::next() {
    if (!running_) {
        return std::nullopt;
    }
    AudioChunk chunk;
    if (!backend_->next(chunk)) {
        running_ = false;
        return std::nullopt;
    }
    if (sink_) {
        sink_->on_chunk(chunk);
    }
    if (chunk.is_last) {
        running_ = false;
    }
    return chunk;
}

void StreamingSynthesizer::synthesize_to_sink(const std::string& text) {
    start(text);
    while (next().has_value()) {
    }
}

void StreamingSynthesizer::reset() {
    backend_->reset();
    running_ = false;
}

void StreamingSynthesizer::set_sink(std::shared_ptr<IAudioChunkSink> sink) {
    sink_ = std::move(sink);
}

std::string StreamingSynthesizer::backend_name() const {
    return backend_ ? backend_->name() : "none";
}

}  // namespace libparoli_rknn
