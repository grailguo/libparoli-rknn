#include "libparoli-rknn/backend.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace libparoli_rknn {

class NullBackend final : public IStreamingBackend {
public:
    void load(const SessionOptions& options, const ModelConfig& config) override {
        options_ = options;
        config_ = config;
        cursor_ = 0;
        rendered_.clear();
    }

    void start(const SynthesisRequest& request) override {
        cursor_ = 0;
        rendered_.clear();

        const int sample_rate = config_.sample_rate;
        const double seconds_per_symbol = 0.04 * static_cast<double>(request.length_scale);
        for (std::size_t i = 0; i < request.phoneme_ids.size(); ++i) {
            const int token = request.phoneme_ids[i];
            const double frequency = 180.0 + static_cast<double>(token % 48) * 10.0;
            const std::size_t samples = static_cast<std::size_t>(seconds_per_symbol * sample_rate);
            for (std::size_t s = 0; s < samples; ++s) {
                const double t = static_cast<double>(s) / static_cast<double>(sample_rate);
                const float sample = static_cast<float>(0.15 * std::sin(2.0 * 3.14159265358979323846 * frequency * t));
                rendered_.push_back(sample);
            }
        }
        phonemes_ = request.phonemes;
        phoneme_ids_ = request.phoneme_ids;
    }

    bool next(AudioChunk& chunk) override {
        if (cursor_ >= rendered_.size()) {
            return false;
        }
        const std::size_t n = std::min(options_.frames_per_chunk, rendered_.size() - cursor_);
        chunk.samples.assign(rendered_.begin() + static_cast<std::ptrdiff_t>(cursor_),
                             rendered_.begin() + static_cast<std::ptrdiff_t>(cursor_ + n));
        chunk.sample_rate = config_.sample_rate;
        chunk.phonemes = phonemes_;
        chunk.phoneme_ids = phoneme_ids_;
        chunk.alignments.assign(phoneme_ids_.size(), static_cast<int32_t>(n / std::max<std::size_t>(1, phoneme_ids_.size())));
        cursor_ += n;
        chunk.is_last = cursor_ >= rendered_.size();
        return true;
    }

    void reset() override {
        cursor_ = 0;
        rendered_.clear();
    }

    [[nodiscard]] const char* name() const noexcept override {
        return "null";
    }

private:
    SessionOptions options_;
    ModelConfig config_;
    std::vector<float> rendered_;
    std::vector<char32_t> phonemes_;
    std::vector<int32_t> phoneme_ids_;
    std::size_t cursor_ = 0;
};

std::unique_ptr<IStreamingBackend> make_null_backend() {
    return std::make_unique<NullBackend>();
}

}  // namespace libparoli_rknn
