#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace libparoli_rknn {

struct AudioChunk {
    std::vector<float> samples;
    int sample_rate = 22050;
    bool is_last = false;
    std::vector<char32_t> phonemes;
    std::vector<int32_t> phoneme_ids;
    std::vector<int32_t> alignments;

    [[nodiscard]] bool empty() const noexcept {
        return samples.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return samples.size();
    }
};

}  // namespace libparoli_rknn
