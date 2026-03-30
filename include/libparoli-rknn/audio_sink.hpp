#pragma once

#include "libparoli-rknn/audio_chunk.hpp"

#include <cstddef>
#include <mutex>
#include <vector>

namespace libparoli_rknn {

class IAudioChunkSink {
public:
    virtual ~IAudioChunkSink() = default;
    virtual void on_chunk(const AudioChunk& chunk) = 0;
};

class AudioRingBufferSink final : public IAudioChunkSink {
public:
    explicit AudioRingBufferSink(std::size_t capacity_frames = 48000);

    void on_chunk(const AudioChunk& chunk) override;
    std::size_t pop(float* dst, std::size_t max_frames);
    std::size_t size() const;
    std::size_t capacity() const noexcept { return buffer_.size(); }
    void clear();

private:
    std::vector<float> buffer_;
    std::size_t write_pos_ = 0;
    std::size_t read_pos_ = 0;
    std::size_t size_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace libparoli_rknn
