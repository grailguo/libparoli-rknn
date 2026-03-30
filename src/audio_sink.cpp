#include "libparoli-rknn/audio_sink.hpp"

#include <algorithm>
#include <stdexcept>

namespace libparoli_rknn {

AudioRingBufferSink::AudioRingBufferSink(std::size_t capacity_frames)
    : buffer_(std::max<std::size_t>(1, capacity_frames), 0.0f) {}

void AudioRingBufferSink::on_chunk(const AudioChunk& chunk) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (float sample : chunk.samples) {
        buffer_[write_pos_] = sample;
        write_pos_ = (write_pos_ + 1) % buffer_.size();
        if (size_ == buffer_.size()) {
            read_pos_ = (read_pos_ + 1) % buffer_.size();
        } else {
            ++size_;
        }
    }
}

std::size_t AudioRingBufferSink::pop(float* dst, std::size_t max_frames) {
    if (dst == nullptr && max_frames > 0) {
        throw std::invalid_argument("AudioRingBufferSink::pop destination must not be null");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t n = std::min(max_frames, size_);
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = buffer_[read_pos_];
        read_pos_ = (read_pos_ + 1) % buffer_.size();
    }
    size_ -= n;
    return n;
}

std::size_t AudioRingBufferSink::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

void AudioRingBufferSink::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    write_pos_ = 0;
    read_pos_ = 0;
    size_ = 0;
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
}

}  // namespace libparoli_rknn
