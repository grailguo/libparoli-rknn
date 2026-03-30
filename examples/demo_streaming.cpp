#include "libparoli-rknn/audio_sink.hpp"
#include "libparoli-rknn/streaming_synthesizer.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    using namespace libparoli_rknn;

    SessionOptions options;
    options.backend = BackendKind::Null;
    options.frames_per_chunk = 1024;

    StreamingSynthesizer synth(options);
    auto sink = std::make_shared<AudioRingBufferSink>(options.frames_per_chunk * 8);
    synth.set_sink(sink);

    synth.load();
    synth.start("Hello from libparoli-rknn real-time streaming demo.");

    std::vector<float> playback_window(512, 0.0f);
    std::size_t played_frames = 0;
    while (auto chunk = synth.next()) {
        std::cout << "chunk=" << chunk->size() << " buffered=" << sink->size() << " last=" << chunk->is_last << '\n';
        const std::size_t drained = sink->pop(playback_window.data(), playback_window.size());
        played_frames += drained;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    while (sink->size() > 0) {
        const std::size_t drained = sink->pop(playback_window.data(), playback_window.size());
        if (drained == 0) {
            break;
        }
        played_frames += drained;
    }

    std::cout << "Finished streaming with backend: " << synth.backend_name()
              << ", played_frames=" << played_frames << '\n';
    return 0;
}
