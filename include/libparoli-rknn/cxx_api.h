#pragma once

#include "libparoli-rknn/c_api.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace libparoli_rknn {

enum class BackendKind {
    Auto = LPR_BACKEND_AUTO,
    Null = LPR_BACKEND_NULL,
    OnnxRuntime = LPR_BACKEND_ONNXRUNTIME,
    Rknn = LPR_BACKEND_RKNN,
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

struct AudioChunkView {
    const float* samples = nullptr;
    std::size_t sample_count = 0;
    int sample_rate = 0;
    bool is_last = false;
};

class Session final {
public:
    Session()
        : Session(default_options()) {}

    explicit Session(const SessionOptions& options) {
        lpr_session_options c_options{};
        fill_c_options(options, c_options);
        session_ = lpr_create(&c_options);
        if (session_ == nullptr) {
            throw std::runtime_error("lpr_create failed");
        }
    }

    ~Session() {
        if (session_ != nullptr) {
            lpr_destroy(session_);
            session_ = nullptr;
        }
    }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    Session(Session&& other) noexcept
        : session_(other.session_) {
        other.session_ = nullptr;
    }

    Session& operator=(Session&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (session_ != nullptr) {
            lpr_destroy(session_);
        }
        session_ = other.session_;
        other.session_ = nullptr;
        return *this;
    }

    void load() {
        check_status(lpr_load(session_));
    }

    void start(const std::string& text) {
        check_status(lpr_start(session_, text.c_str()));
    }

    bool next(AudioChunkView& chunk) {
        lpr_audio_chunk c_chunk{};
        const int rc = lpr_next(session_, &c_chunk);
        if (rc < 0) {
            check_status(rc);
        }
        chunk.samples = c_chunk.samples;
        chunk.sample_count = static_cast<std::size_t>(c_chunk.sample_count);
        chunk.sample_rate = c_chunk.sample_rate;
        chunk.is_last = c_chunk.is_last != 0;
        return rc == 1;
    }

    const char* last_error() const {
        return session_ != nullptr ? lpr_last_error(session_) : "session is null";
    }

    lpr_session* native_handle() noexcept { return session_; }
    const lpr_session* native_handle() const noexcept { return session_; }

    static SessionOptions default_options() {
        const lpr_session_options c_options = lpr_default_session_options();
        SessionOptions options;
        options.backend = static_cast<BackendKind>(c_options.backend);
        options.speaker_id = c_options.speaker_id;
        options.length_scale = c_options.length_scale;
        options.noise_scale = c_options.noise_scale;
        options.noise_w = c_options.noise_w;
        options.frames_per_chunk = static_cast<std::size_t>(c_options.frames_per_chunk);
        options.enable_alignment = c_options.enable_alignment != 0;
        options.prefer_rknn_for_decoder = c_options.prefer_rknn_for_decoder != 0;
        return options;
    }

private:
    static void fill_c_options(const SessionOptions& in, lpr_session_options& out) {
        out = lpr_default_session_options();
        out.backend = static_cast<int>(in.backend);
        out.encoder_path = in.encoder_path.empty() ? nullptr : in.encoder_path.c_str();
        out.decoder_path = in.decoder_path.empty() ? nullptr : in.decoder_path.c_str();
        out.config_path = in.config_path.empty() ? nullptr : in.config_path.c_str();
        out.espeak_data_path = in.espeak_data_path.empty() ? nullptr : in.espeak_data_path.c_str();
        out.speaker_id = in.speaker_id;
        out.length_scale = in.length_scale;
        out.noise_scale = in.noise_scale;
        out.noise_w = in.noise_w;
        out.frames_per_chunk = in.frames_per_chunk;
        out.enable_alignment = in.enable_alignment ? 1 : 0;
        out.prefer_rknn_for_decoder = in.prefer_rknn_for_decoder ? 1 : 0;
    }

    void check_status(int rc) const {
        if (rc >= 0) {
            return;
        }
        throw std::runtime_error(last_error());
    }

    lpr_session* session_ = nullptr;
};

}  // namespace libparoli_rknn
