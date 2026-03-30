#include "libparoli-rknn/c_api.h"

#include "libparoli-rknn/streaming_synthesizer.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
using libparoli_rknn::AudioChunk;
using libparoli_rknn::BackendKind;
using libparoli_rknn::SessionOptions;
using libparoli_rknn::StreamingSynthesizer;

BackendKind to_backend_kind(int backend) {
    switch (backend) {
        case LPR_BACKEND_NULL:
            return BackendKind::Null;
        case LPR_BACKEND_ONNXRUNTIME:
            return BackendKind::OnnxRuntime;
        case LPR_BACKEND_RKNN:
            return BackendKind::Rknn;
        case LPR_BACKEND_AUTO:
        default:
            return BackendKind::Auto;
    }
}

SessionOptions to_session_options(const lpr_session_options* options) {
    SessionOptions result;
    if (options == nullptr) {
        return result;
    }

    result.backend = to_backend_kind(options->backend);
    result.encoder_path = options->encoder_path != nullptr ? options->encoder_path : "";
    result.decoder_path = options->decoder_path != nullptr ? options->decoder_path : "";
    result.config_path = options->config_path != nullptr ? options->config_path : "";
    result.espeak_data_path = options->espeak_data_path != nullptr ? options->espeak_data_path : "";
    result.speaker_id = options->speaker_id;
    result.length_scale = options->length_scale;
    result.noise_scale = options->noise_scale;
    result.noise_w = options->noise_w;
    result.frames_per_chunk = options->frames_per_chunk;
    result.enable_alignment = options->enable_alignment != 0;
    result.prefer_rknn_for_decoder = options->prefer_rknn_for_decoder != 0;
    return result;
}
}  // namespace

struct lpr_session {
    explicit lpr_session(SessionOptions options)
        : synth(std::move(options)) {}

    StreamingSynthesizer synth;
    std::optional<AudioChunk> last_chunk;
    std::string last_error;
};

lpr_session_options lpr_default_session_options(void) {
    const SessionOptions defaults{};
    lpr_session_options options{};
    options.backend = LPR_BACKEND_AUTO;
    options.encoder_path = nullptr;
    options.decoder_path = nullptr;
    options.config_path = nullptr;
    options.espeak_data_path = nullptr;
    options.speaker_id = defaults.speaker_id;
    options.length_scale = defaults.length_scale;
    options.noise_scale = defaults.noise_scale;
    options.noise_w = defaults.noise_w;
    options.frames_per_chunk = defaults.frames_per_chunk;
    options.enable_alignment = defaults.enable_alignment ? 1 : 0;
    options.prefer_rknn_for_decoder = defaults.prefer_rknn_for_decoder ? 1 : 0;
    return options;
}

lpr_session* lpr_create(const lpr_session_options* options) {
    try {
        return new lpr_session(to_session_options(options));
    } catch (...) {
        return nullptr;
    }
}

void lpr_destroy(lpr_session* session) {
    delete session;
}

int lpr_load(lpr_session* session) {
    if (session == nullptr) {
        return -1;
    }
    try {
        session->synth.load();
        session->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        session->last_error = e.what();
    } catch (...) {
        session->last_error = "unknown error";
    }
    return -1;
}

int lpr_start(lpr_session* session, const char* text) {
    if (session == nullptr || text == nullptr) {
        return -1;
    }
    try {
        session->synth.start(text);
        session->last_chunk.reset();
        session->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        session->last_error = e.what();
    } catch (...) {
        session->last_error = "unknown error";
    }
    return -1;
}

int lpr_next(lpr_session* session, lpr_audio_chunk* out_chunk) {
    if (session == nullptr || out_chunk == nullptr) {
        return -1;
    }
    try {
        session->last_chunk = session->synth.next();
        if (!session->last_chunk.has_value()) {
            out_chunk->samples = nullptr;
            out_chunk->sample_count = 0;
            out_chunk->sample_rate = 0;
            out_chunk->is_last = 1;
            return 0;
        }
        out_chunk->samples = session->last_chunk->samples.data();
        out_chunk->sample_count = session->last_chunk->samples.size();
        out_chunk->sample_rate = session->last_chunk->sample_rate;
        out_chunk->is_last = session->last_chunk->is_last ? 1 : 0;
        session->last_error.clear();
        return 1;
    } catch (const std::exception& e) {
        session->last_error = e.what();
    } catch (...) {
        session->last_error = "unknown error";
    }
    return -1;
}

const char* lpr_last_error(const lpr_session* session) {
    if (session == nullptr) {
        return "session is null";
    }
    return session->last_error.c_str();
}
