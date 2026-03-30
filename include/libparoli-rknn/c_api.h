#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#define LPR_SIZE_T std::size_t
#else
#include <stddef.h>
#define LPR_SIZE_T size_t
#endif

typedef enum lpr_backend_kind {
    LPR_BACKEND_AUTO = 0,
    LPR_BACKEND_NULL = 1,
    LPR_BACKEND_ONNXRUNTIME = 2,
    LPR_BACKEND_RKNN = 3,
} lpr_backend_kind;

typedef struct lpr_session_options {
    int backend;
    const char* encoder_path;
    const char* decoder_path;
    const char* config_path;
    const char* espeak_data_path;
    int speaker_id;
    float length_scale;
    float noise_scale;
    float noise_w;
    LPR_SIZE_T frames_per_chunk;
    int enable_alignment;
    int prefer_rknn_for_decoder;
} lpr_session_options;

typedef struct lpr_audio_chunk {
    const float* samples;
    LPR_SIZE_T sample_count;
    int sample_rate;
    int is_last;
} lpr_audio_chunk;

typedef struct lpr_session lpr_session;

lpr_session_options lpr_default_session_options(void);
lpr_session* lpr_create(const lpr_session_options* options);
void lpr_destroy(lpr_session* session);
int lpr_load(lpr_session* session);
int lpr_start(lpr_session* session, const char* text);
int lpr_next(lpr_session* session, lpr_audio_chunk* out_chunk);
const char* lpr_last_error(const lpr_session* session);

#ifdef __cplusplus
}
#endif

#undef LPR_SIZE_T
