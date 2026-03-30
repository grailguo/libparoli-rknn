#pragma once

#include <string>
#include <vector>

namespace libparoli_rknn {

struct SpeakerConfig {
    int id = 0;
    std::string name;
};

struct ModelConfig {
    int sample_rate = 22050;
    int num_speakers = 1;
    float default_noise_scale = 0.667f;
    float default_noise_w = 0.8f;
    float default_length_scale = 1.0f;
    std::string phonemizer = "espeak";
    std::vector<SpeakerConfig> speakers;

    static ModelConfig from_json_text(const std::string& json_text);
};

}  // namespace libparoli_rknn
