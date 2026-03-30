#include "libparoli-rknn/model_config.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace libparoli_rknn {

namespace {
template <typename T>
T read_number(const nlohmann::json& root, const std::vector<std::string>& path, T fallback) {
    const nlohmann::json* cursor = &root;
    for (const std::string& key : path) {
        if (!cursor->is_object()) {
            return fallback;
        }
        auto it = cursor->find(key);
        if (it == cursor->end()) {
            return fallback;
        }
        cursor = &(*it);
    }
    if constexpr (std::is_same_v<T, int>) {
        if (cursor->is_number_integer()) {
            return cursor->get<int>();
        }
    } else {
        if (cursor->is_number()) {
            return cursor->get<T>();
        }
    }
    return fallback;
}

std::string read_string(const nlohmann::json& root, const std::vector<std::string>& path, const std::string& fallback) {
    const nlohmann::json* cursor = &root;
    for (const std::string& key : path) {
        if (!cursor->is_object()) {
            return fallback;
        }
        auto it = cursor->find(key);
        if (it == cursor->end()) {
            return fallback;
        }
        cursor = &(*it);
    }
    if (cursor->is_string()) {
        return cursor->get<std::string>();
    }
    return fallback;
}

void parse_speakers(const nlohmann::json& root, ModelConfig& cfg) {
    cfg.speakers.clear();
    auto it = root.find("speaker_id_map");
    if (it != root.end() && it->is_object()) {
        for (auto map_it = it->begin(); map_it != it->end(); ++map_it) {
            if (!map_it.value().is_array() || map_it.value().empty() || !map_it.value()[0].is_number_integer()) {
                continue;
            }
            SpeakerConfig speaker;
            speaker.id = map_it.value()[0].get<int>();
            speaker.name = map_it.key();
            cfg.speakers.push_back(std::move(speaker));
        }
        return;
    }

    it = root.find("speakers");
    if (it == root.end() || !it->is_array()) {
        return;
    }
    for (const auto& speaker_json : *it) {
        if (!speaker_json.is_object()) {
            continue;
        }
        SpeakerConfig speaker;
        if (speaker_json.contains("id") && speaker_json["id"].is_number_integer()) {
            speaker.id = speaker_json["id"].get<int>();
        }
        if (speaker_json.contains("name") && speaker_json["name"].is_string()) {
            speaker.name = speaker_json["name"].get<std::string>();
        }
        cfg.speakers.push_back(std::move(speaker));
    }
}
}  // namespace

ModelConfig ModelConfig::from_json_text(const std::string& json_text) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(std::string("invalid model config JSON: ") + e.what());
    }

    ModelConfig cfg;
    cfg.sample_rate = read_number<int>(root, {"sample_rate"}, cfg.sample_rate);
    cfg.sample_rate = read_number<int>(root, {"audio", "sample_rate"}, cfg.sample_rate);
    cfg.num_speakers = read_number<int>(root, {"num_speakers"}, cfg.num_speakers);
    cfg.default_noise_scale = read_number<float>(root, {"noise_scale"}, cfg.default_noise_scale);
    cfg.default_noise_scale = read_number<float>(root, {"inference", "noise_scale"}, cfg.default_noise_scale);
    cfg.default_noise_w = read_number<float>(root, {"noise_w"}, cfg.default_noise_w);
    cfg.default_noise_w = read_number<float>(root, {"inference", "noise_w"}, cfg.default_noise_w);
    cfg.default_length_scale = read_number<float>(root, {"length_scale"}, cfg.default_length_scale);
    cfg.default_length_scale = read_number<float>(root, {"inference", "length_scale"}, cfg.default_length_scale);
    cfg.phonemizer = read_string(root, {"phoneme_type"}, cfg.phonemizer);
    cfg.phonemizer = read_string(root, {"phonemizer", "type"}, cfg.phonemizer);
    parse_speakers(root, cfg);
    if (!cfg.speakers.empty()) {
        cfg.num_speakers = static_cast<int>(cfg.speakers.size());
    }
    return cfg;
}

}  // namespace libparoli_rknn
