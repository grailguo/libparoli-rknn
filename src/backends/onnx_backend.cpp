#include "libparoli-rknn/backend.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace libparoli_rknn {

class OnnxBackend final : public IStreamingBackend {
public:
    void load(const SessionOptions& options, const ModelConfig& config) override {
        options_ = options;
        config_ = config;
        reset();

        if (options_.encoder_path.empty()) {
            throw std::runtime_error("onnx backend requires encoder_path");
        }
        if (options_.decoder_path.empty()) {
            throw std::runtime_error("onnx backend requires decoder_path");
        }

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "libparoli-rknn");
        session_options_ = Ort::SessionOptions{};
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        encoder_ = std::make_unique<Ort::Session>(*env_, options_.encoder_path.c_str(), session_options_);
        decoder_ = std::make_unique<Ort::Session>(*env_, options_.decoder_path.c_str(), session_options_);

        cache_io_names(*encoder_, true, encoder_input_name_storage_, encoder_input_names_);
        cache_io_names(*encoder_, false, encoder_output_name_storage_, encoder_output_names_);
        cache_io_names(*decoder_, true, decoder_input_name_storage_, decoder_input_names_);
        cache_io_names(*decoder_, false, decoder_output_name_storage_, decoder_output_names_);
    }

    void start(const SynthesisRequest& request) override {
        if (!encoder_ || !decoder_) {
            throw std::runtime_error("onnx backend not loaded");
        }

        rendered_.clear();
        cursor_ = 0;
        phonemes_ = request.phonemes;
        phoneme_ids_ = request.phoneme_ids;

        const std::vector<int64_t> token_ids(phoneme_ids_.begin(), phoneme_ids_.end());
        const std::vector<int64_t> lengths = {static_cast<int64_t>(token_ids.size())};
        const std::vector<int64_t> speaker = {static_cast<int64_t>(request.speaker_id)};
        const std::vector<float> scales = {request.noise_scale, request.length_scale, request.noise_w};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> encoder_inputs;
        std::vector<const char*> encoder_input_names;
        append_encoder_inputs(memory_info, token_ids, lengths, speaker, scales, encoder_inputs, encoder_input_names);

        auto encoder_outputs = encoder_->Run(
            Ort::RunOptions{nullptr},
            encoder_input_names.data(),
            encoder_inputs.data(),
            encoder_inputs.size(),
            encoder_output_names_.data(),
            encoder_output_names_.size());

        std::vector<float> latent_values;
        std::vector<int64_t> latent_shape;
        extract_largest_float_tensor(encoder_outputs, latent_values, latent_shape);
        if (latent_values.empty() || latent_shape.empty()) {
            throw std::runtime_error("encoder did not produce a usable latent tensor");
        }

        std::vector<Ort::Value> decoder_inputs;
        std::vector<const char*> decoder_input_names;
        append_decoder_inputs(memory_info, latent_values, latent_shape, speaker, decoder_inputs, decoder_input_names);

        auto decoder_outputs = decoder_->Run(
            Ort::RunOptions{nullptr},
            decoder_input_names.data(),
            decoder_inputs.data(),
            decoder_inputs.size(),
            decoder_output_names_.data(),
            decoder_output_names_.size());

        extract_largest_float_tensor(decoder_outputs, rendered_, decoded_shape_);
        if (rendered_.empty()) {
            throw std::runtime_error("decoder did not produce waveform output");
        }

        if (options_.enable_alignment && !phoneme_ids_.empty()) {
            const int32_t width = static_cast<int32_t>(rendered_.size() / std::max<std::size_t>(1, phoneme_ids_.size()));
            alignments_.assign(phoneme_ids_.size(), std::max<int32_t>(1, width));
        }
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
        chunk.alignments = alignments_;
        cursor_ += n;
        chunk.is_last = cursor_ >= rendered_.size();
        return true;
    }

    void reset() override {
        rendered_.clear();
        phonemes_.clear();
        phoneme_ids_.clear();
        alignments_.clear();
        decoded_shape_.clear();
        cursor_ = 0;
    }

    [[nodiscard]] const char* name() const noexcept override {
        return "onnxruntime";
    }

private:
    static void cache_io_names(const Ort::Session& session,
                               bool inputs,
                               std::vector<std::string>& storage,
                               std::vector<const char*>& names) {
        storage.clear();
        names.clear();
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
        storage.reserve(count);
        names.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto name_alloc = inputs ? session.GetInputNameAllocated(i, allocator)
                                     : session.GetOutputNameAllocated(i, allocator);
            storage.emplace_back(name_alloc.get());
        }
        for (const auto& name : storage) {
            names.push_back(name.c_str());
        }
    }

    static int find_name_index(const std::vector<std::string>& names, const std::vector<std::string>& candidates) {
        for (const auto& candidate : candidates) {
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (names[i] == candidate) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }

    static void extract_largest_float_tensor(const std::vector<Ort::Value>& outputs,
                                             std::vector<float>& values,
                                             std::vector<int64_t>& shape) {
        values.clear();
        shape.clear();
        std::size_t best_size = 0;
        for (const auto& output : outputs) {
            if (!output.IsTensor()) {
                continue;
            }
            auto info = output.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                continue;
            }
            const auto current_shape = info.GetShape();
            const std::size_t current_size = info.GetElementCount();
            if (current_size <= best_size) {
                continue;
            }
            const float* data = output.GetTensorData<float>();
            values.assign(data, data + static_cast<std::ptrdiff_t>(current_size));
            shape = current_shape;
            best_size = current_size;
        }
    }

    void append_encoder_inputs(const Ort::MemoryInfo& memory_info,
                               const std::vector<int64_t>& token_ids,
                               const std::vector<int64_t>& lengths,
                               const std::vector<int64_t>& speaker,
                               const std::vector<float>& scales,
                               std::vector<Ort::Value>& inputs,
                               std::vector<const char*>& input_names) {
        const std::vector<int64_t> token_shape = {1, static_cast<int64_t>(token_ids.size())};
        const std::vector<int64_t> scalar_shape = {1};
        const std::vector<int64_t> scales_shape = {1, 3};

        auto push_tensor = [&](int index, Ort::Value value) {
            if (index < 0) {
                return;
            }
            input_names.push_back(encoder_input_names_[static_cast<std::size_t>(index)]);
            inputs.push_back(std::move(value));
        };

        const int ids_index = find_name_index(encoder_input_name_storage_, {"input", "input_ids", "tokens", "phoneme_ids"});
        push_tensor(ids_index, Ort::Value::CreateTensor<int64_t>(
                                   memory_info,
                                   const_cast<int64_t*>(token_ids.data()),
                                   token_ids.size(),
                                   token_shape.data(),
                                   token_shape.size()));

        const int len_index = find_name_index(encoder_input_name_storage_, {"input_lengths", "lengths", "text_lengths"});
        push_tensor(len_index, Ort::Value::CreateTensor<int64_t>(
                                   memory_info,
                                   const_cast<int64_t*>(lengths.data()),
                                   lengths.size(),
                                   scalar_shape.data(),
                                   scalar_shape.size()));

        const int sid_index = find_name_index(encoder_input_name_storage_, {"sid", "speaker_id"});
        push_tensor(sid_index, Ort::Value::CreateTensor<int64_t>(
                                   memory_info,
                                   const_cast<int64_t*>(speaker.data()),
                                   speaker.size(),
                                   scalar_shape.data(),
                                   scalar_shape.size()));

        const int scales_index = find_name_index(encoder_input_name_storage_, {"scales", "noise_scale"});
        push_tensor(scales_index, Ort::Value::CreateTensor<float>(
                                      memory_info,
                                      const_cast<float*>(scales.data()),
                                      scales.size(),
                                      scales_shape.data(),
                                      scales_shape.size()));

        if (input_names.empty()) {
            throw std::runtime_error("unable to match encoder input names");
        }
    }

    void append_decoder_inputs(const Ort::MemoryInfo& memory_info,
                               std::vector<float>& latent_values,
                               const std::vector<int64_t>& latent_shape,
                               const std::vector<int64_t>& speaker,
                               std::vector<Ort::Value>& inputs,
                               std::vector<const char*>& input_names) {
        auto push_tensor = [&](int index, Ort::Value value) {
            if (index < 0) {
                return;
            }
            input_names.push_back(decoder_input_names_[static_cast<std::size_t>(index)]);
            inputs.push_back(std::move(value));
        };

        const int latent_index = find_name_index(decoder_input_name_storage_, {"z", "input", "latent", "decoder_input"});
        push_tensor(latent_index, Ort::Value::CreateTensor<float>(
                                      memory_info,
                                      latent_values.data(),
                                      latent_values.size(),
                                      latent_shape.data(),
                                      latent_shape.size()));

        const std::vector<int64_t> scalar_shape = {1};
        const int sid_index = find_name_index(decoder_input_name_storage_, {"sid", "speaker_id"});
        push_tensor(sid_index, Ort::Value::CreateTensor<int64_t>(
                                   memory_info,
                                   const_cast<int64_t*>(speaker.data()),
                                   speaker.size(),
                                   scalar_shape.data(),
                                   scalar_shape.size()));

        if (input_names.empty()) {
            throw std::runtime_error("unable to match decoder input names");
        }
    }

    SessionOptions options_;
    ModelConfig config_;
    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;
    std::vector<std::string> encoder_input_name_storage_;
    std::vector<std::string> encoder_output_name_storage_;
    std::vector<std::string> decoder_input_name_storage_;
    std::vector<std::string> decoder_output_name_storage_;
    std::vector<const char*> encoder_input_names_;
    std::vector<const char*> encoder_output_names_;
    std::vector<const char*> decoder_input_names_;
    std::vector<const char*> decoder_output_names_;
    std::vector<float> rendered_;
    std::vector<int64_t> decoded_shape_;
    std::vector<char32_t> phonemes_;
    std::vector<int32_t> phoneme_ids_;
    std::vector<int32_t> alignments_;
    std::size_t cursor_ = 0;
};

std::unique_ptr<IStreamingBackend> make_onnx_backend() {
    return std::make_unique<OnnxBackend>();
}

}  // namespace libparoli_rknn
