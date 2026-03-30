#pragma once

#include <string>
#include <vector>

namespace libparoli_rknn {

class Phonemizer {
public:
    virtual ~Phonemizer() = default;
    virtual std::vector<int> phonemize_ids(const std::string& text) = 0;
    virtual std::vector<char32_t> phonemize_codepoints(const std::string& text) = 0;
};

class PiperPhonemizer final : public Phonemizer {
public:
    explicit PiperPhonemizer(std::string espeak_data_path = {});

    std::vector<int> phonemize_ids(const std::string& text) override;
    std::vector<char32_t> phonemize_codepoints(const std::string& text) override;

private:
    std::string espeak_data_path_;
};

}  // namespace libparoli_rknn
