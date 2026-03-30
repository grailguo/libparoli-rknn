#include "libparoli-rknn/phonemizer.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace libparoli_rknn {

namespace {
std::vector<char32_t> utf8_to_codepoints(const std::string& text) {
    std::vector<char32_t> cps;
    cps.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch < 0x80) {
            cps.push_back(static_cast<char32_t>(ch));
            ++i;
            continue;
        }
        if ((ch & 0xE0) == 0xC0 && i + 1 < text.size()) {
            const auto c1 = static_cast<uint32_t>(ch & 0x1F);
            const auto c2 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F);
            cps.push_back(static_cast<char32_t>((c1 << 6U) | c2));
            i += 2;
            continue;
        }
        if ((ch & 0xF0) == 0xE0 && i + 2 < text.size()) {
            const auto c1 = static_cast<uint32_t>(ch & 0x0F);
            const auto c2 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F);
            const auto c3 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3F);
            cps.push_back(static_cast<char32_t>((c1 << 12U) | (c2 << 6U) | c3));
            i += 3;
            continue;
        }
        if ((ch & 0xF8) == 0xF0 && i + 3 < text.size()) {
            const auto c1 = static_cast<uint32_t>(ch & 0x07);
            const auto c2 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F);
            const auto c3 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3F);
            const auto c4 = static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3F);
            cps.push_back(static_cast<char32_t>((c1 << 18U) | (c2 << 12U) | (c3 << 6U) | c4));
            i += 4;
            continue;
        }
        cps.push_back(static_cast<char32_t>('?'));
        ++i;
    }
    return cps;
}

std::string run_phonemize_process(const std::string& text, const std::string& espeak_data_path) {
#if defined(_WIN32)
    const std::string quote = "\"";
    const std::string stderr_redir = "2>nul";
#else
    const std::string quote = "'";
    const std::string stderr_redir = "2>/dev/null";
#endif
    std::ostringstream cmd;
    cmd << "piper-phonemize --input " << quote << text << quote;
    if (!espeak_data_path.empty()) {
        cmd << " --espeak-data " << quote << espeak_data_path << quote;
    }
    cmd << " " << stderr_redir;

#if defined(_WIN32)
    FILE* pipe = _popen(cmd.str().c_str(), "r");
#else
    FILE* pipe = popen(cmd.str().c_str(), "r");
#endif
    if (pipe == nullptr) {
        return {};
    }

    std::array<char, 512> out{};
    std::string result;
    while (fgets(out.data(), static_cast<int>(out.size()), pipe) != nullptr) {
        result.append(out.data());
    }

#if defined(_WIN32)
    const int exit_code = _pclose(pipe);
#else
    const int exit_code = pclose(pipe);
#endif
    if (exit_code != 0) {
        return {};
    }
    return result;
}

std::vector<char32_t> sanitize_phoneme_text(const std::string& phoneme_text) {
    std::vector<char32_t> cps;
    const std::vector<char32_t> raw = utf8_to_codepoints(phoneme_text);
    cps.reserve(raw.size());
    for (char32_t cp : raw) {
        if (cp == U'\r' || cp == U'\n' || cp == U'\t') {
            continue;
        }
        cps.push_back(cp);
    }
    return cps;
}
}  // namespace

PiperPhonemizer::PiperPhonemizer(std::string espeak_data_path)
    : espeak_data_path_(std::move(espeak_data_path)) {}

std::vector<int> PiperPhonemizer::phonemize_ids(const std::string& text) {
    const auto cps = phonemize_codepoints(text);
    std::vector<int> ids;
    ids.reserve(cps.size() + 2);
    ids.push_back(1);
    for (char32_t cp : cps) {
        ids.push_back(static_cast<int>(cp));
        ids.push_back(0);
    }
    ids.push_back(2);
    return ids;
}

std::vector<char32_t> PiperPhonemizer::phonemize_codepoints(const std::string& text) {
    const std::string from_piper = run_phonemize_process(text, espeak_data_path_);
    if (!from_piper.empty()) {
        return sanitize_phoneme_text(from_piper);
    }
    return utf8_to_codepoints(text);
}

}  // namespace libparoli_rknn
