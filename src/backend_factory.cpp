#include "libparoli-rknn/backend.hpp"

#include <memory>
#include <stdexcept>

namespace libparoli_rknn {
std::unique_ptr<IStreamingBackend> make_null_backend();
std::unique_ptr<IStreamingBackend> make_onnx_backend();
std::unique_ptr<IStreamingBackend> make_rknn_backend();

std::unique_ptr<IStreamingBackend> create_backend(const SessionOptions& options) {
    switch (options.backend) {
        case BackendKind::Null:
            return make_null_backend();
        case BackendKind::OnnxRuntime:
#if LIBPAROLI_RKNN_HAS_ONNXRUNTIME
            return make_onnx_backend();
#else
            throw std::runtime_error("ONNX Runtime backend requested but not compiled in");
#endif
        case BackendKind::Rknn:
#if LIBPAROLI_RKNN_HAS_RKNN
            return make_rknn_backend();
#else
            throw std::runtime_error("RKNN backend requested but not compiled in");
#endif
        case BackendKind::Auto:
        default:
#if LIBPAROLI_RKNN_HAS_RKNN
            if (options.prefer_rknn_for_decoder) {
                return make_rknn_backend();
            }
#endif
#if LIBPAROLI_RKNN_HAS_ONNXRUNTIME
            return make_onnx_backend();
#else
            return make_null_backend();
#endif
    }
}

}  // namespace libparoli_rknn
