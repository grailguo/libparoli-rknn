# libparoli-rknn

`libparoli-rknn` is a **C++17 project skeleton** for a streaming Piper-compatible TTS library with optional RK3588 NPU acceleration.

It is intentionally modeled after two upstream ideas:

- `libpiper`: a small reusable C/C++ synthesis library with chunked `start/next` style APIs. Piper's `libpiper` exposes a `piper_synthesize_start` / `piper_synthesize_next` flow and returns audio chunks plus alignment metadata. citeturn824931view0turn106934view2
- `paroli`: a streaming Piper implementation that separates encoder/decoder artifacts and can offload the decoder to RK3588 NPU through RKNN, while the encoder remains CPU-side because of dynamic graph constraints. citeturn293885view2turn293885view3

## What this repo contains

This scaffold gives you:

- a reusable library target: `libparoli-rknn`
- a `StreamingSynthesizer` API
- pluggable backend abstraction for:
  - null/demo backend
  - ONNX Runtime backend placeholder
  - RKNN backend placeholder
- a tiny example program that writes `demo_output.wav`
- a CMake layout that can be extended for real deployment on x86_64, ARM64, and RK3588

## What is intentionally **not** finished yet

This project is a strong starting point, but it is **not** a complete drop-in Piper runtime yet.

The following pieces still need real implementation work:

1. actual Piper streaming encoder graph execution
2. actual decoder inference for ONNX Runtime
3. RKNN decoder binding and tensor marshalling
4. real phonemization via `piper-phonemize` / `espeak-ng`
5. full JSON parsing for Piper voice config
6. chunk-level alignment logic matching upstream semantics exactly

## Why this architecture

### 1) `libpiper` style public API

Piper's C API is deliberately small: create synthesizer, get defaults, start synthesis, then pull audio chunks until done. That pattern is a good fit for embedding in GUI apps, CLI tools, and network servers. citeturn824931view0turn106934view2

### 2) `paroli` style streaming split

Paroli expects a split streaming export with separate encoder and decoder artifacts, and its README explicitly documents ONNX encoder/decoder files plus optional RKNN use for the decoder stage. citeturn293885view2turn293885view3

### 3) RK3588 limitation captured in the design

Paroli notes that the encoder is dynamic, so RKNN is not suitable for that stage; only the decoder is a realistic NPU-offload target on RK3588. This scaffold keeps that boundary explicit. citeturn293885view3turn106934view1

## Directory layout

```text
libparoli-rknn/
├── CMakeLists.txt
├── README.md
├── include/libparoli-rknn/
│   ├── audio_chunk.hpp
│   ├── backend.hpp
│   ├── model_config.hpp
│   ├── phonemizer.hpp
│   ├── session_options.hpp
│   └── streaming_synthesizer.hpp
├── src/
│   ├── audio_chunk.cpp
│   ├── backend_factory.cpp
│   ├── model_config.cpp
│   ├── phonemizer.cpp
│   ├── session_options.cpp
│   ├── streaming_synthesizer.cpp
│   └── backends/
│       ├── null_backend.cpp
│       ├── onnx_backend.cpp
│       └── rknn_backend.cpp
└── examples/
    └── demo_streaming.cpp
```

## Build

### Windows (MSVC)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DLIBPAROLI_RKNN_BUILD_EXAMPLES=ON
cmake --build build --config Release
```

### Linux aarch64

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBPAROLI_RKNN_BUILD_EXAMPLES=ON
cmake --build build -j
```

### Enable ONNX Runtime backend

```bash
cmake -S . -B build -DLIBPAROLI_RKNN_BUILD_EXAMPLES=ON
  -DLIBPAROLI_RKNN_ENABLE_ONNXRUNTIME=ON
cmake --build build -j
```

If CMake cannot find ONNX Runtime automatically, set:

- `CMAKE_PREFIX_PATH=/path/to/onnxruntime` (package config install), or
- `ONNXRUNTIME_ROOT=/path/to/onnxruntime` (library/include root)

### Enable RKNN backend (Linux aarch64 only)

Paroli documents `-DUSE_RKNN=ON` and RKNN runtime requirements for RK3588. This scaffold mirrors that as `-DLIBPAROLI_RKNN_ENABLE_RKNN=ON`. citeturn293885view0turn106934view1

```bash
cmake -S . -B build \
  -DLIBPAROLI_RKNN_ENABLE_RKNN=ON
cmake --build build -j
```

## Recommended next implementation steps

1. Replace `NaivePhonemizer` with a thin wrapper over `piper-phonemize`.
2. Parse the Piper model JSON with a real JSON library such as `nlohmann/json`.
3. Implement `OnnxBackend::load/start/next` around exported streaming `encoder.onnx` and `decoder.onnx`.
4. Keep encoder on CPU and move only decoder to `RknnBackend`.
5. Add a ring buffer or callback sink for real-time playback instead of accumulating WAV samples.
6. Add a small C wrapper if you want ABI compatibility close to `libpiper`.

## Suggested public usage

```cpp
libparoli_rknn::SessionOptions options;
options.backend = libparoli_rknn::BackendKind::Auto;
options.encoder_path = "encoder.onnx";
options.decoder_path = "decoder.onnx"; // or decoder.rknn on RK3588
options.config_path = "voice.json";

libparoli_rknn::StreamingSynthesizer synth(options);
synth.load();
synth.start("hello world");

while (auto chunk = synth.next()) {
    // play / queue / stream chunk->samples
}
```

## Notes on upstream references

- `libpiper` says it automatically builds/downloads `espeak-ng` and ONNX Runtime in its own project and exposes a shared library for Piper with a C-style API. citeturn824931view0
- `paroli` targets a web API / CLI use case and also supports WebSocket streaming. That makes it a strong design reference if you later want to add a server layer on top of this library. citeturn293885view2turn106934view1

## License

Choose a license only after confirming compatibility with every dependency and with any upstream code you directly copy.
