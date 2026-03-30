cmake_minimum_required(VERSION 3.16)
project(libparoli-rknn VERSION 0.1.0 LANGUAGES CXX)

option(LIBPAROLI_RKNN_BUILD_EXAMPLES "Build example programs" ON)
option(LIBPAROLI_RKNN_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)
option(LIBPAROLI_RKNN_ENABLE_RKNN "Enable RKNN backend" OFF)
option(LIBPAROLI_RKNN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LIBPAROLI_RKNN_SYSTEM_PROCESSOR_LOWER)
set(LIBPAROLI_RKNN_IS_WINDOWS FALSE)
set(LIBPAROLI_RKNN_IS_LINUX FALSE)
set(LIBPAROLI_RKNN_IS_AARCH64 FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIBPAROLI_RKNN_IS_WINDOWS TRUE)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LIBPAROLI_RKNN_IS_LINUX TRUE)
endif()
if(LIBPAROLI_RKNN_SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
    set(LIBPAROLI_RKNN_IS_AARCH64 TRUE)
endif()

message(STATUS "Target OS: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Target CPU: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Linux aarch64: ${LIBPAROLI_RKNN_IS_LINUX}/${LIBPAROLI_RKNN_IS_AARCH64}")

function(libparoli_rknn_find_onnxruntime)
    if(TARGET onnxruntime::onnxruntime)
        return()
    endif()

    find_package(onnxruntime CONFIG QUIET)
    if(TARGET onnxruntime::onnxruntime)
        return()
    endif()

    find_package(onnxruntime QUIET)
    if(TARGET onnxruntime::onnxruntime)
        return()
    endif()

    message(FATAL_ERROR "ONNX Runtime not found. Set CMAKE_PREFIX_PATH to an ONNX Runtime CMake package or set ONNXRUNTIME_ROOT.")
endfunction()

add_library(libparoli-rknn
    src/audio_chunk.cpp
    src/audio_sink.cpp
    src/c_api.cpp
    src/model_config.cpp
    src/phonemizer.cpp
    src/session_options.cpp
    src/streaming_synthesizer.cpp
    src/backend_factory.cpp
    src/backends/null_backend.cpp
)

add_library(libparoli-rknn::libparoli-rknn ALIAS libparoli-rknn)

target_include_directories(libparoli-rknn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_features(libparoli-rknn PUBLIC cxx_std_17)

target_compile_definitions(libparoli-rknn
    PUBLIC
        $<$<BOOL:${LIBPAROLI_RKNN_ENABLE_ONNXRUNTIME}>:LIBPAROLI_RKNN_HAS_ONNXRUNTIME=1>
        $<$<BOOL:${LIBPAROLI_RKNN_ENABLE_RKNN}>:LIBPAROLI_RKNN_HAS_RKNN=1>
)

if(MSVC)
    target_compile_options(libparoli-rknn PRIVATE /W4)
    if(LIBPAROLI_RKNN_WARNINGS_AS_ERRORS)
        target_compile_options(libparoli-rknn PRIVATE /WX)
    endif()
else()
    target_compile_options(libparoli-rknn PRIVATE -Wall -Wextra -Wpedantic)
    if(LIBPAROLI_RKNN_WARNINGS_AS_ERRORS)
        target_compile_options(libparoli-rknn PRIVATE -Werror)
    endif()
endif()

if(LIBPAROLI_RKNN_ENABLE_ONNXRUNTIME)
    libparoli_rknn_find_onnxruntime()
    target_sources(libparoli-rknn PRIVATE src/backends/onnx_backend.cpp)
    target_link_libraries(libparoli-rknn PRIVATE onnxruntime::onnxruntime)
endif()

if(LIBPAROLI_RKNN_ENABLE_RKNN)
    if(NOT LIBPAROLI_RKNN_IS_LINUX OR NOT LIBPAROLI_RKNN_IS_AARCH64)
        message(FATAL_ERROR "RKNN backend is only supported on Linux aarch64 targets")
    endif()
    libparoli_rknn_find_onnxruntime()
    find_path(RKNN_INCLUDE_DIR rknn_api.h)
    find_library(RKNN_LIBRARY NAMES rknnrt)
    if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIBRARY)
        message(FATAL_ERROR "RKNN requested but rknn_api.h or librknnrt not found")
    endif()
    target_sources(libparoli-rknn PRIVATE src/backends/rknn_backend.cpp)
    target_include_directories(libparoli-rknn PRIVATE ${RKNN_INCLUDE_DIR})
    target_link_libraries(libparoli-rknn PRIVATE ${RKNN_LIBRARY} onnxruntime::onnxruntime)
endif()

if(LIBPAROLI_RKNN_BUILD_EXAMPLES)
    add_executable(libparoli-rknn-demo examples/demo_streaming.cpp)
    target_link_libraries(libparoli-rknn-demo PRIVATE libparoli-rknn::libparoli-rknn)
endif()

include(GNUInstallDirs)
install(TARGETS libparoli-rknn
    EXPORT libparoli-rknnTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT libparoli-rknnTargets
    FILE libparoli-rknnTargets.cmake
    NAMESPACE libparoli-rknn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libparoli-rknn
)
