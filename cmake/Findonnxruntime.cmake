find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES include include/onnxruntime
)

if(WIN32)
    set(_ONNXRUNTIME_LIBRARY_NAMES onnxruntime)
else()
    set(_ONNXRUNTIME_LIBRARY_NAMES onnxruntime libonnxruntime)
endif()

find_library(ONNXRUNTIME_LIBRARY
    NAMES ${_ONNXRUNTIME_LIBRARY_NAMES}
    HINTS
        ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
    REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
)

if(onnxruntime_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
