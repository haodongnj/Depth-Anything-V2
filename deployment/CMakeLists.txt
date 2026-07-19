cmake_minimum_required(VERSION 3.10)
project(DepthAnythingTRT VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile_commands.json for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# CUDA architecture auto-detection
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(GPU_CC)
        string(REPLACE "." "" GPU_SM "${GPU_CC}")
        set(CMAKE_CUDA_ARCHITECTURES "${GPU_SM}")
        message(STATUS "CUDA arch from nvidia-smi: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Jetson detection (no nvidia-smi, read device-tree)
        if(EXISTS /proc/device-tree/model)
            file(READ /proc/device-tree/model NVIDIA_MODEL)
            if(NVIDIA_MODEL MATCHES "Jetson")
                set(CMAKE_CUDA_ARCHITECTURES "53")   # Maxwell SM 5.3
                message(STATUS "Jetson platform — CUDA arch: 53")
            endif()
        endif()
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            # Desktop/server fallback (Turing through Ada)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
            message(STATUS "Default CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# CUDA toolkit  (cmake ≥ 3.17 → CUDAToolkit,  older → manual)
# ---------------------------------------------------------------------------
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    find_package(CUDAToolkit REQUIRED)
else()
    find_path(CUDA_INCLUDE_DIRS cuda_runtime.h
        PATHS /usr/local/cuda/include
    )
    find_library(CUDA_CUDART_LIBRARY cudart
        PATHS /usr/local/cuda/lib64 /usr/local/cuda/lib
    )
    if(NOT CUDA_INCLUDE_DIRS OR NOT CUDA_CUDART_LIBRARY)
        message(FATAL_ERROR "CUDA toolkit not found (set CUDA_PATH if custom location)")
    endif()
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA cudart:  ${CUDA_CUDART_LIBRARY}")

    add_library(CUDA::cudart SHARED IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "${CUDA_CUDART_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
    )
endif()

# ---------------------------------------------------------------------------
# TensorRT
# ---------------------------------------------------------------------------
find_package(TensorRT REQUIRED)

# Detect TensorRT major version from NvInferVersion.h
if(TensorRT_INCLUDE_DIR)
    set(_TRT_VER_HDR "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
else()
    # Search common locations if FindTensorRT didn't cache the include dir
    foreach(_dir /usr/include /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
                  /usr/local/tensorrt /opt/tensorrt)
        if(EXISTS "${_dir}/NvInferVersion.h")
            set(_TRT_VER_HDR "${_dir}/NvInferVersion.h")
            break()
        endif()
    endforeach()
endif()

if(EXISTS "${_TRT_VER_HDR}")
    file(STRINGS "${_TRT_VER_HDR}" _TRT_MAJOR_LINE
        REGEX "^#define NV_TENSORRT_MAJOR ")
    string(REGEX MATCH "[0-9]+" TRT_VERSION_MAJOR "${_TRT_MAJOR_LINE}")

    file(STRINGS "${_TRT_VER_HDR}" _TRT_MINOR_LINE
        REGEX "^#define NV_TENSORRT_MINOR ")
    string(REGEX MATCH "[0-9]+" TRT_VERSION_MINOR "${_TRT_MINOR_LINE}")

    set(TRT_VERSION "${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}")
    message(STATUS "TensorRT version: ${TRT_VERSION}")
else()
    set(TRT_VERSION "unknown")
    message(WARNING "NvInferVersion.h not found — assuming TRT ≥ 10")
endif()

if(TRT_VERSION VERSION_LESS "10")
    add_definitions(-DDEPTH_TRT_LEGACY_API)
    message(STATUS "TRT < 10 → using legacy binding API")
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(depth_trt)
add_subdirectory(depth_image)
add_subdirectory(depth_camera)
