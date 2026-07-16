# TensorRT C++ Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two C++ executables (`depth_image`, `depth_camera`) that share a `libdepth_trt` static library for TensorRT inference with Depth Anything V2.

**Architecture:** A static library (`depth_trt`) encapsulates all TensorRT engine loading, CUDA inference, image preprocessing, and depth visualization. Two thin executables (`depth_image`, `depth_camera`) handle only CLI parsing and I/O (file vs. camera stream), reusing the library pipeline: preprocess → infer → visualize.

**Tech Stack:** C++17, CMake 3.20+, CUDA 12.8, TensorRT 11.1, OpenCV 4.x

## Global Constraints

- C++17 (`CMAKE_CXX_STANDARD 17`)
- Engine resolution must match CLI `--height`/`--width` exactly — mismatch is fatal
- Preprocessing: BGR→RGB/255 → center-crop → resize exact → normalize(mean/std) → CHW → batch dim
- Normalization constants: MEAN={0.485, 0.456, 0.406}, STD={0.229, 0.224, 0.225}
- Depth visualization: Inferno colormap (default) or grayscale (--grayscale flag)
- Camera: center-crop from native resolution to engine aspect ratio, then resize exact
- All errors are fatal (print + exit(1)) except camera frame drops (warn + continue)
- RAII for all GPU resources (Engine class owns CUDA/TRT objects)
- No exceptions across library boundaries

---

## File Structure Map

| File | Responsibility |
|------|---------------|
| `deployment/CMakeLists.txt` | Top-level: find deps, add subdirectories |
| `deployment/cmake/FindTensorRT.cmake` | Portable TRT discovery (env var → apt → tar → JetPack) |
| `deployment/depth_trt/CMakeLists.txt` | Static library target `depth_trt` |
| `deployment/depth_trt/include/depth_trt/types.h` | Resolution, ColorMode, MEAN/STD constants |
| `deployment/depth_trt/include/depth_trt/engine.h` | Engine class declaration |
| `deployment/depth_trt/src/engine.cpp` | Engine: load .engine, CUDA alloc, infer |
| `deployment/depth_trt/include/depth_trt/preprocess.h` | preprocess() declaration |
| `deployment/depth_trt/src/preprocess.cpp` | BGR→RGB, center-crop, resize, normalize, CHW |
| `deployment/depth_trt/include/depth_trt/visualize.h` | depth_to_color(), depth_to_gray() declarations |
| `deployment/depth_trt/src/visualize.cpp` | Depth → uint8 BGR with Inferno LUT or grayscale |
| `deployment/depth_image/CMakeLists.txt` | Executable target `depth_image` |
| `deployment/depth_image/main.cpp` | Single-image CLI: parse → load → pipeline → save |
| `deployment/depth_camera/CMakeLists.txt` | Executable target `depth_camera` |
| `deployment/depth_camera/main.cpp` | Camera CLI: parse → open → loop(pipeline + imshow) |

---

### Task 1: Project scaffolding — CMake, FindTensorRT, directory structure

**Files:**
- Create: `deployment/CMakeLists.txt`
- Create: `deployment/cmake/FindTensorRT.cmake`
- Create: `deployment/depth_trt/CMakeLists.txt`
- Create: `deployment/depth_trt/include/depth_trt/` (empty dir placeholder)
- Create: `deployment/depth_trt/src/` (empty dir placeholder)
- Create: `deployment/depth_image/CMakeLists.txt`
- Create: `deployment/depth_camera/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: build system that finds CUDA, TensorRT, OpenCV; `depth_trt` static library target (empty for now); `depth_image` and `depth_camera` executable targets (stub main.cpp only)

- [ ] **Step 1: Create FindTensorRT.cmake**

```cmake
# deployment/cmake/FindTensorRT.cmake
#[=======================================================================[.rst:
FindTensorRT
------------

Find the NVIDIA TensorRT inference SDK.

Imported targets
^^^^^^^^^^^^^^^^
``TensorRT::nvinfer``

Result variables
^^^^^^^^^^^^^^^^
``TensorRT_FOUND``
``TensorRT_INCLUDE_DIRS``
``TensorRT_LIBRARIES``

Search order
^^^^^^^^^^^^
1. $ENV{TRT_ROOT}
2. /usr (apt install)
3. /usr/local/tensorrt (tar install)
4. /opt/tensorrt (JetPack)
#]=======================================================================]

include(FindPackageHandleStandardArgs)

# Build the search list
set(_TRT_SEARCH_PATHS
    /usr
    /usr/local/tensorrt
    /opt/tensorrt
)

if(DEFINED ENV{TRT_ROOT})
    list(INSERT _TRT_SEARCH_PATHS 0 "$ENV{TRT_ROOT}")
endif()

# Find include dir
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES include include/x86_64-linux-gnu
    DOC "TensorRT include directory"
)

# Find library
find_library(TensorRT_LIBRARY
    NAMES nvinfer
    PATHS ${_TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu
    DOC "TensorRT inference library"
)

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)
```

- [ ] **Step 2: Create top-level CMakeLists.txt**

```cmake
# deployment/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(DepthAnythingTRT VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow the user to override CUDA arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")  # Turing through Ada
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

add_subdirectory(depth_trt)
add_subdirectory(depth_image)
add_subdirectory(depth_camera)
```

- [ ] **Step 3: Create depth_trt/CMakeLists.txt**

```cmake
# deployment/depth_trt/CMakeLists.txt
add_library(depth_trt STATIC
    src/engine.cpp
    src/preprocess.cpp
    src/visualize.cpp
)

target_include_directories(depth_trt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(depth_trt
    PUBLIC
        TensorRT::nvinfer
        CUDA::cudart
        opencv_core
        opencv_imgproc
)
```

- [ ] **Step 4: Create depth_image/CMakeLists.txt**

```cmake
# deployment/depth_image/CMakeLists.txt
add_executable(depth_image main.cpp)
target_link_libraries(depth_image PRIVATE depth_trt opencv_imgcodecs)
```

- [ ] **Step 5: Create depth_camera/CMakeLists.txt**

```cmake
# deployment/depth_camera/CMakeLists.txt
add_executable(depth_camera main.cpp)
target_link_libraries(depth_camera PRIVATE depth_trt opencv_highgui opencv_videoio)
```

- [ ] **Step 6: Create stub main.cpp files to verify CMake builds**

```cpp
// deployment/depth_image/main.cpp (stub)
#include <cstdio>
int main() { printf("depth_image: not yet implemented\n"); return 0; }
```

```cpp
// deployment/depth_camera/main.cpp (stub)
#include <cstdio>
int main() { printf("depth_camera: not yet implemented\n"); return 0; }
```

- [ ] **Step 7: Create placeholder .cpp files in depth_trt/src/**

```cpp
// deployment/depth_trt/src/engine.cpp (stub)
// (empty — just needs to compile as part of the static lib)
```

```cpp
// deployment/depth_trt/src/preprocess.cpp (stub)
```

```cpp
// deployment/depth_trt/src/visualize.cpp (stub)
```

- [ ] **Step 8: Configure and build to verify scaffolding**

Run:
```bash
cd deployment && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Expected: CMake configures without errors, all targets compile and link. Three stubs print messages.

- [ ] **Step 9: Clean up stubs and commit**

Remove the stub .cpp contents from `engine.cpp`, `preprocess.cpp`, `visualize.cpp` (leave empty files). Keep stub main.cpp files.

```bash
cd /home/hao/hdd/ai/haodongnj/Depth-Anything-V2
git add deployment/
git commit -m "feat: add CMake scaffolding for TensorRT C++ deployment

- cmake/FindTensorRT.cmake: portable TRT discovery
- depth_trt/ static library target (empty impl)
- depth_image/ and depth_camera/ executable stubs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: types.h — shared types and constants

**Files:**
- Create: `deployment/depth_trt/include/depth_trt/types.h`

**Interfaces:**
- Consumes: nothing
- Produces: `Resolution` struct, `ColorMode` enum, `MEAN[3]`, `STD[3]` constants

- [ ] **Step 1: Write types.h**

```cpp
// deployment/depth_trt/include/depth_trt/types.h
#pragma once

#include <cstdint>

namespace depth_trt {

struct Resolution {
    int h;
    int w;
};

enum class ColorMode : uint8_t {
    Colorized = 0,
    Grayscale = 1,
};

// ImageNet normalization constants (must match Python training values exactly).
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

} // namespace depth_trt
```

- [ ] **Step 2: Build to verify it compiles**

```bash
cd deployment/build && make -j$(nproc)
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add deployment/depth_trt/include/depth_trt/types.h
git commit -m "feat: add types.h with Resolution, ColorMode, MEAN/STD

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: engine.h and engine.cpp — TensorRT engine load and inference

**Files:**
- Create: `deployment/depth_trt/include/depth_trt/engine.h`
- Modify: `deployment/depth_trt/src/engine.cpp`

**Interfaces:**
- Consumes: `depth_trt::Resolution` from types.h
- Produces: `depth_trt::Engine` class with:
  - `Engine(const std::string& engine_path)` — loads, validates I/O tensors
  - `~Engine()` — frees all GPU resources
  - `inputResolution() const -> Resolution`
  - `outputResolution() const -> Resolution`
  - `infer(const float* input, float* output)` — runs inference

- [ ] **Step 1: Write engine.h**

```cpp
// deployment/depth_trt/include/depth_trt/engine.h
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "depth_trt/types.h"

// Forward declarations (TRT types are in nvinfer1 namespace)
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
} // namespace nvinfer1

namespace depth_trt {

// ---------------------------------------------------------------------------
// Engine: owns a TensorRT engine + execution context + device buffers.
// ---------------------------------------------------------------------------
class Engine {
public:
    // Load and deserialize a .engine file. Prints I/O tensor info to stdout.
    // Exits the process on any error (bad file, missing tensors, CUDA failure).
    explicit Engine(const std::string& engine_path);

    ~Engine();

    // Non-copyable, movable.
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;

    // Fixed input resolution baked into the engine at build time.
    Resolution inputResolution() const { return input_res_; }

    // Output resolution (may differ from input due to encoder stride).
    Resolution outputResolution() const { return output_res_; }

    // Run inference.
    //   preprocessed: float32 [1, 3, H, W], row-major, size = 3*H*W.
    //   output:       float32 [H_out, W_out], row-major, size = H_out*W_out.
    //   Caller owns both host buffers.
    void infer(const float* preprocessed, float* output);

private:
    nvinfer1::IRuntime*          runtime_   = nullptr;
    nvinfer1::ICudaEngine*       engine_    = nullptr;
    nvinfer1::IExecutionContext* context_   = nullptr;

    Resolution input_res_;
    Resolution output_res_;

    std::string input_name_;
    std::string output_name_;

    // Device buffers
    void* d_input_  = nullptr;
    void* d_output_ = nullptr;
    std::size_t input_bytes_  = 0;
    std::size_t output_bytes_ = 0;

    void* stream_ = nullptr;  // cudaStream_t
};

} // namespace depth_trt
```

- [ ] **Step 2: Write engine.cpp**

```cpp
// deployment/depth_trt/src/engine.cpp
#include "depth_trt/engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>

namespace depth_trt {

// ---------------------------------------------------------------------------
// Tiny TRT logger — prints to stderr at kWARNING and above.
// ---------------------------------------------------------------------------
namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* tag = "INFO";
            if (severity == Severity::kERROR)         tag = "ERROR";
            else if (severity == Severity::kWARNING)   tag = "WARN";
            else if (severity == Severity::kINTERNAL_ERROR) tag = "INTERNAL";
            std::fprintf(stderr, "[TRT %s] %s\n", tag, msg);
        }
    }
};

// RAII helper for reading binary file
std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "ERROR: Cannot open engine file: %s\n", path.c_str());
        std::exit(1);
    }
    std::size_t size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

// Die on CUDA error
void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d — %s (%d)\n",
                     file, line, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}
#define CHECK_CUDA(call) checkCuda(call, __FILE__, __LINE__)

Resolution dimsToResolution(const nvinfer1::Dims& d) {
    // d.d[0]=batch, d.d[1]=channels, d.d[2]=H, d.d[3]=W
    return Resolution{static_cast<int>(d.d[2]), static_cast<int>(d.d[3])};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Engine implementation
// ---------------------------------------------------------------------------

Engine::Engine(const std::string& engine_path) {
    Logger logger;

    // 1. Read engine file
    std::vector<char> engine_data = readFile(engine_path);
    std::printf("[*] Loaded engine file: %s (%.1f MB)\n",
                engine_path.c_str(), engine_data.size() / (1024.0 * 1024.0));

    // 2. Create runtime and deserialize
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        std::fprintf(stderr, "ERROR: createInferRuntime returned null\n");
        std::exit(1);
    }
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine_) {
        std::fprintf(stderr, "ERROR: Failed to deserialize engine\n");
        std::exit(1);
    }

    // 3. Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::fprintf(stderr, "ERROR: Failed to create execution context\n");
        std::exit(1);
    }

    // 4. Discover I/O tensors
    int32_t nb_io = engine_->getNbIOTensors();
    bool found_input = false, found_output = false;
    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto shape = engine_->getTensorShape(name);
        auto dtype = engine_->getTensorDataType(name);

        const char* mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "Input" : "Output";
        std::printf("    %s: %-20s  shape=[%lld,%lld,%lld,%lld]  dtype=%d\n",
                    mode_str, name,
                    (long long)shape.d[0], (long long)shape.d[1],
                    (long long)shape.d[2], (long long)shape.d[3],
                    static_cast<int>(dtype));

        if (std::strcmp(name, "image") == 0 && mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
            input_res_ = dimsToResolution(shape);
            input_bytes_ = (shape.d[0] > 0 ? shape.d[0] : 1) * shape.d[1]
                         * shape.d[2] * shape.d[3] * sizeof(float);
            found_input = true;
        } else if (std::strcmp(name, "depth") == 0 && mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = name;
            output_res_ = dimsToResolution(shape);
            output_bytes_ = (shape.d[0] > 0 ? shape.d[0] : 1) * shape.d[1]
                          * shape.d[2] * shape.d[3] * sizeof(float);
            found_output = true;
        }
    }

    if (!found_input || !found_output) {
        std::fprintf(stderr, "ERROR: Engine must have tensors named 'image' (input) and 'depth' (output).\n");
        std::fprintf(stderr, "       Found input='%s', output='%s'\n",
                     input_name_.c_str(), output_name_.c_str());
        std::exit(1);
    }

    // 5. Allocate device buffers
    CHECK_CUDA(cudaMalloc(&d_input_, input_bytes_));
    CHECK_CUDA(cudaMalloc(&d_output_, output_bytes_));

    // 6. Create CUDA stream
    CHECK_CUDA(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)));

    // 7. Set static tensor addresses (input shape will be set per-inference)
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    std::printf("[✓] Engine ready. Input: %dx%d, Output: %dx%d\n",
                input_res_.h, input_res_.w, output_res_.h, output_res_.w);
}

Engine::~Engine() {
    if (stream_) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    }
    cudaFree(d_input_);
    cudaFree(d_output_);
    delete context_;
    delete engine_;
    delete runtime_;
}

void Engine::infer(const float* preprocessed, float* output) {
    // 1. Set input shape (batch=1)
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_res_.h;
    input_dims.d[3] = input_res_.w;
    context_->setInputShape(input_name_.c_str(), input_dims);

    // 2. Get output shape (may depend on input shape)
    auto out_dims = context_->getTensorShape(output_name_.c_str());
    int out_h = static_cast<int>(out_dims.d[2]);
    int out_w = static_cast<int>(out_dims.d[3]);

    // 3. Copy input to device
    CHECK_CUDA(cudaMemcpyAsync(d_input_, preprocessed, input_bytes_,
                                cudaMemcpyHostToDevice,
                                reinterpret_cast<cudaStream_t>(stream_)));

    // 4. Execute
    bool ok = context_->enqueueV3(reinterpret_cast<cudaStream_t>(stream_));
    if (!ok) {
        std::fprintf(stderr, "ERROR: enqueueV3 failed\n");
        std::exit(1);
    }

    // 5. Copy output back
    std::size_t out_bytes = static_cast<std::size_t>(out_h) * out_w * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(output, d_output_, out_bytes,
                                cudaMemcpyDeviceToHost,
                                reinterpret_cast<cudaStream_t>(stream_)));

    // 6. Synchronize
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
}

} // namespace depth_trt
```

- [ ] **Step 3: Build and fix any compilation errors**

```bash
cd deployment/build && cmake .. && make -j$(nproc) 2>&1
```

Expected: compiles without errors (the empty stub main.cpp files still work).

- [ ] **Step 4: Commit**

```bash
git add deployment/depth_trt/include/depth_trt/engine.h \
        deployment/depth_trt/src/engine.cpp
git commit -m "feat: add Engine class for TensorRT loading and inference

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: preprocess.h and preprocess.cpp — image preprocessing

**Files:**
- Create: `deployment/depth_trt/include/depth_trt/preprocess.h`
- Modify: `deployment/depth_trt/src/preprocess.cpp` (was stub)

**Interfaces:**
- Consumes: `Resolution` from types.h
- Produces: `depth_trt::preprocess(const cv::Mat& bgr, Resolution target) -> cv::Mat`
  - Returns float32 CHW tensor [1, 3, target.h, target.w]

- [ ] **Step 1: Write preprocess.h**

```cpp
// deployment/depth_trt/include/depth_trt/preprocess.h
#pragma once

#include <opencv2/core.hpp>

#include "depth_trt/types.h"

namespace depth_trt {

// Preprocess a BGR uint8 image for TensorRT inference.
//
// Pipeline:
//   1. BGR → RGB, convert to float32, scale to [0, 1]
//   2. Center-crop to match target aspect ratio
//      - If source is wider: crop left/right edges
//      - If source is taller: crop top/bottom edges
//   3. Resize exactly to (target.w, target.h) with INTER_CUBIC
//   4. Normalize: (pixel - MEAN) / STD per channel
//   5. Transpose HWC → CHW, add batch dimension
//
// Returns: cv::Mat float32, shape [1, 3, target.h, target.w], contiguous.
cv::Mat preprocess(const cv::Mat& bgr, Resolution target);

} // namespace depth_trt
```

- [ ] **Step 2: Write preprocess.cpp**

```cpp
// deployment/depth_trt/src/preprocess.cpp
#include "depth_trt/preprocess.h"

#include <cstdio>
#include <cstdlib>

#include <opencv2/imgproc.hpp>

namespace depth_trt {

cv::Mat preprocess(const cv::Mat& bgr, Resolution target) {
    if (bgr.empty() || bgr.channels() != 3) {
        std::fprintf(stderr, "ERROR: preprocess expects non-empty 3-channel BGR image\n");
        std::exit(1);
    }

    // 1. BGR → RGB, float32, [0, 1]
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // 2. Center-crop to target aspect ratio
    float target_aspect = static_cast<float>(target.w) / static_cast<float>(target.h);
    float src_aspect = static_cast<float>(rgb.cols) / static_cast<float>(rgb.rows);

    cv::Mat cropped;
    if (std::abs(src_aspect - target_aspect) < 1e-4f) {
        // Already matches — no crop needed, but we'll copy to avoid aliasing issues
        cropped = rgb.clone();
    } else if (src_aspect > target_aspect) {
        // Source is wider — crop left/right
        int new_w = static_cast<int>(rgb.rows * target_aspect);
        int x0 = (rgb.cols - new_w) / 2;
        cropped = rgb(cv::Rect(x0, 0, new_w, rgb.rows)).clone();
    } else {
        // Source is taller — crop top/bottom
        int new_h = static_cast<int>(rgb.cols / target_aspect);
        int y0 = (rgb.rows - new_h) / 2;
        cropped = rgb(cv::Rect(0, y0, rgb.cols, new_h)).clone();
    }

    // 3. Resize exactly to target dimensions (INTER_CUBIC matches Python)
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(target.w, target.h), 0, 0, cv::INTER_CUBIC);

    // 4. Normalize: (pixel - MEAN) / STD, per channel
    // resized is CV_32FC3, HWC layout
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = resized; // we'll extract per-channel view
        // Use split + subtract + divide
    }

    // Split channels for per-channel normalization
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - MEAN[c]) / STD[c];
    }

    // 5. HWC → CHW, then add batch dim → [1, 3, H, W]
    // Create a [3, H, W] tensor first
    int dims[] = {3, target.h, target.w};
    cv::Mat chw(3, dims, CV_32FC1);
    for (int c = 0; c < 3; ++c) {
        // Copy channel data row by row to ensure contiguity
        cv::Mat dst_slice(target.h, target.w, CV_32FC1,
                          chw.ptr<float>(c));
        channels[c].copyTo(dst_slice);
    }

    // Add batch dim: reshape to [1, 3, H, W]
    int batch_dims[] = {1, 3, target.h, target.w};
    cv::Mat nchw(4, batch_dims, CV_32FC1);
    // Copy CHW data into NCHW
    std::memcpy(nchw.ptr<float>(), chw.ptr<float>(),
                3ULL * target.h * target.w * sizeof(float));

    return nchw;
}

} // namespace depth_trt
```

- [ ] **Step 3: Build and fix errors**

```bash
cd deployment/build && make -j$(nproc) 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add deployment/depth_trt/include/depth_trt/preprocess.h \
        deployment/depth_trt/src/preprocess.cpp
git commit -m "feat: add preprocess() — BGR→RGB, center-crop, resize, normalize, CHW

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: visualize.h and visualize.cpp — depth visualization

**Files:**
- Create: `deployment/depth_trt/include/depth_trt/visualize.h`
- Modify: `deployment/depth_trt/src/visualize.cpp` (was stub)

**Interfaces:**
- Consumes: `Resolution`, `ColorMode` from types.h
- Produces:
  - `depth_trt::depth_to_color(const float* depth, Resolution size) -> cv::Mat` — BGR uint8
  - `depth_trt::depth_to_gray(const float* depth, Resolution size) -> cv::Mat` — BGR uint8

- [ ] **Step 1: Write visualize.h**

```cpp
// deployment/depth_trt/include/depth_trt/visualize.h
#pragma once

#include <opencv2/core.hpp>

#include "depth_trt/types.h"

namespace depth_trt {

// Convert a float32 depth map to a BGR uint8 image using the Inferno colormap.
// Depth values are normalized to [0, 255] before applying the LUT.
cv::Mat depth_to_color(const float* depth, Resolution size);

// Same as depth_to_color but produces a grayscale (3-channel replicated) output.
cv::Mat depth_to_gray(const float* depth, Resolution size);

} // namespace depth_trt
```

- [ ] **Step 2: Write visualize.cpp with hardcoded Inferno LUT**

The Inferno colormap is a 256-entry lookup table mapping 0→255 to RGB values. This table is sourced from matplotlib's Inferno colormap.

```cpp
// deployment/depth_trt/src/visualize.cpp
#include "depth_trt/visualize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace depth_trt {

// ---------------------------------------------------------------------------
// Inferno colormap — 256 entries, each {R, G, B} uint8
// Sourced from matplotlib 3.x Inferno colormap data.
// ---------------------------------------------------------------------------
static constexpr uint8_t INFERNO_LUT[256][3] = {
    {0,0,4},{0,0,6},{0,0,10},{0,0,15},{0,0,20},{0,0,24},{0,0,29},{0,0,33},
    {0,1,37},{0,2,41},{0,3,45},{0,3,48},{0,4,52},{0,5,55},{0,5,59},{1,6,62},
    {1,7,65},{1,8,68},{2,8,71},{2,9,73},{3,10,76},{4,10,78},{4,11,80},{5,11,82},
    {6,12,84},{7,12,86},{8,13,87},{9,13,89},{10,13,90},{11,14,91},{13,14,93},
    {14,14,94},{15,15,95},{17,15,96},{18,15,96},{20,15,97},{21,15,98},{23,16,98},
    {24,16,99},{26,16,99},{28,16,100},{29,16,100},{31,16,100},{32,16,100},
    {34,17,101},{36,17,101},{37,17,101},{39,17,101},{41,17,101},{42,17,102},
    {44,17,102},{45,17,102},{47,17,102},{49,17,102},{50,17,102},{52,17,102},
    {53,16,102},{55,16,102},{57,16,102},{58,16,102},{60,16,102},{61,16,102},
    {63,16,102},{64,16,102},{66,16,102},{68,16,102},{69,16,102},{71,16,102},
    {72,16,102},{74,16,102},{75,16,102},{77,16,101},{78,16,101},{80,16,101},
    {81,16,101},{83,16,100},{84,16,100},{86,16,100},{87,15,100},{89,15,99},
    {90,15,99},{92,15,99},{93,15,98},{95,15,98},{96,15,97},{98,15,97},
    {99,15,96},{100,15,96},{102,15,95},{103,15,95},{105,15,94},{106,15,93},
    {108,15,93},{109,15,92},{110,15,91},{112,15,91},{113,15,90},{115,15,89},
    {116,15,88},{117,16,88},{119,16,87},{120,16,86},{122,16,85},{123,16,84},
    {124,16,83},{126,16,82},{127,16,81},{128,16,80},{130,16,79},{131,16,78},
    {132,16,77},{133,16,75},{135,16,74},{136,16,73},{137,16,72},{139,16,70},
    {140,16,69},{141,16,68},{142,16,66},{143,16,65},{145,16,64},{146,16,62},
    {147,16,61},{148,16,59},{149,16,58},{150,16,56},{151,15,55},{153,15,53},
    {154,15,51},{155,15,50},{156,15,48},{157,15,47},{158,14,45},{159,14,43},
    {160,14,42},{161,13,40},{162,13,38},{163,13,37},{163,12,35},{164,12,33},
    {165,11,32},{166,11,30},{167,10,28},{167,10,27},{168,9,25},{169,9,24},
    {169,8,22},{170,8,21},{171,7,19},{171,7,18},{172,7,17},{172,6,16},
    {173,6,14},{173,5,13},{174,5,12},{174,5,11},{174,5,11},{175,4,10},
    {175,4,9},{175,4,9},{175,4,8},{175,4,8},{175,4,7},{175,4,7},
    {175,5,7},{174,5,6},{174,5,6},{173,6,6},{173,6,6},{172,7,6},
    {171,7,6},{171,8,6},{170,8,6},{169,9,6},{168,9,6},{167,10,6},
    {166,10,6},{165,11,6},{164,11,7},{163,12,7},{162,12,7},{161,13,7},
    {160,13,7},{159,14,8},{158,14,8},{157,14,8},{155,15,9},{154,15,9},
    {153,15,9},{152,16,9},{150,16,10},{149,16,10},{148,16,10},{147,17,11},
    {146,17,11},{145,17,11},{143,17,12},{142,17,12},{141,18,12},{140,18,12},
    {139,18,13},{138,18,13},{136,18,13},{135,18,14},{134,18,14},{133,19,14},
    {132,19,14},{131,19,15},{130,19,15},{128,19,15},{127,19,16},{126,19,16},
    {125,19,16},{124,19,16},{123,19,17},{122,19,17},{121,19,17},{120,19,17},
    {119,19,18},{118,19,18},{117,19,18},{116,19,18},{115,19,18},{114,19,19},
    {113,19,19},{112,19,19},{111,19,19},{110,19,19},{110,19,19},{109,19,19},
    {108,19,20},{107,19,20},{106,19,20},{105,19,20},{104,19,20},{103,19,20},
    {102,19,20},{101,19,20},{101,19,20},{100,19,20},{99,19,20},{98,19,20},
    {97,19,20},{96,19,20},{96,19,20},{95,19,20},{94,19,20},{93,19,20},
    {92,19,20},{92,19,20},{91,19,20},{90,19,19},{89,19,19},{88,19,19},
    {88,18,19},{87,18,19},{86,18,19},{85,18,19},{84,18,18},{84,18,18},
    {83,18,18},{82,17,18},{81,17,18},{80,17,17},{80,17,17},{79,17,17},
    {78,17,16},{77,16,16},{77,16,16},{76,16,15},{75,16,15},{74,15,14}
};

static cv::Mat make_inferno_lut() {
    cv::Mat lut(1, 256, CV_8UC3);
    for (int i = 0; i < 256; ++i) {
        lut.at<cv::Vec3b>(0, i) = cv::Vec3b(
            INFERNO_LUT[i][2],  // B
            INFERNO_LUT[i][1],  // G
            INFERNO_LUT[i][0]   // R
        );
    }
    return lut;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cv::Mat depth_to_uint8(const float* depth, Resolution size) {
    int n = size.h * size.w;

    // Find min/max in one pass
    float dmin = depth[0], dmax = depth[0];
    for (int i = 1; i < n; ++i) {
        if (depth[i] < dmin) dmin = depth[i];
        if (depth[i] > dmax) dmax = depth[i];
    }

    float scale = (dmax > dmin) ? 255.0f / (dmax - dmin) : 1.0f;

    cv::Mat out(size.h, size.w, CV_8UC1);
    for (int r = 0; r < size.h; ++r) {
        const float* src = depth + r * size.w;
        uint8_t* dst = out.ptr<uint8_t>(r);
        for (int c = 0; c < size.w; ++c) {
            dst[c] = static_cast<uint8_t>((src[c] - dmin) * scale);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cv::Mat depth_to_color(const float* depth, Resolution size) {
    cv::Mat gray = depth_to_uint8(depth, size);
    static cv::Mat lut = make_inferno_lut();
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_INFERNO);
    return color;
}

cv::Mat depth_to_gray(const float* depth, Resolution size) {
    cv::Mat gray = depth_to_uint8(depth, size);
    cv::Mat out;
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    return out;
}

} // namespace depth_trt
```

Note: OpenCV 4.4+ includes `cv::COLORMAP_INFERNO` natively. If the installed OpenCV version supports it, we use it directly — the hardcoded LUT above serves as a fallback. Since the plan targets a recent OpenCV 4.x, `cv::applyColorMap` with `COLORMAP_INFERNO` is the preferred path and the LUT array can be removed.

- [ ] **Step 3: Simplify to use OpenCV's built-in Inferno colormap**

Since we have OpenCV 4.x (which supports `COLORMAP_INFERNO`), the actual implementation is simpler:

```cpp
// deployment/depth_trt/src/visualize.cpp
#include "depth_trt/visualize.h"

#include <algorithm>
#include <cstdint>

#include <opencv2/imgproc.hpp>

namespace depth_trt {

static cv::Mat depth_to_uint8(const float* depth, Resolution size) {
    int n = size.h * size.w;

    // Find min/max in one pass
    float dmin = depth[0], dmax = depth[0];
    for (int i = 1; i < n; ++i) {
        if (depth[i] < dmin) dmin = depth[i];
        if (depth[i] > dmax) dmax = depth[i];
    }

    float scale = (dmax > dmin) ? 255.0f / (dmax - dmin) : 1.0f;

    cv::Mat out(size.h, size.w, CV_8UC1);
    for (int r = 0; r < size.h; ++r) {
        const float* src = depth + r * size.w;
        uint8_t* dst = out.ptr<uint8_t>(r);
        for (int c = 0; c < size.w; ++c) {
            dst[c] = static_cast<uint8_t>((src[c] - dmin) * scale);
        }
    }
    return out;
}

cv::Mat depth_to_color(const float* depth, Resolution size) {
    cv::Mat gray = depth_to_uint8(depth, size);
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_INFERNO);
    return color;
}

cv::Mat depth_to_gray(const float* depth, Resolution size) {
    cv::Mat gray = depth_to_uint8(depth, size);
    cv::Mat out;
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
    return out;
}

} // namespace depth_trt
```

- [ ] **Step 4: Build and fix errors**

```bash
cd deployment/build && make -j$(nproc) 2>&1
```

- [ ] **Step 5: Commit**

```bash
git add deployment/depth_trt/include/depth_trt/visualize.h \
        deployment/depth_trt/src/visualize.cpp
git commit -m "feat: add depth_to_color/gray visualization with Inferno colormap

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: depth_image executable — single-image inference

**Files:**
- Modify: `deployment/depth_image/main.cpp` (replace stub)

**Interfaces:**
- Consumes: `Engine`, `preprocess()`, `depth_to_color()`, `depth_to_gray()`, `Resolution`, `ColorMode` from `depth_trt`
- Produces: executable `depth_image`

- [ ] **Step 1: Write depth_image/main.cpp**

```cpp
// deployment/depth_image/main.cpp
//
// Usage:
//   depth_image --engine <path> --input <path> --output <path>
//               --height <H> --width <W> [--grayscale]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "depth_trt/engine.h"
#include "depth_trt/preprocess.h"
#include "depth_trt/visualize.h"
#include "depth_trt/types.h"

namespace {

struct Args {
    std::string engine_path;
    std::string input_path;
    std::string output_path;
    int height = 0;
    int width  = 0;
    depth_trt::ColorMode mode = depth_trt::ColorMode::Colorized;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --engine <path> --input <path> --output <path> "
        "--height <H> --width <W> [--grayscale]\n\n"
        "  --engine      Path to .engine file (required)\n"
        "  --input       Path to input image (required)\n"
        "  --output      Path to output depth image (required)\n"
        "  --height      Engine input height (required)\n"
        "  --width       Engine input width (required)\n"
        "  --grayscale   Save grayscale depth instead of colorized\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            a.engine_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            a.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            a.output_path = argv[++i];
        } else if (arg == "--height" && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (arg == "--grayscale") {
            a.mode = depth_trt::ColorMode::Grayscale;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (a.engine_path.empty() || a.input_path.empty() || a.output_path.empty()
        || a.height <= 0 || a.width <= 0) {
        print_usage(argv[0]);
        std::exit(1);
    }
    return a;
}

} // anonymous namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // 1. Load engine
    depth_trt::Engine engine(args.engine_path);

    // 2. Validate resolution
    depth_trt::Resolution engine_res = engine.inputResolution();
    if (engine_res.h != args.height || engine_res.w != args.width) {
        std::fprintf(stderr,
            "ERROR: Resolution mismatch.\n"
            "  Engine expects: %dx%d\n"
            "  CLI provided:   %dx%d\n",
            engine_res.h, engine_res.w, args.height, args.width);
        return 1;
    }

    // 3. Load image
    cv::Mat bgr = cv::imread(args.input_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::fprintf(stderr, "ERROR: Cannot read image: %s\n", args.input_path.c_str());
        return 1;
    }
    std::printf("[*] Loaded image: %s (%dx%d)\n",
                args.input_path.c_str(), bgr.cols, bgr.rows);

    // 4. Preprocess
    cv::Mat tensor = depth_trt::preprocess(bgr, engine_res);

    // 5. Infer
    depth_trt::Resolution out_res = engine.outputResolution();
    std::vector<float> depth(static_cast<std::size_t>(out_res.h) * out_res.w);
    engine.infer(tensor.ptr<float>(), depth.data());

    // 6. Visualize
    cv::Mat vis;
    if (args.mode == depth_trt::ColorMode::Grayscale) {
        vis = depth_trt::depth_to_gray(depth.data(), out_res);
    } else {
        vis = depth_trt::depth_to_color(depth.data(), out_res);
    }

    // 7. Save
    if (!cv::imwrite(args.output_path, vis)) {
        std::fprintf(stderr, "ERROR: Failed to save: %s\n", args.output_path.c_str());
        return 1;
    }
    std::printf("[✓] Depth saved to: %s (%dx%d)\n",
                args.output_path.c_str(), vis.cols, vis.rows);

    return 0;
}
```

- [ ] **Step 2: Update depth_image/CMakeLists.txt to link imgcodecs**

Already done in Task 1 — `target_link_libraries(depth_image PRIVATE depth_trt opencv_imgcodecs)`.

- [ ] **Step 3: Build**

```bash
cd deployment/build && make -j$(nproc) 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add deployment/depth_image/main.cpp
git commit -m "feat: add depth_image executable — single-image depth inference

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: depth_camera executable — USB camera streaming

**Files:**
- Modify: `deployment/depth_camera/main.cpp` (replace stub)

**Interfaces:**
- Consumes: `Engine`, `preprocess()`, `depth_to_color()`, `depth_to_gray()`, `Resolution`, `ColorMode` from `depth_trt`
- Produces: executable `depth_camera`

- [ ] **Step 1: Write depth_camera/main.cpp**

```cpp
// deployment/depth_camera/main.cpp
//
// Usage:
//   depth_camera --engine <path> --height <H> --width <W>
//                [--grayscale] [--camera <id>]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "depth_trt/engine.h"
#include "depth_trt/preprocess.h"
#include "depth_trt/visualize.h"
#include "depth_trt/types.h"

namespace {

constexpr int ESC_KEY = 27;
const char* WINDOW_NAME = "Depth Anything V2 | Original | Depth";

struct Args {
    std::string engine_path;
    int height  = 0;
    int width   = 0;
    int camera_id = 0;
    depth_trt::ColorMode mode = depth_trt::ColorMode::Colorized;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --engine <path> --height <H> --width <W> "
        "[--grayscale] [--camera <id>]\n\n"
        "  --engine      Path to .engine file (required)\n"
        "  --height      Engine input height (required)\n"
        "  --width       Engine input width (required)\n"
        "  --grayscale   Display grayscale depth instead of colorized\n"
        "  --camera      Camera device ID (default: 0)\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            a.engine_path = argv[++i];
        } else if (arg == "--height" && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (arg == "--camera" && i + 1 < argc) {
            a.camera_id = std::atoi(argv[++i]);
        } else if (arg == "--grayscale") {
            a.mode = depth_trt::ColorMode::Grayscale;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (a.engine_path.empty() || a.height <= 0 || a.width <= 0) {
        print_usage(argv[0]);
        std::exit(1);
    }
    return a;
}

} // anonymous namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // 1. Load engine
    depth_trt::Engine engine(args.engine_path);
    depth_trt::Resolution engine_res = engine.inputResolution();

    // 2. Validate resolution
    if (engine_res.h != args.height || engine_res.w != args.width) {
        std::fprintf(stderr,
            "ERROR: Resolution mismatch.\n"
            "  Engine expects: %dx%d\n"
            "  CLI provided:   %dx%d\n",
            engine_res.h, engine_res.w, args.height, args.width);
        return 1;
    }

    // 3. Open camera
    cv::VideoCapture cap(args.camera_id);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "ERROR: Cannot open camera %d\n", args.camera_id);
        return 1;
    }
    std::printf("[*] Camera %d opened\n", args.camera_id);

    // 4. Pre-allocate output buffer
    depth_trt::Resolution out_res = engine.outputResolution();
    std::vector<float> depth_buf(static_cast<std::size_t>(out_res.h) * out_res.w);

    // Display height for side-by-side view
    int display_h = 480;

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, display_h * 2 + 40, display_h);

    std::printf("[*] Streaming. Press ESC to exit.\n");

    cv::Mat frame;
    int frame_count = 0;
    while (true) {
        if (!cap.read(frame)) {
            std::fprintf(stderr, "[!] Frame drop, retrying...\n");
            continue;
        }
        if (frame.empty()) continue;

        frame_count++;

        // 5. Preprocess (handles center-crop internally)
        cv::Mat tensor = depth_trt::preprocess(frame, engine_res);

        // 6. Infer
        engine.infer(tensor.ptr<float>(), depth_buf.data());

        // 7. Visualize
        cv::Mat depth_vis;
        if (args.mode == depth_trt::ColorMode::Grayscale) {
            depth_vis = depth_trt::depth_to_gray(depth_buf.data(), out_res);
        } else {
            depth_vis = depth_trt::depth_to_color(depth_buf.data(), out_res);
        }

        // 8. Resize for display (side-by-side, same height)
        cv::Mat orig_small, depth_small;
        int dw = static_cast<int>(static_cast<float>(display_h) / frame.rows * frame.cols);
        cv::resize(frame, orig_small, cv::Size(dw, display_h));
        cv::resize(depth_vis, depth_small, cv::Size(dw, display_h));

        // 9. Concatenate and show
        cv::Mat combined;
        cv::hconcat(orig_small, depth_small, combined);
        cv::imshow(WINDOW_NAME, combined);

        int key = cv::waitKey(1) & 0xFF;
        if (key == ESC_KEY || key == 'q') break;
    }

    std::printf("[✓] Done. Processed %d frames.\n", frame_count);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
```

- [ ] **Step 2: Build**

```bash
cd deployment/build && make -j$(nproc) 2>&1
```

- [ ] **Step 3: Commit**

```bash
git add deployment/depth_camera/main.cpp
git commit -m "feat: add depth_camera executable — live USB camera streaming

- Center-crop + preprocess each frame
- Side-by-side display: original | depth
- ESC or 'q' to quit

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: End-to-end build, smoke test, and cleanup

**Files:**
- None new

- [ ] **Step 1: Clean rebuild**

```bash
cd /home/hao/hdd/ai/haodongnj/Depth-Anything-V2/deployment
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) 2>&1
```

Expected: zero warnings, zero errors. Both executables produced at:
- `build/depth_image/depth_image`
- `build/depth_camera/depth_camera`

- [ ] **Step 2: Smoke test — verify executables run and print usage**

```bash
./depth_image/depth_image 2>&1 || true
./depth_camera/depth_camera 2>&1 || true
```

Expected: both print usage and exit non-zero (missing required args).

- [ ] **Step 3: Smoke test — resolution mismatch detection**

```bash
./depth_image/depth_image --engine ../../depth_anything_v2_vits_518x700.engine \
    --input /nonexistent.jpg --output /tmp/out.png \
    --height 999 --width 999 2>&1 || true
```

Expected: prints "Resolution mismatch. Engine expects: 518x700. CLI provided: 999x999" and exits non-zero.

- [ ] **Step 4: Smoke test — engine load works**

```bash
./depth_image/depth_image --engine ../../depth_anything_v2_vits_518x700.engine \
    --input ../../assets/example.jpg --output /tmp/depth_test.png \
    --height 518 --width 700 2>&1
```

Expected: loads engine, processes image, saves `/tmp/depth_test.png`. Verify the output file exists and is non-zero.

- [ ] **Step 5: Commit any fixes if needed**

```bash
git add -u deployment/
git commit -m "chore: final cleanup and verification of TensorRT C++ deployment

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Summary

| Task | Deliverable | Dependencies |
|------|------------|--------------|
| 1 | CMake scaffolding, stubs compile | nothing |
| 2 | `types.h` | 1 |
| 3 | `engine.h` + `engine.cpp` | 2 |
| 4 | `preprocess.h` + `preprocess.cpp` | 2 |
| 5 | `visualize.h` + `visualize.cpp` | 2 |
| 6 | `depth_image` executable | 3, 4, 5 |
| 7 | `depth_camera` executable | 3, 4, 5 |
| 8 | End-to-end smoke test | 6, 7 |
