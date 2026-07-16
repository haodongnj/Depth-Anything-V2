# Depth Anything V2 — TensorRT C++ Deployment Design

**Date**: 2026-07-16
**Branch**: feat/donghao/practice
**Context**: Deploy a pre-built TensorRT engine (`depth_anything_v2_vits_518x700.engine`) as reusable C++ tools.

---

## Overview

Two standalone executables that share a common `libdepth_trt` static library for engine loading, preprocessing, inference, and depth visualization.

- **`depth_image`** — reads a single image file, runs inference, saves the depth map.
- **`depth_camera`** — streams from a USB camera, shows original + depth side-by-side in a live window.

---

## Constraints & Environment

| Item | Value |
|------|-------|
| TensorRT | 11.1.0.106 |
| CUDA | 12.8 |
| OpenCV | 4.x |
| CMake | ≥ 3.20 |
| C++ standard | C++17 |
| Engine | `depth_anything_v2_vits_518x700.engine` (FP32, ViT-S) |
| Engine I/O | input `"image"` [N,3,H,W] fp32, output `"depth"` [N,1,H',W'] fp32 |

---

## Directory Structure

```
deployment/
├── CMakeLists.txt                    # top-level: find deps, add_subdirectory
├── cmake/
│   └── FindTensorRT.cmake            # portable TRT discovery
├── depth_trt/                        # static library: libdepth_trt.a
│   ├── CMakeLists.txt
│   ├── include/depth_trt/
│   │   ├── engine.h                  # Engine class
│   │   ├── preprocess.h             # preprocess() free function
│   │   ├── visualize.h              # depth_to_color() / depth_to_gray()
│   │   └── types.h                  # Resolution, ColorMode, constants
│   └── src/
│       ├── engine.cpp
│       ├── preprocess.cpp
│       └── visualize.cpp
├── depth_image/
│   ├── CMakeLists.txt
│   └── main.cpp                      # single-image CLI
└── depth_camera/
    ├── CMakeLists.txt
    └── main.cpp                      # USB camera streaming CLI
```

---

## Shared Library API

### `types.h`
```cpp
struct Resolution { int h, w; };
enum class ColorMode { Colorized, Grayscale };

// Preprocessing constants (must match Python training values)
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
```

### `engine.h`
```cpp
class Engine {
public:
    explicit Engine(const std::string& engine_path);
    ~Engine();

    // Non-copyable, movable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Resolution inputResolution() const;
    Resolution outputResolution() const;

    // preprocessed: float32 NCHW [1, 3, H, W], size = 3*H*W
    // output:       float32 [1, 1, H_out, W_out], size = H_out*W_out
    // Caller owns both host buffers.
    void infer(const float* preprocessed, float* output);

private:
    // Owns: nvinfer1::IRuntime, ICudaEngine, IExecutionContext
    //       cudaStream_t, device buffers, host output buffer
};
```

Construction validates the engine file exists, deserializes it, locates I/O tensors named `"image"` and `"depth"`, and prints their shapes/dtypes to stdout. Any failure is fatal (print + exit).

### `preprocess.h`
```cpp
// Input:  BGR uint8 image at arbitrary resolution, CV_8UC3.
// Output: float32 CHW tensor [1, 3, target.h, target.w], resized exactly
//         via center-crop then resize (INTER_CUBIC).
// Pipeline: BGR→RGB /255 → center-crop to target aspect ratio →
//           resize exact to (target.w, target.h) → normalize → CHW → batch dim.
cv::Mat preprocess(const cv::Mat& bgr, Resolution target);
```

### `visualize.h`
```cpp
// depth: float32 [H, W], size = H*W
// Returns BGR uint8 image at `size` resolution.
cv::Mat depth_to_color(const float* depth, Resolution size);
cv::Mat depth_to_gray(const float* depth, Resolution size);

// Applies Inferno colormap (reproduces visual quality of Python Spectral_r).
// Normalizes depth to [0,255] internally.
```

---

## CLI Design

### `depth_image`

```
Usage: depth_image --engine <path> --input <path> --output <path>
                  --height <H> --width <W> [--grayscale]

  --engine     Path to .engine file (required)
  --input      Path to input image (required)
  --output     Path to output depth image (required)
  --height     Expected engine input height (required)
  --width      Expected engine input width (required)
  --grayscale  Save grayscale depth instead of colorized (optional)
```

Validation: `--height` and `--width` must match `Engine::inputResolution()` exactly — otherwise print error and exit(1). Input image must exist and be readable — otherwise print error and exit(1).

### `depth_camera`

```
Usage: depth_camera --engine <path> --height <H> --width <W>
                    [--grayscale] [--camera <id>]

  --engine     Path to .engine file (required)
  --height     Expected engine input height (required)
  --width      Expected engine input width (required)
  --grayscale  Show grayscale depth instead of colorized (optional)
  --camera     Camera device ID (default: 0)
```

Behavior:
- Open `cv::VideoCapture(id)`, capture at native resolution.
- Each frame: center-crop to engine aspect ratio → resize exactly to target → preprocess → infer → visualize.
- Display: `cv::hconcat(original_resized_to_match_display, depth)` in a single window.
- Exit on ESC key.
- Transient frame drops (`cap.read()` returns false) are logged and skipped (no crash).

---

## Preprocessing Pipeline (must match Python exactly)

1. **Input**: `cv::Mat` BGR uint8, arbitrary (orig_h, orig_w)
2. **Convert**: `cv::cvtColor(BGR → RGB)`, then `.convertTo(float32) / 255.0`
3. **Center-crop**: compute largest center rectangle at aspect ratio `target.w / target.h`. If image is smaller than target in either dimension, resize directly (no crop) with `INTER_CUBIC`.
4. **Resize exact**: `cv::resize` to `(target.w, target.h)` with `INTER_CUBIC`
5. **Normalize**: `(pixel - MEAN) / STD` per channel
6. **Transpose**: HWC → CHW using `cv::split` + `cv::vconcat` or manual loop
7. **Add batch dim**: reshape/slice as [1, 3, H, W]
8. **Return**: `cv::Mat` float32, shape [1, 3, H, W], contiguous

---

## Postprocessing Pipeline

1. **Engine output**: float32 [H_out, W_out] depth map
2. **Normalize**: `(d - min) / (max - min + 1e-8) * 255` → uint8
3. **Colorize** (default): apply Inferno LUT → BGR uint8
4. **Grayscale** (if `--grayscale`): replicate to 3 channels → BGR uint8

---

## Error Handling

| Layer | Strategy |
|-------|----------|
| CLI validation | Resolution mismatch → print expected vs given → `exit(1)` |
| File I/O | `cv::imread` empty → print path → `exit(1)`. `cv::imwrite` fails → print → `exit(1)` |
| Camera | `!cap.isOpened()` → print → `exit(1)`. Frame drop → stderr warn, continue |
| CUDA | `CHECK_CUDA(call)` macro → print file:line + cudaGetErrorString → `exit(1)` |
| TRT | All API failures → print diagnostic → `exit(1)` |
| Memory | RAII throughout: `Engine` owns device allocations, `cv::Mat` ref-counts host buffers |

---

## Build

```bash
cd deployment
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# produces: build/depth_image/depth_image, build/depth_camera/depth_camera
```

`FindTensorRT.cmake` searches (in order):
1. `$ENV{TRT_ROOT}` if set
2. `/usr/include/x86_64-linux-gnu` + `/usr/lib/x86_64-linux-gnu` (Debian/Ubuntu apt)
3. `/usr/local/tensorrt` (tar install)
4. `/opt/tensorrt` (JetPack)

---

## What's NOT in Scope

- Batch processing (single-image only)
- Audio or recording from camera
- Multi-GPU or INT8/FP16 support (engine is already built as FP32)
- Dynamic resolution — engine resolution is fixed at build time
- ONNX loading — only `.engine` files
