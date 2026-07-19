// deployment/depth_trt/include/depth_trt/engine.h
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <string>

#ifdef DEPTH_TRT_LEGACY_API
#include <vector>
#endif

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
#ifdef DEPTH_TRT_LEGACY_API
    int32_t input_binding_ = -1;
    int32_t output_binding_ = -1;
    std::vector<void*> bindings_;
#endif

    // Device buffers
    void* d_input_  = nullptr;
    void* d_output_ = nullptr;
    std::size_t input_bytes_  = 0;
    std::size_t output_bytes_ = 0;

    cudaStream_t stream_ = nullptr;
};

} // namespace depth_trt
