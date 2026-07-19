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
#ifndef DEPTH_TRT_LEGACY_API
#include <NvInferRuntime.h>
#endif

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

static Logger s_logger;

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

void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d -- %s (%d)\n",
                     file, line, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}
#define CHECK_CUDA(call) checkCuda(call, __FILE__, __LINE__)

Resolution dimsToResolution(const nvinfer1::Dims& d) {
    if (d.nbDims == 4) {
        return Resolution{static_cast<int>(d.d[2]), static_cast<int>(d.d[3])};
    } else if (d.nbDims == 3) {
        return Resolution{static_cast<int>(d.d[1]), static_cast<int>(d.d[2])};
    }
    return Resolution{static_cast<int>(d.d[d.nbDims - 2]),
                      static_cast<int>(d.d[d.nbDims - 1])};
}

int64_t tensorElementCount(const nvinfer1::Dims& d) {
    int64_t count = 1;
    for (int32_t i = 0; i < d.nbDims; ++i) {
        count *= (d.d[i] > 0 ? d.d[i] : 1);
    }
    return count;
}

// Print a Dims to stdout as [a,b,c,...]
void printDims(const nvinfer1::Dims& shape) {
    for (int32_t j = 0; j < shape.nbDims; ++j) {
        std::printf("%lld%s", (long long)shape.d[j],
                    j + 1 < shape.nbDims ? "," : "");
    }
}

} // anonymous namespace

// ===================================================================
// Engine implementation
// ===================================================================

Engine::Engine(const std::string& engine_path) {
    // 1. Read engine file
    std::vector<char> engine_data = readFile(engine_path);
    std::printf("[*] Loaded engine file: %s (%.1f MB)\n",
                engine_path.c_str(), engine_data.size() / (1024.0 * 1024.0));

    // 2. Create runtime and deserialize
    runtime_ = nvinfer1::createInferRuntime(s_logger);
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

    bool found_input = false, found_output = false;

#ifdef DEPTH_TRT_LEGACY_API
    // -----------------------------------------------------------------------
    // TensorRT 8.x — legacy binding API
    // -----------------------------------------------------------------------
    int32_t nb = engine_->getNbBindings();
    bindings_.resize(nb, nullptr);

    for (int32_t i = 0; i < nb; ++i) {
        char const* name     = engine_->getBindingName(i);
        auto        shape    = engine_->getBindingDimensions(i);
        auto        dtype    = engine_->getBindingDataType(i);
        bool        is_input = engine_->bindingIsInput(i);

        const char* io_str = is_input ? "Input" : "Output";
        std::printf("    %s: %-20s  shape=[", io_str, name);
        printDims(shape);
        std::printf("]  dtype=%d\n", static_cast<int>(dtype));

        if (std::strcmp(name, "image") == 0 && is_input) {
            input_name_    = name;
            input_binding_ = i;
            input_res_     = dimsToResolution(shape);
            input_bytes_   = tensorElementCount(shape) * sizeof(float);
            found_input    = true;
        } else if (std::strcmp(name, "depth") == 0 && !is_input) {
            output_name_    = name;
            output_binding_ = i;
            output_res_     = dimsToResolution(shape);
            output_bytes_   = tensorElementCount(shape) * sizeof(float);
            found_output    = true;
        }
    }
#else
    // -----------------------------------------------------------------------
    // TensorRT 10.x — I/O Tensor API
    // -----------------------------------------------------------------------
    int32_t nb_io = engine_->getNbIOTensors();

    for (int32_t i = 0; i < nb_io; ++i) {
        char const* name  = engine_->getIOTensorName(i);
        auto        mode  = engine_->getTensorIOMode(name);
        auto        shape = engine_->getTensorShape(name);
        auto        dtype = engine_->getTensorDataType(name);

        const char* mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "Input" : "Output";
        std::printf("    %s: %-20s  shape=[", mode_str, name);
        printDims(shape);
        std::printf("]  dtype=%d\n", static_cast<int>(dtype));

        if (std::strcmp(name, "image") == 0 && mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_  = name;
            input_res_   = dimsToResolution(shape);
            input_bytes_ = tensorElementCount(shape) * sizeof(float);
            found_input  = true;
        } else if (std::strcmp(name, "depth") == 0 && mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_  = name;
            output_res_   = dimsToResolution(shape);
            output_bytes_ = tensorElementCount(shape) * sizeof(float);
            found_output  = true;
        }
    }
#endif

    if (!found_input || !found_output) {
        std::fprintf(stderr, "ERROR: Engine must have tensors named 'image' (input) and 'depth' (output).\n");
        std::fprintf(stderr, "       Found input='%s', output='%s'\n",
                     input_name_.c_str(), output_name_.c_str());
        std::exit(1);
    }

    // 5. Allocate device buffers
    CHECK_CUDA(cudaMalloc(&d_input_, input_bytes_));
    CHECK_CUDA(cudaMalloc(&d_output_, output_bytes_));

#ifdef DEPTH_TRT_LEGACY_API
    // Bind device pointers into the bindings array
    bindings_[input_binding_]  = d_input_;
    bindings_[output_binding_] = d_output_;
#endif

    // 6. Create CUDA stream
    CHECK_CUDA(cudaStreamCreate(&stream_));

#ifndef DEPTH_TRT_LEGACY_API
    // 7. Set static tensor addresses (TRT 10.x)
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
#endif

    std::printf("[*] Engine ready. Input: %dx%d, Output: %dx%d\n",
                input_res_.h, input_res_.w, output_res_.h, output_res_.w);
}

Engine::~Engine() {
    cudaStreamSynchronize(stream_);
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaStreamDestroy(stream_);
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

#ifdef DEPTH_TRT_LEGACY_API
    // TensorRT 8.x
    context_->setBindingDimensions(input_binding_, input_dims);
    auto out_dims = context_->getBindingDimensions(output_binding_);
#else
    // TensorRT 10.x
    context_->setInputShape(input_name_.c_str(), input_dims);
    auto out_dims = context_->getTensorShape(output_name_.c_str());
#endif

    Resolution out_res = dimsToResolution(out_dims);

    // 3. Copy input to device
    CHECK_CUDA(cudaMemcpyAsync(d_input_, preprocessed, input_bytes_,
                                cudaMemcpyHostToDevice, stream_));

    // 4. Execute
#ifdef DEPTH_TRT_LEGACY_API
    bool ok = context_->enqueueV2(bindings_.data(), stream_, nullptr);
#else
    bool ok = context_->enqueueV3(stream_);
#endif
    if (!ok) {
        std::fprintf(stderr, "ERROR: inference enqueue failed\n");
        std::exit(1);
    }

    // 5. Copy output back
    std::size_t out_bytes = static_cast<std::size_t>(out_res.h) * out_res.w * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(output, d_output_, out_bytes, cudaMemcpyDeviceToHost, stream_));

    // 6. Synchronize
    cudaStreamSynchronize(stream_);
}

} // namespace depth_trt
