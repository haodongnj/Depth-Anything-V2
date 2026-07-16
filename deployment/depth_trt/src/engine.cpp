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
        std::fprintf(stderr, "CUDA error at %s:%d -- %s (%d)\n",
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

    std::printf("[*] Engine ready. Input: %dx%d, Output: %dx%d\n",
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
