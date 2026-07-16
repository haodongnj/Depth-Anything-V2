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
