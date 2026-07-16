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
