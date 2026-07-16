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
