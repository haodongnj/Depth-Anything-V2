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
