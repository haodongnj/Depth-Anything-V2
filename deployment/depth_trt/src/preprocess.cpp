// deployment/depth_trt/src/preprocess.cpp
#include "depth_trt/preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
