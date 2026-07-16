// deployment/depth_image/main.cpp
//
// Usage:
//   depth_image --engine <path> --input <path> --output <path>
//               --height <H> --width <W> [--grayscale]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
