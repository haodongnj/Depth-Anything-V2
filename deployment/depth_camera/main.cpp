// deployment/depth_camera/main.cpp
//
// Usage:
//   depth_camera --engine <path> --height <H> --width <W>
//                [--grayscale] [--camera <id>]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "depth_trt/engine.h"
#include "depth_trt/preprocess.h"
#include "depth_trt/visualize.h"
#include "depth_trt/types.h"

namespace {

constexpr int ESC_KEY = 27;
const char* WINDOW_NAME = "Depth Anything V2 | Original | Depth";

struct Args {
    std::string engine_path;
    int height  = 0;
    int width   = 0;
    int camera_id = 0;
    depth_trt::ColorMode mode = depth_trt::ColorMode::Colorized;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --engine <path> --height <H> --width <W> "
        "[--grayscale] [--camera <id>]\n\n"
        "  --engine      Path to .engine file (required)\n"
        "  --height      Engine input height (required)\n"
        "  --width       Engine input width (required)\n"
        "  --grayscale   Display grayscale depth instead of colorized\n"
        "  --camera      Camera device ID (default: 0)\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            a.engine_path = argv[++i];
        } else if (arg == "--height" && i + 1 < argc) {
            a.height = std::atoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            a.width = std::atoi(argv[++i]);
        } else if (arg == "--camera" && i + 1 < argc) {
            a.camera_id = std::atoi(argv[++i]);
        } else if (arg == "--grayscale") {
            a.mode = depth_trt::ColorMode::Grayscale;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (a.engine_path.empty() || a.height <= 0 || a.width <= 0) {
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
    depth_trt::Resolution engine_res = engine.inputResolution();

    // 2. Validate resolution
    if (engine_res.h != args.height || engine_res.w != args.width) {
        std::fprintf(stderr,
            "ERROR: Resolution mismatch.\n"
            "  Engine expects: %dx%d\n"
            "  CLI provided:   %dx%d\n",
            engine_res.h, engine_res.w, args.height, args.width);
        return 1;
    }

    // 3. Open camera
    cv::VideoCapture cap(args.camera_id);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "ERROR: Cannot open camera %d\n", args.camera_id);
        return 1;
    }
    std::printf("[*] Camera %d opened\n", args.camera_id);

    // 4. Pre-allocate output buffer
    depth_trt::Resolution out_res = engine.outputResolution();
    std::vector<float> depth_buf(static_cast<std::size_t>(out_res.h) * out_res.w);

    // Display height for side-by-side view
    int display_h = 480;

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, display_h * 2 + 40, display_h);

    std::printf("[*] Streaming. Press ESC to exit.\n");

    cv::Mat frame;
    int frame_count = 0;
    while (true) {
        if (!cap.read(frame)) {
            std::fprintf(stderr, "[!] Frame drop, retrying...\n");
            continue;
        }
        if (frame.empty()) continue;

        frame_count++;

        // 5. Preprocess (handles center-crop internally)
        cv::Mat tensor = depth_trt::preprocess(frame, engine_res);

        // 6. Infer
        engine.infer(tensor.ptr<float>(), depth_buf.data());

        // 7. Visualize
        cv::Mat depth_vis;
        if (args.mode == depth_trt::ColorMode::Grayscale) {
            depth_vis = depth_trt::depth_to_gray(depth_buf.data(), out_res);
        } else {
            depth_vis = depth_trt::depth_to_color(depth_buf.data(), out_res);
        }

        // 8. Resize for display (side-by-side, same height)
        cv::Mat orig_small, depth_small;
        int dw = static_cast<int>(static_cast<float>(display_h) / frame.rows * frame.cols);
        cv::resize(frame, orig_small, cv::Size(dw, display_h));
        cv::resize(depth_vis, depth_small, cv::Size(dw, display_h));

        // 9. Concatenate and show
        cv::Mat combined;
        cv::hconcat(orig_small, depth_small, combined);
        cv::imshow(WINDOW_NAME, combined);

        int key = cv::waitKey(1) & 0xFF;
        if (key == ESC_KEY || key == 'q') break;
    }

    std::printf("[*] Done. Processed %d frames.\n", frame_count);
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
