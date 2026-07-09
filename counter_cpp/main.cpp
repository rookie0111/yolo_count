#include "yolo_detector.h"
#include "sam3_detector.h"
#include "bytetrack.h"
#include "counter.h"
#include "config.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <filesystem>
#include <chrono>
#include <csignal>

static volatile sig_atomic_t g_stop = 0;
static void signal_handler(int) { g_stop = 1; }

Config load_config(const std::string& path) {
    Config cfg;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open config: " << path << ", using defaults\n";
        return cfg;
    }

    fs["model"]["mode"] >> cfg.model.mode;
    fs["model"]["onnx_path"] >> cfg.model.onnx_path;
    fs["model"]["rknn_path"] >> cfg.model.rknn_path;
    fs["model"]["input_size"] >> cfg.model.input_size;
    fs["model"]["conf_thres"] >> cfg.model.conf_thres;
    fs["model"]["iou_thres"] >> cfg.model.iou_thres;
    fs["model"]["backend"] >> cfg.model.backend;

    // SAM3 specific
    fs["model"]["sam3_backbone_path"] >> cfg.model.sam3_backbone_path;
    fs["model"]["sam3_detector_path"] >> cfg.model.sam3_detector_path;
    fs["model"]["sam3_text_embed_path"] >> cfg.model.sam3_text_embed_path;
    fs["model"]["sam3_text_prompt"] >> cfg.model.sam3_text_prompt;

    fs["tracker"]["high_thresh"] >> cfg.tracker.high_thresh;
    fs["tracker"]["track_thresh"] >> cfg.tracker.track_thresh;
    fs["tracker"]["match_thresh"] >> cfg.tracker.match_thresh;
    fs["tracker"]["track_buffer"] >> cfg.tracker.track_buffer;

    fs["show_window"] >> cfg.show_window;
    fs["save_video"] >> cfg.save_video;
    fs["output_path"] >> cfg.output_path;
    fs["source"] >> cfg.source;

    // Parse line points: "180,430 330,250" or "200,100 500,100 500,350 200,350"
    std::string region_str;
    fs["counter"]["region"] >> region_str;
    if (!region_str.empty()) {
        std::replace(region_str.begin(), region_str.end(), ',', ' ');
        std::istringstream iss(region_str);
        int x, y;
        while (iss >> x >> y) {
            cfg.counter.line_points.push_back({x, y});
        }
    }

    fs["counter"]["direction"] >> cfg.counter.direction;
    fs["counter"]["enable_direction_check"] >> cfg.counter.enable_direction_check;
    fs["counter"]["polygon_mode"] >> cfg.counter.polygon_mode;

    fs["camera"]["source"] >> cfg.camera.source;
    fs["camera"]["width"] >> cfg.camera.width;
    fs["camera"]["height"] >> cfg.camera.height;
    fs["camera"]["fps"] >> cfg.camera.fps;

    fs.release();
    return cfg;
}

void print_usage() {
    std::cout << "Usage: bag_counter [options]\n"
              << "  --config <file>    Config file path (YAML/JSON format)\n"
              << "  --source <src>     Video source (file path or camera index)\n"
              << "  --model <path>     ONNX model path (YOLO mode)\n"
              << "  --mode <type>      Detection mode: yolo or sam3\n"
              << "  --backend <type>   Backend: onnx or rknn\n"
              << "  --max-frames <n>   Max frames to process (0 = unlimited)\n"
              << "  --save             Save output video\n"
              << "  --show             Show output window\n"
              << "  -h, --help         Print help\n";
}

int main(int argc, char** argv) {
    std::string config_path = "config.yaml";
    std::string source;
    std::string model_path;
    std::string backend;
    std::string mode;
    std::string output_path;
    bool save = false;
    bool show = false;
    int max_frames = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--max-frames" && i + 1 < argc) max_frames = std::atoi(argv[++i]);
        else if (arg == "--save") save = true;
        else if (arg == "--show") show = true;
        else if (arg == "-h" || arg == "--help") { print_usage(); return 0; }
        else if (arg[0] != '-' && output_path.empty()) output_path = arg;  // bare path → output
    }

    Config cfg = load_config(config_path);
    if (!source.empty()) cfg.source = source;
    if (!model_path.empty()) {
        if (cfg.model.backend == "rknn") cfg.model.rknn_path = model_path;
        else cfg.model.onnx_path = model_path;
    }
    if (!backend.empty()) cfg.model.backend = backend;
    if (!mode.empty()) cfg.model.mode = mode;
    if (!output_path.empty()) cfg.output_path = output_path;
    if (save) cfg.save_video = true;
    if (show) cfg.show_window = true;

    if (cfg.source.empty()) {
        cfg.source = "0";
    }

    // Open video source
    cv::VideoCapture cap;
    if (cfg.source.size() == 1 && std::isdigit(cfg.source[0])) {
        cap.open(std::stoi(cfg.source));
    } else {
        cap.open(cfg.source);
    }

    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << cfg.source << std::endl;
        return 1;
    }

    // Only set camera properties for live camera (not video files)
    if (cfg.source.size() == 1 && std::isdigit(cfg.source[0])) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.camera.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera.height);
        cap.set(cv::CAP_PROP_FPS, cfg.camera.fps);
    }

    BagCounter counter(cfg);

    cv::VideoWriter writer;
    if (cfg.save_video) {
        std::filesystem::path out_path(cfg.output_path);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;
        if (w <= 0) w = cfg.camera.width;
        if (h <= 0) h = cfg.camera.height;

        std::cout << "[Video] Source: " << w << "x" << h << " @" << fps << "fps" << std::endl;

        // Try multiple codecs: mp4v, then x264, then mjpg
        // NOTE: MP4 container requires release() to finalize (moov atom at end).
        //       AVI/MJPEG streams are playable even if interrupted.
        int codecs[] = {
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            cv::VideoWriter::fourcc('X', '2', '6', '4'),
        };
        const char* codec_names[] = {"mp4v", "MJPG", "X264"};
        const char* exts[] = {".mp4", ".avi", ".mp4"};

        bool writer_ok = false;
        for (int i = 0; i < 3 && !writer_ok; i++) {
            std::string out_file = cfg.output_path;
            // Adjust extension based on codec
            if (i >= 1) {
                std::string base = cfg.output_path;
                size_t dot = base.rfind('.');
                if (dot != std::string::npos) base = base.substr(0, dot);
                out_file = base + exts[i];
            }
            writer.open(out_file, codecs[i], fps, cv::Size(w, h));
            if (writer.isOpened()) {
                std::cout << "Saving to: " << out_file << " (codec: " << codec_names[i] << ")" << std::endl;
                cfg.output_path = out_file;
                writer_ok = true;
            }
        }
        if (!writer_ok) {
            std::cerr << "Warning: Cannot open output with any codec: " << cfg.output_path << std::endl;
        }
    }

    // Register signal handler for clean shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "Bag Counter C++ started. Press Ctrl+C to stop.\n";
    int last_count = 0;
    int frame_idx = 0;

    cv::Mat frame;
    while (cap.read(frame) && !g_stop) {
        if (max_frames > 0 && frame_idx >= max_frames) {
            std::cout << "Reached max frames (" << max_frames << "), stopping." << std::endl;
            break;
        }

        auto t0 = std::chrono::steady_clock::now();
        cv::Mat output = counter.process(frame);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (writer.isOpened()) {
            writer.write(output);
            if (frame_idx < 3 || frame_idx % 10 == 0) {
                std::cout << "[Frame " << frame_idx << "] " << ms << "ms, written to writer" << std::endl;
            }
        }
        frame_idx++;

        if (cfg.show_window) {
            cv::imshow("Bag Counter", output);
            if (cv::waitKey(1) == 'q') break;
        }

        if (counter.total_count() != last_count) {
            last_count = counter.total_count();
            std::cout << "Count: " << last_count << std::endl;
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();
    std::cout << "Final count: " << last_count << std::endl;
    return 0;
}
