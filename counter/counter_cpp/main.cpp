#include "yolo_detector.h"
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

Config load_config(const std::string& path) {
    Config cfg;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open config: " << path << ", using defaults\n";
        return cfg;
    }

    fs["model"]["onnx_path"] >> cfg.model.onnx_path;
    fs["model"]["rknn_path"] >> cfg.model.rknn_path;
    fs["model"]["input_size"] >> cfg.model.input_size;
    fs["model"]["conf_thres"] >> cfg.model.conf_thres;
    fs["model"]["iou_thres"] >> cfg.model.iou_thres;
    fs["model"]["backend"] >> cfg.model.backend;

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
              << "  --model <path>     ONNX model path\n"
              << "  --backend <type>   Backend: onnx or rknn\n"
              << "  --save             Save output video\n"
              << "  --show             Show output window\n"
              << "  -h, --help         Print help\n";
}

int main(int argc, char** argv) {
    std::string config_path = "config.yaml";
    std::string source;
    std::string model_path;
    std::string backend;
    std::string output_path;
    bool save = false;
    bool show = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
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

    // Only set camera properties for live camera, not video files
    if (cfg.source.size() == 1 && std::isdigit(cfg.source[0])) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.camera.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera.height);
        cap.set(cv::CAP_PROP_FPS, cfg.camera.fps);
    }

    BagCounter counter(cfg);

    cv::VideoWriter writer;
    if (cfg.save_video) {
        std::filesystem::path out_path(cfg.output_path);
        std::filesystem::create_directories(out_path.parent_path());

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;
        if (w <= 0) w = cfg.camera.width;
        if (h <= 0) h = cfg.camera.height;

        writer.open(cfg.output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(w, h));
        if (writer.isOpened())
            std::cout << "Saving to: " << cfg.output_path << std::endl;
        else
            std::cerr << "Warning: Cannot open output: " << cfg.output_path << std::endl;
    }

    std::cout << "Bag Counter C++ started. Press Ctrl+C to stop.\n";
    int last_count = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat output = counter.process(frame);

        if (cfg.show_window) {
            cv::imshow("Bag Counter", output);
            if (cv::waitKey(1) == 'q') break;
        }

        if (writer.isOpened()) writer.write(output);

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
