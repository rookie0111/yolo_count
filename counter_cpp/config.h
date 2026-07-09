#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct CameraConfig {
    int source = 0;
    int width = 1280;
    int height = 720;
    int fps = 30;
};

struct ModelConfig {
    std::string mode = "yolo";     // "yolo" or "sam3"
    std::string type = "yolov8";
    std::string weight;          // .pt path (not used in C++)
    std::string onnx_path;       // ONNX model path (YOLO)
    std::string rknn_path;       // RKNN model path
    int input_size = 640;
    float conf_thres = 0.35f;
    float iou_thres = 0.50f;
    std::string backend = "onnx"; // "onnx" or "rknn"

    // SAM3 specific
    std::string sam3_backbone_path;      // sam3_backbone.onnx
    std::string sam3_detector_path;      // sam3_detector.onnx
    std::string sam3_text_embed_path;    // sam3_text_embeddings.yml
    std::string sam3_text_prompt = "bag on the conveyor belt";
};

struct TrackerConfig {
    std::string type = "bytetrack";
    float track_thresh = 0.45f;
    float high_thresh = 0.60f;
    float match_thresh = 0.75f;
    int track_buffer = 30;
};

struct CounterConfig {
    std::vector<std::pair<int, int>> line_points;  // 2-point line or 4-point polygon
    std::string direction = "any";
    bool enable_direction_check = true;
    std::string polygon_mode = "enter_exit";
};

struct Config {
    CameraConfig camera;
    ModelConfig model;
    TrackerConfig tracker;
    CounterConfig counter;
    bool show_window = true;
    bool save_video = false;
    std::string output_path = "outputs/predict.mp4";
    std::string source;  // video file or camera index
};

Config load_config(const std::string& path);
