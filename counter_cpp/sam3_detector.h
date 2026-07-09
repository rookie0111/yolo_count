#pragma once

#include "detector.h"

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef ONNX_ENABLE
#include <onnxruntime_cxx_api.h>
#endif

class SAM3Detector : public BaseDetector {
public:
    SAM3Detector(const std::string& backbone_path,
                 const std::string& detector_path,
                 const std::string& text_embed_path,
                 float conf_thres,
                 float iou_thres);
    ~SAM3Detector() = default;

    std::vector<Detection> detect(const cv::Mat& image) override;
    cv::Size input_size() const override { return cv::Size(1008, 1008); }
    std::string backend_name() const override { return "sam3"; }

private:
    float _conf_thres;
    float _iou_thres;
    bool _verbose = true;
    static constexpr int IMGSZ = 1008;
    static constexpr int FEAT_SZ = 72;
    static constexpr int NUM_QUERIES = 200;

    // ONNX sessions
#ifdef ONNX_ENABLE
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _backbone_session;
    std::unique_ptr<Ort::Session> _detector_session;

    // Backbone
    std::vector<std::string> _bb_input_names_str;
    std::vector<std::string> _bb_output_names_str;
    std::vector<const char*> _bb_input_names;
    std::vector<const char*> _bb_output_names;
    std::vector<const char*> _bb_output_name_ptrs;

    // Detector
    std::vector<std::string> _det_input_names_str;
    std::vector<std::string> _det_output_names_str;
    std::vector<const char*> _det_input_names;
    std::vector<const char*> _det_output_names;

    // Pre-computed text embeddings
    std::vector<float> _lang_feats;    // (32, 1, 256) flattened
    std::vector<bool> _lang_mask;       // (1, 32)
    std::vector<int64_t> _text_ids;     // (1,) = [0]

    void load_text_embeddings(const std::string& path);
    void preprocess(const cv::Mat& img, std::vector<float>& blob);
    std::vector<Detection> postprocess(const float* boxes, const float* logits,
                                        const float* presence, int nc, int nq,
                                        const cv::Size& orig);
#endif

    float iou(const Detection& a, const Detection& b);
    void nms(std::vector<Detection>& dets);

    static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
