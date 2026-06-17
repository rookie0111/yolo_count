#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef ONNX_ENABLE
#include <onnxruntime_cxx_api.h>
#endif

#ifdef RKNN_ENABLE
#include "rknn_api.h"
#endif

struct Detection {
    float x1, y1, x2, y2;  // bbox
    float conf;              // confidence
    int cls;                 // class index
};

class YOLODetector {
public:
    YOLODetector(const std::string& model_path, const std::string& backend,
                 int input_size, float conf_thres, float iou_thres);
    ~YOLODetector();

    std::vector<Detection> detect(const cv::Mat& image);

    cv::Size input_size() const { return cv::Size(_input_size, _input_size); }

private:
    std::string _backend;
    int _input_size;
    float _conf_thres;
    float _iou_thres;
    bool _verbose = true;
    cv::Size _orig_size;

    // ONNX
#ifdef ONNX_ENABLE
    std::unique_ptr<Ort::Env> _env;
    std::unique_ptr<Ort::Session> _session;
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;
    std::vector<const char*> _input_name_ptrs;
    std::vector<const char*> _output_name_ptrs;
    std::vector<int64_t> _input_shape;

    std::vector<Detection> postprocess_onnx(const std::vector<float*>& outputs);
    void preprocess_onnx(const cv::Mat& img, std::vector<float>& blob);
#endif

    // RKNN
#ifdef RKNN_ENABLE
    rknn_context _rknn_ctx = 0;
    rknn_input_output_num _rknn_io_num{};
    rknn_tensor_attr* _rknn_input_attrs = nullptr;
    rknn_tensor_attr* _rknn_output_attrs = nullptr;
    unsigned char* _rknn_model = nullptr;
    int _rknn_model_size = 0;

    std::vector<Detection> postprocess_rknn(float* outputs, const cv::Size& orig);
#endif

    std::vector<Detection> common_postprocess(const float* pred, int num_boxes,
                                               int num_attrs, int num_classes, const cv::Size& orig);

    float iou(const Detection& a, const Detection& b);
    void nms(std::vector<Detection>& dets);

    static void softmax(const float* x, int length, float* result);
    static int argmax(const float* x, int length);
    static cv::Mat letterbox(const cv::Mat& img, const cv::Size& new_shape,
                              cv::Scalar color = cv::Scalar(114, 114, 114));
};
