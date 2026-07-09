#pragma once

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef ONNX_ENABLE
#include <onnxruntime_cxx_api.h>
#endif

struct Detection {
    float x1, y1, x2, y2;  // bbox in original image coordinates
    float conf;              // confidence
    int cls;                 // class index
};

// Abstract detector interface
class BaseDetector {
public:
    virtual ~BaseDetector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
    virtual cv::Size input_size() const = 0;
    virtual std::string backend_name() const = 0;
};
