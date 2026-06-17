#include "yolo_detector.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

// ======================== ONNX Implementation ========================

#ifdef ONNX_ENABLE
YOLODetector::YOLODetector(const std::string& model_path, const std::string& backend,
                           int input_size, float conf_thres, float iou_thres)
    : _backend(backend), _input_size(input_size), _conf_thres(conf_thres), _iou_thres(iou_thres) {
    _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "bag_counter");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (backend == "onnx") {
        _session = std::make_unique<Ort::Session>(*_env, model_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < _session->GetInputCount(); i++) {
            auto name = _session->GetInputNameAllocated(i, alloc);
            _input_names.push_back(name.get());
            _input_name_ptrs.push_back(_input_names.back().c_str());
        }
        for (size_t i = 0; i < _session->GetOutputCount(); i++) {
            auto name = _session->GetOutputNameAllocated(i, alloc);
            _output_names.push_back(name.get());
            _output_name_ptrs.push_back(_output_names.back().c_str());
        }

        auto shape = _session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        _input_shape = shape;
    }
}

YOLODetector::~YOLODetector() = default;

void YOLODetector::preprocess_onnx(const cv::Mat& img, std::vector<float>& blob) {
    cv::Mat resized = letterbox(img, cv::Size(_input_size, _input_size));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    int h = _input_size, w = _input_size, c = 3;
    blob.resize(h * w * c);
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    for (int k = 0; k < c; k++) {
        float* dst = blob.data() + k * h * w;
        std::memcpy(dst, channels[k].data, h * w * sizeof(float));
    }
}

std::vector<Detection> YOLODetector::postprocess_onnx(const std::vector<float*>& outputs) {
    if (outputs.empty()) return {};
    float* output = outputs[0];
    auto out_shape = _session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int num_attrs = (int)out_shape[1];   // features per box: 4 + num_classes
    int num_boxes = (int)out_shape[2];   // number of predictions (e.g. 8400)
    if (num_attrs < 4) num_attrs = 84;
    if (num_boxes < 1) num_boxes = 8400;

    if (_verbose) {
        std::cout << "[YOLO] Output shape: [1, " << num_attrs << ", " << num_boxes << "]"
                  << " num_classes=" << (num_attrs - 4) << std::endl;
    }

    int num_classes = num_attrs - 4;
    return common_postprocess(output, num_boxes, num_attrs, num_classes, _orig_size);
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    _orig_size = image.size();
    std::vector<float> blob;
    preprocess_onnx(image, blob);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {1, 3, _input_size, _input_size};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(), shape.data(), shape.size());

    auto outputs = _session->Run(Ort::RunOptions{nullptr}, _input_name_ptrs.data(), &input, 1,
                                  _output_name_ptrs.data(), _output_name_ptrs.size());

    std::vector<float*> raw_outs;
    for (auto& o : outputs) {
        raw_outs.push_back(o.GetTensorMutableData<float>());
    }
    return postprocess_onnx(raw_outs);
}
#endif // ONNX_ENABLE

// ======================== RKNN Implementation ========================

#ifdef RKNN_ENABLE
YOLODetector::YOLODetector(const std::string& model_path, const std::string& backend,
                           int input_size, float conf_thres, float iou_thres)
    : _backend(backend), _input_size(input_size), _conf_thres(conf_thres), _iou_thres(iou_thres) {
    if (backend == "rknn") {
        std::ifstream f(model_path, std::ios::binary);
        f.seekg(0, std::ios::end);
        _rknn_model_size = (int)f.tellg();
        f.seekg(0, std::ios::beg);
        _rknn_model = new unsigned char[_rknn_model_size];
        f.read((char*)_rknn_model, _rknn_model_size);
        f.close();

        rknn_init(&_rknn_ctx, _rknn_model, _rknn_model_size, 0, nullptr);
        rknn_query(_rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &_rknn_io_num, sizeof(_rknn_io_num));

        _rknn_input_attrs = new rknn_tensor_attr[_rknn_io_num.n_input];
        for (uint32_t i = 0; i < _rknn_io_num.n_input; i++) {
            _rknn_input_attrs[i].index = i;
            rknn_query(_rknn_ctx, RKNN_QUERY_INPUT_ATTR, &_rknn_input_attrs[i], sizeof(rknn_tensor_attr));
        }
        _rknn_output_attrs = new rknn_tensor_attr[_rknn_io_num.n_output];
        for (uint32_t i = 0; i < _rknn_io_num.n_output; i++) {
            _rknn_output_attrs[i].index = i;
            rknn_query(_rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &_rknn_output_attrs[i], sizeof(rknn_tensor_attr));
        }
    }
}

YOLODetector::~YOLODetector() {
    if (_rknn_ctx) { rknn_destroy(_rknn_ctx); _rknn_ctx = 0; }
    delete[] _rknn_model;
    delete[] _rknn_input_attrs;
    delete[] _rknn_output_attrs;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    _orig_size = image.size();
    if (_backend != "rknn" || !_rknn_ctx) return {};

    cv::Mat resized = letterbox(image, cv::Size(_input_size, _input_size));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = _input_size * _input_size * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = rgb.data;

    rknn_inputs_set(_rknn_ctx, 1, inputs);

    rknn_output outputs[_rknn_io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < _rknn_io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }
    rknn_run(_rknn_ctx, nullptr);
    rknn_outputs_get(_rknn_ctx, _rknn_io_num.n_output, outputs, nullptr);

    std::vector<float> out_data;
    int out_size = outputs[0].size / sizeof(float);
    out_data.resize(out_size);
    memcpy(out_data.data(), outputs[0].buf, out_size * sizeof(float));
    rknn_outputs_release(_rknn_ctx, _rknn_io_num.n_output, outputs);

    int n_attrs = (int)_rknn_output_attrs[0].dims[1];   // features per box
    int n_anchor = (int)_rknn_output_attrs[0].dims[2];   // number of predictions
    if (n_attrs < 4) n_attrs = 84;
    if (n_anchor < 1) n_anchor = 8400;
    return common_postprocess(out_data.data(), n_anchor, n_attrs, n_attrs - 4, _orig_size);
}
#endif // RKNN_ENABLE

// ======================== Common (both ONNX and RKNN) ========================

cv::Mat YOLODetector::letterbox(const cv::Mat& img, const cv::Size& shape, cv::Scalar color) {
    float r = std::min((float)shape.width / img.cols, (float)shape.height / img.rows);
    int new_w = (int)(img.cols * r), new_h = (int)(img.rows * r);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat out(shape, img.type(), color);
    int dx = (shape.width - new_w) / 2, dy = (shape.height - new_h) / 2;
    resized.copyTo(out(cv::Rect(dx, dy, new_w, new_h)));
    return out;
}

std::vector<Detection> YOLODetector::common_postprocess(
    const float* pred, int num_boxes, int num_attrs, int num_classes, const cv::Size& orig
) {
    std::vector<Detection> dets;
    float r = std::min((float)_input_size / orig.width, (float)_input_size / orig.height);
    int new_w = (int)(orig.width * r);
    int new_h = (int)(orig.height * r);
    int pad_x = (_input_size - new_w) / 2;
    int pad_y = (_input_size - new_h) / 2;

    int raw_count = 0;
    for (int i = 0; i < num_boxes; i++) {
        // Output is planar (channel-first): [cx*N, cy*N, w*N, h*N, conf*N]
        float cx   = pred[0 * num_boxes + i];
        float cy   = pred[1 * num_boxes + i];
        float w    = pred[2 * num_boxes + i];
        float h    = pred[3 * num_boxes + i];
        float conf = pred[4 * num_boxes + i];  // single class score (no need for argmax)

        if (conf < _conf_thres) continue;
        raw_count++;

        // bbox is in input-image pixel coordinates
        // Convert to original image coordinates accounting for letterbox
        float orig_cx = (cx - pad_x) / r;
        float orig_cy = (cy - pad_y) / r;
        float orig_w  = w / r;
        float orig_h  = h / r;

        Detection d;
        d.x1 = std::max(0.0f, orig_cx - orig_w / 2.0f);
        d.y1 = std::max(0.0f, orig_cy - orig_h / 2.0f);
        d.x2 = std::min((float)orig.width, orig_cx + orig_w / 2.0f);
        d.y2 = std::min((float)orig.height, orig_cy + orig_h / 2.0f);
        d.conf = conf;
        d.cls = 0;
        dets.push_back(d);
    }

    if (_verbose) {
        std::cout << "[YOLO] " << num_boxes << " proposals, "
                  << raw_count << " above conf=" << _conf_thres;
        if (raw_count > 0 && raw_count <= 20) {
            std::cout << ", samples: ";
            for (int k = 0; k < std::min(raw_count, 3); k++) {
                std::cout << "[" << dets[k].x1 << "," << dets[k].y1
                          << "," << dets[k].x2 << "," << dets[k].y2
                          << " c=" << dets[k].conf << "] ";
            }
        }
        if (raw_count > 20) {
            std::cout << ", first box: [" << dets[0].x1 << "," << dets[0].y1
                      << "," << dets[0].x2 << "," << dets[0].y2 << "]";
        }
        std::cout << std::endl;
    }

    nms(dets);

    if (_verbose) {
        std::cout << "[YOLO] After NMS: " << dets.size() << " detections" << std::endl;
    }
    return dets;
}

float YOLODetector::iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

void YOLODetector::nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<Detection> keep;
    while (!dets.empty()) {
        keep.push_back(dets[0]);
        std::vector<Detection> rest;
        for (size_t i = 1; i < dets.size(); i++) {
            if (iou(dets[0], dets[i]) <= _iou_thres) rest.push_back(dets[i]);
        }
        dets = std::move(rest);
    }
    dets = std::move(keep);
}
