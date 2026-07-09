#include "sam3_detector.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#ifdef ONNX_ENABLE

SAM3Detector::SAM3Detector(const std::string& backbone_path,
                           const std::string& detector_path,
                           const std::string& text_embed_path,
                           float conf_thres,
                           float iou_thres)
    : _conf_thres(conf_thres), _iou_thres(iou_thres) {
    _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sam3_counter");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try CUDA provider first (reduces memory pressure on CPU)
    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.gpu_mem_limit = 10 * 1024 * 1024 * 1024UL;  // 10GB GPU memory limit
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[SAM3] Using CUDA execution provider" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cout << "[SAM3] CUDA not available, using CPU: " << e.what() << std::endl;
    }
    // Load backbone ONNX
    _backbone_session = std::make_unique<Ort::Session>(*_env, backbone_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;

    // Backbone input: "image"
    {
        auto name = _backbone_session->GetInputNameAllocated(0, alloc);
        _bb_input_names_str.push_back(name.get());
    }
    for (const auto& s : _bb_input_names_str) _bb_input_names.push_back(s.c_str());

    // Backbone outputs: 13 outputs (some have internal ONNX names like "10417")
    for (size_t i = 0; i < _backbone_session->GetOutputCount(); i++) {
        auto name = _backbone_session->GetOutputNameAllocated(i, alloc);
        _bb_output_names_str.push_back(name.get());
    }
    for (const auto& s : _bb_output_names_str) _bb_output_name_ptrs.push_back(s.c_str());

    if (_verbose) {
        std::cout << "[SAM3] Backbone loaded: " << backbone_path
                  << " inputs=" << _bb_input_names.size()
                  << " outputs=" << _bb_output_names_str.size() << std::endl;
    }

    // Load detector ONNX
    _detector_session = std::make_unique<Ort::Session>(*_env, detector_path.c_str(), opts);
    for (size_t i = 0; i < _detector_session->GetInputCount(); i++) {
        auto name = _detector_session->GetInputNameAllocated(i, alloc);
        _det_input_names_str.push_back(name.get());
    }
    for (const auto& s : _det_input_names_str) _det_input_names.push_back(s.c_str());

    for (size_t i = 0; i < _detector_session->GetOutputCount(); i++) {
        auto name = _detector_session->GetOutputNameAllocated(i, alloc);
        _det_output_names_str.push_back(name.get());
    }
    for (const auto& s : _det_output_names_str) _det_output_names.push_back(s.c_str());

    if (_verbose) {
        std::cout << "[SAM3] Detector loaded: " << detector_path
                  << " inputs=" << _det_input_names.size()
                  << " outputs=" << _det_output_names.size() << std::endl;
    }

    // Load pre-computed text embeddings
    load_text_embeddings(text_embed_path);
}

void SAM3Detector::load_text_embeddings(const std::string& path) {
    // Load .npz file using OpenCV FileStorage (we saved as npz, but we'll use a simple binary format)
    // Actually, we'll load from a simple binary file format that we create
    // For now, use numpy format via a simple approach: read raw float data

    // The npz file contains:
    //   language_features: (32, 1, 256) float32
    //   language_mask: (1, 32) bool

    // We'll read the npz using a minimal parser or use a pre-converted binary format
    // For simplicity, let's use OpenCV to read a .yml file with the data

    std::string yml_path = path;
    // Check if path ends with .npz, replace with .yml
    if (path.size() > 4 && path.substr(path.size() - 4) == ".npz") {
        yml_path = path.substr(0, path.size() - 4) + ".yml";
    }

    cv::FileStorage fs(yml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[SAM3] Cannot open text embeddings: " << yml_path << std::endl;
        std::cerr << "[SAM3] Please run: python -c \"...\" to convert .npz to .yml" << std::endl;
        return;
    }

    cv::Mat lang_feats_flat, lang_mask_flat;
    cv::Mat lang_feats_size, lang_mask_size;
    fs["lang_feats_flat"] >> lang_feats_flat;
    fs["lang_feats_size"] >> lang_feats_size;
    fs["lang_mask_flat"] >> lang_mask_flat;
    fs["lang_mask_size"] >> lang_mask_size;
    fs.release();

    if (lang_feats_flat.empty() || lang_mask_flat.empty()) {
        std::cerr << "[SAM3] Failed to read text embeddings from " << yml_path << std::endl;
        return;
    }

    // Convert to flat vectors
    // lang_feats_flat: (1, 8192) CV_32F -> 8192 floats
    // lang_mask_flat: (1, 32) CV_8U -> 32 bools
    lang_feats_flat.convertTo(lang_feats_flat, CV_32F);
    _lang_feats.resize(lang_feats_flat.total());
    std::memcpy(_lang_feats.data(), lang_feats_flat.data, lang_feats_flat.total() * sizeof(float));

    _lang_mask.resize(lang_mask_flat.total());
    for (size_t i = 0; i < lang_mask_flat.total(); i++) {
        _lang_mask[i] = lang_mask_flat.at<uchar>(i) != 0;
    }

    _text_ids = {0};  // single class

    if (_verbose) {
        std::cout << "[SAM3] Text embeddings loaded: feats_vec=" << _lang_feats.size()
                  << " mask_vec=" << _lang_mask.size() << std::endl;
    }
}

void SAM3Detector::preprocess(const cv::Mat& img, std::vector<float>& blob) {
    // SAM3 preprocessing: resize to 1008x1008, normalize to [-1, 1]
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(IMGSZ, IMGSZ));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 127.5, -1.0);  // (x - 127.5) / 127.5

    // HWC to CHW
    int h = IMGSZ, w = IMGSZ, c = 3;
    blob.resize(h * w * c);
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    for (int k = 0; k < c; k++) {
        float* dst = blob.data() + k * h * w;
        std::memcpy(dst, channels[k].data, h * w * sizeof(float));
    }
}

std::vector<Detection> SAM3Detector::detect(const cv::Mat& image) {
    cv::Size orig = image.size();

    // ── Step 1: Run backbone ──
    std::vector<float> bb_blob;
    preprocess(image, bb_blob);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> bb_shape = {1, 3, IMGSZ, IMGSZ};
    Ort::Value bb_input = Ort::Value::CreateTensor<float>(mem, bb_blob.data(), bb_blob.size(),
                                                           bb_shape.data(), bb_shape.size());

    // Run backbone
    auto bb_outputs = _backbone_session->Run(Ort::RunOptions{nullptr},
                                              _bb_input_names.data(), &bb_input, 1,
                                              _bb_output_name_ptrs.data(), _bb_output_name_ptrs.size());

    // Map backbone outputs by index (some ONNX outputs have empty names)
    // Known output order from export:
    //  0: vision_features
    //  1: vision_pos_enc_0 (288x288)
    //  2: vision_pos_enc_1 (144x144)
    //  3: vision_pos_enc_2 (72x72)    <-- we need this
    //  4: backbone_fpn_0 (288x288)    <-- we need this
    //  5: backbone_fpn_1 (144x144)    <-- we need this
    //  6: backbone_fpn_2 (72x72)      <-- we need this
    //  7: sam2_backbone_fpn
    //  8: sam2_vision_pos_enc
    //  9-12: internal shape/constant outputs

    const int IDX_POS_ENC_2 = 3;
    const int IDX_FPN_0 = 4;
    const int IDX_FPN_1 = 5;
    const int IDX_FPN_2 = 6;

    float* pos_enc_data = bb_outputs[IDX_POS_ENC_2].GetTensorMutableData<float>();
    float* fpn0_data = bb_outputs[IDX_FPN_0].GetTensorMutableData<float>();
    float* fpn1_data = bb_outputs[IDX_FPN_1].GetTensorMutableData<float>();
    float* fpn2_data = bb_outputs[IDX_FPN_2].GetTensorMutableData<float>();

    // Get shapes
    auto pos_enc_info = bb_outputs[IDX_POS_ENC_2].GetTensorTypeAndShapeInfo();
    auto fpn0_info = bb_outputs[IDX_FPN_0].GetTensorTypeAndShapeInfo();
    auto fpn1_info = bb_outputs[IDX_FPN_1].GetTensorTypeAndShapeInfo();
    auto fpn2_info = bb_outputs[IDX_FPN_2].GetTensorTypeAndShapeInfo();

    auto pos_shape = pos_enc_info.GetShape();
    auto fpn0_shape = fpn0_info.GetShape();
    auto fpn1_shape = fpn1_info.GetShape();
    auto fpn2_shape = fpn2_info.GetShape();

    // Create detector inputs
    int nc = 1;  // single class
    std::vector<int64_t> pos_enc_shape(pos_shape.begin(), pos_shape.end());
    std::vector<int64_t> fpn0_shape_vec(fpn0_shape.begin(), fpn0_shape.end());
    std::vector<int64_t> fpn1_shape_vec(fpn1_shape.begin(), fpn1_shape.end());
    std::vector<int64_t> fpn2_shape_vec(fpn2_shape.begin(), fpn2_shape.end());

    size_t pos_enc_size = pos_enc_info.GetElementCount();
    size_t fpn0_size = fpn0_info.GetElementCount();
    size_t fpn1_size = fpn1_info.GetElementCount();
    size_t fpn2_size = fpn2_info.GetElementCount();

    // Copy backbone outputs to new tensors (since ONNX Runtime may reuse buffers)
    std::vector<float> pos_enc_copy(pos_enc_data, pos_enc_data + pos_enc_size);
    std::vector<float> fpn0_copy(fpn0_data, fpn0_data + fpn0_size);
    std::vector<float> fpn1_copy(fpn1_data, fpn1_data + fpn1_size);
    std::vector<float> fpn2_copy(fpn2_data, fpn2_data + fpn2_size);

    Ort::Value det_pos_enc = Ort::Value::CreateTensor<float>(mem, pos_enc_copy.data(), pos_enc_size,
                                                              pos_enc_shape.data(), pos_enc_shape.size());
    Ort::Value det_fpn0 = Ort::Value::CreateTensor<float>(mem, fpn0_copy.data(), fpn0_size,
                                                           fpn0_shape_vec.data(), fpn0_shape_vec.size());
    Ort::Value det_fpn1 = Ort::Value::CreateTensor<float>(mem, fpn1_copy.data(), fpn1_size,
                                                           fpn1_shape_vec.data(), fpn1_shape_vec.size());
    Ort::Value det_fpn2 = Ort::Value::CreateTensor<float>(mem, fpn2_copy.data(), fpn2_size,
                                                           fpn2_shape_vec.data(), fpn2_shape_vec.size());

    // Language features: (seq=32, nc=1, dim=256) = 8192 floats total
    std::vector<int64_t> lang_feats_shape = {32, 1, 256};
    std::vector<int64_t> lang_mask_shape = {1, 32};
    std::vector<int64_t> text_ids_shape = {1};

    Ort::Value det_lang_feats = Ort::Value::CreateTensor<float>(mem, _lang_feats.data(), _lang_feats.size(),
                                                                 lang_feats_shape.data(), lang_feats_shape.size());

    // language_mask: bool tensor (ONNX Runtime needs contiguous bool* array)
    // std::vector<bool> is bit-packed, so use a raw bool array
    std::vector<char> lang_mask_raw(_lang_mask.size());
    for (size_t i = 0; i < _lang_mask.size(); i++) {
        lang_mask_raw[i] = _lang_mask[i] ? 1 : 0;
    }
    // Cast char* to bool* for ONNX Runtime (bool is 1 byte in most ABIs)
    Ort::Value det_lang_mask = Ort::Value::CreateTensor<bool>(
        mem, reinterpret_cast<bool*>(lang_mask_raw.data()), lang_mask_raw.size(),
        lang_mask_shape.data(), lang_mask_shape.size());

    Ort::Value det_text_ids = Ort::Value::CreateTensor<int64_t>(mem, _text_ids.data(), _text_ids.size(),
                                                                 text_ids_shape.data(), text_ids_shape.size());

    // Run detector
    std::array<Ort::Value, 7> det_inputs = {
        std::move(det_pos_enc),
        std::move(det_fpn0),
        std::move(det_fpn1),
        std::move(det_fpn2),
        std::move(det_lang_feats),
        std::move(det_lang_mask),
        std::move(det_text_ids),
    };

    auto det_outputs = _detector_session->Run(Ort::RunOptions{nullptr},
                                               _det_input_names.data(), det_inputs.data(), det_inputs.size(),
                                               _det_output_names.data(), _det_output_names.size());

    // ── Step 3: Post-process ──
    float* pred_boxes = det_outputs[0].GetTensorMutableData<float>();    // (nc, 200, 4) cxcywh normalized
    float* pred_logits = det_outputs[1].GetTensorMutableData<float>();   // (nc, 200, 1)
    float* presence = det_outputs[3].GetTensorMutableData<float>();      // (nc, 1)

    auto boxes_shape = det_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int nq = (int)boxes_shape[1];  // 200

    return postprocess(pred_boxes, pred_logits, presence, nc, nq, orig);
}

std::vector<Detection> SAM3Detector::postprocess(const float* boxes, const float* logits,
                                                   const float* presence, int nc, int nq,
                                                   const cv::Size& orig) {
    std::vector<Detection> dets;

    for (int c = 0; c < nc; c++) {
        float presence_score = sigmoid(presence[c]);

        for (int q = 0; q < nq; q++) {
            int idx = c * nq + q;
            float score = sigmoid(logits[idx]) * presence_score;

            if (score < _conf_thres) continue;

            // boxes are cxcywh normalized [0, 1]
            float cx = boxes[idx * 4 + 0];
            float cy = boxes[idx * 4 + 1];
            float w  = boxes[idx * 4 + 2];
            float h  = boxes[idx * 4 + 3];

            // Convert to original image pixel coordinates
            Detection d;
            d.x1 = std::max(0.0f, (cx - w / 2.0f) * orig.width);
            d.y1 = std::max(0.0f, (cy - h / 2.0f) * orig.height);
            d.x2 = std::min((float)orig.width, (cx + w / 2.0f) * orig.width);
            d.y2 = std::min((float)orig.height, (cy + h / 2.0f) * orig.height);
            d.conf = score;
            d.cls = c;
            dets.push_back(d);
        }
    }

    if (_verbose) {
        std::cout << "[SAM3] " << dets.size() << " detections above conf=" << _conf_thres;
        if (!dets.empty()) {
            std::cout << ", first: [" << dets[0].x1 << "," << dets[0].y1
                      << "," << dets[0].x2 << "," << dets[0].y2
                      << " c=" << dets[0].conf << "]";
        }
        std::cout << std::endl;
    }

    nms(dets);

    if (_verbose) {
        std::cout << "[SAM3] After NMS: " << dets.size() << " detections" << std::endl;
    }

    return dets;
}

float SAM3Detector::iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

void SAM3Detector::nms(std::vector<Detection>& dets) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });
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

#endif // ONNX_ENABLE
