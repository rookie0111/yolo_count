#include "counter.h"
#include "bytetrack.h"
#include "detector.h"
#include "yolo_detector.h"
#include "sam3_detector.h"
#include "config.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iostream>

BagCounter::BagCounter(const Config& config) : _config(config) {
    const auto& pts = config.counter.line_points;
    for (const auto& p : pts) {
        _region.push_back(cv::Point(p.first, p.second));
    }
    _is_line = (_region.size() == 2);
}

float BagCounter::point_line_side(float px, float py, cv::Point p1, cv::Point p2) {
    return (p2.x - p1.x) * (py - p1.y) - (p2.y - p1.y) * (px - p1.x);
}

cv::Mat BagCounter::process(const cv::Mat& frame) {
    static BYTETracker::Args tracker_args;
    static bool tracker_init = false;
    static std::unique_ptr<BYTETracker> tracker;
    static std::unique_ptr<BaseDetector> detector;

    if (!detector) {
        const std::string& mode = _config.model.mode;
        std::string backend = _config.model.backend;

        if (mode == "sam3") {
#ifdef ONNX_ENABLE
            std::string text_embed_path = _config.model.sam3_text_embed_path;
            if (text_embed_path.empty()) {
                text_embed_path = "sam3_text_embeddings.yml";
            }
            detector = std::make_unique<SAM3Detector>(
                _config.model.sam3_backbone_path,
                _config.model.sam3_detector_path,
                text_embed_path,
                _config.model.conf_thres,
                _config.model.iou_thres
            );
            std::cout << "[Counter] Using SAM3 detector mode" << std::endl;
#else
            std::cerr << "[Counter] SAM3 mode requires ONNX Runtime support" << std::endl;
            return frame.clone();
#endif
        } else {
            // Default: YOLO mode
            std::string model_path;
            if (backend == "rknn") {
                model_path = _config.model.rknn_path;
            } else {
                model_path = _config.model.onnx_path;
            }
            detector = std::make_unique<YOLODetector>(
                model_path, backend,
                _config.model.input_size,
                _config.model.conf_thres,
                _config.model.iou_thres
            );
            std::cout << "[Counter] Using YOLO detector mode" << std::endl;
        }
    }

    if (!tracker_init) {
        tracker_args.track_high_thresh = _config.tracker.high_thresh;
        tracker_args.track_low_thresh = 0.1f;
        tracker_args.new_track_thresh = _config.tracker.track_thresh;
        tracker_args.match_thresh = _config.tracker.match_thresh;
        tracker_args.track_buffer = _config.tracker.track_buffer;
        tracker_args.frame_rate = _config.camera.fps;
        tracker = std::make_unique<BYTETracker>(tracker_args);
        tracker_init = true;
    }

    cv::Mat output = frame.clone();

    auto detections = detector->detect(frame);
    auto tracks = tracker->update(detections, frame);

    static int frame_idx = 0;
    if (frame_idx < 3 || frame_idx % 30 == 0) {
        std::cout << "[F" << frame_idx << "] dets=" << detections.size()
                  << " tracks=" << tracks.size()
                  << " count=" << _total_count
                  << " entered=" << _entered_ids.size()
                  << " region=" << (_is_line?"line":"polygon");
        if (_is_line) std::cout << " [" << _region[0].x<<","<<_region[0].y
                                << "]-["<<_region[1].x<<","<<_region[1].y<<"]";
        std::cout << std::endl;
    }
    frame_idx++;

    update_track_history(tracks);
    count_objects(tracks);

    draw_region(output);
    if (tracks.empty() && !detections.empty()) {
        // Fallback: draw detections when ByteTrack hasn't activated tracks yet
        for (const auto& d : detections) {
            cv::rectangle(output,
                cv::Point((int)d.x1, (int)d.y1),
                cv::Point((int)d.x2, (int)d.y2),
                cv::Scalar(0, 0, 255), 2);
        }
    } else {
        draw_tracks(output, tracks);
    }
    draw_count_overlay(output);
    draw_analytics(output);

    return output;
}

void BagCounter::update_track_history(const std::vector<STrack>& tracks) {
    for (const auto& track : tracks) {
        float cx = (track.tlwh[0] + track.tlwh[0] + track.tlwh[2]) / 2.0f;
        float cy = (track.tlwh[1] + track.tlwh[1] + track.tlwh[3]) / 2.0f;
        _track_history[track.track_id].push_back(cv::Point2f(cx, cy));
        if (_track_history[track.track_id].size() > 30) {
            _track_history[track.track_id].erase(_track_history[track.track_id].begin());
        }
    }
}

void BagCounter::count_objects(const std::vector<STrack>& tracks) {
    if (_region.empty()) return;

    static int count_frame = 0;

    for (const auto& track : tracks) {
        int tid = track.track_id;
        if (_counted_ids.count(tid)) continue;

        auto& history = _track_history[tid];
        if (history.size() < 2) {
            if (count_frame < 2 && history.size() == 1) {
                std::cout << "  [CNT] track " << tid << " first frame, waiting" << std::endl;
            }
            continue;
        }

        cv::Point2f curr = history.back();
        cv::Point2f prev = history[history.size() - 2];

        if (_is_line) {
            cv::Point p1 = _region[0], p2 = _region[1];
            if (!_config.counter.enable_direction_check) {
                _counted_ids.insert(tid);
                _total_count++;
                std::cout << "  [CNT] COUNT " << _total_count << " (no dir)" << std::endl;
                continue;
            }

            const auto& dir = _config.counter.direction;
            if (dir == "any") {
                float s1 = point_line_side(prev.x, prev.y, p1, p2);
                float s2 = point_line_side(curr.x, curr.y, p1, p2);

                if (count_frame < 5) {
                    std::cout << "  [CNT] tid=" << tid
                              << " prev=(" << (int)prev.x << "," << (int)prev.y
                              << " s1=" << (int)s1
                              << ") curr=(" << (int)curr.x << "," << (int)curr.y
                              << " s2=" << (int)s2
                              << ") line_y=" << p1.y << "-" << p2.y
                              << " cross=" << (s1 * s2 < 0 ? "YES" : "no")
                              << std::endl;
                }

                if (s1 * s2 >= 0) continue;
            } else {
                float prev_side = (point_line_side(prev.x, prev.y, p1, p2) > 0) ? 1.0f : -1.0f;
                float curr_side = (point_line_side(curr.x, curr.y, p1, p2) > 0) ? 1.0f : -1.0f;
                if (prev_side == curr_side) continue;

                static const std::unordered_map<std::string, std::pair<float, float>> dir_rules = {
                    {"up_right", {-1, 1}},
                    {"down_right", {1, -1}},
                    {"up_left", {1, -1}},
                    {"down_left", {-1, 1}},
                };
                auto it = dir_rules.find(dir);
                if (it == dir_rules.end()) continue;
                if (prev_side != it->second.first || curr_side != it->second.second) continue;
            }
            _counted_ids.insert(tid);
            _total_count++;
        } else {
            // Polygon: enter+exit
            std::vector<cv::Point2f> poly;
            for (const auto& p : _region) poly.push_back(cv::Point2f(p.x, p.y));
            bool inside = cv::pointPolygonTest(poly, curr, false) >= 0;

            if (_config.counter.polygon_mode == "enter_exit") {
                if (inside) {
                    _entered_ids.insert(tid);
                } else if (_entered_ids.count(tid)) {
                    _entered_ids.erase(tid);
                    _counted_ids.insert(tid);
                    _total_count++;
                }
            } else {
                if (inside) {
                    _counted_ids.insert(tid);
                    _total_count++;
                }
            }
        }
    }
    count_frame++;
}

void BagCounter::draw_region(cv::Mat& img) {
    if (_region.size() < 2) return;
    cv::Scalar color(104, 0, 123);
    for (size_t i = 0; i < _region.size(); i++) {
        cv::line(img, _region[i], _region[(i + 1) % _region.size()], color, 4);
        cv::circle(img, _region[i], 8, color, -1);
    }
}

void BagCounter::draw_tracks(cv::Mat& img, const std::vector<STrack>& tracks) {
    for (const auto& track : tracks) {
        int x1 = (int)track.tlwh[0];
        int y1 = (int)track.tlwh[1];
        int x2 = (int)(track.tlwh[0] + track.tlwh[2]);
        int y2 = (int)(track.tlwh[1] + track.tlwh[3]);

        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = "bag " + std::to_string(track.track_id);
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        cv::rectangle(img, cv::Point(x1, y1 - text_size.height - 10),
                      cv::Point(x1 + text_size.width, y1), color, -1);
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    }
}

void BagCounter::draw_count_overlay(cv::Mat& img) {
    std::string label = "COUNT: " + std::to_string(_total_count);
    float scale = std::max(1.5f, img.cols / 800.0f);
    int thickness = std::max(3, (int)(scale * 2));
    int baseline;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    int x = 20, y = 50 + text_size.height;

    cv::rectangle(img, cv::Point(x - 10, y - text_size.height - 15),
                  cv::Point(x + text_size.width + 10, y + 15),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(img, label, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
}

void BagCounter::draw_analytics(cv::Mat& img) {
    if (_total_count == 0) return;
    std::string txt = "Total: " + std::to_string(_total_count);
    int margin = 10, baseline;
    cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    int rx = img.cols - ts.width - margin * 4 - 20;
    int ry = 5 + ts.height + margin * 2;

    cv::rectangle(img, cv::Point(rx, 5),
                  cv::Point(img.cols - 20, ry),
                  cv::Scalar(104, 31, 17), -1);
    cv::putText(img, txt, cv::Point(rx + margin * 2, ry - margin),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}
