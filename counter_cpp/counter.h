#pragma once

#include <vector>
#include <set>
#include <unordered_map>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "detector.h"
#include "bytetrack.h"

class BagCounter {
public:
    BagCounter(const Config& config);
    ~BagCounter() = default;

    cv::Mat process(const cv::Mat& frame);

    int total_count() const { return _total_count; }
    const Config& config() const { return _config; }

private:
    Config _config;
    int _total_count = 0;
    int _frame_counter = 0;
    std::set<int> _counted_ids;
    std::set<int> _entered_ids;
    std::unordered_map<int, std::vector<cv::Point2f>> _track_history;

    // counting region as cv::Point list
    std::vector<cv::Point> _region;
    bool _is_line;  // true if 2-point line, false if polygon

    // Direction check
    static float point_line_side(float px, float py, cv::Point p1, cv::Point p2);

    void count_objects(const std::vector<STrack>& tracks);
    void draw_region(cv::Mat& img);
    void draw_tracks(cv::Mat& img, const std::vector<STrack>& tracks);
    void draw_count_overlay(cv::Mat& img);
    void draw_analytics(cv::Mat& img);
    void update_track_history(const std::vector<STrack>& tracks);
};
