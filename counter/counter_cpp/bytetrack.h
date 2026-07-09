#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

struct Detection;
struct STrack {
    enum State { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

    int track_id = 0;
    State state = New;
    bool is_activated = false;

    // bbox: [x1, y1, x2, y2]  (image coordinates)
    float tlwh[4] = {};   // top-left x, top-left y, width, height
    float score = 0;
    int cls = -1;
    int start_frame = 0;
    int frame_id = 0;
    int tracklet_len = 0;

    // Kalman state: [x, y, a, h, vx, vy, va, vh]
    Eigen::Matrix<float, 8, 1> mean;
    Eigen::Matrix<float, 8, 8> covariance;

    static int _count;

    STrack() {
        mean.setZero();
        covariance.setZero();
    }

    void activate(int frame_id);
    void re_activate(const STrack& new_track, int frame_id, bool new_id = false);
    void update(const STrack& new_track, int frame_id);
    void predict();

    // get bbox in [x1, y1, x2, y2]
    std::vector<float> tlbr() const;

    static void multi_predict(std::vector<STrack>& stracks);
    static void multi_gmc(std::vector<STrack>& stracks, const Eigen::Matrix<float, 2, 3>& H);

    static float tlwh_to_xyah(const float* tlwh, float* xyah);
    static float tlwh_to_xywh(const float* tlwh, float* xywh);
};

class BYTETracker {
public:
    struct Args {
        float track_high_thresh = 0.25f;
        float track_low_thresh = 0.1f;
        float new_track_thresh = 0.25f;
        int track_buffer = 30;
        float match_thresh = 0.8f;
        bool fuse_score = true;
        int frame_rate = 30;
    };

    BYTETracker(const Args& args);
    ~BYTETracker() = default;

    std::vector<STrack> update(const std::vector<Detection>& detections,
                                const cv::Mat& img = cv::Mat());
    void reset();

private:
    Args _args;
    int _frame_id = 0;
    int _max_time_lost;

    std::vector<STrack> _tracked_stracks;
    std::vector<STrack> _lost_stracks;
    std::vector<STrack> _removed_stracks;

    std::vector<STrack> joint_stracks(const std::vector<STrack>& a, const std::vector<STrack>& b);
    std::vector<STrack> sub_stracks(const std::vector<STrack>& a, const std::vector<STrack>& b);
    void remove_duplicate_stracks(std::vector<STrack>& a, std::vector<STrack>& b);

    // IoU + matching
    std::vector<std::vector<float>> iou_distance(const std::vector<STrack>& tracks,
                                                  const std::vector<STrack>& detections);
    void linear_assignment(const std::vector<std::vector<float>>& cost,
                           float thresh, int n_cols,
                           std::vector<std::pair<int, int>>& matches,
                           std::vector<int>& unmatched_a,
                           std::vector<int>& unmatched_b);
};
