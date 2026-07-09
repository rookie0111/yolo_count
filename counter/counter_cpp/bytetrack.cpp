#include "bytetrack.h"
#include "yolo_detector.h"

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

// ======================== STrack ========================

int STrack::_count = 0;

static Eigen::Matrix<float, 8, 8> make_kalman_init_cov() {
    Eigen::Matrix<float, 8, 8> P = Eigen::Matrix<float, 8, 8>::Zero();
    P.diagonal() << 4.0, 4.0, 1.0, 4.0, 100.0, 100.0, 100.0, 100.0;
    return P;
}

void STrack::activate(int frame_id) {
    if (is_activated) return;
    track_id = _count++;
    state = Tracked;
    is_activated = true;

    // mean = [x, y, a, h, 0, 0, 0, 0]
    mean = Eigen::Matrix<float, 8, 1>::Zero();
    float cx = tlwh[0] + tlwh[2] / 2.0f;
    float cy = tlwh[1] + tlwh[3] / 2.0f;
    float a = tlwh[2] / std::max(tlwh[3], 1e-6f);
    float h = tlwh[3];
    mean << cx, cy, a, h, 0, 0, 0, 0;

    covariance = make_kalman_init_cov();
    start_frame = frame_id;
    this->frame_id = frame_id;
    tracklet_len = 0;
}

void STrack::re_activate(const STrack& new_track, int frame_id, bool new_id) {
    // Kalman update with new measurement
    Eigen::Matrix<float, 4, 1> z;
    float cx = new_track.tlwh[0] + new_track.tlwh[2] / 2.0f;
    float cy = new_track.tlwh[1] + new_track.tlwh[3] / 2.0f;
    float a = new_track.tlwh[2] / std::max(new_track.tlwh[3], 1e-6f);
    float h = new_track.tlwh[3];
    z << cx, cy, a, h;

    // Observation matrix H (4x8)
    Eigen::Matrix<float, 4, 8> H = Eigen::Matrix<float, 4, 8>::Zero();
    H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1; H(3, 3) = 1;

    // Measurement noise R
    Eigen::Matrix<float, 4, 4> R = Eigen::Matrix<float, 4, 4>::Zero();
    R(0, 0) = 4; R(1, 1) = 4; R(2, 2) = 1; R(3, 3) = 4;

    // Innovation
    Eigen::Matrix<float, 4, 1> y = z - H * mean;
    Eigen::Matrix<float, 4, 4> S = H * covariance * H.transpose() + R;
    Eigen::Matrix<float, 8, 4> K = covariance * H.transpose() * S.inverse();

    mean = mean + K * y;
    covariance = (Eigen::Matrix<float, 8, 8>::Identity() - K * H) * covariance;

    state = Tracked;
    is_activated = true;
    frame_id = frame_id;
    if (new_id) track_id = _count++;
    score = new_track.score;
    cls = new_track.cls;
    tracklet_len = 0;
    tlwh[0] = new_track.tlwh[0];
    tlwh[1] = new_track.tlwh[1];
    tlwh[2] = new_track.tlwh[2];
    tlwh[3] = new_track.tlwh[3];
}

void STrack::update(const STrack& new_track, int frame_id) {
    this->frame_id = frame_id;
    tracklet_len++;

    // Copy new detection position
    tlwh[0] = new_track.tlwh[0];
    tlwh[1] = new_track.tlwh[1];
    tlwh[2] = new_track.tlwh[2];
    tlwh[3] = new_track.tlwh[3];

    Eigen::Matrix<float, 4, 1> z;
    float cx = new_track.tlwh[0] + new_track.tlwh[2] / 2.0f;
    float cy = new_track.tlwh[1] + new_track.tlwh[3] / 2.0f;
    float a = new_track.tlwh[2] / std::max(new_track.tlwh[3], 1e-6f);
    float h = new_track.tlwh[3];
    z << cx, cy, a, h;

    Eigen::Matrix<float, 4, 8> H = Eigen::Matrix<float, 4, 8>::Zero();
    H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1; H(3, 3) = 1;

    Eigen::Matrix<float, 4, 4> R = Eigen::Matrix<float, 4, 4>::Zero();
    R(0, 0) = 4; R(1, 1) = 4; R(2, 2) = 1; R(3, 3) = 4;

    Eigen::Matrix<float, 4, 1> y = z - H * mean;
    Eigen::Matrix<float, 4, 4> S = H * covariance * H.transpose() + R;
    Eigen::Matrix<float, 8, 4> K = covariance * H.transpose() * S.inverse();

    mean = mean + K * y;
    covariance = (Eigen::Matrix<float, 8, 8>::Identity() - K * H) * covariance;

    state = Tracked;
    is_activated = true;
    score = new_track.score;
    cls = new_track.cls;

    tlwh[0] = new_track.tlwh[0];
    tlwh[1] = new_track.tlwh[1];
    tlwh[2] = new_track.tlwh[2];
    tlwh[3] = new_track.tlwh[3];
}

void STrack::predict() {
    // State transition matrix F (8x8)
    Eigen::Matrix<float, 8, 8> F = Eigen::Matrix<float, 8, 8>::Identity();
    F(0, 4) = 1; F(1, 5) = 1; F(2, 6) = 1; F(3, 7) = 1;

    // Process noise Q
    Eigen::Matrix<float, 8, 8> Q = Eigen::Matrix<float, 8, 8>::Zero();
    float dt = 1.0f;
    Q(0, 0) = dt * dt * dt / 3; Q(0, 4) = dt * dt / 2;
    Q(4, 0) = dt * dt / 2;       Q(4, 4) = dt;
    Q(1, 1) = dt * dt * dt / 3; Q(1, 5) = dt * dt / 2;
    Q(5, 1) = dt * dt / 2;       Q(5, 5) = dt;
    Q(2, 2) = dt * dt * dt / 3; Q(2, 6) = dt * dt / 2;
    Q(6, 2) = dt * dt / 2;       Q(6, 6) = dt;
    Q(3, 3) = dt * dt * dt / 3; Q(3, 7) = dt * dt / 2;
    Q(7, 3) = dt * dt / 2;       Q(7, 7) = dt;
    Q *= 0.05f;

    if (state != Tracked) {
        mean(7) = 0;
    }
    mean = F * mean;
    covariance = F * covariance * F.transpose() + Q;
}

std::vector<float> STrack::tlbr() const {
    (void)mean; // tlwh is used directly, mean is updated via Kalman
    return {tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]};
}

void STrack::multi_predict(std::vector<STrack>& stracks) {
    for (auto& t : stracks) t.predict();
}

void STrack::multi_gmc(std::vector<STrack>& stracks, const Eigen::Matrix<float, 2, 3>& H) {
    if (stracks.empty()) return;
    Eigen::Matrix2f R = H.block<2, 2>(0, 0);
    Eigen::Vector2f t = H.col(2);
    for (auto& st : stracks) {
        Eigen::Vector2f pos(st.mean(0), st.mean(1));
        pos = R * pos + t;
        st.mean(0) = pos(0);
        st.mean(1) = pos(1);
    }
}

// ======================== BYTETracker ========================

BYTETracker::BYTETracker(const Args& args) : _args(args), _max_time_lost((int)(args.frame_rate / 30.0f * args.track_buffer)) {}

static float compute_iou(const std::vector<float>& a, const std::vector<float>& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float denom = area_a + area_b - inter;
    return (denom > 0) ? inter / denom : 0;
}

std::vector<std::vector<float>> BYTETracker::iou_distance(
    const std::vector<STrack>& tracks,
    const std::vector<STrack>& detections
) {
    if (tracks.empty() || detections.empty()) return {};
    std::vector<std::vector<float>> cost(tracks.size(), std::vector<float>(detections.size()));
    for (size_t i = 0; i < tracks.size(); i++) {
        auto ta = tracks[i].tlbr();
        for (size_t j = 0; j < detections.size(); j++) {
            auto da = detections[j].tlbr();
            float iou = compute_iou(ta, da);
            cost[i][j] = 1.0f - iou;
        }
    }
    return cost;
}

// Hungarian algorithm (linear_sum_assignment) — O(n^3)
static std::vector<int> hungarian(const std::vector<std::vector<float>>& cost) {
    int rows = (int)cost.size();
    int cols = rows > 0 ? (int)cost[0].size() : 0;
    if (rows == 0 || cols == 0) return {};

    int n = std::max(rows, cols);
    std::vector<std::vector<double>> a(n, std::vector<double>(n, 1e9));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            a[i][j] = cost[i][j];

    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, 1e18);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            double delta = 1e18;
            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(rows, -1);
    for (int j = 1; j <= n; j++) {
        if (p[j] > 0 && p[j] <= rows && j <= cols) {
            assignment[p[j] - 1] = j - 1;
        }
    }
    return assignment;
}

void BYTETracker::linear_assignment(
    const std::vector<std::vector<float>>& cost,
    float thresh, int n_cols,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_a,
    std::vector<int>& unmatched_b
) {
    matches.clear();
    unmatched_a.clear();
    unmatched_b.clear();

    if (cost.empty()) {
        for (int j = 0; j < n_cols; j++) unmatched_b.push_back(j);
        return;
    }

    int n_rows = (int)cost.size();
    auto assign = hungarian(cost);

    std::set<int> matched_cols;
    for (int i = 0; i < n_rows; i++) {
        int j = assign[i];
        if (j >= 0 && j < n_cols && cost[i][j] <= thresh) {
            matches.push_back({i, j});
            matched_cols.insert(j);
        } else {
            unmatched_a.push_back(i);
        }
    }
    for (int j = 0; j < n_cols; j++) {
        if (!matched_cols.count(j)) unmatched_b.push_back(j);
    }
}

std::vector<STrack> BYTETracker::joint_stracks(const std::vector<STrack>& a, const std::vector<STrack>& b) {
    std::set<int> exists;
    std::vector<STrack> res;
    for (const auto& t : a) { exists.insert(t.track_id); res.push_back(t); }
    for (const auto& t : b) { if (!exists.count(t.track_id)) res.push_back(t); }
    return res;
}

std::vector<STrack> BYTETracker::sub_stracks(const std::vector<STrack>& a, const std::vector<STrack>& b) {
    std::set<int> ids;
    for (const auto& t : b) ids.insert(t.track_id);
    std::vector<STrack> res;
    for (const auto& t : a) if (!ids.count(t.track_id)) res.push_back(t);
    return res;
}

void BYTETracker::remove_duplicate_stracks(std::vector<STrack>& a, std::vector<STrack>& b) {
    auto dist = iou_distance(a, b);
    std::set<int> dupa, dupb;
    for (size_t i = 0; i < dist.size(); i++) {
        for (size_t j = 0; j < dist[i].size(); j++) {
            if (dist[i][j] < 0.15f) {
                int ta = a[i].frame_id - a[i].start_frame;
                int tb = b[j].frame_id - b[j].start_frame;
                if (ta > tb) dupb.insert((int)j); else dupa.insert((int)i);
            }
        }
    }
    std::vector<STrack> na, nb;
    for (size_t i = 0; i < a.size(); i++) if (!dupa.count((int)i)) na.push_back(a[i]);
    for (size_t i = 0; i < b.size(); i++) if (!dupb.count((int)i)) nb.push_back(b[i]);
    a = na; b = nb;
}

std::vector<STrack> BYTETracker::update(const std::vector<Detection>& detections, const cv::Mat&) {
    _frame_id++;
    std::vector<STrack> activated, refind, lost, removed;

    // Build STrack from high-score detections
    std::vector<STrack> det_stracks;
    std::vector<STrack> det_second;
    for (const auto& det : detections) {
        STrack st;
        st.tlwh[0] = det.x1;
        st.tlwh[1] = det.y1;
        st.tlwh[2] = det.x2 - det.x1;
        st.tlwh[3] = det.y2 - det.y1;
        st.score = det.conf;
        st.cls = det.cls;

        if (det.conf >= _args.track_high_thresh) {
            det_stracks.push_back(st);
        } else if (det.conf >= _args.track_low_thresh) {
            det_second.push_back(st);
        }
    }

    std::vector<STrack> unconfirmed;
    std::vector<STrack> tracked;
    for (const auto& t : _tracked_stracks) {
        if (!t.is_activated) unconfirmed.push_back(t);
        else tracked.push_back(t);
    }

    // Step 2: First association (high score)
    auto strack_pool = joint_stracks(tracked, _lost_stracks);
    STrack::multi_predict(strack_pool);

    auto dists = iou_distance(strack_pool, det_stracks);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_det;
    linear_assignment(dists, _args.match_thresh, (int)det_stracks.size(), matches, u_track, u_det);

    for (auto [it, id] : matches) {
        auto& track = strack_pool[it];
        auto& det = det_stracks[id];
        if (track.state == STrack::Tracked) {
            track.update(det, _frame_id);
            activated.push_back(track);
        } else {
            track.re_activate(det, _frame_id, false);
            refind.push_back(track);
        }
    }

    // Step 3: Second association (low score)
    std::vector<STrack> r_tracked;
    for (int i : u_track) {
        if (strack_pool[i].state == STrack::Tracked) r_tracked.push_back(strack_pool[i]);
    }
    dists = iou_distance(r_tracked, det_second);
    matches.clear();
    std::vector<int> u_track2, u_det2;
    linear_assignment(dists, 0.5f, (int)det_second.size(), matches, u_track2, u_det2);

    for (auto [it, id] : matches) {
        r_tracked[it].update(det_second[id], _frame_id);
        activated.push_back(r_tracked[it]);
    }
    for (int i : u_track2) {
        auto& track = r_tracked[i];
        if (track.state != STrack::Lost) {
            track.state = STrack::Lost;
            lost.push_back(track);
        }
    }

    // Unconfirmed tracks
    std::vector<STrack> det_unmatched;
    for (int i : u_det) det_unmatched.push_back(det_stracks[i]);
    dists = iou_distance(unconfirmed, det_unmatched);
    matches.clear();
    std::vector<int> u_unc, u_udet;
    linear_assignment(dists, 0.7f, (int)det_unmatched.size(), matches, u_unc, u_udet);

    for (auto [it, id] : matches) {
        unconfirmed[it].update(det_unmatched[id], _frame_id);
        activated.push_back(unconfirmed[it]);
    }
    for (int i : u_unc) {
        unconfirmed[i].state = STrack::Removed;
        removed.push_back(unconfirmed[i]);
    }

    // Init new tracks
    for (int i : u_udet) {
        auto& det = det_unmatched[i];
        if (det.score < _args.new_track_thresh) continue;
        det.activate(_frame_id);
        activated.push_back(det);
    }

    // Remove lost tracks past buffer
    for (auto& t : _lost_stracks) {
        if (_frame_id - t.frame_id > _max_time_lost) {
            t.state = STrack::Removed;
            removed.push_back(t);
        }
    }

    // Update state lists
    std::vector<STrack> new_tracked;
    for (const auto& t : _tracked_stracks)
        if (t.state == STrack::Tracked) new_tracked.push_back(t);

    _tracked_stracks = joint_stracks(activated, new_tracked);  // new positions take priority
    _tracked_stracks = joint_stracks(refind, _tracked_stracks);    // refind (new) takes priority
    _lost_stracks = sub_stracks(_lost_stracks, _tracked_stracks);
    for (auto& t : lost) _lost_stracks.push_back(t);
    _lost_stracks = sub_stracks(_lost_stracks, _removed_stracks);
    remove_duplicate_stracks(_tracked_stracks, _lost_stracks);
    for (auto& t : removed) _removed_stracks.push_back(t);
    if (_removed_stracks.size() > 1000)
        _removed_stracks = std::vector<STrack>(_removed_stracks.end() - 1000, _removed_stracks.end());

    // Build result
    std::vector<STrack> result;
    for (const auto& t : _tracked_stracks)
        if (t.is_activated) result.push_back(t);
    return result;
}

void BYTETracker::reset() {
    _frame_id = 0;
    _tracked_stracks.clear();
    _lost_stracks.clear();
    _removed_stracks.clear();
    STrack::_count = 0;
}
