/**
 * OracleTrack - Multi-Object Tracker with Kalman Filtering + Cascaded Association
 */

#include <motcpp/trackers/oracletrack.hpp>
#include <motcpp/utils/matching.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace motcpp::trackers {

// Static member initialization
Eigen::Matrix<float, 7, 7> OracleState::F_;
Eigen::Matrix<float, 4, 7> OracleState::H_;
Eigen::Matrix<float, 7, 7> OracleState::Q_;
Eigen::Matrix<float, 4, 4> OracleState::R_;
bool OracleState::matrices_initialized_ = false;

void OracleState::init_matrices() {
    if (matrices_initialized_) return;
    
    // State transition (constant velocity model)
    F_.setIdentity();
    F_(0, 4) = 1.0f;  // x += vx
    F_(1, 5) = 1.0f;  // y += vy
    F_(2, 6) = 1.0f;  // s += vs
    
    // Observation matrix
    H_.setZero();
    H_(0, 0) = 1.0f;
    H_(1, 1) = 1.0f;
    H_(2, 2) = 1.0f;
    H_(3, 3) = 1.0f;
    
    // Q and R will be set adaptively per-track based on object size
    Q_.setIdentity();
    R_.setIdentity();
    
    matrices_initialized_ = true;
}

OracleState::OracleState() 
    : id(-1), cls(-1), det_ind(-1), conf(0.0f)
    , age(0), hits(0), time_since_update(0)
    , avg_iou(0.0f), consecutive_hits(0)
    , state(TrackState::Tentative), prev_state(TrackState::Tentative)
    , track_score(0.0f), frames_since_birth(0)
    , covariance_frozen(false)
{
    init_matrices();
    mean.setZero();
    covariance.setIdentity();
    frozen_covariance.setIdentity();
}

OracleState::OracleState(float x1, float y1, float x2, float y2,
                         float confidence, int class_id, int detection_index, int track_id)
    : id(track_id), cls(class_id), det_ind(detection_index), conf(confidence)
    , age(1), hits(1), time_since_update(0)
    , avg_iou(1.0f), consecutive_hits(1)
    , state(TrackState::Tentative), prev_state(TrackState::Tentative)
    , track_score(confidence), frames_since_birth(0)
    , covariance_frozen(false)
{
    init_matrices();
    
    // State: [cx, cy, s, r, vx, vy, vs] where s = area (w*h), r = w/h (like SORT)
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = x2 - x1;
    float h = y2 - y1;
    float s = std::max(w * h, 1.0f);  // s = area
    float r = w / std::max(h, 1e-6f);
    
    mean << cx, cy, s, r, 0.0f, 0.0f, 0.0f;
    
    // Adaptive initial covariance (like ByteTrack)
    const float std_weight_position = 1.0f / 20.0f;
    const float std_weight_velocity = 1.0f / 160.0f;
    
    covariance.setZero();
    float init_pos_std = 2.0f * std_weight_position * h;
    float init_vel_std = 10.0f * std_weight_velocity * h;
    
    covariance(0, 0) = init_pos_std * init_pos_std;  // cx
    covariance(1, 1) = init_pos_std * init_pos_std;  // cy
    covariance(2, 2) = 1e-2f * h * h;                // scale (area)
    covariance(3, 3) = 1e-2f;                        // aspect ratio
    covariance(4, 4) = init_vel_std * init_vel_std;  // vx
    covariance(5, 5) = init_vel_std * init_vel_std;  // vy
    covariance(6, 6) = 1e-5f * h * h;                // vs
    
    frozen_covariance = covariance;  // Initialize frozen copy for OC-SORT recovery
}

float OracleState::get_priority_score() const {
    // Priority score combines state and track quality
    float state_weight = 1.0f;
    switch (state) {
        case TrackState::Mature: state_weight = 3.0f; break;
        case TrackState::Confirmed: state_weight = 2.0f; break;
        case TrackState::Tentative: state_weight = 1.0f; break;
    }
    return state_weight * track_score * static_cast<float>(consecutive_hits) / 
           (static_cast<float>(time_since_update) + 1.0f);
}

bool OracleState::should_output() const {
    // Only output Confirmed or Mature tracks (consecutive_hits >= 3)
    // This eliminates "ghost" tracks from noise
    return (state == TrackState::Confirmed || state == TrackState::Mature);
}

bool OracleState::has_positive_confidence_gradient() const {
    if (conf_history.size() < 2) return true;  // Not enough data, allow
    
    // Check if confidence is rising or stable
    float prev_conf = conf_history[conf_history.size() - 2];
    float curr_conf = conf_history[conf_history.size() - 1];
    
    // Allow if: confidence is rising, OR both are reasonably high
    return (curr_conf >= prev_conf - 0.05f) && (curr_conf >= 0.55f || curr_conf > prev_conf);
}

void OracleState::update_state() {
    // Save previous state for transition detection
    prev_state = state;
    
    // Update track state based on CONSECUTIVE hits (not total)
    // This prevents ghost tracks from being output
    if (consecutive_hits >= 10) {
        state = TrackState::Mature;
    } else if (consecutive_hits >= 2) {
        state = TrackState::Confirmed;
    } else {
        state = TrackState::Tentative;
    }
}

std::array<float, 4> OracleState::to_xyxy() const {
    // Convert from [cx, cy, s, r] to [x1, y1, x2, y2]
    // where s = area (w*h), r = w/h (like SORT)
    float cx = mean(0);
    float cy = mean(1);
    float s = std::max(mean(2), 1.0f);  // s = area
    float r = std::max(mean(3), 0.01f);  // r = w/h
    
    // s = w*h, r = w/h => w = sqrt(s*r), h = s/w
    float w = std::sqrt(s * r);
    float h = s / std::max(w, 1e-6f);
    
    return {cx - w/2, cy - h/2, cx + w/2, cy + h/2};
}

std::array<float, 4> OracleState::predicted_xyxy() const {
    // One-step ahead prediction: add velocities to position
    float cx = mean(0) + mean(4);
    float cy = mean(1) + mean(5);
    float s = std::max(mean(2) + mean(6), 1.0f);  // s = area
    float r = std::max(mean(3), 0.01f);  // r = w/h
    
    float w = std::sqrt(s * r);
    float h = s / std::max(w, 1e-6f);
    
    return {cx - w/2, cy - h/2, cx + w/2, cy + h/2};
}

void OracleState::predict() {
    // Constant velocity prediction
    mean = F_ * mean;
    
    // Adaptive process noise scaled by object size
    const float std_weight_position = 1.0f / 20.0f;
    const float std_weight_velocity = 1.0f / 160.0f;
    
    float s = std::max(mean(2), 1.0f);
    float r = std::max(mean(3), 0.01f);
    float h = std::sqrt(s / r);
    
    Eigen::Matrix<float, 7, 7> Q_adaptive = Eigen::Matrix<float, 7, 7>::Zero();
    float pos_std = std_weight_position * h;
    float vel_std = std_weight_velocity * h;
    
    Q_adaptive(0, 0) = pos_std * pos_std;
    Q_adaptive(1, 1) = pos_std * pos_std;
    Q_adaptive(2, 2) = 1e-2f * h * h;
    Q_adaptive(3, 3) = 1e-4f;
    Q_adaptive(4, 4) = vel_std * vel_std;
    Q_adaptive(5, 5) = vel_std * vel_std;
    Q_adaptive(6, 6) = 1e-5f * h * h;
    
    // OC-SORT style: Freeze covariance growth after 5 frames of being lost
    if (time_since_update >= 5 && !covariance_frozen) {
        frozen_covariance = covariance;
        covariance_frozen = true;
    }
    
    if (covariance_frozen) {
        // Use frozen covariance (no growth)
        covariance = F_ * frozen_covariance * F_.transpose();
        covariance = (covariance + covariance.transpose()) * 0.5f;
    } else {
        // Normal covariance growth with adaptive noise
        if (time_since_update > 0) {
            float uncertainty_scale = 1.0f + 0.5f * static_cast<float>(time_since_update);
            Q_adaptive *= uncertainty_scale * uncertainty_scale;
        }
        
        covariance = F_ * covariance * F_.transpose() + Q_adaptive;
        covariance = (covariance + covariance.transpose()) * 0.5f;
    }
    
    age++;
    time_since_update++;
    frames_since_birth++;
}

void OracleState::update(float x1, float y1, float x2, float y2, 
                         float new_conf, int new_cls, int new_det_ind) {
    // Measurement: [cx, cy, s, r] where s = area (like SORT)
    float det_cx = (x1 + x2) * 0.5f;
    float det_cy = (y1 + y2) * 0.5f;
    float det_w = x2 - x1;
    float det_h = y2 - y1;
    float det_s = std::max(det_w * det_h, 1.0f);  // s = area
    float det_r = det_w / std::max(det_h, 1e-6f);
    
    Eigen::Matrix<float, 4, 1> z;
    z << det_cx, det_cy, det_s, det_r;
    
    // Adaptive measurement noise scaled by object size
    const float std_weight_position = 1.0f / 20.0f;
    Eigen::Matrix<float, 4, 4> R_adaptive = Eigen::Matrix<float, 4, 4>::Zero();
    float meas_std = std_weight_position * det_h;
    R_adaptive(0, 0) = meas_std * meas_std;  // cx
    R_adaptive(1, 1) = meas_std * meas_std;  // cy
    R_adaptive(2, 2) = 1e-1f * det_h * det_h;  // scale
    R_adaptive(3, 3) = 1e-1f;  // aspect ratio
    
    Eigen::Matrix<float, 4, 1> y = z - H_ * mean;
    Eigen::Matrix<float, 4, 4> S = H_ * covariance * H_.transpose() + R_adaptive;
    Eigen::Matrix<float, 7, 4> K = covariance * H_.transpose() * S.inverse();
    
    mean = mean + K * y;
    
    Eigen::Matrix<float, 7, 7> I = Eigen::Matrix<float, 7, 7>::Identity();
    covariance = (I - K * H_) * covariance;
    covariance = (covariance + covariance.transpose()) * 0.5f;
    
    // OC-SORT: Unfreeze covariance when track is successfully re-associated
    if (covariance_frozen) {
        covariance_frozen = false;
        frozen_covariance = covariance;
    }
    
    // Update avg_iou
    auto pred_box = to_xyxy();
    float xi1 = std::max(pred_box[0], x1);
    float yi1 = std::max(pred_box[1], y1);
    float xi2 = std::min(pred_box[2], x2);
    float yi2 = std::min(pred_box[3], y2);
    float inter_area = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
    float box1_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]);
    float box2_area = (x2 - x1) * (y2 - y1);
    float union_area = box1_area + box2_area - inter_area;
    float iou = (union_area > 0) ? inter_area / union_area : 0.0f;
    
    avg_iou = (avg_iou * (hits - 1) + iou) / hits;
    
    // Update track score (exponential moving average)
    float quality = new_conf * iou;  // Confidence × match quality
    track_score = 0.8f * track_score + 0.2f * quality;
    
    // Update state based on hits
    update_state();
    
    conf = new_conf;
    cls = new_cls;
    det_ind = new_det_ind;
    
    // Update confidence history for gradient check
    if (conf_history.size() >= 3) conf_history.pop_front();
    conf_history.push_back(new_conf);
    
    hits++;
    consecutive_hits++;
    time_since_update = 0;
    frames_since_birth++;
}

float OracleState::mahalanobis_distance(float x1, float y1, float x2, float y2) const {
    // Measurement: [cx, cy, s, r] where s = area (like SORT)
    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = x2 - x1;
    float h = y2 - y1;
    float s = std::max(w * h, 1.0f);  // s = area
    float r = w / std::max(h, 1e-6f);
    
    Eigen::Matrix<float, 4, 1> z;
    z << cx, cy, s, r;
    
    Eigen::Matrix<float, 4, 1> y = z - H_ * mean;
    Eigen::Matrix<float, 4, 4> S = H_ * covariance * H_.transpose() + R_;
    
    float d2 = (y.transpose() * S.inverse() * y)(0, 0);
    return std::sqrt(std::max(d2, 0.0f));
}

float OracleState::gating_distance(float x1, float y1, float x2, float y2) const {
    return mahalanobis_distance(x1, y1, x2, y2);
}

bool OracleState::within_gate(float x1, float y1, float x2, float y2, float chi2_threshold) const {
    float d = mahalanobis_distance(x1, y1, x2, y2);
    return d * d < chi2_threshold;
}

// ============================================================================
// OracleTrack Implementation
// ============================================================================

OracleTrack::OracleTrack(
    float det_thresh,
    int max_age,
    int min_hits,
    float gating_threshold,
    float max_mahalanobis
)
    : BaseTracker(det_thresh, max_age, 50, min_hits, 0.3f, false, 80, "iou", false)
    , gating_threshold_(gating_threshold)
    , max_mahalanobis_(max_mahalanobis)
    , next_id_(0)
    , frame_id_(0)
    , cmc_enabled_(true)  // CMC helps significantly
{
    camera_motion_.setIdentity();
}

void OracleTrack::reset() {
    tracks_.clear();
    next_id_ = 0;
    frame_id_ = 0;
    prev_frame_ = cv::Mat();
    camera_motion_.setIdentity();
}

void OracleTrack::estimate_camera_motion(const cv::Mat& frame) {
    if (!cmc_enabled_ || frame.empty()) {
        camera_motion_.setIdentity();
        return;
    }
    
    if (prev_frame_.empty()) {
        prev_frame_ = frame.clone();
        camera_motion_.setIdentity();
        return;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_prev, gray_curr;
    if (prev_frame_.channels() == 3) {
        cv::cvtColor(prev_frame_, gray_prev, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, gray_curr, cv::COLOR_BGR2GRAY);
    } else {
        gray_prev = prev_frame_;
        gray_curr = frame;
    }
    
    // Detect ORB keypoints in both frames
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kp_prev, kp_curr;
    cv::Mat desc_prev, desc_curr;
    
    orb->detectAndCompute(gray_prev, cv::Mat(), kp_prev, desc_prev);
    orb->detectAndCompute(gray_curr, cv::Mat(), kp_curr, desc_curr);
    
    if (kp_prev.size() < 10 || kp_curr.size() < 10) {
        camera_motion_.setIdentity();
        prev_frame_ = frame.clone();
        return;
    }
    
    // Match features using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(desc_prev, desc_curr, knn_matches, 2);
    
    // Apply ratio test
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts_prev, pts_curr;
    
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2 && match_pair[0].distance < 0.75f * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
            pts_prev.push_back(kp_prev[match_pair[0].queryIdx].pt);
            pts_curr.push_back(kp_curr[match_pair[0].trainIdx].pt);
        }
    }
    
    // Estimate affine transformation (2x3 matrix)
    if (good_matches.size() >= 10) {
        cv::Mat H = cv::estimateAffinePartial2D(pts_prev, pts_curr, cv::noArray(), cv::RANSAC, 3.0);
        
        if (!H.empty()) {
            // Convert 2x3 affine to 3x3 homogeneous
            camera_motion_(0, 0) = H.at<double>(0, 0);
            camera_motion_(0, 1) = H.at<double>(0, 1);
            camera_motion_(0, 2) = H.at<double>(0, 2);
            camera_motion_(1, 0) = H.at<double>(1, 0);
            camera_motion_(1, 1) = H.at<double>(1, 1);
            camera_motion_(1, 2) = H.at<double>(1, 2);
            camera_motion_(2, 0) = 0.0f;
            camera_motion_(2, 1) = 0.0f;
            camera_motion_(2, 2) = 1.0f;
        } else {
            camera_motion_.setIdentity();
        }
    } else {
        camera_motion_.setIdentity();
    }
    
    prev_frame_ = frame.clone();
}

void OracleTrack::apply_camera_motion_to_tracks() {
    if (!cmc_enabled_) return;
    
    // Apply inverse camera motion to all track predictions
    for (auto& track : tracks_) {
        float cx = track.mean(0);
        float cy = track.mean(1);
        
        // Transform center point
        Eigen::Vector3f point(cx, cy, 1.0f);
        Eigen::Vector3f transformed = camera_motion_ * point;
        
        track.mean(0) = transformed(0);
        track.mean(1) = transformed(1);
        
        // Scale velocity by transformation (approximate)
        float vx = track.mean(4);
        float vy = track.mean(5);
        
        Eigen::Vector3f vel_point(cx + vx, cy + vy, 1.0f);
        Eigen::Vector3f vel_transformed = camera_motion_ * vel_point;
        
        track.mean(4) = vel_transformed(0) - transformed(0);
        track.mean(5) = vel_transformed(1) - transformed(1);
    }
}

void OracleTrack::predict_all() {
    for (auto& track : tracks_) {
        track.predict();
    }
}

void OracleTrack::compute_gated_associations(const Eigen::MatrixXf& dets) {
    gated_pairs_.clear();
    gated_pairs_.resize(tracks_.size());
    
    for (size_t t = 0; t < tracks_.size(); ++t) {
        for (int d = 0; d < dets.rows(); ++d) {
            if (tracks_[t].within_gate(dets(d, 0), dets(d, 1), dets(d, 2), dets(d, 3), gating_threshold_)) {
                gated_pairs_[t].push_back(d);
            }
        }
    }
}

void OracleTrack::build_cost_matrix(const Eigen::MatrixXf& dets) {
    int n_tracks = static_cast<int>(tracks_.size());
    int n_dets = static_cast<int>(dets.rows());
    
    cost_matrix_.resize(n_tracks, n_dets);
    cost_matrix_.setConstant(1.0f);
    
    for (int t = 0; t < n_tracks; ++t) {
        auto track_box = tracks_[t].to_xyxy();
        for (int d = 0; d < n_dets; ++d) {
            std::array<float, 4> det_box = {dets(d, 0), dets(d, 1), dets(d, 2), dets(d, 3)};
            float iou = compute_iou(track_box, det_box);
            cost_matrix_(t, d) = 1.0f - iou;
        }
    }
}

void OracleTrack::associate_and_update(const Eigen::MatrixXf& dets) {
    int n_tracks = static_cast<int>(tracks_.size());
    int n_dets = static_cast<int>(dets.rows());
    
    if (n_tracks == 0 || n_dets == 0) {
        det_matched_.assign(n_dets, false);
        track_matched_.assign(n_tracks, false);
        return;
    }
    
    auto result = utils::linear_assignment(cost_matrix_, 0.7f);
    
    det_matched_.assign(n_dets, false);
    track_matched_.assign(n_tracks, false);
    
    for (const auto& match : result.matches) {
        int track_idx = match[0];
        int det_idx = match[1];
        
        if (cost_matrix_(track_idx, det_idx) > 0.7f) continue;
        
        track_matched_[track_idx] = true;
        det_matched_[det_idx] = true;
        
        tracks_[track_idx].update(
            dets(det_idx, 0), dets(det_idx, 1),
            dets(det_idx, 2), dets(det_idx, 3),
            dets(det_idx, 4), static_cast<int>(dets(det_idx, 5)),
            det_idx
        );
    }
    
    for (size_t t = 0; t < tracks_.size(); ++t) {
        if (!track_matched_[t]) {
            tracks_[t].consecutive_hits = 0;
        }
    }
}

void OracleTrack::create_new_tracks(const Eigen::MatrixXf& dets) {
    for (int d = 0; d < dets.rows(); ++d) {
        if (!det_matched_[d]) {
            tracks_.emplace_back(
                dets(d, 0), dets(d, 1), dets(d, 2), dets(d, 3),
                dets(d, 4), static_cast<int>(dets(d, 5)),
                d, ++next_id_
            );
        }
    }
}

void OracleTrack::remove_dead_tracks() {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const OracleState& track) {
                return track.time_since_update > max_age_;
            }),
        tracks_.end()
    );
}

Eigen::MatrixXf OracleTrack::build_output() const {
    std::vector<int> output_indices;
    
    for (size_t i = 0; i < tracks_.size(); ++i) {
        const auto& track = tracks_[i];
        // Output tracks that:
        // 1. Were updated this frame (time_since_update == 0)
        // 2. Are Confirmed/Mature (should_output), OR
        // 3. Are Tentative but have consecutive_hits >= 1 AND positive confidence gradient
        if (track.time_since_update == 0) {
            if (track.should_output()) {
                // Always output Confirmed/Mature tracks
                output_indices.push_back(static_cast<int>(i));
            } else if (track.state == TrackState::Tentative && track.consecutive_hits >= 1) {
                // Tentative: only output if confidence is rising/stable (FP suppression)
                if (track.has_positive_confidence_gradient()) {
                    output_indices.push_back(static_cast<int>(i));
                }
            }
        }
    }
    
    if (output_indices.empty()) {
        return Eigen::MatrixXf(0, 7);
    }
    
    Eigen::MatrixXf output(output_indices.size(), 7);
    
    for (size_t i = 0; i < output_indices.size(); ++i) {
        const auto& track = tracks_[output_indices[i]];
        auto box = track.to_xyxy();
        
        output(i, 0) = box[0];
        output(i, 1) = box[1];
        output(i, 2) = box[2];
        output(i, 3) = box[3];
        output(i, 4) = static_cast<float>(track.id);
        output(i, 5) = track.conf;
        output(i, 6) = static_cast<float>(track.cls);
    }
    
    return output;
}

float OracleTrack::compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    float xi1 = std::max(a[0], b[0]);
    float yi1 = std::max(a[1], b[1]);
    float xi2 = std::min(a[2], b[2]);
    float yi2 = std::min(a[3], b[3]);
    
    float inter_area = std::max(0.0f, xi2 - xi1) * std::max(0.0f, yi2 - yi1);
    float a_area = (a[2] - a[0]) * (a[3] - a[1]);
    float b_area = (b[2] - b[0]) * (b[3] - b[1]);
    float union_area = a_area + b_area - inter_area;
    
    return (union_area > 0) ? inter_area / union_area : 0.0f;
}

Eigen::MatrixXf OracleTrack::update(
    const Eigen::MatrixXf& dets,
    const cv::Mat& img,
    const Eigen::MatrixXf& /* embs */
) {
    frame_id_++;
    
    // ===== CAMERA MOTION COMPENSATION =====
    // Estimate camera motion from frame-to-frame transformation
    estimate_camera_motion(img);
    
    // Predict all tracks first
    predict_all();
    
    // Apply camera motion compensation to predictions
    apply_camera_motion_to_tracks();
    
    // ByteTrack-style: Separate high and low confidence detections  
    const float high_thresh = 0.6f;   // Balanced threshold for MOTA  
    const float low_thresh = 0.1f;    // Low confidence threshold
    
    // Split detections into high/low confidence
    std::vector<int> high_conf_indices, low_conf_indices;
    for (int i = 0; i < dets.rows(); ++i) {
        float conf = dets(i, 4);
        if (conf >= high_thresh) {
            high_conf_indices.push_back(i);
        } else if (conf >= low_thresh) {
            low_conf_indices.push_back(i);
        }
    }
    
    // Initialize tracking state
    int n_tracks = static_cast<int>(tracks_.size());
    std::vector<bool> track_matched(n_tracks, false);
    std::vector<bool> det_matched_high(high_conf_indices.size(), false);
    std::vector<bool> det_matched_low(low_conf_indices.size(), false);
    
    // ===== STAGE 1: Cascaded priority matching with high-confidence detections =====
    // Priority order: Mature > Confirmed > Tentative
    const float match_thresh = 0.75f;  // Balanced threshold (IoU > 0.25 required)
    
    if (!tracks_.empty() && !high_conf_indices.empty()) {
        // Group tracks by state
        std::vector<int> mature_indices, confirmed_indices, tentative_indices;
        for (int t = 0; t < n_tracks; ++t) {
            switch (tracks_[t].state) {
                case TrackState::Mature:
                    mature_indices.push_back(t);
                    break;
                case TrackState::Confirmed:
                    confirmed_indices.push_back(t);
                    break;
                case TrackState::Tentative:
                    tentative_indices.push_back(t);
                    break;
            }
        }
        
        // Track which high-conf detections are still available
        std::vector<int> available_high_dets;
        for (size_t i = 0; i < high_conf_indices.size(); ++i) {
            available_high_dets.push_back(static_cast<int>(i));
        }
        
        // Lambda for cascaded matching by priority
        auto match_cascade = [&](const std::vector<int>& track_group) {
            if (track_group.empty() || available_high_dets.empty()) return;
            
            int n_group = static_cast<int>(track_group.size());
            int n_avail = static_cast<int>(available_high_dets.size());
            
            Eigen::MatrixXf cost(n_group, n_avail);
            cost.setConstant(1.0f);
            
            // Build cost matrix with predicted IoU bonus
            for (int t = 0; t < n_group; ++t) {
                int track_idx = track_group[t];
                auto track_box = tracks_[track_idx].to_xyxy();
                auto predicted_box = tracks_[track_idx].predicted_xyxy();
                
                for (int d = 0; d < n_avail; ++d) {
                    int det_local = available_high_dets[d];
                    int det_idx = high_conf_indices[det_local];
                    std::array<float, 4> det_box = {dets(det_idx, 0), dets(det_idx, 1),
                                                     dets(det_idx, 2), dets(det_idx, 3)};
                    
                    // IoU with predicted position bonus (handles fast motion)
                    float iou = compute_iou(track_box, det_box);
                    float predicted_iou = compute_iou(predicted_box, det_box);
                    
                    // 80% current, 20% predicted as tiebreaker
                    float combined_iou = 0.8f * iou + 0.2f * predicted_iou;
                    cost(t, d) = 1.0f - combined_iou;
                }
            }
            
            auto result = utils::linear_assignment(cost, match_thresh);
            
            // Update matched tracks and mark detections as unavailable
            std::vector<int> newly_matched_dets;
            for (const auto& match : result.matches) {
                int t_local = match[0];
                int d_local = match[1];
                
                if (cost(t_local, d_local) > match_thresh) continue;
                
                int track_idx = track_group[t_local];
                int det_avail_idx = available_high_dets[d_local];
                int det_idx = high_conf_indices[det_avail_idx];
                
                track_matched[track_idx] = true;
                det_matched_high[det_avail_idx] = true;
                newly_matched_dets.push_back(d_local);
                
                tracks_[track_idx].update(
                    dets(det_idx, 0), dets(det_idx, 1),
                    dets(det_idx, 2), dets(det_idx, 3),
                    dets(det_idx, 4), static_cast<int>(dets(det_idx, 5)),
                    det_idx
                );
            }
            
            // Remove matched detections from available pool
            std::sort(newly_matched_dets.rbegin(), newly_matched_dets.rend());
            for (int idx : newly_matched_dets) {
                available_high_dets.erase(available_high_dets.begin() + idx);
            }
        };
        
        // Cascade: Mature -> Confirmed -> Tentative
        match_cascade(mature_indices);
        match_cascade(confirmed_indices);
        match_cascade(tentative_indices);
    }
    
    // ===== STAGE 2: Match low-confidence detections with unmatched tracks =====
    std::vector<int> unmatched_track_indices;
    for (int t = 0; t < n_tracks; ++t) {
        if (!track_matched[t]) {
            unmatched_track_indices.push_back(t);
        }
    }
    
    if (!unmatched_track_indices.empty() && !low_conf_indices.empty()) {
        int n_unmatched = static_cast<int>(unmatched_track_indices.size());
        int n_low = static_cast<int>(low_conf_indices.size());
        Eigen::MatrixXf cost_low(n_unmatched, n_low);
        cost_low.setConstant(1.0f);
        
        for (int ut = 0; ut < n_unmatched; ++ut) {
            int track_idx = unmatched_track_indices[ut];
            auto track_box = tracks_[track_idx].to_xyxy();
            for (int d = 0; d < n_low; ++d) {
                int det_idx = low_conf_indices[d];
                std::array<float, 4> det_box = {dets(det_idx, 0), dets(det_idx, 1),
                                                 dets(det_idx, 2), dets(det_idx, 3)};
                float iou = compute_iou(track_box, det_box);
                cost_low(ut, d) = 1.0f - iou;
            }
        }
        
        auto result_low = utils::linear_assignment(cost_low, 0.5f);  // Lower threshold for low-conf
        
        for (const auto& match : result_low.matches) {
            int ut = match[0];
            int det_local = match[1];
            
            if (cost_low(ut, det_local) > 0.5f) continue;
            
            int track_idx = unmatched_track_indices[ut];
            int det_idx = low_conf_indices[det_local];
            track_matched[track_idx] = true;
            det_matched_low[det_local] = true;
            
            tracks_[track_idx].update(
                dets(det_idx, 0), dets(det_idx, 1),
                dets(det_idx, 2), dets(det_idx, 3),
                dets(det_idx, 4), static_cast<int>(dets(det_idx, 5)),
                det_idx
            );
        }
    }
    
    // ===== STAGE 3: Try to recover lost tracks with remaining high-conf detections =====
    // Collect remaining unmatched high-conf detections
    std::vector<int> remaining_high_det_indices;
    for (size_t d = 0; d < high_conf_indices.size(); ++d) {
        if (!det_matched_high[d]) {
            remaining_high_det_indices.push_back(static_cast<int>(d));
        }
    }
    
    // Find lost tracks (unmatched and lost for 1+ frames)
    std::vector<int> lost_track_indices;
    for (int t = 0; t < n_tracks; ++t) {
        if (!track_matched[t] && tracks_[t].time_since_update >= 1) {
            lost_track_indices.push_back(t);
        }
    }
    
    // Try to match remaining high-conf detections with lost tracks
    if (!lost_track_indices.empty() && !remaining_high_det_indices.empty()) {
        int n_lost = static_cast<int>(lost_track_indices.size());
        int n_remain = static_cast<int>(remaining_high_det_indices.size());
        Eigen::MatrixXf cost_lost(n_lost, n_remain);
        cost_lost.setConstant(1.0f);
        
        for (int lt = 0; lt < n_lost; ++lt) {
            int track_idx = lost_track_indices[lt];
            auto track_box = tracks_[track_idx].to_xyxy();
            Eigen::Vector2f track_vel(tracks_[track_idx].mean(4), tracks_[track_idx].mean(5));
            float track_speed = track_vel.norm();
            
            for (int rd = 0; rd < n_remain; ++rd) {
                int local_d = remaining_high_det_indices[rd];
                int det_idx = high_conf_indices[local_d];
                std::array<float, 4> det_box = {dets(det_idx, 0), dets(det_idx, 1),
                                                 dets(det_idx, 2), dets(det_idx, 3)};
                
                // Base IoU cost
                float iou = compute_iou(track_box, det_box);
                
                // OC-SORT: Velocity-based matching for lost tracks
                // Check if detection is in the direction of track's velocity
                float det_cx = (det_box[0] + det_box[2]) * 0.5f;
                float det_cy = (det_box[1] + det_box[3]) * 0.5f;
                float track_cx = (track_box[0] + track_box[2]) * 0.5f;
                float track_cy = (track_box[1] + track_box[3]) * 0.5f;
                
                Eigen::Vector2f displacement(det_cx - track_cx, det_cy - track_cy);
                float displacement_norm = displacement.norm();
                
                // Velocity alignment bonus: reward if detection is in velocity direction
                float velocity_bonus = 0.0f;
                if (track_speed > 1.0f && displacement_norm > 1.0f) {
                    float dot_product = track_vel.dot(displacement);
                    float alignment = dot_product / (track_speed * displacement_norm);
                    velocity_bonus = std::max(0.0f, alignment) * 0.3f;  // Up to 0.3 bonus
                }
                
                cost_lost(lt, rd) = 1.0f - (iou + velocity_bonus);
            }
        }
        
        auto result_lost = utils::linear_assignment(cost_lost, 0.7f);  // More permissive with velocity bonus
        
        for (const auto& match : result_lost.matches) {
            int lt = match[0];
            int rd = match[1];
            
            if (cost_lost(lt, rd) > 0.7f) continue;
            
            int track_idx = lost_track_indices[lt];
            int local_d = remaining_high_det_indices[rd];
            int det_idx = high_conf_indices[local_d];
            
            track_matched[track_idx] = true;
            det_matched_high[local_d] = true;
            
            tracks_[track_idx].update(
                dets(det_idx, 0), dets(det_idx, 1),
                dets(det_idx, 2), dets(det_idx, 3),
                dets(det_idx, 4), static_cast<int>(dets(det_idx, 5)),
                det_idx
            );
        }
    }
    
    // Reset consecutive_hits for unmatched tracks
    for (int t = 0; t < n_tracks; ++t) {
        if (!track_matched[t]) {
            tracks_[t].consecutive_hits = 0;
        }
    }
    
    // Create new tracks only from high-confidence unmatched detections
    for (size_t d = 0; d < high_conf_indices.size(); ++d) {
        if (!det_matched_high[d]) {
            int det_idx = high_conf_indices[d];
            tracks_.emplace_back(
                dets(det_idx, 0), dets(det_idx, 1),
                dets(det_idx, 2), dets(det_idx, 3),
                dets(det_idx, 4), static_cast<int>(dets(det_idx, 5)),
                det_idx, ++next_id_
            );
        }
    }
    
    remove_dead_tracks();
    return build_output();
}

} // namespace motcpp::trackers
