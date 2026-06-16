#pragma once

#include "camera.hpp"
#include "counter.hpp"

#include <yolos/tasks/detection.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <vector>

class Pipeline {
public:
    Pipeline(Camera& camera,
             yolos::det::YOLODetector& detector,
             motcpp::trackers::ByteTrack& tracker,
             LineCounter& counter,
             const std::vector<int>& count_classes,
             float conf_thresh,
             float iou_thresh)
        : camera_(camera)
        , detector_(detector)
        , tracker_(tracker)
        , counter_(counter)
        , count_classes_(count_classes.begin(), count_classes.end())
        , conf_thresh_(conf_thresh)
        , iou_thresh_(iou_thresh)
    {}

    // Runs one frame. Returns false when camera stops delivering frames.
    bool tick() {
        cv::Mat frame;
        if (!camera_.read(frame) || frame.empty())
            return false;

        // Detect
        auto detections = detector_.detect(frame, conf_thresh_, iou_thresh_);

        // Filter to target classes only (empty set = accept all)
        if (!count_classes_.empty()) {
            detections.erase(
                std::remove_if(detections.begin(), detections.end(),
                    [&](const yolos::det::Detection& d) {
                        return count_classes_.find(d.classId) == count_classes_.end();
                    }),
                detections.end());
        }

        // Build motcpp detection matrix [x1,y1,x2,y2,conf,cls]
        Eigen::MatrixXf dets(static_cast<int>(detections.size()), 6);
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            const auto& d = detections[i];
            dets(i, 0) = static_cast<float>(d.box.x);
            dets(i, 1) = static_cast<float>(d.box.y);
            dets(i, 2) = static_cast<float>(d.box.x + d.box.width);
            dets(i, 3) = static_cast<float>(d.box.y + d.box.height);
            dets(i, 4) = d.conf;
            dets(i, 5) = static_cast<float>(d.classId);
        }

        // Track
        Eigen::MatrixXf tracks = tracker_.update(dets, frame);

        // Count
        counter_.update(tracks);

        const auto& c = counter_.counts();
        spdlog::debug("frame | detections={} tracks={} entered={} exited={}",
                      detections.size(), tracks.rows(), c.entered, c.exited);

        return true;
    }

    // Log current counts at info level — call periodically from main loop
    void logCounts() const {
        const auto& c = counter_.counts();
        spdlog::info("counts | entered={} exited={} net={}", c.entered, c.exited, c.entered - c.exited);
    }

private:
    Camera& camera_;
    yolos::det::YOLODetector& detector_;
    motcpp::trackers::ByteTrack& tracker_;
    LineCounter& counter_;
    std::unordered_set<int> count_classes_;
    float conf_thresh_;
    float iou_thresh_;
};
