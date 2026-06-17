#pragma once

#include "camera.hpp"

#include <yolos/tasks/detection.hpp>
#include <yolos/tasks/segmentation.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

enum class Mode { Detection, Segmentation };

class Pipeline {
public:
    // Detection mode
    Pipeline(Camera& camera,
             yolos::det::YOLODetector& detector,
             motcpp::trackers::ByteTrack& tracker,
             float conf_thresh,
             float iou_thresh)
        : camera_(camera)
        , tracker_(tracker)
        , conf_thresh_(conf_thresh)
        , iou_thresh_(iou_thresh)
        , mode_(Mode::Detection)
        , det_detector_(&detector)
        , seg_detector_(nullptr)
    {}

    // Segmentation mode
    Pipeline(Camera& camera,
             yolos::seg::YOLOSegDetector& detector,
             motcpp::trackers::ByteTrack& tracker,
             float conf_thresh,
             float iou_thresh)
        : camera_(camera)
        , tracker_(tracker)
        , conf_thresh_(conf_thresh)
        , iou_thresh_(iou_thresh)
        , mode_(Mode::Segmentation)
        , det_detector_(nullptr)
        , seg_detector_(&detector)
    {}

    Mode mode() const { return mode_; }

    // Runs one frame. Returns false when camera stops delivering frames.
    bool tick() {
        cv::Mat frame;
        if (!camera_.read(frame) || frame.empty())
            return false;

        Eigen::MatrixXf dets = buildDetMatrix(frame);
        Eigen::MatrixXf tracks = tracker_.update(dets, frame);

        spdlog::debug("frame | detections={} tracks={}", dets.rows(), tracks.rows());
        return true;
    }

protected:
    // Shared helper — runs inference and returns motcpp detection matrix [x1,y1,x2,y2,conf,cls]
    Eigen::MatrixXf buildDetMatrix(const cv::Mat& frame) {
        if (mode_ == Mode::Detection) {
            auto detections = det_detector_->detect(frame, conf_thresh_, iou_thresh_);
            Eigen::MatrixXf m(static_cast<int>(detections.size()), 6);
            for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
                const auto& d = detections[i];
                m(i, 0) = static_cast<float>(d.box.x);
                m(i, 1) = static_cast<float>(d.box.y);
                m(i, 2) = static_cast<float>(d.box.x + d.box.width);
                m(i, 3) = static_cast<float>(d.box.y + d.box.height);
                m(i, 4) = d.conf;
                m(i, 5) = static_cast<float>(d.classId);
            }
            return m;
        } else {
            auto results = seg_detector_->segment(frame, conf_thresh_, iou_thresh_);
            Eigen::MatrixXf m(static_cast<int>(results.size()), 6);
            for (int i = 0; i < static_cast<int>(results.size()); ++i) {
                const auto& r = results[i];
                m(i, 0) = static_cast<float>(r.box.x);
                m(i, 1) = static_cast<float>(r.box.y);
                m(i, 2) = static_cast<float>(r.box.x + r.box.width);
                m(i, 3) = static_cast<float>(r.box.y + r.box.height);
                m(i, 4) = r.conf;
                m(i, 5) = static_cast<float>(r.classId);
            }
            return m;
        }
    }

    Camera& camera_;
    motcpp::trackers::ByteTrack& tracker_;
    float conf_thresh_;
    float iou_thresh_;
    Mode mode_;
    yolos::det::YOLODetector*    det_detector_;
    yolos::seg::YOLOSegDetector* seg_detector_;
};
