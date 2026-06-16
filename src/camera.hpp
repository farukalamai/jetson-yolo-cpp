#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

class Camera {
public:
    Camera(const std::string& device, int width, int height, int fps) {
        // GStreamer pipeline: nvvidconv gives HW-accelerated color conversion
        std::string pipeline =
            "v4l2src device=" + device +
            " ! image/jpeg"
            ",width=" + std::to_string(width) +
            ",height=" + std::to_string(height) +
            ",framerate=" + std::to_string(fps) + "/1"
            " ! jpegdec"
            " ! nvvidconv"
            " ! video/x-raw,format=BGRx"
            " ! videoconvert"
            " ! video/x-raw,format=BGR"
            " ! appsink drop=1 sync=false";

        cap_.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap_.isOpened()) {
            // Fallback to plain V4L2 if GStreamer pipeline fails
            cap_.open(device, cv::CAP_V4L2);
            if (!cap_.isOpened())
                throw std::runtime_error("Cannot open camera: " + device);
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            cap_.set(cv::CAP_PROP_FPS, fps);
        }
    }

    ~Camera() { cap_.release(); }

    // Returns false when camera disconnects or stream ends
    bool read(cv::Mat& frame) { return cap_.read(frame); }

    bool isOpened() const { return cap_.isOpened(); }

private:
    cv::VideoCapture cap_;
};
