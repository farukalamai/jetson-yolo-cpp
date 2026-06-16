#include "camera.hpp"

#include <yolos/tasks/detection.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <chrono>
#include <csignal>
#include <atomic>
#include <unordered_set>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

static std::atomic<bool> g_running{true};
static void onSignal(int) { g_running = false; }

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    const std::string config_path = (argc > 1) ? argv[1] : "config.yaml";
    YAML::Node cfg = YAML::LoadFile(config_path);

    // ── Camera ──────────────────────────────────────────────────────────────
    Camera camera(
        cfg["camera"]["device"].as<std::string>("/dev/video0"),
        cfg["camera"]["width"].as<int>(1920),
        cfg["camera"]["height"].as<int>(1080),
        cfg["camera"]["fps"].as<int>(30));

    // ── Detector ────────────────────────────────────────────────────────────
    const std::string engine_path = cfg["model"]["engine_path"].as<std::string>();
    const std::string labels_path = cfg["model"]["labels_path"].as<std::string>();
    const float conf_thresh = cfg["model"]["confidence_threshold"].as<float>(0.5f);
    const float iou_thresh  = cfg["model"]["iou_threshold"].as<float>(0.45f);

    // Load class names for labels on bounding boxes
    std::vector<std::string> class_names;
    {
        std::ifstream f(labels_path);
        std::string line;
        while (std::getline(f, line))
            if (!line.empty()) class_names.push_back(line);
    }

    std::cout << "[test] Loading model: " << engine_path << std::endl;
    yolos::det::YOLODetector detector(engine_path, labels_path);

    // ── Tracker ─────────────────────────────────────────────────────────────
    motcpp::trackers::ByteTrack tracker(
        cfg["tracker"]["det_thresh"].as<float>(0.3f),
        cfg["tracker"]["max_age"].as<int>(30),
        50,
        cfg["tracker"]["min_hits"].as<int>(3),
        cfg["tracker"]["iou_threshold"].as<float>(0.3f),
        false, 80, "iou", false, 0.1f, 0.45f, 0.8f,
        cfg["tracker"]["track_buffer"].as<int>(25),
        cfg["tracker"]["frame_rate"].as<int>(30));

    // ── Snapshot output dir ─────────────────────────────────────────────────
    const std::string snap_dir = "snapshots";
    std::filesystem::create_directories(snap_dir);
    std::cout << "[test] New-track snapshots will be saved to: " << snap_dir << "/" << std::endl;

    std::cout << "[test] Running. Press Ctrl+C to stop.\n" << std::endl;
    std::cout << std::left
              << std::setw(12) << "CamFPS"
              << std::setw(14) << "InferenceFPS"
              << std::setw(12) << "Detections"
              << std::setw(10) << "Tracks"
              << "New IDs"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::unordered_set<int> seen_ids;

    // FPS counters
    using Clock = std::chrono::steady_clock;
    auto fps_window_start = Clock::now();
    int  frame_count  = 0;
    int  report_every = 30; // frames between each terminal print

    while (g_running) {
        // ── Capture ─────────────────────────────────────────────────────────
        auto t_frame_start = Clock::now();
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "[test] Camera read failed" << std::endl;
            break;
        }
        auto t_after_capture = Clock::now();

        // ── Inference ───────────────────────────────────────────────────────
        auto t_infer_start = Clock::now();
        auto detections = detector.detect(frame, conf_thresh, iou_thresh);
        auto t_infer_end = Clock::now();

        // ── Track ───────────────────────────────────────────────────────────
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
        Eigen::MatrixXf tracks = tracker.update(dets, frame);

        // ── New IDs + snapshots ──────────────────────────────────────────────
        std::string new_ids_str;
        bool has_new_id = false;
        for (int i = 0; i < tracks.rows(); ++i) {
            int id = static_cast<int>(tracks(i, 4));
            if (seen_ids.find(id) == seen_ids.end()) {
                seen_ids.insert(id);
                new_ids_str += "ID#" + std::to_string(id) + " ";
                has_new_id = true;
            }
        }

        if (has_new_id) {
            // Draw all current bounding boxes with class, ID and confidence
            cv::Mat annotated = frame.clone();
            for (int i = 0; i < tracks.rows(); ++i) {
                int   x1    = static_cast<int>(tracks(i, 0));
                int   y1    = static_cast<int>(tracks(i, 1));
                int   x2    = static_cast<int>(tracks(i, 2));
                int   y2    = static_cast<int>(tracks(i, 3));
                int   id    = static_cast<int>(tracks(i, 4));
                float conf  = tracks(i, 5);
                int   cls   = static_cast<int>(tracks(i, 6));

                std::string cls_name = (cls >= 0 && cls < static_cast<int>(class_names.size()))
                                       ? class_names[cls] : "obj";

                // Format: "car  ID#3  0.87"
                std::ostringstream label;
                label << cls_name << "  ID#" << id << "  "
                      << std::fixed << std::setprecision(2) << conf;

                // Draw box
                cv::rectangle(annotated, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);

                // Draw filled label background so text is readable
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label.str(),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                int label_y = std::max(y1 - 4, text_size.height + 4);
                cv::rectangle(annotated,
                    {x1, label_y - text_size.height - 4},
                    {x1 + text_size.width, label_y + baseline},
                    {0, 255, 0}, cv::FILLED);
                cv::putText(annotated, label.str(),
                    {x1, label_y - 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 2);
            }

            // Filename: snapshots/frame_<frame_count>_<ids>.png
            std::string ids_clean = new_ids_str;
            std::replace(ids_clean.begin(), ids_clean.end(), ' ', '_');
            std::string filename = snap_dir + "/frame_" +
                                   std::to_string(frame_count) + "_" +
                                   ids_clean + ".png";
            cv::imwrite(filename, annotated);
            std::cout << "  >> New track: " << new_ids_str
                      << "-> saved: " << filename << std::endl;
        }

        // ── FPS calculation ─────────────────────────────────────────────────
        float infer_ms = std::chrono::duration<float, std::milli>(
                             t_infer_end - t_infer_start).count();
        float infer_fps = 1000.0f / infer_ms;

        frame_count++;
        if (frame_count % report_every == 0) {
            auto now = Clock::now();
            float elapsed = std::chrono::duration<float>(
                                now - fps_window_start).count();
            float cam_fps = report_every / elapsed;
            fps_window_start = now;

            std::cout << std::left << std::fixed << std::setprecision(1)
                      << std::setw(12) << cam_fps
                      << std::setw(14) << infer_fps
                      << std::setw(12) << detections.size()
                      << std::setw(10) << tracks.rows()
                      << new_ids_str
                      << std::endl;
        }
    }

    std::cout << "\n[test] Stopped. Total unique IDs seen: " << seen_ids.size() << std::endl;
    return 0;
}
