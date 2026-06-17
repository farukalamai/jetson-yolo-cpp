#include "camera.hpp"

#include <yolos/tasks/detection.hpp>
#include <yolos/tasks/segmentation.hpp>
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

// Draws box + label (class, ID, conf) onto annotated frame.
// For segmentation mode, also blends the mask.
static void drawTrack(cv::Mat& img,
                      int x1, int y1, int x2, int y2,
                      int id, float conf, int cls,
                      const std::vector<std::string>& class_names,
                      const cv::Mat* mask = nullptr) {
    // Blend mask if present
    if (mask && !mask->empty()) {
        cv::Mat colored(img.size(), CV_8UC3, cv::Scalar(0, 200, 100));
        colored.copyTo(img, *mask);
    }

    cv::rectangle(img, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);

    std::string cls_name = (cls >= 0 && cls < static_cast<int>(class_names.size()))
                           ? class_names[cls] : "obj";
    std::ostringstream label;
    label << cls_name << "  ID#" << id << "  "
          << std::fixed << std::setprecision(2) << conf;

    int baseline = 0;
    cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    int ly = std::max(y1 - 4, ts.height + 4);
    cv::rectangle(img, {x1, ly - ts.height - 4}, {x1 + ts.width, ly + baseline},
                  {0, 255, 0}, cv::FILLED);
    cv::putText(img, label.str(), {x1, ly - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 0}, 2);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    const std::string config_path = (argc > 1) ? argv[1] : "config.yaml";
    YAML::Node cfg = YAML::LoadFile(config_path);

    const std::string mode_str    = cfg["model"]["mode"].as<std::string>("detection");
    const std::string engine_path = cfg["model"]["engine_path"].as<std::string>();
    const std::string labels_path = cfg["model"]["labels_path"].as<std::string>();
    const float conf_thresh       = cfg["model"]["confidence_threshold"].as<float>(0.5f);
    const float iou_thresh        = cfg["model"]["iou_threshold"].as<float>(0.45f);
    const bool  is_seg            = (mode_str == "segmentation");

    // Load class names for labels
    std::vector<std::string> class_names;
    {
        std::ifstream f(labels_path);
        std::string line;
        while (std::getline(f, line))
            if (!line.empty()) class_names.push_back(line);
    }

    // ── Camera ──────────────────────────────────────────────────────────────
    Camera camera(
        cfg["camera"]["device"].as<std::string>("/dev/video0"),
        cfg["camera"]["width"].as<int>(1920),
        cfg["camera"]["height"].as<int>(1080),
        cfg["camera"]["fps"].as<int>(30));

    // ── Detector ────────────────────────────────────────────────────────────
    std::cout << "[test] Mode: " << mode_str << std::endl;
    std::cout << "[test] Loading model: " << engine_path << std::endl;

    std::unique_ptr<yolos::det::YOLODetector>    det_detector;
    std::unique_ptr<yolos::seg::YOLOSegDetector> seg_detector;

    if (is_seg)
        seg_detector = std::make_unique<yolos::seg::YOLOSegDetector>(engine_path, labels_path);
    else
        det_detector = std::make_unique<yolos::det::YOLODetector>(engine_path, labels_path);

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

    // ── Snapshot dir ─────────────────────────────────────────────────────────
    const std::string snap_dir = "snapshots";
    std::filesystem::create_directories(snap_dir);
    std::cout << "[test] Snapshots -> " << snap_dir << "/" << std::endl;
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

    using Clock = std::chrono::steady_clock;
    auto fps_window_start = Clock::now();
    int  frame_count      = 0;
    int  report_every     = 30;

    while (g_running) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "[test] Camera read failed" << std::endl;
            break;
        }

        // ── Inference ───────────────────────────────────────────────────────
        auto t0 = Clock::now();

        Eigen::MatrixXf dets;
        // Also keep segmentation results around so we can draw masks on snapshots
        std::vector<yolos::seg::Segmentation> seg_results;

        if (is_seg) {
            seg_results = seg_detector->segment(frame, conf_thresh, iou_thresh);
            dets.resize(static_cast<int>(seg_results.size()), 6);
            for (int i = 0; i < static_cast<int>(seg_results.size()); ++i) {
                const auto& r = seg_results[i];
                dets(i, 0) = static_cast<float>(r.box.x);
                dets(i, 1) = static_cast<float>(r.box.y);
                dets(i, 2) = static_cast<float>(r.box.x + r.box.width);
                dets(i, 3) = static_cast<float>(r.box.y + r.box.height);
                dets(i, 4) = r.conf;
                dets(i, 5) = static_cast<float>(r.classId);
            }
        } else {
            auto det_results = det_detector->detect(frame, conf_thresh, iou_thresh);
            dets.resize(static_cast<int>(det_results.size()), 6);
            for (int i = 0; i < static_cast<int>(det_results.size()); ++i) {
                const auto& d = det_results[i];
                dets(i, 0) = static_cast<float>(d.box.x);
                dets(i, 1) = static_cast<float>(d.box.y);
                dets(i, 2) = static_cast<float>(d.box.x + d.box.width);
                dets(i, 3) = static_cast<float>(d.box.y + d.box.height);
                dets(i, 4) = d.conf;
                dets(i, 5) = static_cast<float>(d.classId);
            }
        }

        auto t1 = Clock::now();

        // ── Track ───────────────────────────────────────────────────────────
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
            cv::Mat annotated = frame.clone();

            // Build a map from bbox to mask for segmentation mode
            // (tracks bbox may differ slightly from raw detection bbox after NMS — match by class+proximity)
            for (int i = 0; i < tracks.rows(); ++i) {
                int   x1  = static_cast<int>(tracks(i, 0));
                int   y1  = static_cast<int>(tracks(i, 1));
                int   x2  = static_cast<int>(tracks(i, 2));
                int   y2  = static_cast<int>(tracks(i, 3));
                int   id  = static_cast<int>(tracks(i, 4));
                float conf = tracks(i, 5);
                int   cls  = static_cast<int>(tracks(i, 6));

                const cv::Mat* mask_ptr = nullptr;

                if (is_seg) {
                    // Find closest segmentation result by box overlap
                    float best_iou = 0.f;
                    for (const auto& sr : seg_results) {
                        float ix1 = std::max(x1, sr.box.x);
                        float iy1 = std::max(y1, sr.box.y);
                        float ix2 = std::min(x2, sr.box.x + sr.box.width);
                        float iy2 = std::min(y2, sr.box.y + sr.box.height);
                        float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
                        float area1 = static_cast<float>((x2-x1)*(y2-y1));
                        float area2 = static_cast<float>(sr.box.width * sr.box.height);
                        float iou   = inter / (area1 + area2 - inter + 1e-6f);
                        if (iou > best_iou) { best_iou = iou; mask_ptr = &sr.mask; }
                    }
                }

                drawTrack(annotated, x1, y1, x2, y2, id, conf, cls, class_names, mask_ptr);
            }

            std::string ids_clean = new_ids_str;
            std::replace(ids_clean.begin(), ids_clean.end(), ' ', '_');
            std::string filename = snap_dir + "/frame_" +
                                   std::to_string(frame_count) + "_" +
                                   ids_clean + ".png";
            cv::imwrite(filename, annotated);
            std::cout << "  >> New track: " << new_ids_str
                      << "-> saved: " << filename << std::endl;
        }

        // ── FPS ─────────────────────────────────────────────────────────────
        float infer_ms  = std::chrono::duration<float, std::milli>(t1 - t0).count();
        float infer_fps = 1000.f / infer_ms;

        frame_count++;
        if (frame_count % report_every == 0) {
            auto now = Clock::now();
            float elapsed = std::chrono::duration<float>(now - fps_window_start).count();
            float cam_fps = report_every / elapsed;
            fps_window_start = now;

            std::cout << std::left << std::fixed << std::setprecision(1)
                      << std::setw(12) << cam_fps
                      << std::setw(14) << infer_fps
                      << std::setw(12) << dets.rows()
                      << std::setw(10) << tracks.rows()
                      << std::endl;
        }
    }

    std::cout << "\n[test] Stopped. Total unique IDs seen: " << seen_ids.size() << std::endl;
    return 0;
}
