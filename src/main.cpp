#include "camera.hpp"
#include "pipeline.hpp"

#include <yolos/tasks/detection.hpp>
#include <yolos/tasks/segmentation.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <atomic>
#include <thread>
#include <memory>

static std::atomic<bool> g_running{true};
static void onSignal(int) { g_running = false; }

static void setupLogger(const std::string& log_file,
                        const std::string& level,
                        int flush_seconds) {
    std::filesystem::create_directories(
        std::filesystem::path(log_file).parent_path());

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink    = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file, 10 * 1024 * 1024, 5);

    auto logger = std::make_shared<spdlog::logger>(
        "app", spdlog::sinks_init_list{console_sink, file_sink});

    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(flush_seconds));

    if      (level == "debug") spdlog::set_level(spdlog::level::debug);
    else if (level == "warn")  spdlog::set_level(spdlog::level::warn);
    else if (level == "error") spdlog::set_level(spdlog::level::err);
    else                        spdlog::set_level(spdlog::level::info);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    const std::string config_path = (argc > 1) ? argv[1] : "config.yaml";

    YAML::Node cfg;
    try {
        cfg = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        spdlog::critical("Failed to load config: {}", e.what());
        return 1;
    }

    setupLogger(
        cfg["logging"]["log_file"].as<std::string>("logs/app.log"),
        cfg["logging"]["level"].as<std::string>("info"),
        cfg["logging"]["flush_every_seconds"].as<int>(5));

    const std::string mode_str    = cfg["model"]["mode"].as<std::string>("detection");
    const std::string engine_path = cfg["model"]["engine_path"].as<std::string>();
    const std::string labels_path = cfg["model"]["labels_path"].as<std::string>();
    const float conf_thresh       = cfg["model"]["confidence_threshold"].as<float>(0.5f);
    const float iou_thresh        = cfg["model"]["iou_threshold"].as<float>(0.45f);

    spdlog::info("jetson-yolo-cpp starting | mode={} model={}", mode_str, engine_path);

    Camera camera(
        cfg["camera"]["device"].as<std::string>("/dev/video0"),
        cfg["camera"]["width"].as<int>(1920),
        cfg["camera"]["height"].as<int>(1080),
        cfg["camera"]["fps"].as<int>(30));

    motcpp::trackers::ByteTrack tracker(
        cfg["tracker"]["det_thresh"].as<float>(0.3f),
        cfg["tracker"]["max_age"].as<int>(30),
        50,
        cfg["tracker"]["min_hits"].as<int>(3),
        cfg["tracker"]["iou_threshold"].as<float>(0.3f),
        false, 80, "iou", false, 0.1f, 0.45f, 0.8f,
        cfg["tracker"]["track_buffer"].as<int>(25),
        cfg["tracker"]["frame_rate"].as<int>(30));

    std::unique_ptr<Pipeline> pipeline;

    if (mode_str == "segmentation") {
        auto detector = std::make_unique<yolos::seg::YOLOSegDetector>(engine_path, labels_path);
        pipeline = std::make_unique<Pipeline>(camera, *detector, tracker, conf_thresh, iou_thresh);
        // detector must outlive pipeline — transfer ownership via shared storage below
        // Use static to keep it alive for the duration of main
        static std::unique_ptr<yolos::seg::YOLOSegDetector> seg_owner = std::move(detector);
        pipeline = std::make_unique<Pipeline>(camera, *seg_owner, tracker, conf_thresh, iou_thresh);
    } else {
        static std::unique_ptr<yolos::det::YOLODetector> det_owner =
            std::make_unique<yolos::det::YOLODetector>(engine_path, labels_path);
        pipeline = std::make_unique<Pipeline>(camera, *det_owner, tracker, conf_thresh, iou_thresh);
    }

    spdlog::info("Pipeline running. Press Ctrl+C to stop.");

    while (g_running) {
        if (!pipeline->tick()) {
            spdlog::error("Camera read failed. Retrying in 2s...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    spdlog::info("jetson-yolo-cpp stopped");
    return 0;
}
