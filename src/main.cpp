#include "camera.hpp"
#include "counter.hpp"
#include "pipeline.hpp"

#include <yolos/tasks/detection.hpp>
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

static std::atomic<bool> g_running{true};

static void onSignal(int) { g_running = false; }

static bool isWithinSchedule(int start_hour, int stop_hour) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&t);
    int hour = tm->tm_hour;
    return hour >= start_hour && hour < stop_hour;
}

static void setupLogger(const std::string& log_file,
                        const std::string& level,
                        int flush_seconds) {
    std::filesystem::create_directories(
        std::filesystem::path(log_file).parent_path());

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_file, 10 * 1024 * 1024, 5);  // 10 MB per file, keep 5

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
        cfg["logging"]["log_file"].as<std::string>("logs/detections.log"),
        cfg["logging"]["level"].as<std::string>("info"),
        cfg["logging"]["flush_every_seconds"].as<int>(5));

    const int start_hour = cfg["schedule"]["start_hour"].as<int>(6);
    const int stop_hour  = cfg["schedule"]["stop_hour"].as<int>(18);

    if (!isWithinSchedule(start_hour, stop_hour)) {
        spdlog::info("Outside operating hours ({:02d}:00 – {:02d}:00). Exiting.",
                     start_hour, stop_hour);
        return 0;
    }

    spdlog::info("jetson-yolo-cpp starting");

    // Camera
    Camera camera(
        cfg["camera"]["device"].as<std::string>("/dev/video0"),
        cfg["camera"]["width"].as<int>(1280),
        cfg["camera"]["height"].as<int>(720),
        cfg["camera"]["fps"].as<int>(30));

    // Detector — build TRT engine on first run if not cached
    const std::string engine_path = cfg["model"]["engine_path"].as<std::string>();
    const std::string onnx_path   = cfg["model"]["onnx_path"].as<std::string>();
    const std::string labels_path = cfg["model"]["labels_path"].as<std::string>();

    const std::string model_path = std::filesystem::exists(engine_path)
                                   ? engine_path
                                   : onnx_path;

    spdlog::info("Loading model: {}", model_path);
    yolos::det::YOLODetector detector(model_path, labels_path);

    // Tracker
    motcpp::trackers::ByteTrack tracker(
        cfg["tracker"]["det_thresh"].as<float>(0.3f),
        cfg["tracker"]["max_age"].as<int>(30),
        50,
        cfg["tracker"]["min_hits"].as<int>(3),
        cfg["tracker"]["iou_threshold"].as<float>(0.3f),
        false,
        80,
        "iou",
        false,
        0.1f,
        0.45f,
        0.8f,
        cfg["tracker"]["track_buffer"].as<int>(25),
        cfg["tracker"]["frame_rate"].as<int>(30));

    // Counter
    CountLine line{
        cfg["counting_line"]["x1"].as<float>(),
        cfg["counting_line"]["y1"].as<float>(),
        cfg["counting_line"]["x2"].as<float>(),
        cfg["counting_line"]["y2"].as<float>()};
    LineCounter counter(line);

    // Target classes
    std::vector<int> count_classes;
    if (cfg["count_classes"] && cfg["count_classes"].IsSequence())
        for (const auto& n : cfg["count_classes"])
            count_classes.push_back(n.as<int>());

    Pipeline pipeline(
        camera, detector, tracker, counter, count_classes,
        cfg["model"]["confidence_threshold"].as<float>(0.5f),
        cfg["model"]["iou_threshold"].as<float>(0.45f));

    spdlog::info("Pipeline running. Operating hours {:02d}:00-{:02d}:00", start_hour, stop_hour);

    auto last_log = std::chrono::steady_clock::now();

    while (g_running) {
        if (!isWithinSchedule(start_hour, stop_hour)) {
            spdlog::info("Operating hours ended. Shutting down.");
            break;
        }

        if (!pipeline.tick()) {
            spdlog::error("Camera read failed. Retrying in 2s...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        // Log counts every 30 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 30) {
            pipeline.logCounts();
            last_log = now;
        }
    }

    pipeline.logCounts();
    spdlog::info("jetson-yolo-cpp stopped");
    return 0;
}
