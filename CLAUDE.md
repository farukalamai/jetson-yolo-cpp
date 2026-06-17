# jetson-yolo-cpp — Project Context

## What this project is

Real-time object detection and tracking running on NVIDIA Jetson Orin NX 16GB using YOLO + TensorRT in C++. A webcam is connected to the Jetson and the app detects and tracks objects in real time. Deployable via Docker or run natively.

## Hardware

- **Device:** NVIDIA Jetson Orin NX 16GB (Seeed reComputer J401)
- **JetPack:** 6.2.1 (L4T R36.4.3, Ubuntu 22.04, aarch64)
- **Camera:** HTI-UC320 USB webcam at `/dev/video0` — 1920x1080 resolution
- **CUDA:** 12.6
- **TensorRT:** 10.3.0.30

## Key architectural decisions

- **Inference:** YOLOs-CPP-TensorRT library (vendored in `third_party/yolos/`) — provides CUDA preprocessing kernel, TRT engine management, detection API
- **Tracking:** motcpp library (vendored in `third_party/motcpp/`) — ByteTrack algorithm assigns persistent IDs across frames
- **Camera capture:** GStreamer pipeline with `nvvidconv` for hardware-accelerated color conversion. Falls back to plain V4L2 if GStreamer fails
- **No FetchContent:** Both libraries are vendored locally in `third_party/` so users don't need internet at build time
- **Model format:** ONNX → TensorRT engine (FP16). Engine is device-specific and gitignored. Generated once via `scripts/build_engine.sh`
- **No display window:** App runs headless — logs to terminal and file only. No `cv::imshow`

## Docker notes

- Base image should be `nvcr.io/nvidia/l4t-tensorrt` for JetPack 6.x
- `"iptables": false` is required in `/etc/docker/daemon.json` — Seeed Jetson kernel does not ship the `iptable_raw` module
- All containers must use `--network=host`
- GPU verified working: `docker run --rm --network=host --gpus all ubuntu bash -c "ls /dev/nvidia*"`

## Project structure

```
jetson-yolo-cpp/
├── CMakeLists.txt
├── config.yaml                  # camera, model, tracker settings
├── src/
│   ├── main.cpp                 # entry point, logger init, main loop
│   ├── camera.hpp               # GStreamer V4L2 capture
│   ├── pipeline.hpp             # frame → detect → track loop
│   └── test_detection.cpp       # test binary: FPS display + snapshots on new track ID
├── third_party/
│   ├── yolos/include/           # YOLOs-CPP-TensorRT headers + cuda_preprocessing.cu
│   └── motcpp/                  # motcpp headers + source
├── models/
│   ├── coco.names               # 80 COCO class labels
│   └── engines/                 # TRT engines (gitignored, generated locally)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── scripts/
    └── build_engine.sh          # converts .onnx → .engine via trtexec
```

## Two modes

Controlled by `model.mode` in `config.yaml`:

| Mode | Model suffix | Extra output |
|---|---|---|
| `detection` | `yolo26s.onnx` | bounding boxes + class + confidence |
| `segmentation` | `yolo26s-seg.onnx` | bounding boxes + pixel masks |

Tracker (ByteTrack) is mode-agnostic — it receives bounding boxes in both cases.

## Two binaries

| Binary | Purpose |
|---|---|
| `build/jetson-yolo-cpp` | Production — headless detect+track, logs to file |
| `build/test_detection` | Testing — prints FPS table, saves annotated snapshots on new track IDs (with masks in seg mode) |

## Build commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

## Known issues / quirks

- OpenCV 4.8.0 on this Jetson was built without CUDA modules (`libopencv_cuda*` absent) — preprocessing runs via the custom CUDA kernel in YOLOs-CPP-TensorRT instead
- `trtexec` is at `/usr/src/tensorrt/bin/trtexec` (not on PATH by default)
- TensorRT 10 removed `--workspace` flag — use `--memPoolSize=workspace:4096M` instead
- GStreamer warning `Cannot query video position` on camera open is harmless
- Camera runs at ~14 FPS pipeline throughput (bottleneck is 1920x1080 MJPEG decode), model inference alone is ~48 FPS
