# jetson-yolo-cpp

Real-time object detection, segmentation and tracking on NVIDIA Jetson using YOLO + TensorRT in C++. Supports detection and instance segmentation modes. Drop in any YOLO ONNX model, plug in a camera, and run — natively or via Docker.

---

## Requirements

- NVIDIA Jetson Orin NX (JetPack 6.2+)
- USB or CSI camera
- YOLO ONNX model

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/faruk/jetson-yolo-cpp.git
cd jetson-yolo-cpp
```

### 2. Install dependencies

```bash
sudo apt install -y cmake libeigen3-dev libspdlog-dev libyaml-cpp-dev
```

See [SETUP_DEPENDENCIES.md](SETUP_DEPENDENCIES.md) for the full setup guide including Docker.

### 3. Download a YOLO model

Export an ONNX model using the Ultralytics package:

```bash
pip install ultralytics

# Detection model
yolo export model=yolo26s.pt format=onnx imgsz=640
mv yolo26s.onnx models/

# Segmentation model (adds pixel masks)
yolo export model=yolo26s-seg.pt format=onnx imgsz=640
mv yolo26s-seg.onnx models/
```

Available variants: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l` — each has a `-seg` variant for segmentation.

### 4. Build the TensorRT engine

Run once per model per device. Takes 5–15 minutes.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolo26s.onnx \
    --saveEngine=models/engines/yolo26s.engine \
    --fp16 \
    --memPoolSize=workspace:4096M \
    --iterations=10 \
    --warmUp=500
```

Or use the helper script:

```bash
./scripts/build_engine.sh models/yolo26s.onnx
```

### 5. Configure

Edit `config.yaml` to match your camera and model:

```yaml
camera:
  device: "/dev/video0"
  width: 1920
  height: 1080

model:
  mode: "detection"          # "detection" or "segmentation"
  engine_path: "models/engines/yolo26s.engine"
```

Switch to segmentation by changing mode and pointing to a `-seg` engine:

```yaml
model:
  mode: "segmentation"
  engine_path: "models/engines/yolo26s-seg.engine"
```

### 6. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

---

## Run

### Test first (recommended)

Run the test tool to verify camera, model and tracker are working before going to production:

```bash
./build/test_detection config.yaml
```

Terminal output:

```
CamFPS      InferenceFPS  Detections  Tracks    New IDs
----------------------------------------------------------------------
  >> New track: ID#1 ID#2  -> saved: snapshots/frame_45_ID#1_ID#2_.png
28.4        47.8          2           2
  >> New track: ID#4       -> saved: snapshots/frame_120_ID#4_.png
27.9        47.6          3           3
```

- **CamFPS** — full pipeline speed (camera + inference + tracking)
- **InferenceFPS** — YOLO model speed only
- Every new track ID triggers an annotated PNG snapshot saved to `snapshots/`
- In segmentation mode, snapshots include pixel masks drawn over detected objects

Stop with `Ctrl+C`.

### Production (native)

```bash
./build/jetson-yolo-cpp config.yaml
```

Runs headless — detects and tracks continuously, logs to terminal and `logs/app.log`. Stop with `Ctrl+C`.

### Production (Docker)

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## Acknowledgements

- **[YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT)** — C++ TensorRT inference library for YOLO v5–v26
- **[motcpp](https://github.com/Geekgineer/motcpp)** — C++ multi-object tracking (ByteTrack and others)
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — YOLO model family
- **[spdlog](https://github.com/gabime/spdlog)** — fast C++ logging
- **[yaml-cpp](https://github.com/jbeder/yaml-cpp)** — YAML configuration parsing
- **[OpenCV](https://opencv.org)** — camera capture and image I/O
- **[Eigen](https://eigen.tuxfamily.org)** — linear algebra for the tracker
