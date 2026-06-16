# jetson-yolo-cpp

Production-ready real-time object detection and counting on NVIDIA Jetson using YOLO + TensorRT in C++. Supports any YOLO model. Deployable via Docker.

---

## Requirements

- NVIDIA Jetson Orin NX (JetPack 6.2+)
- USB or CSI camera
- Docker with NVIDIA Container Toolkit
- YOLO ONNX model (e.g. `yolo26s.onnx`)

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/faruk/jetson-yolo-cpp.git
cd jetson-yolo-cpp
```

### 2. Download and add your ONNX model

ONNX models are not included in the repo (large binary files). Download from Ultralytics HuggingFace:

```bash
# Download yolo26s (small) — recommended for Jetson
wget -O models/yolo26s.onnx \
  https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26s.onnx
```

Or export from the PyTorch model:

```bash
pip install ultralytics
yolo export model=yolo26s.pt format=onnx imgsz=640
mv yolo26s.onnx models/
```

Available variants: `yolo26n` (nano), `yolo26s` (small), `yolo26m` (medium), `yolo26l` (large)

### 3. Build the TensorRT engine

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

### 4. Configure

Edit `config.yaml` to match your setup:

```yaml
camera:
  device: "/dev/video0"     # your camera device
  width: 1280
  height: 720

model:
  engine_path: "models/engines/yolo26s.engine"

# COCO class IDs to count: 2=car, 3=motorcycle, 5=bus, 7=truck
count_classes: [2, 3, 5, 7]

# Counting line — two points across the road in pixel coordinates
counting_line:
  x1: 0
  y1: 360
  x2: 1280
  y2: 360

schedule:
  start_hour: 6    # 6 AM
  stop_hour: 18    # 6 PM
```

**Tip:** To find the right counting line coordinates, take a snapshot from your camera:

```bash
python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video0')
ret, frame = cap.read()
if ret:
    cv2.imwrite('/tmp/camera_test.jpg', frame)
    print('Saved to /tmp/camera_test.jpg')
cap.release()
"
```

Open `/tmp/camera_test.jpg`, decide where across the road the line should be, and update `y1`/`y2` in `config.yaml`.

### 5. Build the Docker image

```bash
docker compose -f docker/docker-compose.yml build
```

### 6. Run

```bash
docker compose -f docker/docker-compose.yml up
```

Counts are logged every 30 seconds:

```
[info] counts | entered=5 exited=3 net=2
```

Stop with `Ctrl+C` or:

```bash
docker compose -f docker/docker-compose.yml down
```

---

## Build Natively (without Docker)

Faster for development and testing.

```bash
# Install dependencies first (see SETUP_DEPENDENCIES.md)

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
./build/jetson-yolo-cpp config.yaml
```

---

## 24/7 Scheduled Deployment (6 AM – 6 PM)

Install systemd timers that automatically start the container at 6 AM and stop it at 6 PM daily:

```bash
sudo ./scripts/install_service.sh
```

Check status:

```bash
sudo systemctl status vehicle-counter.service
sudo systemctl list-timers
```

Start or stop manually:

```bash
sudo systemctl start vehicle-counter.service
sudo systemctl stop vehicle-counter.service
```

---

## Swap the Model

1. Drop the new `.onnx` into `models/`
2. Build its engine:
    ```bash
    /usr/src/tensorrt/bin/trtexec \
        --onnx=models/yolo26n.onnx \
        --saveEngine=models/engines/yolo26n.engine \
        --fp16 \
        --workspace=4096 \
        --iterations=10 \
        --warmUp=500
    ```
3. Update `config.yaml`:
    ```yaml
    model:
      onnx_path: "models/yolo26n.onnx"
      engine_path: "models/engines/yolo26n.engine"
    ```
4. Restart the container:
    ```bash
    docker compose -f docker/docker-compose.yml restart
    ```

---

## Swap the Camera

Update `config.yaml`:

```yaml
camera:
  device: "/dev/video1"     # different USB camera
```

Or for an RTSP stream, edit `src/camera.hpp` to replace the GStreamer pipeline source.

---

## Project Structure

```
jetson-yolo-cpp/
├── CMakeLists.txt
├── config.yaml                  # all settings — model, camera, line, schedule
├── src/
│   ├── main.cpp                 # entry point, schedule check, main loop
│   ├── camera.hpp               # GStreamer V4L2 capture
│   ├── counter.hpp              # line crossing → enter/exit counts
│   └── pipeline.hpp             # frame → detect → track → count
├── models/
│   ├── coco.names               # class labels
│   └── engines/                 # TRT engines go here (gitignored)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── deploy/
│   ├── vehicle-counter.service  # systemd service
│   ├── start.timer              # 06:00 daily
│   └── stop.timer               # 18:00 daily
└── scripts/
    ├── build_engine.sh          # ONNX → TRT engine
    └── install_service.sh       # install systemd timers
```

---

## Dependencies

See [SETUP_DEPENDENCIES.md](SETUP_DEPENDENCIES.md) for full installation instructions.

| Component | Version |
|---|---|
| JetPack | 6.2.1 |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| OpenCV | 4.8.0 |
| Docker | 29.5+ |

---

## Environment

See [CURRENT_ENVIRONMENT.md](CURRENT_ENVIRONMENT.md) for a full audit of the Jetson environment.
