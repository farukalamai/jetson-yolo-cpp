# Current Environment — Jetson Orin NX

**Device:** NVIDIA Jetson Orin NX 16GB (Seeed reComputer J401)
**JetPack:** 6.2.1 (L4T R36.4.3, Ubuntu 22.04, aarch64)

---

## Pre-installed Components

| Component | Version | Notes |
|---|---|---|
| JetPack | 6.2.1 | L4T R36.4.3 |
| CUDA | 12.6 | `/usr/local/cuda-12.6` |
| TensorRT | 10.3.0.30 | Dev headers + ONNX parser included |
| OpenCV | 4.8.0 | NVIDIA build — **no CUDA modules** (no `libopencv_cuda*`) |
| GStreamer | 1.20.3 | `nvvidconv` plugin present |
| NVIDIA Container Toolkit | 1.16.2 | `nvidia-container-runtime` at `/usr/bin/` |
| g++ | 11.4.0 | C++17 supported |
| Make | 4.3 | |

## Installed During Setup

| Component | Version | Notes |
|---|---|---|
| Docker Engine | 29.5.3 | From Docker's official repo |
| Docker Compose | v5.1.4 | Plugin-based |
| CMake | 3.22.1 | From Ubuntu 22.04 apt |
| spdlog | 1.9.2 | `libspdlog-dev` |
| yaml-cpp | 0.7.0 | `libyaml-cpp-dev` |

## Hardware

| Item | Details |
|---|---|
| RAM | 16 GB (unified CPU/GPU memory) |
| Swap | 7.6 GB |
| Disk | 468 GB NVMe — 428 GB free |
| CPU Cores | 8 |
| Camera | HTI-UC320 USB webcam → `/dev/video0`, `/dev/video1` |

## Known Limitations

| Issue | Detail |
|---|---|
| OpenCV has no CUDA modules | Pre-built NVIDIA OpenCV 4.8.0 does not include `cudawarping`/`cudaimgproc`. Preprocessing will use custom CUDA kernels instead. |
| `iptable_raw` kernel module missing | Seeed Jetson kernel does not ship this module. Docker is configured with `"iptables": false` in `/etc/docker/daemon.json`. All containers must use `--network=host`. |
| `nvcc` not on PATH by default | Fixed by adding `/usr/local/cuda/bin` to `~/.bashrc`. |
