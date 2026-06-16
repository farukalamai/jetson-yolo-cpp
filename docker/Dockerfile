# Base: NVIDIA L4T with CUDA 12 + TensorRT 10 for JetPack 6.x
FROM nvcr.io/nvidia/l4t-tensorrt:r36.4.0-devel

ENV DEBIAN_FRONTEND=noninteractive

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# ─── Build application ────────────────────────────────────────────────────────
WORKDIR /app
COPY CMakeLists.txt .
COPY src/ src/

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel $(nproc)

# ─── Runtime image ────────────────────────────────────────────────────────────
# models/ config.yaml and logs/ are mounted at runtime — not baked in.
CMD ["./build/jetson-yolo-cpp", "config.yaml"]
