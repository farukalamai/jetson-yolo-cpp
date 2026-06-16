# Jetson Orin NX — Mandatory Dependencies Setup

**Device:** NVIDIA Jetson Orin NX 16GB  
**JetPack:** 6.2.1 (L4T R36.4.3, Ubuntu 22.04)  
**Already installed:** CUDA 12.6, TensorRT 10.3.0, OpenCV 4.8.0, GStreamer 1.20.3, NVIDIA Container Toolkit 1.16.2, g++ 11.4.0

---

## 1. Fix CUDA PATH

`nvcc` is installed but not on your shell PATH. This is required to compile any CUDA code.

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version
```

---

## 2. Install CMake

Required as the build system. Ubuntu 22.04 provides CMake 3.22.1 which is sufficient (>= 3.20 needed for TensorRT + CUDA).

```bash
sudo apt update
sudo apt install -y cmake
```

Verify:

```bash
cmake --version
# Expected: cmake version 3.22.1
```

---

## 3. Install Docker Engine

Required for containerized 24/7 deployment. The NVIDIA Container Toolkit is already installed — you just need Docker itself.

```bash
# Install prerequisites
sudo apt install -y ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER
```

Apply the group change without logging out (if connected via SSH, run in current shell):

```bash
newgrp docker
```

Configure Docker to use NVIDIA runtime and fix iptables issue (Seeed Jetson kernel does not ship the `iptable_raw` module):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

This creates `/etc/docker/daemon.json`. Now overwrite it to also disable iptables (required on Seeed Jetson kernel):

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia",
    "iptables": false
}
EOF

sudo systemctl restart docker
```

> **Note:** `"iptables": false` is needed because the Seeed Jetson kernel does not include the `iptable_raw` module. This is safe for edge deployment — use `--network=host` on all containers.

Verify:

```bash
docker --version
docker compose version
```

Test that the GPU is accessible inside a container:

```bash
docker run --rm --network=host --gpus all ubuntu bash -c "ls /dev/nvidia*"
# Expected output: /dev/nvidia0  /dev/nvidia-modeset  /dev/nvidiactl
```

---

## 4. Install C++ Libraries

### spdlog (structured async logging)

```bash
sudo apt install -y libspdlog-dev
```

Provides version 1.9.2 — fully supports C++17, production-grade async logging.

### yaml-cpp (configuration file parsing)

```bash
sudo apt install -y libyaml-cpp-dev
```

Provides version 0.7.0 — needed to load model paths, camera sources, and thresholds from config files without recompiling.

### Eigen3 (linear algebra — required by tracker)

```bash
sudo apt install -y libeigen3-dev
```

Required by the motcpp tracker library for matrix operations (Kalman filter, IoU matching).

---

## One-Shot Install (copy-paste)

After fixing CUDA PATH (step 1), run everything else in one go:

```bash
sudo apt update && sudo apt install -y \
    cmake \
    ca-certificates \
    curl \
    libspdlog-dev \
    libyaml-cpp-dev \
    libeigen3-dev

# Docker (separate due to repo setup)
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
sudo nvidia-ctk runtime configure --runtime=docker

sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia",
    "iptables": false
}
EOF

sudo systemctl restart docker
```

---

## Post-Install Verification Checklist

| Check | Command | Expected |
|---|---|---|
| CUDA compiler | `nvcc --version` | CUDA 12.6 |
| CMake | `cmake --version` | 3.22.1+ |
| Docker | `docker --version` | 24.x+ |
| Docker GPU | `docker run --rm --network=host --gpus all ubuntu bash -c "ls /dev/nvidia*"` | `/dev/nvidia0` visible |
| spdlog | `dpkg -s libspdlog-dev \| grep Version` | 1.9.2 |
| yaml-cpp | `dpkg -s libyaml-cpp-dev \| grep Version` | 0.7.0 |
| Eigen3 | `dpkg -s libeigen3-dev \| grep Version` | 3.4.x |
| TensorRT (pre-existing) | `dpkg -s libnvinfer-dev \| grep Version` | 10.3.0.30 |
| Camera (pre-existing) | `ls /dev/video0` | exists |

