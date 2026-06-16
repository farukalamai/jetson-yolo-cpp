cmake_minimum_required(VERSION 3.22)
project(jetson-yolo-cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# ─── Dependencies ────────────────────────────────────────────────────────────

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT — installed via JetPack
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include)

find_library(NVINFER_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

find_library(NVONNXPARSER_LIB nvonnxparser
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found. Ensure JetPack is installed.")
endif()

# ─── YOLOs-CPP-TensorRT (local third_party) ──────────────────────────────────

set(YOLOS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/yolos/include)

add_library(cuda_preprocessing STATIC
    ${YOLOS_INCLUDE_DIR}/yolos/core/cuda_preprocessing.cu
)
target_include_directories(cuda_preprocessing PRIVATE
    ${YOLOS_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(cuda_preprocessing PRIVATE CUDA::cudart)
set_target_properties(cuda_preprocessing PROPERTIES
    CUDA_ARCHITECTURES "87"   # Jetson Orin = SM 8.7
)

# ─── motcpp (local third_party) ───────────────────────────────────────────────

set(MOTCPP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/motcpp/include)
set(MOTCPP_SRC_DIR     ${CMAKE_SOURCE_DIR}/third_party/motcpp/src)

file(GLOB_RECURSE MOTCPP_SOURCES
    ${MOTCPP_SRC_DIR}/tracker.cpp
    ${MOTCPP_SRC_DIR}/config.cpp
    ${MOTCPP_SRC_DIR}/motion/*.cpp
    ${MOTCPP_SRC_DIR}/trackers/bytetrack.cpp
    ${MOTCPP_SRC_DIR}/trackers/sort.cpp
    ${MOTCPP_SRC_DIR}/utils/*.cpp
)

add_library(motcpp STATIC ${MOTCPP_SOURCES})
target_include_directories(motcpp PUBLIC ${MOTCPP_INCLUDE_DIR})
target_link_libraries(motcpp PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)
add_library(motcpp::motcpp ALIAS motcpp)

# ─── Helper macro to avoid repeating link/include for each binary ─────────────

macro(add_yolo_binary name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${YOLOS_INCLUDE_DIR}
        ${TENSORRT_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE
        ${OpenCV_LIBS}
        ${NVINFER_LIB}
        ${NVONNXPARSER_LIB}
        CUDA::cudart
        cuda_preprocessing
        motcpp::motcpp
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp
        Threads::Threads
    )
endmacro()

# ─── Binaries ────────────────────────────────────────────────────────────────

add_yolo_binary(jetson-yolo-cpp     src/main.cpp)
add_yolo_binary(test_detection      src/test_detection.cpp)
