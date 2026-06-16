#!/bin/bash
# Converts an ONNX model to a TensorRT engine for this Jetson device.
# Usage: ./scripts/build_engine.sh models/yolo26s.onnx
# Output: models/engines/yolo26s.engine

set -e

ONNX_PATH="${1}"
if [ -z "${ONNX_PATH}" ]; then
    echo "Usage: $0 <path_to_model.onnx>"
    exit 1
fi

if [ ! -f "${ONNX_PATH}" ]; then
    echo "Error: file not found: ${ONNX_PATH}"
    exit 1
fi

BASENAME=$(basename "${ONNX_PATH}" .onnx)
ENGINE_DIR="models/engines"
ENGINE_PATH="${ENGINE_DIR}/${BASENAME}.engine"

mkdir -p "${ENGINE_DIR}"

echo "Building TensorRT engine (FP16)..."
echo "  Input : ${ONNX_PATH}"
echo "  Output: ${ENGINE_PATH}"
echo "  This may take 5–15 minutes on first run."

TRTEXEC=$(command -v trtexec 2>/dev/null || echo "/usr/src/tensorrt/bin/trtexec")
if [ ! -x "${TRTEXEC}" ]; then
    echo "Error: trtexec not found"
    exit 1
fi

"${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --fp16 \
    --memPoolSize=workspace:4096M \
    --iterations=10 \
    --warmUp=500

echo ""
echo "Done. Engine saved to: ${ENGINE_PATH}"
echo "Update engine_path in config.yaml to point to this file."
