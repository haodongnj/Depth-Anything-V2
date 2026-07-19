#!/bin/bash
# Build TensorRT engine from ONNX on Jetson Nano
# Usage: ./build_engine.sh <onnx_path> [output_engine_path]
#
# Key flags explained:
#   --fp16                   Enable FP16 precision (faster, lower memory)
#   --tacticSources=-CUDNN   Disable cuDNN tactics (saves ~80 MiB GPU RAM on 4GB Jetson)
#   --workspace=512          Give Myelin ForeignNode runner enough scratch space
#                            (default 16 MiB is too small for ViT reshape patterns)

set -e

TRTEXEC=/usr/src/tensorrt/bin/trtexec
ONNX="${1:?Usage: $0 <onnx_path> [output_engine_path]}"
ENGINE="${2:-${ONNX%.onnx}.engine}"

echo "[*] ONNX:  $ONNX"
echo "[*] Engine: $ENGINE"
echo "[*] Device: $(cat /proc/device-tree/model 2>/dev/null || echo unknown)"
echo ""

$TRTEXEC \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --tacticSources=-CUDNN \
    --workspace=512

echo ""
echo "[✓] Engine saved: $ENGINE ($(du -h "$ENGINE" | cut -f1))"
