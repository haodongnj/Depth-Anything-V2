#!/usr/bin/env bash
#
# Convert a Depth Anything V2 ONNX model to a TensorRT engine (trtexec wrapper).
#
# Usage:
#   ./convert_trt.sh model.onnx                          # → model.engine
#   ./convert_trt.sh model.onnx -o out.engine            # → custom output path
#   ./convert_trt.sh model.onnx --jetson                 # → Jetson Nano defaults
#   ./convert_trt.sh model.onnx --dynamic                # for fully dynamic H×W models
#
# Notes:
#   - Engine files are architecture-specific; build on the target machine.
#   - TensorRT 11.x removed the --fp16 flag; mixed precision is handled
#     automatically by the builder.  On older TRT (≤10.x, Jetson Nano),
#     add --fp16 back to the trtexec command below.
#   - --jetson sets 1 GB workspace and conservative batch/shape limits.

set -euo pipefail

# --- defaults ---
WORKSPACE_MB=4096
MIN_BATCH=1
OPT_BATCH=1
MAX_BATCH=4
DYNAMIC=""
OUTPUT=""

# --- helpers ---
die() { echo "[!] $*" >&2; exit 1; }

usage() {
    sed -n '2,/^$/s/^# //p' "$0"
    exit 0
}

# --- parse args ---
ONNX=""
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        --workspace)  WORKSPACE_MB="$2"; shift 2 ;;
        --min-batch)  MIN_BATCH="$2"; shift 2 ;;
        --opt-batch)  OPT_BATCH="$2"; shift 2 ;;
        --max-batch)  MAX_BATCH="$2"; shift 2 ;;
        --dynamic)    DYNAMIC="1"; shift ;;
        --jetson)
            WORKSPACE_MB=1024
            MIN_BATCH=1; OPT_BATCH=1; MAX_BATCH=1
            DYNAMIC="1"
            shift
            ;;
        -*)
            EXTRA+=("$1")
            shift
            ;;
        *)
            [[ -z "$ONNX" ]] || die "Unexpected argument: $1"
            ONNX="$1"
            shift
            ;;
    esac
done

[[ -n "${ONNX:-}" ]] || die "Missing ONNX path.  Usage: $0 <model.onnx> [-o out.engine]"
[[ -f "$ONNX" ]]       || die "ONNX file not found: $ONNX"

# --- output path ---
[[ -n "${OUTPUT:-}" ]] && OUTPUT="$OUTPUT" || OUTPUT="${ONNX%.onnx}.engine"

if [[ -f "$OUTPUT" ]]; then
    read -r -p "[!] '$OUTPUT' already exists. Overwrite? [y/N] " ANS
    [[ "$ANS" =~ ^[Yy]$ ]] || die "Aborted by user."
fi

# --- shape arguments ---
if [[ -n "$DYNAMIC" ]]; then
    # Full dynamic: H/W vary at runtime.  Must call set_input_shape() in your code.
    SHAPE_ARGS="--minShapes=image:${MIN_BATCH}x3x14x14 \
                --optShapes=image:${OPT_BATCH}x3x518x518 \
                --maxShapes=image:${MAX_BATCH}x3x1260x1260"
else
    # Batch-only dynamic (H/W fixed); trtexec handles this naturally.
    SHAPE_ARGS="--minShapes=image:${MIN_BATCH}x3x518x700 \
                --optShapes=image:${OPT_BATCH}x3x518x700 \
                --maxShapes=image:${MAX_BATCH}x3x518x700"
fi

# --- build ---
echo "=============================================="
echo " ONNX:       $ONNX"
echo " Output:     $OUTPUT"
echo " Workspace:  $((WORKSPACE_MB / 1024)) GB"
echo " Batch:      [$MIN_BATCH .. $OPT_BATCH .. $MAX_BATCH]"
echo " Dynamic:    ${DYNAMIC:+yes}${DYNAMIC:-no (HxW fixed)}"
echo "=============================================="

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$OUTPUT" \
    --memPoolSize="workspace:${WORKSPACE_MB}" \
    $SHAPE_ARGS \
    "${EXTRA[@]}" \
    2>&1 | grep -E '^\[|PASSED|FAILED|ERROR'

echo ""
echo "[✓] Engine saved: $OUTPUT"
