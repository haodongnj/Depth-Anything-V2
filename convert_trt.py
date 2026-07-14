#!/usr/bin/env python3
"""
Convert a Depth Anything V2 ONNX model to a TensorRT engine.

Usage:
  python convert_trt.py depth_anything_v2_vits_518x700.onnx -o output.engine
  python convert_trt.py model.onnx -o model.engine --fp16
  python convert_trt.py model.onnx -o model.engine --min-batch 1 --max-batch 4

Platform notes:
  - Engine files are architecture-specific — build on the target machine.
  - For Jetson Nano, use --workspace 1 to stay within the 4 GB RAM budget.
"""

import argparse
import sys
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX → TensorRT engine converter")
    p.add_argument("onnx", type=Path, help="Path to the ONNX model")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output engine path (default: <onnx_stem>.engine)")
    p.add_argument("--workspace", type=int, default=4,
                   help="Max workspace memory in GB (default: 4; use 1 on Jetson Nano)")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1,
                   help="Optimal batch size (default: same as --min-batch)")
    p.add_argument("--max-batch", type=int, default=4)
    p.add_argument("--fp16", action="store_true",
                   help="Enable FP16 precision (halves memory, faster on modern GPUs)")
    p.add_argument("--tf32", action="store_true",
                   help="Enable TF32 (Ampere+ GPUs)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    onnx_path = args.onnx
    if not onnx_path.exists():
        sys.exit(f"[!] ONNX file not found: {onnx_path}")

    engine_path = args.output or onnx_path.with_suffix(".engine")
    if engine_path.exists():
        resp = input(f"[!] '{engine_path}' already exists. Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # ---- Parse ONNX ----
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    print(f"[*] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        data = f.read()
        if not parser.parse(data):
            print("[!] ONNX parse FAILED:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            sys.exit(1)
    print("[✓] ONNX parsed successfully")

    # Print model I/O info
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    Input:  {inp.name:20s}  shape={inp.shape}  dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    Output: {out.name:20s}  shape={out.shape}  dtype={out.dtype}")

    # ---- Build config ----
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 30)

    # Optimization profile (dynamic batch)
    input_name = network.get_input(0).name
    inp_shape = network.get_input(0).shape  # e.g. (-1, 3, 518, 700)

    # Start from parsed shape, replace batch dim
    min_shape = [args.min_batch] + list(inp_shape[1:])
    opt_shape = [args.opt_batch or args.min_batch] + list(inp_shape[1:])
    max_shape = [args.max_batch] + list(inp_shape[1:])

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    config.add_optimization_profile(profile)
    print(f"[*] Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")

    # Precision flags
    if args.fp16:
        try:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[*] FP16 enabled")
        except AttributeError:
            print("[!] FP16 not available in this TensorRT build — using FP32")

    if args.tf32:
        try:
            config.clear_flag(trt.BuilderFlag.TF32)  # TF32 is on by default, re-enable explicitly
            config.set_flag(trt.BuilderFlag.TF32)
            print("[*] TF32 enabled")
        except AttributeError:
            pass

    # ---- Build ----
    print(f"[*] Building TensorRT engine (workspace={args.workspace} GB)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit("[!] Build failed — engine is None")

    buf = bytes(serialized)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(buf)

    print(f"[✓] Engine saved: {engine_path}  ({len(buf) / 1024**2:.1f} MB)")
    print(f"[✓] Done.")


if __name__ == "__main__":
    main()
