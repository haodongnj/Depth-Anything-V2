#!/usr/bin/env python3
"""
Unified Depth Anything V2 verification — auto-detects mode from provided options.

Modes (auto-detected):
  1. ONNX inspection   — --onnx --inspect | --onnx --check-integrity  (no --image needed)
  2. Single backend    — exactly one of --torch, --onnx, --trt: run inference + save depths
  3. Multi-backend     — 2+ backends: compare all pairs with full metrics

Usage:
  # Inspection
  python verify.py --onnx model.onnx --inspect
  python verify.py --onnx model.onnx --check-integrity

  # Standalone inference (single backend)
  python verify.py --onnx model.onnx --image test.jpg
  python verify.py --trt model.engine --image test.jpg
  python verify.py --torch vitl --checkpoint ckpt.pth --image test.jpg

  # Comparison (2+ backends)
  python verify.py --torch vitl --checkpoint ckpt.pth --onnx model.onnx --image test.jpg
  python verify.py --onnx model.onnx --trt model.engine --image test.jpg
  python verify.py --torch vits --checkpoint ckpt.pth \\
      --onnx model.onnx --trt model.engine --image test.jpg

  # Batch processing
  python verify.py --torch vits --checkpoint ckpt.pth --onnx model.onnx \\
      --image assets/examples/ --save-diff
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from verify_common import (
    ONNXSession, TRTInference, check_integrity, compute_metrics, gather_images,
    get_onnx_input_shape, inspect_onnx, load_torch_model, infer_torch,
    preprocess_image, preprocess_image_exact, print_metrics, save_depth_vis,
    save_diff_heatmap, save_report, summary_metrics, verdict,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Depth Anything V2 verification — auto-detects mode from provided options",
    )

    # Backends
    p.add_argument("--torch", choices=["vits", "vitb", "vitl", "vitg"],
                   help="PyTorch encoder variant")
    p.add_argument("--checkpoint", help="Path to PyTorch .pth checkpoint (required with --torch)")
    p.add_argument("--onnx", help="Path to ONNX model file")
    p.add_argument("--trt", help="Path to TensorRT engine file")

    # Data
    p.add_argument("--image", default=None,
                   help="Path to test image or directory")
    p.add_argument("--ext", default="jpg", help="Image extension filter for directories")

    # PyTorch
    p.add_argument("--device", default="cpu", help="PyTorch device")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Output directory (auto-generated from active backends if omitted)")
    p.add_argument("--save-diff", action="store_true",
                   help="Save difference heatmap images (comparison mode only)")
    p.add_argument("--report", default=None,
                   help="Save report (.txt for text summary, .csv for per-image table)")

    # ONNX inspection (no --image needed)
    p.add_argument("--inspect", action="store_true",
                   help="Print ONNX model structure and exit")
    p.add_argument("--check-integrity", action="store_true",
                   help="Check ONNX model for common issues and exit")

    # TRT
    p.add_argument("--verbose", action="store_true", help="Verbose TensorRT logging")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _get_trt_input_shape(engine_path: str) -> Tuple[int, int]:
    """Read (H, W) from a TensorRT engine file."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    shape = engine.get_tensor_shape("image")
    return shape[2], shape[3]


def _determine_resolution(args: argparse.Namespace) -> Optional[Tuple[int, int]]:
    """Return the comparison resolution (H, W), or None for standalone PyTorch mode.

    Priority: ONNX input shape > TRT input shape > None (aspect-ratio-preserving).
    """
    if args.onnx:
        return get_onnx_input_shape(args.onnx)
    if args.trt:
        return _get_trt_input_shape(args.trt)
    return None


# ---------------------------------------------------------------------------
# Output dir auto-naming
# ---------------------------------------------------------------------------

def _auto_output_dir(args: argparse.Namespace) -> str:
    """Generate output directory name from active backends."""
    parts = []
    if args.torch:
        parts.append(f"torch_{args.torch}")
    if args.onnx:
        parts.append("onnx")
    if args.trt:
        parts.append("trt")
    if not parts:
        parts.append("verify")
    return "_".join(parts) + "_results"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _active_backends(args: argparse.Namespace) -> List[str]:
    """Return list of active backend names: ['torch', 'onnx', 'trt']."""
    return [b for b in ["torch", "onnx", "trt"] if getattr(args, b)]


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and exit with a message on error."""
    # ONNX inspection modes
    if args.inspect or args.check_integrity:
        if not args.onnx:
            sys.exit("[!] --onnx is required with --inspect / --check-integrity")
        return  # these modes don't need --image or backends

    # Inference / comparison modes need --image
    if not args.image:
        sys.exit("[!] --image is required for inference/comparison mode")

    backends = _active_backends(args)
    if not backends:
        sys.exit("[!] At least one of --torch, --onnx, --trt is required")

    if args.torch and not args.checkpoint:
        sys.exit("[!] --checkpoint is required with --torch")


# ---------------------------------------------------------------------------
# Standalone mode: single-backend inference
# ---------------------------------------------------------------------------

def _run_standalone(
    args: argparse.Namespace,
    backend: str,
    resolution: Optional[Tuple[int, int]],
    output_dir: Path,
) -> None:
    """Run a single backend on all images, save depth visualizations."""

    images = gather_images(args.image, args.ext)
    print(f"[*] Found {len(images)} images")
    print(f"[*] Mode: standalone ({backend})")
    if resolution:
        print(f"[*] Resolution: {resolution[0]}x{resolution[1]} (exact-resize)")
    else:
        print(f"[*] Resolution: aspect-ratio-preserving (input_size=518)")
    print()

    # --- Load backend once ---
    if backend == "torch":
        model = load_torch_model(args.torch, args.checkpoint, args.device)
    elif backend == "onnx":
        session = ONNXSession(args.onnx)
    elif backend == "trt":
        runner = TRTInference(args.trt, verbose=args.verbose)
        resolution = runner.input_shape()  # (H, W)
        print(f"[*] TRT input shape: {resolution[0]}x{resolution[1]}\n")

    for idx, img_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {img_path.name}")

        raw = cv2.imread(str(img_path))
        if raw is None:
            print("  [!] Cannot read, skipping.")
            continue

        orig_h, orig_w = raw.shape[:2]

        if backend == "torch" and resolution is None:
            # PyTorch native: aspect-ratio-preserving
            preproc, _, (h, w) = preprocess_image(raw, input_size=518)
            depth = infer_torch(model, preproc, (h, w))
        elif backend == "torch":
            preproc, _ = preprocess_image_exact(raw, *resolution)
            depth = infer_torch(model, preproc, resolution)
        elif backend == "onnx":
            preproc, _ = preprocess_image_exact(raw, *resolution)
            depth = session.run(preproc)
        elif backend == "trt":
            preproc, _ = preprocess_image_exact(raw, *resolution)
            depth = runner.infer(preproc)

        print(f"  output: {depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}]")

        save_depth_vis(depth, str(output_dir / f"{img_path.stem}_{backend}.png"), (orig_w, orig_h))

        del raw, preproc, depth

    print(f"\n[*] Depth maps saved to: {output_dir}")
    print("[*] Done.")


# ---------------------------------------------------------------------------
# Comparison mode: 2+ backends
# ---------------------------------------------------------------------------

def _run_comparison(
    args: argparse.Namespace,
    backends: List[str],
    resolution: Tuple[int, int],
    output_dir: Path,
) -> None:
    """Run 2+ backends and compare all pairs."""

    images = gather_images(args.image, args.ext)
    print(f"[*] Found {len(images)} images")
    print(f"[*] Mode: comparison ({' + '.join(backends)})")
    print(f"[*] Resolution: {resolution[0]}x{resolution[1]} (exact-resize)\n")

    # --- Load backends once ---
    torch_model = None
    onnx_session = None
    trt_runner = None

    if "torch" in backends:
        print(f"[*] Loading PyTorch model ({args.torch}) ...")
        torch_model = load_torch_model(args.torch, args.checkpoint, args.device)
    if "onnx" in backends:
        onnx_session = ONNXSession(args.onnx)
    if "trt" in backends:
        print("[*] Loading TensorRT engine ...")
        trt_runner = TRTInference(args.trt, verbose=args.verbose)

    # --- Determine all pairs to compare ---
    pair_defs = []
    if "torch" in backends and "onnx" in backends:
        pair_defs.append(("PyTorch vs ONNX", "torch", "onnx"))
    if "torch" in backends and "trt" in backends:
        pair_defs.append(("PyTorch vs TensorRT", "torch", "trt"))
    if "onnx" in backends and "trt" in backends:
        pair_defs.append(("ONNX vs TensorRT", "onnx", "trt"))

    all_metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name, _, _ in pair_defs}
    report_rows: List[Dict] = []

    target_h, target_w = resolution

    for idx, img_path in enumerate(images):
        print(f"{'--'*30}")
        print(f"  [{idx+1}/{len(images)}] {img_path.name}")
        print(f"{'--'*30}")

        raw = cv2.imread(str(img_path))
        if raw is None:
            print("  [!] Cannot read, skipping.")
            continue

        orig_h, orig_w = raw.shape[:2]
        preproc, _ = preprocess_image_exact(raw, target_h, target_w)

        # --- Run each backend ---
        results: Dict[str, np.ndarray] = {}

        if "torch" in backends:
            results["torch"] = infer_torch(torch_model, preproc, (target_h, target_w))
            print(f"    PyTorch: range=[{results['torch'].min():.4f}, {results['torch'].max():.4f}]")

        if "onnx" in backends:
            results["onnx"] = onnx_session.run(preproc)
            print(f"    ONNX:    range=[{results['onnx'].min():.4f}, {results['onnx'].max():.4f}]")

        if "trt" in backends:
            results["trt"] = trt_runner.infer(preproc)
            print(f"    TRT:     range=[{results['trt'].min():.4f}, {results['trt'].max():.4f}]")

        del preproc

        # --- Compare pairs ---
        row: Dict = {"image": img_path.name}
        for pair_name, key_a, key_b in pair_defs:
            m = compute_metrics(results[key_a], results[key_b])
            verdict_label = print_metrics(m, pair_name)
            all_metrics[pair_name].append(m)
            row[f"{pair_name.replace(' ', '_').lower()}_max_ae"] = m["max_ae"]
            row[f"{pair_name.replace(' ', '_').lower()}_mean_ae"] = m["mean_ae"]
            row[f"{pair_name.replace(' ', '_').lower()}_rmse"] = m["rmse"]
            row[f"{pair_name.replace(' ', '_').lower()}_cosine_sim"] = m["cosine_sim"]
            row["verdict"] = verdict_label

            if args.save_diff:
                save_diff_heatmap(
                    results[key_a], results[key_b],
                    str(output_dir / f"{img_path.stem}_{pair_name.lower().replace(' ','_')}_diff.png"),
                    ref_label=key_a.upper(),
                    cand_label=key_b.upper(),
                )

        # --- Save depth maps ---
        stem = img_path.stem
        for key, depth in results.items():
            save_depth_vis(depth, str(output_dir / f"{stem}_{key}.png"), (orig_w, orig_h))

        report_rows.append(row)
        del raw, results

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({len(images)} image(s))")
    print(f"{'='*60}")

    for pair_name, _, _ in pair_defs:
        metrics_list = all_metrics.get(pair_name, [])
        if not metrics_list:
            continue
        avg = summary_metrics(metrics_list)
        print(f"\n  [{pair_name}]")
        print(f"    Mean AE avg:     {avg['mean_ae']:.6f}")
        print(f"    Max AE avg:      {avg['max_ae']:.6f}")
        print(f"    RMSE avg:        {avg['rmse']:.6f}")
        print(f"    Mean Rel avg:    {avg['mean_rel']:.6f}")
        print(f"    Max Rel % avg:   {avg['max_rel_pct']:.4f} %")
        print(f"    Cosine sim avg:  {avg['cosine_sim']:.8f}")
        print(f"    Pearson r avg:   {avg['pearson_r']:.8f}")
        print(f"    => {verdict(avg)}")

    # --- CSV report ---
    if args.report and report_rows and args.report.endswith(".csv"):
        with open(args.report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            w.writeheader()
            w.writerows(report_rows)
        print(f"\n[*] CSV report saved: {args.report}")

    # --- Text report ---
    if report_rows:
        report_path = args.report if (args.report and not args.report.endswith(".csv")) else None
        backend_desc = " + ".join(
            ([f"PyTorch({args.torch})"] if "torch" in backends else []) +
            (["ONNX"] if "onnx" in backends else []) +
            (["TRT"] if "trt" in backends else [])
        )
        save_report(report_path, all_metrics, len(images),
                    backends_info=backend_desc, output_dir=str(output_dir))

    print(f"\n[*] Output saved to: {output_dir}")
    print("[*] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _validate_args(args)

    # --- ONNX inspection modes (no --image, no output dir) ---
    if args.inspect:
        inspect_onnx(args.onnx)
        return
    if args.check_integrity:
        sys.exit(check_integrity(args.onnx))

    # --- Determine mode ---
    backends = _active_backends(args)
    resolution = _determine_resolution(args)
    output_dir = Path(args.output_dir or _auto_output_dir(args))
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(backends) == 1:
        _run_standalone(args, backends[0], resolution, output_dir)
    else:
        if resolution is None:
            sys.exit("[!] Comparison mode requires --onnx or --trt to determine resolution")
        _run_comparison(args, backends, resolution, output_dir)


if __name__ == "__main__":
    main()
