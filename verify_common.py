#!/usr/bin/env python3
"""
Shared utilities for Depth Anything V2 export verification.

Used by: verify_onnx.py, verify_trt.py, verify_precision.py

Provides:
  - Preprocessing (aspect-ratio-preserving and exact-resize)
  - ONNX backend (inspect, integrity check, reusable session)
  - PyTorch backend (model loading, inference)
  - TensorRT backend (TRTInference class with CUDA management)
  - Metrics (float32-optimized, single-image streaming)
  - I/O helpers (image gathering, depth-map saving)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# PyTorch model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


# ===========================================================================
# Preprocessing
# ===========================================================================

def resize_dims(h: int, w: int, input_size: int) -> Tuple[int, int]:
    """Compute aspect-ratio-preserving resize dimensions (multiple-of-14)."""
    ratio = max(h, w) / input_size
    if ratio < 1:
        ratio = 1
    new_h = round(h / ratio)
    new_w = round(w / ratio)
    new_h = new_h - (new_h % 14)
    new_w = new_w - (new_w % 14)
    return new_h, new_w


def preprocess_image(
    raw_image: np.ndarray,
    input_size: int = 518,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Aspect-ratio-preserving preprocessing matching the PyTorch model."""
    orig_h, orig_w = raw_image.shape[:2]
    h, w = resize_dims(orig_h, orig_w, input_size)

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    image = image[np.newaxis, ...]

    return image, (orig_h, orig_w), (h, w)


def preprocess_image_exact(
    raw_image: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess a BGR image, resizing exactly to target (H, W)."""
    orig_h, orig_w = raw_image.shape[:2]

    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    image = image[np.newaxis, ...]

    return image, (orig_h, orig_w)


# ===========================================================================
# ONNX backend
# ===========================================================================

def get_onnx_input_shape(onnx_path: str) -> Tuple[int, int]:
    """Read the ONNX model's expected (H, W) input dimensions."""
    import onnx
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in inp.type.tensor_type.shape.dim]
        if len(dims) == 4:
            h, w = dims[2], dims[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return h, w
    raise ValueError(f"Could not determine fixed H,W from ONNX model: {onnx_path}")


class ONNXSession:
    """Reusable ONNX Runtime inference session (created once, reused per image)."""

    def __init__(self, onnx_path: str, providers=None):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )

    def run(self, preprocessed: np.ndarray) -> np.ndarray:
        """Return squeezed depth map (H, W) float32."""
        out = self._session.run(None, {"image": preprocessed.astype(np.float32)})[0]
        return out[0]


def inspect_onnx(onnx_path: str) -> None:
    """Print detailed ONNX model structure."""
    import onnx

    model = onnx.load(onnx_path)

    print(f"\n{'='*60}")
    print(f"  ONNX Model: {onnx_path}")
    print(f"{'='*60}")

    print(f"\n  IR version:       {model.ir_version}")
    print(f"  Opset:            {model.opset_import[0].domain} v{model.opset_import[0].version}")
    print(f"  Producer:         {model.producer_name} v{model.producer_version}")
    print(f"  Model size:       {os.path.getsize(onnx_path) / 1024**2:.1f} MB")
    print(f"  Number of nodes:  {len(model.graph.node)}")
    print(f"  Number of inputs: {len(model.graph.input)}")
    print(f"  Number of outputs:{len(model.graph.output)}")

    print(f"\n  -- Inputs --")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name:20s}  shape={shape}")

    print(f"\n  -- Outputs --")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name:20s}  shape={shape}")

    op_counts: Dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"\n  -- Operators ({len(op_counts)} unique types) --")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op:30s} {count:4d}")

    total_params = 0
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        total_params += arr.size
    if total_params:
        print(f"\n  -- Weights --")
        print(f"    Total parameters:  {total_params:,}")
        print(f"    Total size (FP32): {total_params * 4 / 1024**2:.1f} MB")


def check_integrity(onnx_path: str) -> int:
    """Run integrity checks on an ONNX model. Returns warning count."""
    import onnx

    warnings = 0
    model = onnx.load(onnx_path)

    def check(desc, condition, detail=""):
        nonlocal warnings
        if condition:
            print(f"  {desc}: ✓  {detail}")
        else:
            print(f"  {desc}: [!] {detail}")
            warnings += 1

    print(f"\n  Integrity check for: {onnx_path}\n")
    check("IR version", model.ir_version >= 7, f"v{model.ir_version}")
    opset = model.opset_import[0].version
    check("Opset", opset >= 11, f"v{opset}")
    size_mb = os.path.getsize(onnx_path) / 1024**2
    check("File size", size_mb <= 2048, f"{size_mb:.1f} MB")
    check("Graph nodes", len(model.graph.node) > 0, f"{len(model.graph.node)} nodes")

    input_names = [inp.name for inp in model.graph.input]
    output_names = [out.name for out in model.graph.output]
    check("Input name 'image'", "image" in input_names)
    check("Output name 'depth'", "depth" in output_names)

    try:
        import onnxruntime as ort
        ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        check("onnxruntime load", True, "OK")
    except Exception as e:
        check("onnxruntime load", False, str(e))

    try:
        onnx.checker.check_model(model)
        check("onnx.checker", True, "model is valid")
    except Exception as e:
        check("onnx.checker", False, str(e))

    print(f"\n  Total warnings: {warnings}")
    return warnings


# ===========================================================================
# PyTorch backend
# ===========================================================================

def load_torch_model(encoder: str, checkpoint_path: str, device: str = "cpu"):
    """Load the PyTorch DepthAnythingV2 model."""
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choices: {list(MODEL_CONFIGS)}")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def infer_torch(model, preprocessed: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Run PyTorch inference, return depth at target resolution (H, W) float32."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    tensor = torch.from_numpy(preprocessed).to(device)
    with torch.no_grad():
        depth = model.forward(tensor)
    depth = F.interpolate(depth[:, None], target_hw, mode="bilinear", align_corners=True)[0, 0]
    return depth.cpu().numpy().astype(np.float32)


# ===========================================================================
# TensorRT backend
# ===========================================================================

# pycuda MUST be initialized before TensorRT creates any CUDA context
try:
    import pycuda.driver as cuda_drv
    import pycuda.autoinit  # noqa
    _PYCUDA_OK = True
except ImportError:
    cuda_drv = None
    _PYCUDA_OK = False


class TRTInference:
    """Minimal TensorRT inference wrapper (TRT 10+/11.x API)."""

    def __init__(self, engine_path: str, verbose: bool = False):
        if not _PYCUDA_OK:
            raise RuntimeError("pycuda is required for TRT inference: pip install pycuda")

        import tensorrt as trt

        level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        logger = trt.Logger(level)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self._engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(self._engine_data)
        self.context = self.engine.create_execution_context()

        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT and name == "image":
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT and name == "depth":
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            raise RuntimeError(f"Could not find 'image'/'depth'. Tensors: {names}")

        print(f"[*] TRT engine loaded: {engine_path}")
        print(f"    Input  [{self.input_name}]: "
              f"shape={self.engine.get_tensor_shape(self.input_name)}  "
              f"dtype={self.engine.get_tensor_dtype(self.input_name)}")
        print(f"    Output [{self.output_name}]: "
              f"shape={self.engine.get_tensor_shape(self.output_name)}  "
              f"dtype={self.engine.get_tensor_dtype(self.output_name)}")

    def input_shape(self) -> Tuple[int, int]:
        s = self.engine.get_tensor_shape(self.input_name)
        return s[2], s[3]

    def infer(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run inference. Returns depth (H, W) float32."""
        self.context.set_input_shape(self.input_name, tuple(preprocessed.shape))
        out_shape = self.context.get_tensor_shape(self.output_name)
        out_bytes = int(np.prod(out_shape)) * 4

        h_in = np.ascontiguousarray(preprocessed, dtype=np.float32)
        d_in = cuda_drv.mem_alloc(h_in.nbytes)
        d_out = cuda_drv.mem_alloc(out_bytes)

        cuda_drv.memcpy_htod(d_in, h_in)
        self.context.set_tensor_address(self.input_name, int(d_in))
        self.context.set_tensor_address(self.output_name, int(d_out))

        stream = cuda_drv.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()

        h_out = np.empty(out_shape, dtype=np.float32)
        cuda_drv.memcpy_dtoh(h_out, d_out)

        d_in.free()
        d_out.free()
        return h_out.squeeze(0)


# ===========================================================================
# Metrics (float32-optimized — no unnecessary float64 casts)
# ===========================================================================

def compute_metrics(ref: np.ndarray, cand: np.ndarray) -> Dict[str, float]:
    """
    Compute per-pixel precision metrics. Both arrays must be same shape.
    Uses float32 arithmetic — no double-casting to avoid 2× memory.
    """
    ref = ref.astype(np.float32).ravel()
    cand = cand.astype(np.float32).ravel()
    n = float(ref.size)

    abs_diff = np.abs(ref - cand, dtype=np.float32)
    valid = np.abs(ref) > 1e-8
    rel_diff = np.zeros_like(abs_diff)
    np.divide(abs_diff, np.abs(ref, out=rel_diff), out=rel_diff, where=valid)

    max_ref = float(np.abs(ref).max())

    m = {}
    m["max_ae"] = float(abs_diff.max())
    m["mean_ae"] = float(abs_diff.mean())
    m["median_ae"] = float(np.median(abs_diff))
    m["rmse"] = float(np.sqrt((abs_diff * abs_diff).mean()))
    m["mean_rel"] = float(rel_diff.mean()) if valid.any() else float("inf")
    m["rel_p90"] = float(np.percentile(rel_diff, 90))
    m["rel_p95"] = float(np.percentile(rel_diff, 95))
    m["rel_p99"] = float(np.percentile(rel_diff, 99))
    m["max_rel_pct"] = float(abs_diff.max() / max_ref * 100) if max_ref > 1e-8 else float("inf")

    # Cosine similarity (manual dot product, no copy)
    dot = float(np.dot(ref, cand))
    nr = float(np.linalg.norm(ref))
    nc = float(np.linalg.norm(cand))
    m["cosine_sim"] = float(dot / (nr * nc + 1e-12))

    # Pearson r
    mr, mc = float(ref.mean()), float(cand.mean())
    cov = float(np.dot(ref - mr, cand - mc)) / n
    sr = float(np.std(ref))  # population std
    sc = float(np.std(cand))
    m["pearson_r"] = float(cov / (sr * sc + 1e-12))

    m["pass_1e6"] = float((abs_diff <= 1e-6).mean() * 100)

    # Free intermediates
    del abs_diff, rel_diff, ref, cand
    return m


def print_metrics(m: Dict[str, float], label: str) -> str:
    """Pretty-print metrics, return one-word verdict."""
    print(f"\n  [{label}]")
    print(f"    Max AE:        {m['max_ae']:.6f}")
    print(f"    Mean AE:       {m['mean_ae']:.6f}")
    print(f"    Median AE:     {m['median_ae']:.6f}")
    print(f"    RMSE:          {m['rmse']:.6f}")
    print(f"    Mean Rel:      {m['mean_rel']:.6f}")
    print(f"    Max Rel %:     {m['max_rel_pct']:.4f} %")
    print(f"    Rel P90:       {m['rel_p90']:.6f}")
    print(f"    Rel P99:       {m['rel_p99']:.6f}")
    print(f"    Cosine sim:    {m['cosine_sim']:.8f}")
    print(f"    Pearson r:     {m['pearson_r']:.8f}")
    print(f"    % within 1e-6: {m['pass_1e6']:.2f} %")

    if m["cosine_sim"] > 0.99999 and m["mean_ae"] < 1e-5:
        return "PASS"
    elif m["cosine_sim"] > 0.9999 and m["mean_ae"] < 1e-4:
        return "PASS(minor)"
    elif m["cosine_sim"] > 0.999 and m["mean_ae"] < 1e-3:
        return "ACCEPTABLE"
    return "WARN"


def summary_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean of key metrics across a batch."""
    keys = ["mean_ae", "max_ae", "rmse", "mean_rel", "max_rel_pct", "cosine_sim", "pearson_r"]
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def verdict(avg: Dict[str, float]) -> str:
    """Return overall verdict string."""
    if avg["cosine_sim"] > 0.99999 and avg["mean_ae"] < 1e-5:
        return "✓ EXCELLENT"
    elif avg["cosine_sim"] > 0.9999 and avg["mean_ae"] < 1e-4:
        return "✓ GOOD"
    elif avg["cosine_sim"] > 0.999 and avg["mean_ae"] < 1e-3:
        return "~ ACCEPTABLE"
    return "✗ SIGNIFICANT DEVIATION"


def save_report(
    report_path: str,
    all_metrics: Dict[str, List[Dict[str, float]]],
    image_count: int,
    backends_info: str = "",
    output_dir: str = "",
) -> str:
    """
    Save a text report summarizing verification results.

    If report_path is empty/None, auto-generates a timestamped filename.

    Args:
        report_path:   path to the report file (.txt), or None for auto-name
        all_metrics:   {pair_name: [per_image_metrics, ...]}
        image_count:   number of images tested
        backends_info: human-readable backends description
        output_dir:    where depth visualizations were saved

    Returns:
        The path where the report was saved.
    """
    from datetime import datetime

    if not report_path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        report_path = f"verify_report_{ts}.txt"

    with open(report_path, "w") as f:
        w = f.write

        w(f"Depth Anything V2 — Precision Verification Report\n")
        w(f"{'='*60}\n")
        w(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        w(f"Backends:   {backends_info}\n")
        w(f"Images:     {image_count}\n")
        if output_dir:
            w(f"Output:     {output_dir}\n")
        w(f"\n")

        # Per-pair summary
        for pair_name, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            avg = summary_metrics(metrics_list)
            w(f"{'─'*60}\n")
            w(f"  {pair_name}  ({len(metrics_list)} images)\n")
            w(f"{'─'*60}\n")
            w(f"  Mean AE:        {avg['mean_ae']:.6f}\n")
            w(f"  Max AE:         {avg['max_ae']:.6f}\n")
            w(f"  RMSE:           {avg['rmse']:.6f}\n")
            w(f"  Mean Rel:       {avg['mean_rel']:.6f}\n")
            w(f"  Max Rel %:      {avg['max_rel_pct']:.4f} %\n")
            w(f"  Cosine sim:     {avg['cosine_sim']:.8f}\n")
            w(f"  Pearson r:      {avg['pearson_r']:.8f}\n")
            w(f"  Verdict:        {verdict(avg)}\n")
            w(f"\n")

            # Per-image detail
            for i, m in enumerate(metrics_list, 1):
                w(f"  Image {i}:  max_ae={m['max_ae']:.6f}  "
                  f"mean_ae={m['mean_ae']:.6f}  rmse={m['rmse']:.6f}  "
                  f"cosine={m['cosine_sim']:.8f}\n")
            w(f"\n")

        w(f"{'='*60}\n")
        w(f"End of report.\n")

    print(f"[*] Report saved: {report_path}")
    return report_path


# ===========================================================================
# Visualization
# ===========================================================================

def save_diff_heatmap(
    ref: np.ndarray,
    cand: np.ndarray,
    output_path: str,
    ref_label: str = "Ref",
    cand_label: str = "Cand",
) -> None:
    """3-panel side-by-side: ref | cand | abs diff heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_vis = (ref - ref.min()) / (ref.max() - ref.min() + 1e-8)
    cand_vis = (cand - cand.min()) / (cand.max() - cand.min() + 1e-8)
    diff = np.abs(ref - cand)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(ref_vis, cmap="inferno"); axes[0].set_title(ref_label); axes[0].axis("off")
    axes[1].imshow(cand_vis, cmap="inferno"); axes[1].set_title(cand_label); axes[1].axis("off")
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title(f"|{ref_label} - {cand_label}|  (max={diff.max():.4f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Diff heatmap saved: {output_path}")


# ===========================================================================
# I/O helpers
# ===========================================================================

def gather_images(image_path: str, ext: str = "jpg") -> List[Path]:
    """Collect image paths from file or directory."""
    p = Path(image_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        images = sorted(p.rglob(f"*.{ext}"))
        if not images:
            sys.exit(f"[!] No *.{ext} images found in {image_path}")
        return images
    sys.exit(f"[!] --image path does not exist: {image_path}")


def save_depth_vis(depth: np.ndarray, output_path: str, target_hw: Tuple[int, int]) -> None:
    """Resize depth to target (W, H) and save as uint8 PNG."""
    vis = cv2.resize(depth.astype(np.float32), target_hw, interpolation=cv2.INTER_LINEAR)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8) * 255
    cv2.imwrite(output_path, vis.astype(np.uint8))
