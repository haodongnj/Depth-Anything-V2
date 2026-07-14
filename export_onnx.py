#!/usr/bin/env python3
"""
Export Depth Anything V2 to ONNX format.

Handles three ONNX-incompatible patterns in the original model:
  1. nn.quantized.FloatFunctional (skip-add) → replaced with plain addition
  2. xFormers MemEffAttention → forced to fall back to standard attention
  3. Dynamic F.interpolate sizes → handled via symbolic dynamo export

H and W are fixed at export time (only the batch dimension is dynamic).
Use --input-height / --input-width to produce separate .onnx files for
different resolutions.

Usage:
  python export_onnx.py --encoder vitl --checkpoint checkpoints/depth_anything_v2_vitl.pth
  python export_onnx.py --encoder vits --checkpoint checkpoints/depth_anything_v2_vits.pth --input-height 392 --input-width 392
  python export_onnx.py --encoder vitl --checkpoint ... --input-height 518 --input-width 700 --verify
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------------
# Model patching for ONNX compatibility
# ---------------------------------------------------------------------------

def _make_patched_rcu_forward(module):
    """Return a patched forward for ResidualConvUnit that uses plain + instead of FloatFunctional."""

    def forward(x):
        out = module.activation(x)
        out = module.conv1(out)
        if module.bn:
            out = module.bn1(out)
        out = module.activation(out)
        out = module.conv2(out)
        if module.bn:
            out = module.bn2(out)
        if module.groups > 1:
            out = module.conv_merge(out)
        return out + x  # plain addition – ONNX-safe

    return forward


def _make_patched_ffb_forward(module):
    """Return a patched forward for FeatureFusionBlock that uses plain + instead of FloatFunctional."""

    def forward(*xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = module.resConfUnit1(xs[1])
            output = output + res  # plain addition – ONNX-safe
        output = module.resConfUnit2(output)

        if (size is None) and (module.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": module.size}
        else:
            modifier = {"size": size}

        output = F.interpolate(
            output, **modifier, mode="bilinear", align_corners=module.align_corners
        )
        output = module.out_conv(output)
        return output

    return forward


def patch_model_for_onnx(model: nn.Module) -> nn.Module:
    """
    Walk the model tree and replace ONNX-incompatible ops in-place.

    - ResidualConvUnit: replace FloatFunctional.add with plain +
    - FeatureFusionBlock: same, plus re-bind forward
    """
    from depth_anything_v2.util.blocks import ResidualConvUnit, FeatureFusionBlock

    for module in model.modules():
        if isinstance(module, ResidualConvUnit):
            module.forward = _make_patched_rcu_forward(module)
        elif isinstance(module, FeatureFusionBlock):
            module.forward = _make_patched_ffb_forward(module)

    return model


def disable_xformers():
    """
    Force MemEffAttention to use the standard (traceable) attention path.

    When xFormers is installed, MemEffAttention calls xformers.ops.memory_efficient_attention
    which cannot be traced by torch.onnx. Setting XFORMERS_AVAILABLE = False makes it fall
    back to the plain Attention.forward (matmul + softmax), which exports cleanly.
    """
    from depth_anything_v2.dinov2_layers import attention as attn_mod

    attn_mod.XFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def load_model(encoder: str, checkpoint_path: str, device: str = "cpu") -> DepthAnythingV2:
    """Load and return a DepthAnythingV2 model with pretrained weights."""
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choices: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    input_height: int,
    input_width: int,
    output_path: str,
    opset_version: int = 18,
    verify: bool = False,
) -> None:
    """
    Export to ONNX with the given spatial resolution baked into the graph.

    The model's internal DINOv2 positional encoding and DPT-head interpolate
    logic prevent true dynamic H/W tracing (torch.export requires H==W and
    specialises to a concrete value).  For a reliable graph we fix H and W
    at export time; only the batch dimension stays dynamic.

    To support multiple resolutions, export separate .onnx files with
    different --input-height / --input-width values.
    """
    print(f"[*] Exporting with input size {input_height}×{input_width} …")

    dummy = torch.randn(1, 3, input_height, input_width)

    # Only batch is dynamic; H and W are baked as constants during tracing.
    dynamic_axes = {
        "image": {0: "batch"},
        "depth": {0: "batch"},
    }

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    _merge_external_data(output_path)

    print(f"[✓] ONNX model saved to {output_path}")

    if verify:
        _verify_onnx(output_path, dummy, model)


def _merge_external_data(onnx_path: str) -> None:
    """
    PyTorch ≥ 2.0 dynamo export saves weights into a separate <model>.onnx.data
    file.  Merge it back into a single self-contained .onnx file.
    """
    data_path = onnx_path + ".data"
    if not os.path.exists(data_path):
        return  # already self-contained

    import onnx
    from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data

    print("[*] Merging external weight data into single-file ONNX …")
    model = onnx.load(onnx_path, load_external_data=True)
    # convert_model_to_external_data with size > model size disables external data
    total_size = model.ByteSize()
    convert_model_to_external_data(model, all_tensors_to_one_file=False, size_threshold=total_size + 1)
    onnx.save(model, onnx_path)
    os.remove(data_path)
    print("[✓] Merged — single-file ONNX model ready.")


def _verify_onnx(
    onnx_path: str,
    dummy_input: torch.Tensor,
    torch_model: nn.Module,
) -> None:
    """Compare ONNX Runtime output against the PyTorch model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] onnxruntime not installed – skipping verification.  pip install onnxruntime")
        return

    print("[*] Verifying with onnxruntime …")

    with torch.no_grad():
        pt_out = torch_model(dummy_input).cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"image": dummy_input.numpy()})[0]

    abs_diff = np.abs(pt_out - ort_out)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    denom = np.abs(pt_out).max()
    rel_diff = (abs_diff / denom).mean() * 100 if denom > 1e-8 else float("inf")

    print(f"    [{dummy_input.shape[2]}×{dummy_input.shape[3]}] "
          f"Max ae: {max_diff:.6f}  Mean ae: {mean_diff:.6f}  "
          f"Mean rel: {rel_diff:.4f} %")

    if max_diff < 1e-4 and mean_diff < 1e-5:
        print("[✓] ONNX output matches PyTorch output.")
    else:
        print("[!] Numerical difference is larger than expected.  This may be acceptable")
        print("    depending on your use-case (the model uses FP32 throughout).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Depth Anything V2 to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --encoder vitl --checkpoint checkpoints/depth_anything_v2_vitl.pth
  %(prog)s --encoder vits --checkpoint checkpoints/depth_anything_v2_vits.pth --input-height 392 --input-width 392
  %(prog)s --encoder vitl --checkpoint ... --input-height 518 --input-width 700 --verify
        """,
    )
    p.add_argument("--encoder", required=True, choices=list(MODEL_CONFIGS),
                   help="Encoder variant (vits, vitb, vitl, vitg)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pth checkpoint file")
    p.add_argument("--output", default=None,
                   help="Output ONNX path (default: depth_anything_v2_<encoder>.onnx)")
    p.add_argument("--input-height", type=int, default=518,
                   help="Input height baked into the ONNX graph (default: 518, must be multiple of 14).  "
                        "The exported model is fixed to this resolution at inference time.")
    p.add_argument("--input-width", type=int, default=518,
                   help="Input width baked into the ONNX graph (default: 518, must be multiple of 14).  "
                        "Same semantics as --input-height.")
    p.add_argument("--opset", type=int, default=18,
                   help="ONNX opset version (default: 18)")
    p.add_argument("--verify", action="store_true",
                   help="Verify the exported model against PyTorch using onnxruntime")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_height % 14 != 0 or args.input_width % 14 != 0:
        print(f"[!] Warning: input dimensions ({args.input_height}×{args.input_width}) "
              f"should be multiples of 14 for correct results.")

    # Output path
    output_path = args.output or f"depth_anything_v2_{args.encoder}.onnx"

    if os.path.exists(output_path):
        print(f"[!] '{output_path}' already exists.  Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # ---- Load ----
    print(f"[*] Loading model (encoder={args.encoder}) …")
    model = load_model(args.encoder, args.checkpoint)

    # ---- Patch ----
    print("[*] Patching model for ONNX compatibility …")
    disable_xformers()
    model = patch_model_for_onnx(model)

    # ---- Export ----
    export_onnx(model, args.input_height, args.input_width,
                output_path, args.opset, args.verify)

    print("[✓] Done.")


if __name__ == "__main__":
    main()
