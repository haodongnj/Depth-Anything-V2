#!/usr/bin/env python3
"""
Patch an ONNX model for TensorRT 8.2 compatibility (Jetson Nano / Tegra X1).

Fixes applied:
  1. Resize mode  cubic → linear
     TRT 8.2 does not support cubic interpolation.  The cubic resize comes from
     DINOv2 positional-encoding interpolation and is safe to downgrade — the
     positional-encoding grid is smooth, so bilinear vs bicubic is negligible.

  2. Transpose + Reshape fusion → split with Identity
     TRT 8.2 myelin tries to fuse adjacent Transpose→Reshape into a single
     kernel, but Tegra X1 has no implementation for the fused pattern.  Inserting
     an Identity node between them breaks the fusion.

Usage:
  python patch_for_trt82.py depth_anything_v2_vits_518x700.onnx
  python patch_for_trt82.py model.onnx -o model_trt82.onnx
"""

import argparse
import sys
from pathlib import Path

import onnx
from onnx import helper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch ONNX model for TensorRT 8.2 (Jetson Nano)"
    )
    p.add_argument("input", type=Path, help="Input ONNX model")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path (default: overwrite input)")
    return p.parse_args()


def patch_cubic_to_linear(model: onnx.ModelProto) -> int:
    """Change every Resize node with mode='cubic' to mode='linear'."""
    patched = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        for attr in node.attribute:
            if attr.name == "mode" and attr.s == b"cubic":
                attr.s = b"linear"  # type: ignore[assignment]
                patched += 1
                print(f"    [cubic→linear] {node.name}")
    return patched


def break_transpose_reshape_fusion(model: onnx.ModelProto) -> int:
    """
    Insert an Identity node between every Transpose→Reshape pair in the
    depth head so TensorRT myelin does not fuse them into a single kernel
    that Tegra X1 cannot execute.
    """
    # Build output→producer map
    out_to_node: dict[str, int] = {}  # output_name → node_index
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            out_to_node[out] = idx

    insertions: list[tuple[int, onnx.NodeProto]] = []  # (index_after, identity_node)
    ident_counter = 0
    seen_outputs = set()

    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Transpose":
            continue
        # Check if this Transpose's output feeds into a Reshape
        for out_name in node.output:
            if out_name in seen_outputs:
                continue
            # Find consumers
            consumers = [
                (ci, cn) for ci, cn in enumerate(model.graph.node)
                if out_name in cn.input
            ]
            reshape_consumers = [
                (ci, cn) for ci, cn in consumers if cn.op_type == "Reshape"
            ]
            if not reshape_consumers:
                continue

            # Insert Identity between Transpose and ALL consumers (not just Reshape)
            # to keep the graph consistent
            ident_name = f"/trt82_fusion_break/Identity_{ident_counter}"
            ident_counter += 1
            ident_output = f"{ident_name}_output_0"

            identity = helper.make_node(
                "Identity",
                inputs=[out_name],
                outputs=[ident_output],
                name=ident_name,
            )

            insertions.append((idx, identity))
            seen_outputs.add(out_name)

            # Re-route all consumers of this Transpose output to the Identity output
            for _ci, cn in consumers:
                for i, inp in enumerate(cn.input):
                    if inp == out_name:
                        cn.input[i] = ident_output

    if not insertions:
        return 0

    # Insert in reverse index order (so indices stay valid)
    for idx, ident in reversed(insertions):
        model.graph.node.insert(idx + 1, ident)

    return len(insertions)


def patch_model(onnx_path: Path, output_path: Path) -> None:
    model = onnx.load(str(onnx_path))

    n_cubic = patch_cubic_to_linear(model)
    n_fusion = break_transpose_reshape_fusion(model)

    if n_cubic == 0 and n_fusion == 0:
        print("    nothing to patch")
        return

    onnx.save(model, str(output_path))
    total = n_cubic + n_fusion
    parts = []
    if n_cubic:
        parts.append(f"{n_cubic} cubic→linear")
    if n_fusion:
        parts.append(f"{n_fusion} Transpose+Reshape fusion break(s)")
    print(f"[✓] Saved: {output_path}  ({', '.join(parts)})")

    # Validate
    onnx.checker.check_model(model)
    print("[✓] Model passes onnx.checker validation")
    print(f"[✓] Opset: {model.opset_import[0].version}")


def main() -> None:
    args = parse_args()
    onnx_path: Path = args.input
    output_path: Path = args.output or onnx_path

    if not onnx_path.exists():
        sys.exit(f"[!] File not found: {onnx_path}")

    print(f"[*] Patching: {onnx_path}")
    patch_model(onnx_path, output_path)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
