"""Mechanical ONNX fixes for the IREE Vulkan (adreno) lowering.

1. LayerNormalization with 2 inputs (bias-less ModernBert norms) gets an
   explicit zero bias.
2. Scalar (0-dim) inputs to Where nodes are explicitly Expanded to the output
   shape -- the SPIR-V lowering chokes on implicit scalar broadcast.

Usage: python scripts/fix_onnx.py in.onnx [out.onnx]  (in place when out omitted)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _const_tensors(model):
    consts = {}
    for init in model.graph.initializer:
        consts[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            for attr in node.attribute:
                if attr.name in ("value", "value_float", "value_int"):
                    if attr.type == onnx.AttributeProto.TENSOR:
                        consts[node.output[0]] = numpy_helper.to_array(attr.t)
    return consts


def _shapes(model):
    shapes = {}
    for container in (
        list(model.graph.input),
        list(model.graph.output),
        list(model.graph.value_info),
    ):
        for v in container:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            shapes[v.name] = dims
    return shapes


def fix_file(src: Path | str, dst: Path | str | None = None) -> None:
    src, dst = Path(src), Path(dst) if dst else Path(src)
    model = onnx.load(str(src))
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as error:
        print(f"shape inference failed (continuing): {error}")
    consts, shapes = _const_tensors(model), _shapes(model)
    fixed_norm, fixed_where = 0, 0

    for node in model.graph.node:
        if node.op_type == "LayerNormalization" and len(node.input) == 2:
            scale = consts.get(node.input[1])
            if scale is None:
                continue
            name = node.output[0] + "_zero_bias"
            model.graph.initializer.append(numpy_helper.from_array(np.zeros_like(scale), name))
            node.input.append(name)
            fixed_norm += 1

    edits = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type != "Where" or not node.output:
            continue
        out_shape = shapes.get(node.output[0])
        if not out_shape or any(d <= 0 for d in out_shape):
            continue
        for i, inp in enumerate(list(node.input)):
            arr = consts.get(inp)
            if arr is None or arr.ndim != 0:
                continue
            edits.append((idx, node, i, inp, out_shape))
    for idx, node, i, inp, out_shape in reversed(edits):
        shape_name = f"{node.output[0]}_expand_shape_{i}"
        out_name = f"{node.output[0]}_expanded_{i}"
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(out_shape, dtype=np.int64), shape_name)
        )
        expand = onnx.helper.make_node(
            "Expand", [inp, shape_name], [out_name], name=f"{node.name}_expand_{i}"
        )
        model.graph.node.insert(idx, expand)
        node.input[i] = out_name
        fixed_where += 1

    onnx.save(model, str(dst))
    print(f"fixed LayerNorm: {fixed_norm}, expanded Where scalars: {fixed_where} -> {dst}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    src = argv[0]
    dst = argv[1] if len(argv) > 1 else src
    fix_file(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
