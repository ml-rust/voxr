#!/usr/bin/env python3
"""Extract Silero VAD weights from the upstream ONNX into safetensors.

The upstream `silero_vad.onnx` carries no top-level initializers: the whole
model sits inside a single `If` node whose two branches are the 8 kHz and
16 kHz variants, and every weight is a `Constant` node inside a branch. This
script pulls each branch out into its own safetensors file so Rust can load it
with `VarMap::from_safetensors`.

Usage:
    python extract_silero_weights.py <models-dir>/silero-vad

Writes `silero_vad_16k.safetensors` and `silero_vad_8k.safetensors` beside the
input `silero_vad.onnx`.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper
from safetensors.numpy import save_file

# The `If` predicate is `sr == 16000`, so `then_branch` is the 16 kHz model.
BRANCH_RATE = {"then_branch": 16_000, "else_branch": 8_000}

# Each branch inlines its module path into the constant name.
PREFIXES = ("If_0_then_branch__Inline_0__", "If_0_else_branch__Inline_0__")


def branch_tensors(graph):
    """Every named `Constant` tensor in a branch graph, plus its initializers."""
    out = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                out[node.output[0]] = numpy_helper.to_array(attr.t)
    for init in graph.initializer:
        out[init.name] = numpy_helper.to_array(init)
    return out


def clean(name):
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_weight(name):
    """Keep module parameters, drop the graph's shape/index scratch constants."""
    return any(
        part in name
        for part in ("reparam_conv", "forward_basis_buffer", "rnn.", "decoder.decoder")
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    model = onnx.load(root / "silero_vad.onnx")

    branches = {
        attr.name: attr.g
        for node in model.graph.node
        if node.op_type == "If"
        for attr in node.attribute
    }
    if set(branches) != set(BRANCH_RATE):
        sys.exit(f"unexpected branches {sorted(branches)}; upstream layout changed")

    for branch, graph in branches.items():
        rate = BRANCH_RATE[branch]
        tensors = {
            clean(k): np.ascontiguousarray(v)
            for k, v in branch_tensors(graph).items()
            if is_weight(clean(k))
        }
        if not tensors:
            sys.exit(f"{branch}: no weights matched; upstream layout changed")
        path = root / f"silero_vad_{rate // 1000}k.safetensors"
        save_file(tensors, path, metadata={"sample_rate": str(rate)})
        total = sum(v.size for v in tensors.values())
        print(f"{path.name}: {len(tensors)} tensors, {total} params")
        for name in sorted(tensors):
            print(f"    {str(tensors[name].shape):20s} {name}")


if __name__ == "__main__":
    main()
