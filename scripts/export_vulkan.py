"""Export Laya EncCore to ONNX → fixed → IREE-compiled Vulkan vmfb buckets.

Pipeline per (B, L) bucket (proven on RX 570 / RADV with --iree-vulkan-target=adreno):
  torch export (int32 arange) → fix LayerNorm/Where → iree-import-onnx → iree-compile

Usage:
  python scripts/export_vulkan.py
  python scripts/export_vulkan.py --pairs 3x128 1x512
  python scripts/export_vulkan.py --b 1 4 --l 128 --skip-compile
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from fix_onnx import fix_file  # noqa: E402

OUT = ROOT / "models" / "iree"


@contextlib.contextmanager
def int32_arange():
    """adreno SPIR-V has no int64 compute; force torch.arange → int32 at export."""
    orig = torch.arange

    def patched(*args, **kwargs):
        kwargs.setdefault("dtype", torch.int32)
        return orig(*args, **kwargs)

    torch.arange = patched
    try:
        yield
    finally:
        torch.arange = orig


class EncCore(nn.Module):
    """Encoder without the token-embedding table (stays on CPU torch).

    inputs_embeds + attention_mask → final-normed hidden states. Mirrors
    ModernBertModel.forward so the 786MB vocab gather never enters VRAM.
    """

    def __init__(self, src):
        super().__init__()
        self.encoder = src.encoder

    def forward(self, inputs_embeds, attention_mask):
        from transformers.masking_utils import (
            create_bidirectional_mask,
            create_bidirectional_sliding_window_mask,
        )

        enc = self.encoder
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden = enc.embeddings(inputs_embeds=inputs_embeds)
        mask_kwargs = {
            "config": enc.config,
            "inputs_embeds": hidden,
            "attention_mask": attention_mask,
        }
        mapping = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        pos_emb = {}
        for layer_type in ("full_attention", "sliding_attention"):
            pos_emb[layer_type] = enc.rotary_emb(hidden, position_ids, layer_type)
        for layer in enc.layers:
            hidden = layer(
                hidden,
                attention_mask=mapping[layer.attention_type],
                position_embeddings=pos_emb[layer.attention_type],
            )
        return enc.final_norm(hidden)


def _tool(name: str) -> Path:
    for candidate in (Path(sys.executable).with_name(name), ROOT / ".venv" / "bin" / name):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{name} not found (pip install iree-base-compiler)")


def compile_bucket(onnx_path: Path, vmfb_path: Path) -> None:
    mlir_path = onnx_path.with_suffix(".mlir")
    subprocess.run([str(_tool("iree-import-onnx")), str(onnx_path), "-o", str(mlir_path)], check=True)
    base = [
        str(_tool("iree-compile")),
        str(mlir_path),
        "--iree-hal-target-device=vulkan",
        "--iree-vulkan-target=adreno",
    ]
    # Default opt can emit a spirv.Store type mismatch on some batch sizes
    # (observed B6/B8 at L128); O0 always legalizes on this target.
    for extra in ([], ["--iree-opt-level=O0"]):
        result = subprocess.run(
            [*base, *extra, f"-o={vmfb_path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if extra:
                print(f"  note: compiled with {extra[0]} after default-opt failure")
            return
        if not extra:
            print(result.stderr[-800:], file=sys.stderr)
    raise RuntimeError(f"iree-compile failed for {onnx_path} (tried default and O0)")


def export_bucket(agent, batch: int, length: int, *, skip_compile: bool) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    model = agent.model.eval()
    pad = agent.tok.pad_token_id or 0
    vocab = model.encoder.config.vocab_size
    core = EncCore(model).eval()

    rng = torch.Generator().manual_seed(7)
    ids = torch.randint(5, vocab, (batch, length), generator=rng)
    valid = min(length, 100)
    ids[:, valid:] = pad
    mask = torch.ones((batch, length), dtype=torch.long)
    mask[:, valid:] = 0

    with torch.no_grad():
        embeds = model.encoder.embeddings.tok_embeddings(ids)
        ref_h = core(embeds, mask)
        stock = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        err = float((ref_h - stock).abs().max())
        print(f"core vs stock encoder max abs err: {err:.3e}")
        if err >= 1e-4:
            raise SystemExit(f"encoder mismatch too large: {err}")

    onnx_path = OUT / f"laya_core_b{batch}l{length}k4.onnx"
    with int32_arange():
        torch.onnx.export(
            core,
            (embeds, mask.to(torch.int32)),
            str(onnx_path),
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["hidden"],
            dynamo=True,
        )
    fix_file(onnx_path)
    vmfb_path = OUT / f"laya_core_b{batch}l{length}_vulkan.vmfb"
    if not skip_compile:
        compile_bucket(onnx_path, vmfb_path)
    print(
        f"bucket B{batch}/L{length}: {onnx_path.name}"
        + ("" if skip_compile else f" → {vmfb_path.name}")
    )
    return vmfb_path if not skip_compile else onnx_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--b", type=int, nargs="+", default=None, help="batch buckets (override known-good set)")
    parser.add_argument("--l", type=int, nargs="+", default=None, help="seq buckets (override known-good set)")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default="multilingual")
    parser.add_argument("--skip-compile", action="store_true", help="Stop after fixed ONNX")
    parser.add_argument("--pairs", nargs="*", help="Explicit BxL pairs like 3x128")
    args = parser.parse_args(argv)

    import laya

    agent = laya.load(args.model, device="cpu", subfolder=args.subfolder)
    if args.pairs:
        pairs = [tuple(int(x) for x in item.lower().split("x")) for item in args.pairs]
    elif args.b or args.l:
        batches = args.b or [1, 2, 3, 4, 5, 7]
        lengths = args.l or [128]
        pairs = [(b, l) for b in batches for l in lengths]
    else:
        # Known-good on adreno (RX 570): larger shapes hit spirv.Store bool-width
        # bugs in IREE; see README. B7 pad-covers batch 6; 8+ and L>known → torch.
        pairs = [
            (1, 128), (2, 128), (3, 128), (4, 128), (5, 128), (7, 128),
            (1, 256), (2, 256),
            (1, 512),
        ]

    for b, l in pairs:
        export_bucket(agent, b, l, skip_compile=args.skip_compile)
    print(f"done: {len(pairs)} bucket(s) → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
