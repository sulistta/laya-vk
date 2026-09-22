"""Validate Vulkan-encoded buckets against torch and time them.

Compares only unmasked (valid) positions: IREE and torch can diverge on
right-pad slots where attention is fully masked; those never reach the head.

Usage: python scripts/check_vulkan.py [B L] [B L] ...
Defaults to every vmfb discovered under models/iree/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GATE = 5e-4  # adreno fp32 stack; decision-level parity still holds at ~1e-5


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import iree.runtime as iree_rt

    import laya
    from laya.vulkan import discover_vmfbs

    print("drivers:", iree_rt.query_available_drivers())
    profiles = discover_vmfbs()
    if not profiles:
        raise SystemExit("no vmfb artifacts found; run scripts/export_vulkan.py")

    pairs = []
    if argv:
        for i in range(0, len(argv), 2):
            pairs.append((int(argv[i]), int(argv[i + 1])))
    else:
        pairs = sorted(profiles)

    agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")
    agent.model.eval()
    encoder = agent.model.encoder
    pad = agent.tok.pad_token_id or 0

    failed = False
    for batch, length in pairs:
        if (batch, length) not in profiles:
            print(f"skip B{batch}/L{length}: no vmfb")
            continue
        rng = torch.Generator().manual_seed(7)
        ids = torch.randint(5, encoder.config.vocab_size, (batch, length), generator=rng)
        valid = min(length, 100)
        ids[:, valid:] = pad
        mask = torch.ones((batch, length), dtype=torch.long)
        mask[:, valid:] = 0

        with torch.no_grad():
            t0 = time.perf_counter()
            expected = encoder(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()
            torch_ms = (time.perf_counter() - t0) * 1000

        invoker = iree_rt.load_vm_flatbuffer_file(str(profiles[(batch, length)]), driver="vulkan")
        fn = invoker.main_graph
        embeds = encoder.embeddings.tok_embeddings(ids).detach().numpy().astype(np.float32)
        att = mask.numpy().astype(np.int32)
        for _ in range(3):
            fn(embeds, att)
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(embeds, att)
        iree_ms = (time.perf_counter() - t0) / reps * 1000
        got = np.asarray(out)
        del fn, invoker
        import gc

        gc.collect()
        if got.ndim == 2:
            got = got.reshape(batch, length, -1)

        # Gate on valid tokens only (pad slots are fully masked out downstream).
        valid_mask = mask.numpy().astype(bool)
        diff = np.abs(got - expected)
        err = float(diff[valid_mask].max())
        pad_err = float(diff[~valid_mask].max()) if (~valid_mask).any() else 0.0
        status = "OK" if err <= GATE else "FAIL"
        if err > GATE:
            failed = True
        print(
            f"B{batch}/L{length}: torch {torch_ms:.1f} ms, iree {iree_ms:.1f} ms/call, "
            f"valid max abs err {err:.3e} (pad {pad_err:.1e}) [{status}]"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
