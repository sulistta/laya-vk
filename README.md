# Laya (Vulkan/IREE port)

**Multilingual, non-autoregressive System 1 decision engine** — the official [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) runtime with a Vulkan encoder backend for AMD/Qualcomm GPUs via IREE (tested on RX 570 / RADV). Typed decisions (`choice` / `score` / `noul`) over any state in a single forward pass.

Upstream checkpoints: [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (English), `laya-multilingual` (100+ languages), `laya-typed-decisions`.

## Install

```bash
# runtime (CPU torch wheels on Linux without CUDA)
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ".[dev]"

# Vulkan export toolchain (optional)
uv pip install -e ".[export]"
```

Never run `uv sync` against this tree — hand-installed wheels (torch CPU, iree) are not fully expressed in the lockfile.

## Quickstart

```python
import laya

# device auto-fallback: cuda → mps → vulkan → cpu
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
# or force the IREE encoder (falls back to CPU torch if artifacts/driver missing)
agent = laya.load("convaiinnovations/laya", device="vulkan", subfolder="multilingual")

state = {"subject": "Duplicate charge", "body": "Please refund invoice #4411."}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "invoices", "technical": "bugs", "sales": "pricing"},
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}
print(agent.predict(state, questions))
```

`agent.backend` is `"vulkan"` when the hybrid encoder is active, otherwise `"cpu"` / `"torch"`.

## Vulkan encoder

The encoder runs as an IREE-compiled `EncCore` (no 786MB vocab table in VRAM); token embeddings and the decision head stay on CPU torch. Fixed shape buckets only (pad to bucket, run IREE, unpad):

| axis | known-good buckets on RX 570 / adreno |
|---|---|
| batch @ L128 | 1, 2, 3, 4, 5, 7 (pad-up covers 6; **8 → torch fallback**) |
| batch @ L256 | 1, 2 |
| batch @ L512 | 1 |
| multilingual L1024 | none → torch fallback |

Parity is gated on **valid (unmasked) tokens** (≤5e-4 adreno fp32; observed ~1e-5). Right-pad slots can diverge but are masked out before the head. Larger shapes fail `iree-compile` (`spirv.Store` bool-width mismatch — IREE adreno backend bug). Calls outside the envelope fall back to the stock torch encoder for that call.

Graph instances are LRU-cached (`LAYA_VULKAN_MAX_GRAPHS`, default 1) so multiple ~400MB vmfbs never sit in VRAM at once.

Build artifacts (needs `.[export]`, Vulkan driver):

```bash
python scripts/export_vulkan.py                  # known-good buckets → models/iree/
python scripts/export_vulkan.py --pairs 3x128    # single bucket
python scripts/check_vulkan.py                   # parity gate + latency
```

Compile flags proven on this GPU: `--iree-hal-target-device=vulkan --iree-vulkan-target=adreno`.

Search path: `$LAYA_VULKAN_VMFB_DIR`, `./models/iree`, package-relative `models/iree`.

## Tests

```bash
pytest tests/ -q -m "not integration"   # unit + snake (no downloads)
pytest tests/ -q                        # + integration (downloads weights, needs vmfb)
python tests/test_local_e2e.py          # full offline e2e (expects ~/laya_models)
```

Snake is a **test/benchmark harness** only:

```bash
laya-snake --headless --steps 50
laya-snake --backend planner --headless
laya-snake export recording.jsonl --output out.mp4
pytest tests/test_snake.py -q
```

## Layout

- `laya/` — official package + `vulkan.py` hybrid encoder + `Agent` device resolution
- `snake/` — demo/benchmark harness (`laya-snake`)
- `scripts/` — `export_vulkan.py`, `check_vulkan.py`, `fix_onnx.py`
- `tests/` — pytest suite, fixtures, upstream script-style checks

## License

Apache-2.0 (see `LICENSE`). Upstream: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).
