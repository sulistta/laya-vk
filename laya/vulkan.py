"""Vulkan encoder backend: IREE-compiled EncCore + CPU-torch embedding/head.

Fixed shape buckets only (adreno SPIR-V target, no dynamic shapes). Outside the
envelope the hybrid encoder transparently falls back to the stock torch encoder
for that call. The token-embedding table stays on CPU torch so the 786MB gather
table never lands in VRAM.

Artifacts: models/iree/laya_core_b{B}l{L}_vulkan.vmfb (see scripts/export_vulkan.py).
Override search path with LAYA_VULKAN_VMFB_DIR.

Graph instances are LRU-cached (default 1): each vmfb is ~400MB resident in VRAM,
and loading all buckets at once exceeds a 4GB card.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

VMFB_RE = re.compile(r"laya_core_b(\d+)l(\d+)(?:k\d+)?_vulkan\.vmfb$")
DEFAULT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1, 128),
    (2, 128),
    (3, 128),
    (4, 128),
    (5, 128),
    (7, 128),
    (1, 512),
)
MAX_LOADED_GRAPHS = int(os.environ.get("LAYA_VULKAN_MAX_GRAPHS", "1"))


def discover_vmfbs(directory: Optional[os.PathLike] = None) -> Dict[Tuple[int, int], Path]:
    """Map (batch, length) → vmfb path from a directory of export artifacts."""
    if directory is None:
        env = os.environ.get("LAYA_VULKAN_VMFB_DIR")
        roots = [Path(env)] if env else []
        roots += [
            Path.cwd() / "models" / "iree",
            Path(__file__).resolve().parents[1] / "models" / "iree",
        ]
    else:
        roots = [Path(directory)]
    profiles: Dict[Tuple[int, int], Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("laya_core_b*l*_vulkan.vmfb")):
            match = VMFB_RE.match(path.name)
            if match:
                profiles[(int(match.group(1)), int(match.group(2)))] = path
    return profiles


class HybridVulkanEncoder(nn.Module):
    """Drop-in for DecisionModel.encoder: IREE when shapes fit a bucket, else torch."""

    def __init__(self, torch_encoder: nn.Module, profiles: Dict[Tuple[int, int], Path],
                 max_graphs: int = MAX_LOADED_GRAPHS):
        super().__init__()
        if not profiles:
            raise ValueError("no Vulkan vmfb profiles found (build with scripts/export_vulkan.py)")
        self.torch_encoder = torch_encoder
        self.profiles = dict(profiles)
        self._graphs: "OrderedDict[Tuple[int, int], object]" = OrderedDict()
        self._max_graphs = max(1, max_graphs)
        self.hits = 0
        self.fallbacks = 0

    @property
    def config(self):
        return self.torch_encoder.config

    @property
    def embeddings(self):
        return self.torch_encoder.embeddings

    def available_shapes(self):
        return sorted(self.profiles)

    def pick_bucket(self, batch: int, length: int) -> Optional[Tuple[int, int]]:
        best = None
        for (b, l) in self.profiles:
            if b >= batch and l >= length:
                if best is None or b < best[0] or (b == best[0] and l < best[1]):
                    best = (b, l)
        return best

    def _graph(self, bucket: Tuple[int, int]):
        if bucket in self._graphs:
            self._graphs.move_to_end(bucket)
            return self._graphs[bucket]
        import iree.runtime as iree_rt

        invoker = iree_rt.load_vm_flatbuffer_file(str(self.profiles[bucket]), driver="vulkan")
        graph = invoker.main_graph
        while len(self._graphs) >= self._max_graphs:
            self._graphs.popitem(last=False)
        self._graphs[bucket] = graph
        return graph

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None or attention_mask is None:
            return self.torch_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        batch, length = input_ids.shape[0], input_ids.shape[1]
        bucket = self.pick_bucket(batch, length)
        if bucket is None:
            self.fallbacks += 1
            return self.torch_encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        b, l = bucket
        pad = int(getattr(self, "pad_id", 0) or 0)
        if (batch, length) == (b, l):
            ids, att = input_ids, attention_mask
        else:
            ids = torch.full((b, l), pad, dtype=input_ids.dtype)
            att = torch.zeros((b, l), dtype=attention_mask.dtype)
            ids[:batch, :length] = input_ids
            att[:batch, :length] = attention_mask

        with torch.no_grad():
            embeds = self.torch_encoder.embeddings.tok_embeddings(ids)
            graph = self._graph((b, l))
            mask_i32 = att.to(torch.int32).cpu().numpy()
            hidden = graph(
                embeds.detach().cpu().numpy().astype(np.float32),
                mask_i32,
            )
        hidden = np.asarray(hidden)
        if hidden.ndim == 2:
            hidden = hidden.reshape(b, l, -1)
        hidden_t = torch.from_numpy(np.ascontiguousarray(hidden))
        if (batch, length) != (b, l):
            hidden_t = hidden_t[:batch, :length]
        self.hits += 1
        return SimpleNamespace(last_hidden_state=hidden_t.to(input_ids.device))


def attach_vulkan_encoder(model: nn.Module, profiles: Optional[Dict[Tuple[int, int], Path]] = None,
                          warmup: bool = True, pad_id: int = 0) -> HybridVulkanEncoder:
    """Swap model.encoder for a hybrid Vulkan/torch encoder. Returns the handle."""
    if profiles is None:
        profiles = discover_vmfbs()
    hybrid = HybridVulkanEncoder(model.encoder, profiles)
    hybrid.pad_id = pad_id
    hybrid.eval()
    model.encoder = hybrid
    if warmup and profiles:
        # Warm only the smallest profile — loading every vmfb at once OOMs 4GB cards.
        smallest = min(profiles)
        try:
            b, l = smallest
            hybrid._graph(smallest)(
                np.zeros((b, l, hybrid.config.hidden_size), dtype=np.float32),
                np.full((b, l), pad_id, dtype=np.int32),
            )
        except Exception as error:
            print(f"[laya] Vulkan warmup warning: {error}", flush=True)
    return hybrid


def vulkan_available() -> bool:
    try:
        import iree.runtime as iree_rt
    except ModuleNotFoundError:
        return False
    try:
        drivers = iree_rt.query_available_drivers()
    except Exception:
        return False
    if "vulkan" not in drivers:
        return False
    return bool(discover_vmfbs())
