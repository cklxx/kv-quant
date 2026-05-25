"""Canonical byte dump for persisted KV-cache experiments.

`dump_prompt_cache(cache)` walks each cache object's `.state` tree in layer
order and emits:

1. a fixed magic/version header,
2. a little-endian JSON metadata length,
3. UTF-8 JSON metadata for every leaf, and
4. raw MLX array bytes concatenated in traversal order.

The format is intentionally one-way for now. It is a stable byte stream for
lossless codec experiments, not a serving reload format. MLX arrays are viewed
as `uint8` before copying so bfloat16 payloads stay byte-exact.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

MAGIC = b"KVQDUMP\x01"


@dataclass
class _ArrayLeaf:
    path: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    payload: bytes


def _array_payload(array: mx.array) -> bytes:
    mx.eval(array)
    bytes_view = array.view(mx.uint8)
    return np.asarray(bytes_view).tobytes(order="C")


def _walk(obj: Any, path: str, leaves: list[_ArrayLeaf], metadata: list[dict]) -> None:
    if isinstance(obj, mx.array):
        payload = _array_payload(obj)
        leaves.append(
            _ArrayLeaf(
                path=path,
                dtype=str(obj.dtype),
                shape=tuple(int(x) for x in obj.shape),
                nbytes=int(obj.nbytes),
                payload=payload,
            )
        )
        metadata.append(
            {
                "path": path,
                "kind": "array",
                "dtype": str(obj.dtype),
                "shape": [int(x) for x in obj.shape],
                "nbytes": int(obj.nbytes),
            }
        )
    elif obj is None:
        metadata.append({"path": path, "kind": "none"})
    elif isinstance(obj, (list, tuple)):
        metadata.append({"path": path, "kind": type(obj).__name__, "len": len(obj)})
        for i, child in enumerate(obj):
            _walk(child, f"{path}/{i}", leaves, metadata)
    else:
        metadata.append({"path": path, "kind": type(obj).__name__, "repr": repr(obj)})


def dump_prompt_cache(prompt_cache: list[Any]) -> bytes:
    """Return a canonical bytes blob for the current prompt-cache state."""

    leaves: list[_ArrayLeaf] = []
    entries: list[dict] = []
    for layer_idx, cache in enumerate(prompt_cache):
        state = getattr(cache, "state", None)
        entries.append(
            {
                "path": f"layer/{layer_idx}",
                "kind": "cache",
                "class": type(cache).__name__,
            }
        )
        _walk(state, f"layer/{layer_idx}/state", leaves, entries)

    offset = 0
    array_entries = [entry for entry in entries if entry["kind"] == "array"]
    for leaf, entry in zip(leaves, array_entries, strict=True):
        if leaf.nbytes != len(leaf.payload):
            raise ValueError(f"{leaf.path} nbytes mismatch: {leaf.nbytes} != {len(leaf.payload)}")
        entry["payload_offset"] = offset
        entry["payload_nbytes"] = len(leaf.payload)
        offset += len(leaf.payload)

    metadata = {
        "format": "kv_quant.prompt_cache_dump",
        "version": 1,
        "payload_nbytes": offset,
        "entries": entries,
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = b"".join(leaf.payload for leaf in leaves)
    return MAGIC + struct.pack("<Q", len(header)) + header + payload
