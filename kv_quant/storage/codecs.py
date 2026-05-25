"""Small lossless codec wrappers used by persisted KV experiments."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .dump import MAGIC


@dataclass(frozen=True)
class CodecResult:
    name: str
    encoded: bytes


class RawCodec:
    name = "raw"
    lossless = True

    def encode(self, blob: bytes) -> bytes:
        return memoryview(blob).tobytes()

    def decode(self, blob: bytes) -> bytes:
        return memoryview(blob).tobytes()


class ZstdCodec:
    lossless = True

    def __init__(self, level: int = 3):
        self.level = level
        self.name = f"zstd{level}"

    def encode(self, blob: bytes) -> bytes:
        import zstandard as zstd

        return zstd.ZstdCompressor(level=self.level).compress(blob)

    def decode(self, blob: bytes) -> bytes:
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(blob)


class Lz4Codec:
    name = "lz4"
    lossless = True

    def encode(self, blob: bytes) -> bytes:
        import lz4.frame

        return lz4.frame.compress(blob)

    def decode(self, blob: bytes) -> bytes:
        import lz4.frame

        return lz4.frame.decompress(blob)


class BrotliCodec:
    lossless = True

    def __init__(self, quality: int = 5):
        self.quality = quality
        self.name = f"brotli{quality}"

    def encode(self, blob: bytes) -> bytes:
        import brotli

        return brotli.compress(blob, quality=self.quality)

    def decode(self, blob: bytes) -> bytes:
        import brotli

        return brotli.decompress(blob)


class Blosc2LeafCodec:
    """Compress each typed array leaf in a canonical KV dump with Blosc2.

    Whole-dump byte compressors see the stream as untyped bytes. This codec
    parses the canonical dump header, keeps that metadata uncompressed, and
    applies Blosc2 to each array payload using the payload's logical dtype.
    `decode()` reconstructs the original canonical dump byte-for-byte.
    """

    package_magic = b"KVQBLSC\x01"
    lossless = True

    def __init__(self, codec: str = "zstd", filter_: str = "shuffle", clevel: int = 5):
        self.codec = codec
        self.filter_ = filter_
        self.clevel = clevel
        self.name = f"blosc2_{codec}_{filter_}"

    @staticmethod
    def _numpy_dtype(dtype: str):
        if dtype == "mlx.core.bfloat16":
            return np.uint16
        if dtype == "mlx.core.float16":
            return np.float16
        if dtype == "mlx.core.float32":
            return np.float32
        if dtype == "mlx.core.float64":
            return np.float64
        if dtype == "mlx.core.uint32":
            return np.uint32
        if dtype == "mlx.core.uint16":
            return np.uint16
        if dtype == "mlx.core.uint8":
            return np.uint8
        if dtype == "mlx.core.int32":
            return np.int32
        if dtype == "mlx.core.int16":
            return np.int16
        if dtype == "mlx.core.int8":
            return np.int8
        return np.uint8

    @staticmethod
    def _parse_dump_segments(blob: bytes) -> list[tuple[bytes, memoryview, list[dict]]]:
        segments = []
        offset = 0
        while offset < len(blob):
            if not blob.startswith(MAGIC, offset):
                raise ValueError("not a kv_quant prompt-cache dump stream")
            header_len_offset = offset + len(MAGIC)
            header_len = struct.unpack(
                "<Q", blob[header_len_offset : header_len_offset + 8]
            )[0]
            header_start = header_len_offset + 8
            payload_start = header_start + header_len
            metadata = json.loads(blob[header_start:payload_start])
            payload_nbytes = int(metadata["payload_nbytes"])
            payload_end = payload_start + payload_nbytes
            array_entries = [
                entry for entry in metadata["entries"] if entry.get("kind") == "array"
            ]
            array_entries.sort(key=lambda entry: entry["payload_offset"])
            segments.append(
                (
                    blob[offset:payload_start],
                    memoryview(blob)[payload_start:payload_end],
                    array_entries,
                )
            )
            offset = payload_end
        return segments

    def _blosc_codec(self):
        import blosc2

        return getattr(blosc2.Codec, self.codec.upper())

    def _blosc_filter(self):
        import blosc2

        return getattr(blosc2.Filter, self.filter_.upper())

    def encode(self, blob: bytes) -> bytes:
        import blosc2

        segments = self._parse_dump_segments(blob)
        chunks: list[bytes] = []
        manifest_segments: list[dict] = []
        for prefix, payload, array_entries in segments:
            manifest_chunks: list[dict] = []
            for entry in array_entries:
                offset = int(entry["payload_offset"])
                nbytes = int(entry["payload_nbytes"])
                raw = payload[offset : offset + nbytes]
                dtype = self._numpy_dtype(entry["dtype"])
                array = np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])
                compressed = blosc2.compress2(
                    array,
                    codec=self._blosc_codec(),
                    clevel=self.clevel,
                    filters=[self._blosc_filter()],
                )
                chunks.append(compressed)
                manifest_chunks.append(
                    {
                        "compressed_nbytes": len(compressed),
                        "raw_nbytes": nbytes,
                    }
                )
            manifest_segments.append(
                {
                    "prefix": prefix,
                    "chunks": manifest_chunks,
                }
            )

        manifest = {
            "format": "kv_quant.blosc2_leaf_dump",
            "version": 1,
            "codec": self.codec,
            "filter": self.filter_,
            "clevel": self.clevel,
            "segments": [
                {
                    "prefix_nbytes": len(segment["prefix"]),
                    "chunks": segment["chunks"],
                }
                for segment in manifest_segments
            ],
        }
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return (
            self.package_magic
            + struct.pack("<Q", len(manifest_bytes))
            + manifest_bytes
            + b"".join(segment["prefix"] for segment in manifest_segments)
            + b"".join(chunks)
        )

    def decode(self, blob: bytes) -> bytes:
        import blosc2

        if not blob.startswith(self.package_magic):
            raise ValueError("not a kv_quant Blosc2 leaf dump")
        manifest_len_offset = len(self.package_magic)
        manifest_len = struct.unpack(
            "<Q", blob[manifest_len_offset : manifest_len_offset + 8]
        )[0]
        manifest_start = manifest_len_offset + 8
        prefix_start = manifest_start + manifest_len
        manifest = json.loads(blob[manifest_start:prefix_start])
        offset = prefix_start
        prefixes = []
        for segment in manifest["segments"]:
            prefix_nbytes = int(segment["prefix_nbytes"])
            prefixes.append(blob[offset : offset + prefix_nbytes])
            offset += prefix_nbytes

        pieces = []
        for prefix, segment in zip(prefixes, manifest["segments"], strict=True):
            pieces.append(prefix)
            for chunk_meta in segment["chunks"]:
                compressed_nbytes = int(chunk_meta["compressed_nbytes"])
                raw_nbytes = int(chunk_meta["raw_nbytes"])
                compressed = blob[offset : offset + compressed_nbytes]
                raw = blosc2.decompress2(compressed)
                if len(raw) != raw_nbytes:
                    raise ValueError(
                        f"Blosc2 leaf size mismatch: {len(raw)} != {raw_nbytes}"
                    )
                pieces.append(raw)
                offset += compressed_nbytes
        if offset != len(blob):
            raise ValueError("trailing bytes in Blosc2 leaf package")
        return b"".join(pieces)
