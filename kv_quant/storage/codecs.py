"""Small lossless codec wrappers used by persisted KV experiments."""

from __future__ import annotations

from dataclasses import dataclass


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
