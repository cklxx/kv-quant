import mlx.core as mx
import pytest

from kv_quant.storage import BrotliCodec, Lz4Codec, RawCodec, ZstdCodec, dump_prompt_cache


class DummyCache:
    def __init__(self):
        self.state = (
            mx.array([[1, 2, 3]], dtype=mx.uint32),
            [mx.array([1.0, 2.0], dtype=mx.float16), None],
        )


@pytest.mark.parametrize(
    "codec",
    [RawCodec(), ZstdCodec(1), Lz4Codec(), BrotliCodec(3)],
    ids=lambda codec: codec.name,
)
def test_lossless_codec_roundtrip(codec):
    blob = dump_prompt_cache([DummyCache()])
    encoded = codec.encode(blob)
    decoded = codec.decode(encoded)
    assert decoded == blob


def test_dump_is_stable_for_same_cache():
    cache = [DummyCache()]
    assert dump_prompt_cache(cache) == dump_prompt_cache(cache)
