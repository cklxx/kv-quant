# 2026-05-25 - Lossless compression on Qwen3.5 persisted cache dumps

## Goal

Measure generic lossless codecs on persisted Qwen3.5 cache dumps: compression
ratio, encode/decode bandwidth, byte-equal roundtrip, and theoretical token
throughput implied by the measured bytes/token.

## Setup

- Model + revision: `mlx-community/Qwen3.5-9B-MLX-4bit`.
- SKU: Apple M4 Pro, 14 CPU cores, 48 GB unified memory.
- Software: Python 3.11.14, MLX 0.31.1, mlx-lm 0.31.2, zstandard 0.23.0,
  lz4 4.4.5, brotli 1.1.0.
- Shapes: 3 prompts x 64 max new tokens, greedy decode.
- Reference baseline: persisted bytes from the current mlx-lm prompt-cache
  state. Codecs are lossless on top of that state; byte-equal roundtrip is the
  correctness criterion.

## Method

`kv_quant.storage.dump.dump_prompt_cache()` walks each cache object's `.state`
tree and emits a canonical one-way byte stream: fixed magic/version, JSON
metadata, then raw MLX array bytes in layer order. The experiment generates
fresh caches for `fp`, `int8_g64`, and `int4_g64`, concatenates the per-prompt
dumps for each KV setting, and times `raw`, `lz4`, `zstd1`, `zstd3`, `zstd9`,
and `brotli5`.

Encode/decode bandwidth is reported as uncompressed MB/s. Theoretical token
throughput is `median codec MB/s / raw dump MB per cache token`; it answers
"if the codec were the bottleneck, how many cache tokens/s could it sustain
for this dump format?"

## Results

| KV state | Codec | Correctness | Footprint ratio | Enc MB/s | Dec MB/s | Theory enc tok/s | Theory dec tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp | raw | byte-equal | 1.000 | 58,863 | 56,663 | 88,002 | 84,713 |
| fp | lz4 | byte-equal | 1.000 | 4,229 | 25,920 | 6,322 | 38,752 |
| fp | zstd1 | byte-equal | 0.932 | 1,491 | 1,477 | 2,229 | 2,208 |
| fp | zstd3 | byte-equal | 0.932 | 1,455 | 1,488 | 2,176 | 2,224 |
| fp | zstd9 | byte-equal | 0.932 | 1,244 | 1,495 | 1,860 | 2,236 |
| fp | brotli5 | byte-equal | 0.928 | 187 | 233 | 279 | 349 |
| int8_g64 | raw | byte-equal | 1.000 | 63,877 | 63,382 | 97,342 | 96,588 |
| int8_g64 | lz4 | byte-equal | 1.000 | 4,325 | 31,019 | 6,591 | 47,269 |
| int8_g64 | zstd1 | byte-equal | 0.940 | 1,515 | 1,492 | 2,309 | 2,273 |
| int8_g64 | zstd3 | byte-equal | 0.940 | 1,472 | 1,501 | 2,243 | 2,287 |
| int8_g64 | zstd9 | byte-equal | 0.940 | 1,271 | 1,505 | 1,936 | 2,294 |
| int8_g64 | brotli5 | byte-equal | 0.935 | 181 | 227 | 276 | 346 |
| int4_g64 | raw | byte-equal | 1.000 | 60,114 | 58,078 | 93,144 | 89,989 |
| int4_g64 | lz4 | byte-equal | 1.000 | 4,211 | 28,694 | 6,525 | 44,459 |
| int4_g64 | zstd1 | byte-equal | 0.939 | 1,489 | 1,459 | 2,307 | 2,260 |
| int4_g64 | zstd3 | byte-equal | 0.939 | 1,421 | 1,471 | 2,201 | 2,280 |
| int4_g64 | zstd9 | byte-equal | 0.939 | 1,212 | 1,471 | 1,877 | 2,279 |
| int4_g64 | brotli5 | byte-equal | 0.934 | 183 | 228 | 284 | 353 |

Raw dump sizes and generation context:

| KV state | Raw dump MB | Raw bytes/cache token | Generation tok/s | Token agreement vs FP |
| --- | ---: | ---: | ---: | ---: |
| fp | 162.5 | 668,876 | 46.2 | 100.0% |
| int8_g64 | 158.8 | 656,216 | 47.4 | 90.4% |
| int4_g64 | 156.8 | 645,395 | 44.5 | 78.6% |

## Repro

`python experiments/kv_dump_compress.py --model mlx-community/Qwen3.5-9B-MLX-4bit --limit-prompts 3 --max-new-tokens 64 --repeat 5`

Raw data:
`docs/experiments/data/kv_dump_compress_qwen3_5_9b_3x64.json`

## Notes

- The main result is negative for generic lossless codecs: Qwen3.5 persisted
  cache state is close to incompressible. zstd saves only 6-7%; brotli saves
  about 6.5-7.2% but encode/decode throughput drops to ~180-230 MB/s.
- lz4 is effectively a fast framing/checksum path here, not a compression win:
  footprint stays ~1.000x.
- zstd level choice barely matters for ratio; `zstd1` is the practical point
  for dump-time encode. Higher levels trade encode bandwidth for negligible
  byte savings.
- Pareto front: `raw` if no size reduction is needed, `zstd1` for meaningful
  compression at high throughput, and `brotli5` only if the last ~0.5% size
  reduction is worth a 6-8x throughput hit.
- The theoretical codec token/s numbers are far above the measured 44-47
  generation tok/s for zstd/lz4/raw on this shape. Brotli still clears the
  generation rate, but with much less margin.
