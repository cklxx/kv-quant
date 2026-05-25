# 2026-05-25 - Lossless compression on Qwen3.5 persisted cache dumps

## Goal

Measure generic and typed-leaf lossless codecs on persisted Qwen3.5 cache
dumps: compression ratio, encode/decode bandwidth, byte-equal roundtrip, and
theoretical token throughput implied by the measured bytes/token.

## Setup

- Model + revision: `mlx-community/Qwen3.5-9B-MLX-4bit`.
- SKU: Apple M4 Pro, 14 CPU cores, 48 GB unified memory.
- Software: Python 3.11.14, MLX 0.31.1, mlx-lm 0.31.2, zstandard 0.23.0,
  lz4 4.4.5, brotli 1.1.0, blosc2 4.3.3.
- Shapes: 3 prompts x 64 max new tokens, greedy decode.
- Reference baseline: persisted bytes from the current mlx-lm prompt-cache
  state. Codecs are lossless on top of that state; byte-equal roundtrip is the
  correctness criterion.

## Method

`kv_quant.storage.dump.dump_prompt_cache()` walks each cache object's `.state`
tree and emits a canonical one-way byte stream: fixed magic/version, JSON
metadata, then raw MLX array bytes in layer order. The experiment generates
fresh caches for `fp`, `int8_g64`, and `int4_g64`, concatenates the per-prompt
dumps for each KV setting, and times two codec families:

- whole-stream byte codecs: `raw`, `lz4`, `zstd1`, `zstd3`, `zstd9`,
  `brotli5`;
- typed-leaf Blosc2 codecs: `blosc2_lz4_bitshuffle`,
  `blosc2_zstd_shuffle`, `blosc2_zstd_bitshuffle`.

The Blosc2 codecs parse the canonical dump header, compress each array leaf
with its dtype and shape, then decode back to the exact same canonical dump
bytes. This is still a lossless persisted-stream experiment; it just avoids
throwing away type information before compression.

Encode/decode bandwidth is reported as uncompressed MB/s. Theoretical token
throughput is `median codec MB/s / raw dump MB per cache token`; it answers
"if the codec were the bottleneck, how many cache tokens/s could it sustain
for this dump format?"

## Results

| KV state | Codec | Correctness | Footprint ratio | Enc MB/s | Dec MB/s | Theory enc tok/s | Theory dec tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fp | raw | byte-equal | 1.000 | 63,375 | 61,002 | 94,749 | 91,201 |
| fp | lz4 | byte-equal | 1.000 | 4,626 | 31,141 | 6,916 | 46,558 |
| fp | zstd1 | byte-equal | 0.932 | 1,547 | 1,592 | 2,314 | 2,379 |
| fp | zstd3 | byte-equal | 0.932 | 1,494 | 1,604 | 2,234 | 2,397 |
| fp | zstd9 | byte-equal | 0.932 | 1,370 | 1,589 | 2,049 | 2,376 |
| fp | brotli5 | byte-equal | 0.928 | 179 | 241 | 267 | 361 |
| fp | blosc2_lz4_bitshuffle | byte-equal | 0.887 | 2,069 | 2,444 | 3,093 | 3,654 |
| fp | blosc2_zstd_shuffle | byte-equal | 0.846 | 444 | 2,265 | 664 | 3,386 |
| fp | blosc2_zstd_bitshuffle | byte-equal | 0.869 | 1,152 | 2,395 | 1,723 | 3,581 |
| int8_g64 | raw | byte-equal | 1.000 | 62,432 | 61,357 | 95,140 | 93,501 |
| int8_g64 | lz4 | byte-equal | 1.000 | 4,569 | 30,714 | 6,963 | 46,804 |
| int8_g64 | zstd1 | byte-equal | 0.940 | 1,551 | 1,549 | 2,363 | 2,361 |
| int8_g64 | zstd3 | byte-equal | 0.940 | 1,492 | 1,553 | 2,273 | 2,367 |
| int8_g64 | zstd9 | byte-equal | 0.940 | 1,356 | 1,552 | 2,066 | 2,366 |
| int8_g64 | brotli5 | byte-equal | 0.935 | 175 | 235 | 267 | 358 |
| int8_g64 | blosc2_lz4_bitshuffle | byte-equal | 0.893 | 1,748 | 2,034 | 2,664 | 3,100 |
| int8_g64 | blosc2_zstd_shuffle | byte-equal | 0.855 | 489 | 1,817 | 745 | 2,770 |
| int8_g64 | blosc2_zstd_bitshuffle | byte-equal | 0.876 | 1,114 | 1,942 | 1,698 | 2,959 |
| int4_g64 | raw | byte-equal | 1.000 | 63,787 | 63,370 | 98,834 | 98,188 |
| int4_g64 | lz4 | byte-equal | 1.000 | 4,423 | 31,291 | 6,853 | 48,483 |
| int4_g64 | zstd1 | byte-equal | 0.939 | 1,538 | 1,528 | 2,383 | 2,367 |
| int4_g64 | zstd3 | byte-equal | 0.939 | 1,479 | 1,531 | 2,291 | 2,372 |
| int4_g64 | zstd9 | byte-equal | 0.939 | 1,356 | 1,528 | 2,100 | 2,368 |
| int4_g64 | brotli5 | byte-equal | 0.934 | 176 | 234 | 273 | 362 |
| int4_g64 | blosc2_lz4_bitshuffle | byte-equal | 0.891 | 1,734 | 1,999 | 2,687 | 3,097 |
| int4_g64 | blosc2_zstd_shuffle | byte-equal | 0.855 | 482 | 1,830 | 747 | 2,836 |
| int4_g64 | blosc2_zstd_bitshuffle | byte-equal | 0.875 | 1,098 | 1,954 | 1,702 | 3,028 |

Raw dump sizes and generation context:

| KV state | Raw dump MB | Raw bytes/cache token | Generation tok/s | Token agreement vs FP |
| --- | ---: | ---: | ---: | ---: |
| fp | 162.5 | 668,876 | 49.6 | 100.0% |
| int8_g64 | 158.8 | 656,216 | 49.2 | 90.4% |
| int4_g64 | 156.8 | 645,395 | 49.4 | 78.6% |

## Repro

`python experiments/kv_dump_compress.py --model mlx-community/Qwen3.5-9B-MLX-4bit --limit-prompts 3 --max-new-tokens 64 --repeat 5`

Raw data:
`docs/experiments/data/kv_dump_compress_qwen3_5_9b_3x64.json`

## Notes

- The main result is that type information matters. Whole-stream zstd saves
  only 6-7%, while typed-leaf Blosc2 reaches 10.7-15.4% depending on the
  speed/ratio point.
- lz4 is effectively a fast framing/checksum path here, not a compression win:
  footprint stays ~1.000x. `blosc2_lz4_bitshuffle` is the useful fast LZ4
  variant, giving ~0.887-0.893x at ~1.7-2.1 GB/s encode.
- zstd level choice barely matters for whole-stream ratio; `zstd1` remains the
  practical whole-stream point.
- Pareto front: `raw` if no size reduction is needed,
  `blosc2_lz4_bitshuffle` for high-throughput compression, and
  `blosc2_zstd_shuffle` for best transfer size. `brotli5` is dominated by
  Blosc2 on both throughput and ratio in this run.
- The theoretical codec token/s numbers are still far above the measured
  ~49 generation tok/s for all Blosc2 variants on this shape. The lowest
  Blosc2 encode-side theory point is ~664 tok/s on FP with zstd+shuffle.
