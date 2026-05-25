# 2026-05-25 - KV-quant MLX baseline (Qwen2.5-0.5B, pure attention)

## Goal

Replace the initial Qwen3.5 hybrid-cache smoke baseline with a pure-attention
small model so KV-only quantization produces interpretable cache-byte deltas.
This run answers whether mlx-lm's built-in `QuantizedKVCache` is good enough
to serve as the M0 baseline for the M1 bits x group-size sweep.

## Setup

- Model + revision: `mlx-community/Qwen2.5-0.5B-Instruct-bf16`.
- Architecture check: mlx-lm routes Qwen2.5 through
  `/opt/homebrew/lib/python3.11/site-packages/mlx_lm/models/qwen2.py`;
  that module contains no `ssm`, `mamba`, or `create_ssm_mask` references.
- SKU: Apple M4 Pro, 14 CPU cores, 48 GB unified memory.
- Software: Python 3.11.14, MLX 0.31.1, mlx-lm 0.31.2.
- Shapes: 20 prompts, greedy decode, 64 new tokens.
- Reference baseline: mlx-lm FP/BF16 `KVCache` on the same model, prompts,
  and decode length.

## Method

`experiments/kv_mlx_baseline.py` loads the model once, runs each prompt with a
fresh cache, and compares each quantized cache config against the FP cache
output from the same prompt. The cache configs are `baseline_fp`, `int8_g64`,
`int4_g64`, and `int4_g32`. Cache bytes are counted by recursively walking
`cache.state` instead of using `QuantizedKVCache.nbytes`, which is broken in
mlx-lm 0.31.2 because it references `tree_reduce` without importing it.

I also ran the same 3 prompt x 16 token smoke shape on
`mlx-community/Llama-3.2-1B-Instruct-bf16`. Its mlx-lm `llama.py` module is
also pure attention, but INT8 agreement was lower than Qwen2.5, so it was not
promoted to the baseline.

## Results

| Config | Correctness: mean token agreement | Perfect match rate | Footprint: mean cache bytes | Footprint vs FP | Performance: mean decode tok/s | Decode vs FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_fp` | 100.00% | 100.00% | 980,582.4 | 1.000x | 202.7 | 1.000x |
| `int8_g64` | 37.81% | 5.00% | 520,934.4 | 0.531x | 176.9 | 0.873x |
| `int4_g64` | 6.02% | 0.00% | 275,788.8 | 0.281x | 182.5 | 0.901x |
| `int4_g32` | 13.91% | 0.00% | 306,432.0 | 0.312x | 186.4 | 0.920x |

- Correctness: `int8_g64` fails the T1 acceptance bar of >=99% mean token
  agreement. The result is not close enough to treat as noise.
- Footprint: the pure-attention model fixes the Qwen3.5 methodology problem.
  INT8 is 53.1% of FP cache bytes, and INT4 g64 is 28.1% of FP cache bytes.
- Performance: decode throughput regresses by 8.0-12.7% versus FP on this
  Apple M4 Pro run.

The first experiment therefore does not close M0. It validates the pure-
attention footprint measurement, but falsifies the assumption that mlx-lm's
off-the-shelf INT8 KV cache is a >=99% token-agreement floor under strict
greedy self-generation.

## Repro

`python experiments/kv_mlx_baseline.py`

Raw data:
`docs/experiments/data/kv_mlx_baseline_qwen2_5_0_5b_instruct_bf16.json`

## Notes

- The previous Qwen3.5 bootstrap data remains at
  `docs/experiments/data/kv_mlx_baseline.json` because that model is useful
  for future hybrid SSM + attention cache work.
- A 3 prompt x 16 token Qwen2.5 smoke run still failed the T1 correctness bar:
  INT8 mean agreement was 81.25%, with one prompt diverging at token 7.
- A 3 prompt x 16 token Llama-3.2-1B smoke run failed harder: INT8 mean
  agreement was 33.33%.
- T2 should not start from this cache implementation as a trusted M1 sweep
  baseline. The next useful task is to debug or replace the INT8 KV path until
  it reaches the M0 correctness floor, then rerun this report.
