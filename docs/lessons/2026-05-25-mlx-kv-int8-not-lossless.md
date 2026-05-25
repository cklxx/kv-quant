# 2026-05-25 - mlx-lm INT8 KV was not a lossless floor

## Trigger

T1 expected a pure-attention small model to make `QuantizedKVCache` produce
interpretable footprint deltas while preserving >=99% mean token agreement at
INT8 g64. The footprint part held; the correctness part did not.

## What was tried

- Switched the candidate baseline from hybrid
  `mlx-community/Qwen3.5-0.8B-MLX-4bit` to pure-attention
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16`.
- Confirmed mlx-lm's `qwen2.py` module has no `ssm`, `mamba`, or
  `create_ssm_mask` references.
- Ran the full Qwen2.5 baseline shape: 20 prompts x 64 new tokens.
- Re-ran the shorter bootstrap-comparable shape: 3 prompts x 16 new tokens.
- Tried a second pure-attention model,
  `mlx-community/Llama-3.2-1B-Instruct-bf16`, at 3 prompts x 16 new tokens.
- Checked mlx-lm's `generate_step`, `maybe_quantize_kv_cache`, and
  `QuantizedKVCache` source. The harness is using the public kv_bits,
  kv_group_size, and quantized_kv_start knobs as intended.

## Evidence

Qwen2.5 full shape:

| Config | Mean agreement | Cache vs FP | Decode tok/s |
| --- | ---: | ---: | ---: |
| `baseline_fp` | 100.00% | 1.000x | 202.7 |
| `int8_g64` | 37.81% | 0.531x | 176.9 |
| `int4_g64` | 6.02% | 0.281x | 182.5 |
| `int4_g32` | 13.91% | 0.312x | 186.4 |

Qwen2.5 short smoke: `int8_g64` mean agreement 81.25%.
Llama-3.2 short smoke: `int8_g64` mean agreement 33.33%.

## Lesson

Pure-attention model selection fixes cache-byte interpretability, but it does
not make mlx-lm's built-in INT8 KV cache a correctness-preserving floor under
strict greedy self-generation. Do not start the M1 bits x group-size sweep
from this implementation as if INT8 were validated.

The next task should isolate whether the gap comes from the quantized cache
format itself, groupwise scaling, the self-generation agreement metric, or a
model-specific sensitivity. Until then, T1 remains a negative first
experiment rather than a closed M0 baseline.
