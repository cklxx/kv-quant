# 2026-05-25 — KV-quant MLX harness bootstrap (Qwen3.5-0.8B, hybrid)

## Goal

Validate the measurement harness: load a small MLX-format model, generate
greedy output with FP-baseline + INT8/INT4 KV configs via mlx-lm's
`QuantizedKVCache`, and produce the three numbers (correctness · footprint
· perf) together. This run is **not** the M0 baseline result — it is the
bootstrap that surfaced the model-choice issue documented below.

## Setup

- **Model**: `mlx-community/Qwen3.5-0.8B-MLX-4bit` (weights already 4-bit;
  this run isolates the KV-quant signal on top of fixed-weight precision).
- **SKU**: Apple M4 Pro, MLX 0.31.1, mlx-lm 0.31.2, Python 3.11.14.
- **Shapes**: 3 prompts × 16 new tokens (smoke-sized). Repro with
  `python experiments/kv_mlx_baseline.py --limit-prompts 3
  --max-new-tokens 16`.
- **Reference**: FP-baseline cache (`KVCache` from mlx-lm). All "vs FP"
  ratios are against this run on the same model + prompt.

## Method

`experiments/kv_mlx_baseline.py`. For each of four cache configs —
`baseline_fp`, `int8_g64`, `int4_g64`, `int4_g32` — the script builds a
per-layer cache via `make_prompt_cache(model)` and lets mlx-lm's
`maybe_quantize_kv_cache` swap attention slots in-place (SSM slots are
left untouched because they don't expose `to_quantized`). Greedy sampler
(argmax). Cache bytes are computed by recursively walking `cache.state`
because `QuantizedKVCache.nbytes` is broken in mlx-lm 0.31.2 (calls
`tree_reduce` without importing it).

## Results

```
config           agree  perfect     ttft      decode      cache    vs FP
baseline_fp     100.0%   100.0%     18ms    320.1t/s  19492.0KiB    1.00x
int8_g64         43.8%    33.3%     19ms    314.7t/s  19298.9KiB    0.99x
int4_g64         43.8%    33.3%     52ms    309.9t/s  19195.9KiB    0.98x
int4_g32         56.2%    33.3%     31ms    309.8t/s  19208.8KiB    0.99x
```

- **Correctness**: token agreement collapses from 100% (FP) to ~44% even
  at INT8.
- **Footprint**: cache bytes barely move (≤ 2% reduction even at INT4).
- **Performance**: ≤ 4% decode tok/s regression across all quant configs.

## Problems

The "≤ 2% cache reduction at INT4" number is the headline anomaly. Root
cause is the model, not the harness:

- `mlx-community/Qwen3.5-0.8B-MLX-4bit` is hybrid SSM + attention
  (Qwen3.5 architecture uses interleaved Mamba/attention layers, visible
  in `mlx_lm/models/qwen3_5.py` calling `create_ssm_mask` for SSM-layer
  caches alongside KV caches for attention layers).
- mlx-lm's `to_quantized` is defined on `KVCache` only, not on the SSM
  cache type, so `maybe_quantize_kv_cache` skips SSM slots — correctly.
- The SSM state dominates total cache bytes at this shape, so quantizing
  the attention slots changes total-cache by only the attention fraction.

This is methodologically correct but empirically uninformative — the
harness is reporting accurate numbers about a model where the lever has
limited reach. The correctness collapse compounds: the W4 weights + INT8
KV stack on a small model with mixed cache types is well outside the
zone where INT8 KV is "lossless to within tolerance".

## Learnings

- **Pure-attention small model is the right baseline.** M0's acceptance
  bar (INT8 ≥ 99% agreement, INT4 ≤ 30% of baseline cache bytes) is set
  against a pure-attention model. The next harness run uses
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16` or similar.
- **Hybrid SSM+attention is its own axis.** Quantizing or compressing
  SSM state to match the KV-quant story is a separate research item;
  noted as a candidate future milestone, not a blocker for M0–M2.
- **mlx-lm 0.31.2 has a `QuantizedKVCache.nbytes` bug**
  (uses undefined `tree_reduce`). The harness already walks `.state`
  itself; any agent picking up this code should not rely on the
  property until upstream lands a fix.

## Repro

```bash
cd ~/code/kv-quant
pip install -e ".[dev,mlx]"
python experiments/kv_mlx_baseline.py --limit-prompts 3 --max-new-tokens 16
# raw output at docs/experiments/data/kv_mlx_baseline.json
```

## Status

**Smoke pass — methodology validated.** This entry is not the M0 result.
T1 in `docs/plans/2026-05-25-codex-execution-plan.md` replaces the model
and produces the clean M0 numbers.
