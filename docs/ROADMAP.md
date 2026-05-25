# Roadmap

Nine milestones, ordered by dependency. Each milestone closes when at
least one method on at least one model on at least one SKU has reported
correctness · footprint · perf together, reproducible from a single
command. Methods compose — late milestones stack on earlier ones.

The progression is deliberate: the harness comes first (M0), then the
single most impactful storage lever (KV quant, M1), then everything
that compounds on it (compression M2, eviction M3, sharing M6, tier
formats M7). Weight + activation quant (M4, M5) and compute kernels (M8)
run in parallel because they don't block the KV path.

---

## M0 — Harness validation

**Question.** Does the measurement loop (one prompt set, FP-baseline
vs N alternative cache factories, three numbers per config) actually
produce trustworthy numbers, and does it generalize beyond a single
model?

**Why it's first.** Every later milestone reuses the same harness. If
the harness is wrong, every subsequent result is contaminated.

**Status.** Bootstrap run on `mlx-community/Qwen3.5-0.8B-MLX-4bit`
(2026-05-25) succeeded as a code-path smoke but revealed Qwen3.5 is
SSM/attention hybrid — the cache state is dominated by SSM, so KV-only
quantization moves total cache bytes by ~1%. Methodologically valid;
empirically uninformative.

**Acceptance.** A second harness run on a **pure-attention** small model
(Qwen2.5-0.5B or Llama-3.2-1B in MLX format) shows the expected ~50%
cache reduction at INT8 / ~25% at INT4. Mean token agreement vs the
FP-cache baseline ≥ 99% at INT8. Writeup committed under
`docs/experiments/`.

---

## M1 — KV cache quantization

**Question.** How aggressively can we quantize the KV bitstream and at
what point does correctness fall off the cliff?

**Methods.** INT8 per-channel (baseline). INT4 with multiple group
sizes (32, 64, 128). FP8 E4M3 with per-tensor scale. KIVI-style
per-channel K + per-token V quant. KVQuant non-uniform codebooks at
2/3/4 bits.

**Acceptance.** A sweep across the methods on one model (pure-attention,
≥1B params) at one ctx length, with a single chart showing
bytes/token vs token-agreement. INT8 is lossless to within tolerance;
the chart names the bits/group-size at which a 50%/90%/99%
token-agreement cliff occurs.

---

## M2 — Lossless compression on the quant bitstream

**Question.** Given quantized KV (or quantized weights), how much
further can a generic / float-aware codec shrink the bytes — and does
the encode/decode latency hide under tier-storage latency?

**Methods.**
- Bit-packing density audit — measure waste in naive INT4 packing
  (4-bit values in 8-bit lanes) vs true nibble streams.
- Generic byte compressors: zstd (levels 1, 3, 9), lz4, brotli.
- Float-aware: zfp, fpzip, BLOSC2 + Byteshuffle on FP16 KV and on
  dequant residuals.
- Codebook + entropy: k-means codebook plus entropy coding of residuals.

**Acceptance.** Table for one model × one ctx length × one prompt set:
compression ratio, encode MB/s, decode MB/s for every codec listed.
At least one codec is verified bit-equal roundtrip via a pytest fixture.
The table identifies a Pareto front (no codec strictly dominated).

---

## M3 — Eviction

**Question.** How much live KV can we drop without breaking long-context
generation?

**Methods.** H2O (heavy-hitter oracle). Scissorhands (persistence-of-
importance). StreamingLLM (window + attention sinks).

**Acceptance.** Per method: retained-token fraction vs output-token
agreement on a LongBench subset (≥3 tasks), at a fixed model + ctx
budget. A chart placing the three methods on the same axes; the method
with the best agreement at any given kept-fraction is named.

---

## M4 — Weight quantization

**Question.** Smallest model artifact that still serves at FP16 quality
on standard evals.

**Methods.** W8 symmetric per-channel (baseline). W4 AWQ. W4 GPTQ. W4A16
group-128 with zero-point.

**Acceptance.** Per method: WikiText-2 PPL, MMLU subset (STEM +
humanities) zero-shot, on-disk MB after packing, load-time peak HBM,
prefill + decode tokens/s through a quantized GEMM kernel (M8). One
model, one SKU. Comparison row vs FP16 reference.

---

## M5 — Activation quantization

**Question.** Move activations to INT8 / FP8 without per-layer tuning.

**Methods.** SmoothQuant W8A8. FP8 activations (E4M3) with per-tensor
scale calibration.

**Acceptance.** MMLU subset score, activation-cosine vs FP16 at every
transformer block boundary, end-to-end tokens/s through a W8A8 GEMM
kernel (M8). One model, one SKU.

---

## M6 — Cross-request / cross-block sharing

**Question.** How much duplicate KV does realistic traffic carry, and
what is the effective bytes/token after dedup + delta encoding at the
tier-storage layer?

**Methods.** Prefix-dedup hit-rate measurement on production-shaped
prompt distributions. Hash-based mid-sequence identical-block detection.
Layer-delta and time-delta encoding (lossless via entropy code, lossy
via quant on the delta).

**Acceptance.** A measurement report with: dedup hit rate by prompt
class; mid-sequence-block hash collision rate; layer-delta entropy in
bits/value at one model. One chart of "effective bytes/token vs
sharing technique" stacked on top of M2's compression ratio.

---

## M7 — Storage tier formats

**Question.** Page-aligned dump format that matches ARLE `kv_tier` page
geometry, NVMe-optimized layout for cold tier, RDMA-friendly batched
layout for remote tier.

**Methods.** Page-aligned dumps. Async checkpoint pipeline overlapping
quant + compress + dump with ongoing decode. NVMe sequential-read
layouts. RDMA batched-read layouts.

**Acceptance.** A reload TTFT measurement comparing in-HBM continuation
vs resume-from-tier on cold + warm tiers. Dump bandwidth (MB/s).
Async-pipeline back-pressure characterization: does encode keep up with
token-gen rate, and if not where it stalls.

---

## M8 — Quantized compute kernels

**Question.** Do the quantization formats actually run fast on the
target SKU, or are we trading memory for compute?

**Methods.** W4A16 GEMM in Triton vs CUTLASS / Marlin reference. W8A8
INT8 tensor-core vs `torch._int_mm`. FP8 GEMM via `torch._scaled_mm`
plus a custom kernel only if there is a measured gap. KV dequant fused
inside the attention kernel rather than as a separate pass.

**Acceptance.** Per kernel: numerical match to an FP16-on-dequant-
operands reference within a stated tolerance. Working memory (smem /
register pressure) reported. TFLOPs and tokens/s on a pinned SKU
(every kernel result includes the SKU). Comparison ratio against the
FP16 baseline.

---

## Cross-cutting deliverables (any milestone touches them)

- `kv_quant/metrics/` — correctness + footprint + perf primitives shared
  across harnesses (token-agreement, logits-MSE, PPL, bytes counter,
  prefill/decode timing).
- `tests/` — pytest fixtures: bit-equal roundtrip for lossless codecs;
  small-model end-to-end smoke; cache-bytes accounting against known
  expected values.
- `docs/experiments/data/` — every committed writeup ships its raw JSON
  alongside.

## Out of roadmap

- QAT — out of scope.
- A new serving runtime — that's ARLE.
- KV tier simulation — that's `kvcache-sim`.

## Sequencing notes

- M0 → M1 is hard sequential (harness must be trusted before sweep).
- M1 → M2 is mostly sequential (compression layers on top of a known
  quant format) but a small zstd-on-FP16 baseline can be probed without
  M1 finishing.
- M3, M4, M5, M6, M7, M8 can run in parallel after M0 + M1.
- M7 and M8 depend on at least one M1-class result to give them
  something concrete to dump / compute on.
