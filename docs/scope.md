# Scope

Anything that drops the storage cost of LLM inference is in scope.
Quantization changes how a value is represented. Compression squeezes the
post-quant bitstream. Content reduction skips values entirely.
Cross-request sharing avoids storing the same value twice. The end metric
is identical regardless of which lever moved: bytes per token, bytes on
disk, peak HBM — and whether throughput survived the trade.

Methods compose. A landed experiment under `docs/experiments/` is usually
two or three axes stacked, with correctness · footprint · perf reported
together. A single-axis result is fine for an early ablation but is not the
endpoint.

The bar: a method is not "supported" until correctness · footprint · perf
exist for at least one model at one shape on one SKU, and the result is
reproducible from a single command.

---

## A. Format compression — how each value is represented

### A1. KV cache quantization

**Question.** How much HBM per token can we drop while preserving downstream
quality and decode throughput?

**Methods.** INT8 per-channel (baseline). INT4 KIVI-style (per-channel K,
per-token V, outlier-aware). FP8 E4M3 with per-tensor scale. KVQuant
non-uniform 2/3/4-bit codebooks.

**Metrics.** Output-token agreement (greedy) and logits MSE vs FP16 on a
fixed 256-prompt set. WikiText-2 PPL for long-context. HBM bytes per
(token · layer · head). Decode tokens/s at fixed batch × seq-len ratio
against FP16 KV on the same SKU.

### A2. Weight quantization

**Question.** Smallest model artifact that still serves at FP16 quality.

**Methods.** W8 symmetric per-channel (baseline). W4 AWQ. W4 GPTQ. W4A16
group-128 with zero-point — the de-facto serving format.

**Metrics.** WikiText-2 PPL. MMLU subset (STEM + humanities) zero-shot.
On-disk MB after packing. Load-time peak HBM. Prefill + decode tokens/s
through a quantized GEMM kernel (Axis E).

### A3. Activation quantization

**Question.** Move activations to INT8 / FP8 without per-layer tuning.

**Methods.** SmoothQuant W8A8. FP8 activations (E4M3) with per-tensor scale
calibration.

**Metrics.** MMLU subset. Activation-cosine vs FP16 at every block boundary.
End-to-end tokens/s with W8A8 GEMM.

### A4. Lossless compression on the quant bitstream

**Question.** After quantization, how much further can a generic or
float-aware compressor shrink the bytes — and can decode hide under disk /
network read latency on a tiered reload?

**Methods.**
- *Bit-packing density* — measure waste in naive INT4 packing (4-bit
  values stored in 8-bit lanes) versus true nibble-packed streams.
- *Generic byte compressors* — zstd, lz4, brotli on packed quant blocks.
- *Float-aware compressors* — zfp, fpzip, BLOSC2 + Byteshuffle on FP16 KV
  and on dequant residuals.
- *Codebook + entropy coding* — k-means codebook plus entropy coding of
  residuals.

**Metrics.** Roundtrip bit-equal check (lossless modes). Compression ratio.
Encode MB/s (matters for dump). Decode MB/s (matters for resume); the bar
is decode time ≤ the tier read latency it overlaps with.

---

## B. Content reduction — drop / pick a subset of values

### B1. Eviction

**Question.** How much live KV can we drop without breaking long-context
generation?

**Methods.** H2O (heavy-hitter oracle). Scissorhands (persistence of
importance). StreamingLLM (window + attention sinks).

**Metrics.** Retained-token fraction. Output-token agreement on LongBench
subset. Memory-resident bytes/token at steady state.

### B2. Sparsification

**Question.** Stored values that can be skipped entirely.

**Methods.** Top-k retention per row. Threshold-based block drop. Structured
sparsity (2:4) on KV.

**Metrics.** Kept-fraction vs output-token agreement. Sparse-storage layout
overhead (indices + values vs dense baseline).

### B3. Mixed precision and token merging

**Methods.** Top-N FP16 layers + rest INT4 — find the precision budget
needed for parity. Token merging (ToMe-style) at attention to fuse
redundant tokens.

**Metrics.** Same as A1, with the precision budget itself as the headline
result.

---

## C. Cross-request / cross-block sharing

### C1. Prefix deduplication measurement

Prefix dedup is already a radix-cache feature in real engines. The question
here is empirical: how much duplicate KV does production-shaped traffic
carry, and what is the realistic effective bytes/token at the
tier-storage layer after dedup?

### C2. Identical-block detection beyond prefix

Hash-based detection of repeating mid-sequence KV blocks across requests
(boilerplate prompts, few-shot exemplars).

### C3. Layer-delta and time-delta encoding

**Layer delta.** KV at layer i+1 ≈ KV at layer i + small delta — store the
delta with fewer bits.

**Time delta.** KV at position t+1 ≈ KV at t, particularly for sink
positions and slow-changing channels.

Lossless if entropy-coded, lossy if quantized. Report both.

---

## D. Storage tier formats

### D1. Page-aligned dumps

Match ARLE's `kv_tier` page geometry so dumps don't need re-blocking on
reload. Measure dump bandwidth and resume TTFT.

### D2. Async checkpointing

Overlap quant + compress + dump with ongoing decode. The question is
whether the encode pipeline sustains the token-generation rate; if not,
where the back-pressure surfaces.

### D3. Cold-tier formats

NVMe-optimized layout (sequential reads, no random IOPs on the hot path).
Remote-tier formats (RDMA-friendly, batched). Trade-off: serving-time
read latency vs offline encode cost.

---

## E. Quantized compute kernels

**Question.** Do the quantization formats actually run fast, or are we
trading memory for compute?

**Methods.** W4A16 GEMM in Triton vs a CUTLASS / Marlin reference. W8A8 INT8
tensor-core path vs `torch._int_mm`. FP8 GEMM via `torch._scaled_mm` plus a
custom kernel only if there is a clear gap. KV dequant fused inside the
attention kernel rather than as a separate pass.

**Metrics.** Numerical match to an FP16-on-dequant-operands reference
within a stated tolerance. Working memory (smem / register pressure)
reported from the kernel. TFLOPs and tokens/s on a pinned SKU; every
kernel result includes the SKU it ran on.

---

## What this repo is not doing

- Training-time quantization (QAT). Out of scope until post-training methods
  are exhausted.
- A new serving runtime. Algorithms and kernels that pass the bar feed back
  to `agent-infer` (ARLE).
- A KV-tier simulator. That is `kvcache-sim`.
