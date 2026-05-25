# Scope

Per-axis plan: what we verify, what we measure, what we compare against. Each
landed experiment under `docs/experiments/` cites a reference, reports the three
numbers (correctness · footprint · perf), and links the code that produced them.

The bar: a method is not "supported" until all three numbers exist for at least
one model at one shape on one SKU, and the result is reproducible from a single
command.

---

## 1. KV cache quantization

**Question.** How much HBM per token can we drop while preserving downstream
quality and decode throughput?

**Methods to verify.**
- **INT8 KV** — per-channel scale, baseline.
- **INT4 KV (KIVI-style)** — per-channel for K, per-token for V; mixed precision
  for outlier channels. Ref: KIVI (Liu et al., 2024).
- **FP8 KV (E4M3)** — straight cast, with and without per-tensor scale. Ref:
  vLLM / TRT-LLM FP8 KV.
- **KVQuant non-uniform** — codebook-based 2/3/4-bit. Ref: KVQuant
  (Hooper et al., 2024).

**Metrics.**
- Correctness: output-token agreement (greedy) and logits MSE vs FP16, on a
  fixed 256-prompt eval set; PPL on WikiText-2 for long-context runs.
- Footprint: HBM bytes per (token · layer · head) and total bytes at fixed
  context length.
- Performance: decode tokens/s under continuous batching at fixed batch ×
  seq-len; ratio against FP16 KV baseline on the same SKU.

---

## 2. Weight quantization

**Question.** What's the smallest model artifact that still serves at FP16
quality on standard evals?

**Methods to verify.**
- **W8 symmetric per-channel** — trivial baseline.
- **W4 AWQ** — activation-aware scaling. Ref: AWQ (Lin et al., 2023).
- **W4 GPTQ** — calibration via Hessian inverse. Ref: GPTQ
  (Frantar et al., 2022).
- **W4A16 with group-128 zero-point** — the de-facto serving format.

**Metrics.**
- Correctness: PPL on WikiText-2; MMLU subset (STEM + humanities) zero-shot.
- Footprint: on-disk MB after packing (safetensors); load-time peak HBM.
- Performance: prefill and decode tokens/s through a quantized GEMM kernel
  (see Axis 5).

---

## 3. Activation quantization

**Question.** Can we move activations to INT8 / FP8 without per-layer
fine-tuning?

**Methods to verify.**
- **SmoothQuant W8A8** — migrate outlier scale into weights. Ref: SmoothQuant
  (Xiao et al., 2022).
- **FP8 activations (E4M3)** — straight cast at chosen layers, with per-tensor
  scale calibration.

**Metrics.**
- Correctness: MMLU subset; activation-cosine vs FP16 at every transformer
  block boundary.
- Footprint: not the primary motivation — report for completeness.
- Performance: end-to-end tokens/s with W8A8 GEMM kernel.

---

## 4. KV persistence compression

**Question.** When KV cache spills off HBM (CPU RAM, NVMe, remote tier), how
small can we get the persisted bytes per token while keeping reload-and-resume
bit-equivalent (or measurably close) to in-HBM continuation?

**Methods to verify.**
- Layer-wise INT8 / INT4 dump with per-block scale; lossless codec on top
  (zstd, lz4) to measure compressibility headroom.
- Mixed: top-N layers FP16, rest INT4. Ref: H2O, Scissorhands eviction
  intuition transplanted to compression.
- Page-aligned tier formats that match ARLE's `kv_tier` page geometry.

**Metrics.**
- Correctness: post-reload generation matches the same prompt's in-HBM
  continuation on the next K tokens (greedy-equal or token-disagreement-rate).
- Footprint: bytes per (token · layer) on disk; total dump size for a fixed
  session length.
- Performance: dump and reload bandwidth (MB/s); end-to-end TTFT impact when
  resuming from cold tier.

---

## 5. Quantized compute kernels

**Question.** Do the quantization formats above actually run fast, or are we
trading memory for compute?

**Methods to verify.**
- **W4A16 GEMM** — Triton implementation, compared to a CUTLASS reference and
  a published kernel (e.g. Marlin, machete) when available.
- **W8A8 GEMM** — INT8 tensor-core path; compare against `torch._int_mm` and a
  Triton reimplementation.
- **FP8 GEMM** — `torch._scaled_mm` (E4M3) baseline; custom kernel only if
  there is a clear gap.
- **KV dequant fused with attention** — INT4 / INT8 KV unpack inside the
  attention kernel, not as a separate pass.

**Metrics.**
- Correctness: numerical match to a reference (FP16 GEMM with the dequantized
  operands) within a stated tolerance.
- Footprint: working memory (smem / register pressure) reported from the
  kernel.
- Performance: TFLOPs and tokens/s on a pinned SKU; comparison ratio against
  the FP16 baseline. SKU is part of every result — never report a kernel
  number without the SKU it ran on.

---

## What this repo is not doing

- Training-time quantization (QAT). Out of scope until post-training methods
  are exhausted.
- A new serving runtime. Algorithms and kernels that pass the bar feed back
  to `agent-infer` (ARLE).
- A simulator. Tier modeling lives in `kvcache-sim`.
