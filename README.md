# kv-quant

Experiments for everything that shrinks the storage side of LLM inference:
quantization (KV / weights / activations), lossless and lossy compression,
eviction, sparsification, deduplication, tier-aware layouts — and any
combination thereof. "Storage" here means HBM-resident state, off-HBM
tiers (CPU RAM, NVMe, remote), and on-disk weights.

Three numbers reported for every landed method:

1. **Correctness** — end-to-end inference matches an FP16/BF16 reference
   within a stated tolerance (logits MSE, output-token agreement, PPL,
   downstream task score). Lossless methods must verify bit-equal roundtrip.
2. **Footprint** — peak HBM, on-disk / on-tier bytes, bytes per token in
   the persisted KV cache.
3. **Performance** — quantized + compressed paths still have to deliver
   throughput. "Compresses 4× but halves tokens/s" is a different experiment
   from "compresses 4× and matches tokens/s" — both get reported, never
   conflated.

Verification > novelty. Methods compose, so the interesting question is
usually `INT4 KV + zfp + 25% H2O eviction, all three numbers, one SKU`,
not any single axis in isolation.

## Scope

| Axis | Methods to verify | Primary metric |
|------|-------------------|----------------|
| KV cache quant | INT8 / INT4 (KIVI, KVQuant), FP8 KV | output-token agreement vs FP16, HBM bytes/token |
| Weight quant | W4 (AWQ, GPTQ), W8 | PPL on WikiText-2, on-disk MB |
| Activation quant | SmoothQuant W8A8, FP8 activations | MMLU subset, end-to-end perf |
| Lossless compression | bit-packing density, zstd / lz4 / brotli, zfp / fpzip / BLOSC2 on quant blocks and FP16 | compression ratio, encode + decode MB/s |
| Content reduction | H2O / Scissorhands / StreamingLLM eviction, top-k & threshold sparsity, ToMe-style merging | retained quality vs fraction kept |
| Cross-request sharing | prefix dedup measurement, mid-sequence identical-block detection, layer-delta + time-delta coding | dedup hit rate, effective bytes/token |
| Storage tier formats | page-aligned dumps matching ARLE `kv_tier`, async checkpoint pipeline, NVMe + RDMA layouts | reload TTFT, dump bandwidth, resume throughput |
| Quant kernels | Triton W4A16 / W8A8 / FP8 GEMM, KV dequant fused into attention | TFLOPs and tokens/s vs FP16 baseline |

Axes stack. An experiment that combines four of them is a single entry
under `docs/experiments/` as long as it carries the three numbers.

## Layout

```
kv-quant/
├── kv_quant/         python package (modules added as experiments land)
├── docs/
│   ├── scope.md      per-axis plan: questions, methods, metrics, references
│   └── experiments/  dated per-experiment writeups (correctness + footprint + perf)
├── tests/            pytest — smoke + correctness fixtures
└── pyproject.toml
```

## Quickstart

```
pip install -e ".[dev]"
pytest
```

Heavy deps (`torch`, `transformers`, `triton`, float-aware codecs) install
on first use; CPU-only smoke tests run without them.

## Non-goals

- Training-time quantization (QAT). Out of scope until post-training methods
  are exhausted.
- A new inference engine. Real serving lives in
  [`agent-infer` (ARLE)](https://github.com/cklxx/arle); algorithms and
  kernels that pass the bar feed back to it.
- A KV-tier simulator. That is
  [`kvcache-sim`](https://github.com/cklxx/kvcache-sim) — orthogonal.
