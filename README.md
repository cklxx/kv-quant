# kv-quant

Experiments for KV / weight / activation quantization. Three things we measure
on every algorithm we land:

1. **Correctness** — end-to-end inference matches an FP16/BF16 reference within
   a stated tolerance (logits MSE, output token agreement, downstream task
   score, PPL).
2. **Footprint** — GPU memory at peak, on-disk bytes for persisted KV / weights.
3. **Performance** — quantized GEMM and KV ops hit a published per-SKU
   throughput target, not just "works".

Verification > novelty. Every entry under `docs/experiments/` has all three
numbers or it doesn't land.

## Scope

| Axis | Methods we plan to verify | Primary metric |
|------|---------------------------|----------------|
| KV cache quant | INT8 / INT4 (KIVI, KVQuant), FP8 KV | output-token agreement vs FP16 + HBM bytes/token |
| Weight quant | W4 (AWQ, GPTQ), W8 | PPL on WikiText-2 + on-disk MB |
| Activation quant | SmoothQuant, W8A8 | downstream task score (MMLU subset) |
| KV persistence | tier-aware quant for off-HBM KV | dump bytes/token + reload correctness |
| Quant kernels | Triton + reference CUTLASS for W4A16 / W8A8 / FP8 GEMM, KV dequant | tokens/s vs FP16 baseline on a fixed SKU |

## Layout

```
kv-quant/
├── kv_quant/         python package (modules added as experiments land)
├── docs/
│   ├── scope.md      per-axis plan: methods, metrics, references
│   └── experiments/  dated per-experiment writeups (correctness + footprint + perf)
├── tests/            pytest — smoke + correctness fixtures
└── pyproject.toml
```

## Quickstart

```
pip install -e ".[dev]"
pytest
```

Heavy deps (`torch`, `transformers`, `triton`) install on first use; CPU-only
smoke tests run without them.

## Non-goals

- A new inference engine. Real serving lives in
  [`agent-infer` (ARLE)](https://github.com/cklxx/arle); this repo feeds
  algorithms and kernels back to it once they pass the three-number bar.
- A KV-tier simulator. That is
  [`kvcache-sim`](https://github.com/cklxx/kvcache-sim) — orthogonal.
