# 2026-05-25 — Codex execution plan

Concrete, ordered tasks Codex can pick up without further direction.
Each task = **Goal · Files · Constraints · Acceptance · Output**.

Tasks cover M0 (harness validation) and M1 (KV quant sweep) plus the
opening probes of M2 (lossless compression). Later milestones generate
their own plans once the first three are landed.

Before starting any task, read `AGENTS.md` and the relevant section of
`docs/scope.md` + `docs/ROADMAP.md`. Verification phase is non-optional.

---

## Setup (run once)

```
cd ~/code/kv-quant
pip install -e ".[dev,mlx]"        # adds mlx + mlx-lm
pytest                              # smoke
python experiments/kv_mlx_baseline.py --limit-prompts 3 --max-new-tokens 16
# the smoke run produces docs/experiments/data/kv_mlx_baseline.json
```

If the second command's "decode tok/s" looks plausible (≥ 100 on M-series,
≥ 500 on a recent CUDA box), the harness loads. Proceed.

---

## T1 (M0) — switch the harness baseline to a pure-attention model

**Goal.** The bootstrap smoke runs against Qwen3.5-0.8B which is hybrid
SSM/attention; KV quant only touches the attention slots, so quantized
total cache bytes change by ~1%. That's not a wrong measurement, it's
the wrong baseline. Replace with a pure-attention small model so M1's
sweep produces interpretable cache-byte deltas.

**Files.**
- `experiments/kv_mlx_baseline.py` — change `DEFAULT_MODEL`. Candidate:
  `mlx-community/Qwen2.5-0.5B-Instruct-bf16` (or `-4bit` if bf16 isn't
  available; document the choice).
- `docs/experiments/2026-05-25-kv-mlx-baseline.md` — write the result.

**Constraints.**
- Do **not** delete the Qwen3.5-0.8B path or its data file; leave them
  for the future hybrid-cache experiment.
- Keep the same prompt set + max_new_tokens to enable cross-comparison.
- Confirm the chosen model is pure-attention by checking
  `from mlx_lm.models import <module>` doesn't reference SSM / Mamba.

**Acceptance.**
- INT8 g64 mean token-agreement ≥ 99% (this is the lossless-floor check;
  if it doesn't hold, INT8 is wrong, not the model).
- INT8 g64 mean cache bytes ≤ 55% of baseline.
- INT4 g64 mean cache bytes ≤ 30% of baseline.
- Writeup follows the template in `docs/experiments/README.md`.

**Output.**
- Commit: `feat(harness): switch baseline model to <name>; clean M0 result`
- Files added: `docs/experiments/2026-05-25-kv-mlx-baseline.md`,
  `docs/experiments/data/kv_mlx_baseline_<model-slug>.json`.

---

## T2 (M1) — KV-quant sweep across bits × group_size

**Goal.** Map the (bits, group_size) plane: at what bits does
correctness collapse, and how much does shrinking the group from 128 →
32 buy back?

**Files.**
- `experiments/kv_mlx_sweep_bits.py` — new script, copy of the baseline
  with a `CONFIGS` matrix: bits ∈ {8, 6, 4, 3, 2}, group_size ∈
  {32, 64, 128}. Note that mlx-lm `QuantizedKVCache` may only support a
  subset; for unsupported combos, log + skip rather than crash.
- `docs/experiments/2026-05-25-kv-mlx-sweep-bits.md` — writeup with a
  table (rows = bits, cols = group_size; cells = mean token-agreement /
  bytes-vs-FP).

**Constraints.**
- Same prompt set and max_new_tokens as T1.
- Each config runs from a fresh cache (no shared state across configs).
- If a config OOMs or errors, record the failure mode and continue.

**Acceptance.**
- A table whose rows + cols are the matrix above, all cells filled in
  (or marked `n/a` with reason).
- One sentence in the writeup naming the bits/group-size at which mean
  agreement drops below 99% and below 90%.
- Reproducer: `python experiments/kv_mlx_sweep_bits.py`.

**Output.**
- Commit: `feat(kv): bits×group_size sweep on <model>`
- Files: script + writeup + raw JSON.

---

## T3 (M1) — long-context KV quant correctness

**Goal.** Short prompts under-represent the KV-quant correctness risk:
errors compound over time. Re-run the best (INT4) config from T2 on
prompts that drive the cache to ≥ 4K tokens.

**Files.**
- `experiments/kv_mlx_long_ctx.py` — generate long-context inputs (e.g.
  concatenated WikiText paragraphs or a fixed code file) up to 4K
  prompt tokens, then generate 256 new tokens.
- `docs/experiments/2026-05-25-kv-mlx-long-ctx.md`.

**Constraints.**
- Use the model from T1.
- Use 5–10 long inputs; greedy decode; record divergence position not
  just mean agreement (per-step divergence is the interesting signal at
  4K).

**Acceptance.**
- Plot or table of "fraction of runs still in agreement vs decode
  position" for FP / INT8 / INT4. The position at which INT4 drops
  below 90% is named.
- Repro command in the writeup.

**Output.**
- Commit: `feat(kv): long-context correctness sweep`
- Files: script + writeup + raw JSON.

---

## T4 (M2) — lossless compression on persisted KV bitstream

**Goal.** After we quantize, how much do generic and float-aware codecs
shrink the bytes when we *dump* the cache (the persistence axis), and
what is encode/decode bandwidth?

**Files.**
- `kv_quant/storage/dump.py` — function that takes a `prompt_cache`
  list and returns a `bytes` blob (one canonical layout — document
  it in a docstring).
- `kv_quant/storage/codecs.py` — codecs as small wrappers: `RawCodec`,
  `ZstdCodec(level)`, `Lz4Codec`, `BrotliCodec`. Each has
  `encode(blob) -> bytes` and `decode(bytes) -> blob`.
- `tests/test_dump_roundtrip.py` — for each lossless codec, verify
  encode→decode reproduces the original blob byte-equal.
- `experiments/kv_dump_compress.py` — for one model (T1's),
  for each (kv_bits, codec) combination, report: bytes after quant,
  bytes after codec, compression ratio, encode MB/s, decode MB/s.
- `docs/experiments/2026-05-25-kv-dump-compress.md`.

**Constraints.**
- The codecs are **lossless on top of already-quantized data** — do not
  combine with further lossy compression in this task.
- Pin codec versions in `pyproject.toml`'s `[storage]` extra.
- Encode/decode timing uses `time.perf_counter`, averaged over 5 runs
  per codec; report median + p95.

**Acceptance.**
- All roundtrip tests pass.
- Table in the writeup with one row per (kv_bits ∈ {fp, 8, 4}, codec).
- A "Pareto front" paragraph: which (kv_bits, codec) pairs are not
  strictly dominated.

**Output.**
- Commit: `feat(compress): lossless codecs on quantized KV dumps`
- Files: 4 source files + writeup + raw JSON.

---

## T5 (M2) — float-aware compression on FP16 KV (no quant)

**Goal.** Probe whether float-aware codecs (zfp, fpzip, BLOSC2 +
Byteshuffle) on raw FP16 KV beat "INT8 + zstd". This is the alternative
hypothesis to T4 + INT8: instead of quantize-then-compress, just
compress the floats directly.

**Files.**
- `kv_quant/storage/codecs.py` — extend with `ZfpCodec(precision|rate)`,
  `BloscCodec(filter='bitshuffle')`. Mark each lossless / lossy
  explicitly via a class attribute.
- `experiments/kv_fp16_compress.py` — analogous to T4 but operating on
  the FP16 baseline cache.
- `docs/experiments/2026-05-25-kv-fp16-compress.md`.

**Constraints.**
- Lossy codecs (zfp at low precision): in addition to compression
  ratio + bandwidth, report **token agreement after dequant** on T1's
  prompt set. Failing that, the codec is lossy-too-far for KV.
- Lossless codecs: roundtrip pytest as in T4.

**Acceptance.**
- A combined chart (or table) placing T4 + T5 results on the same axes
  (bytes/token vs token agreement).
- A one-paragraph verdict: does float-aware-on-FP16 beat INT8+zstd, or
  does it lose.

**Output.**
- Commit: `feat(compress): float-aware codecs on FP16 KV; T4 vs T5 verdict`
- Files: extended codec module + script + writeup + raw JSON.

---

## T6 — wire metric helpers into a reusable module

**Goal.** Stop copy-pasting `argmax_sampler`, `compare_tokens`,
`cache_nbytes`, and the timing block across experiments. Move them into
`kv_quant/metrics/` and have T1-T5's scripts import them.

**Files.**
- `kv_quant/metrics/__init__.py` — public re-exports.
- `kv_quant/metrics/agreement.py` — `compare_tokens` + a logits-MSE
  helper.
- `kv_quant/metrics/footprint.py` — `cache_nbytes` with the safe walk.
- `kv_quant/metrics/timing.py` — a `Timer` context-manager that records
  ttft + decode tok/s.
- Each existing `experiments/*.py` script updated to import these.

**Constraints.**
- No behavior change. Per-prompt outputs and JSON layout stay identical
  byte-for-byte (or with documented diff).
- Add unit tests for `compare_tokens` and `cache_nbytes` with hand-built
  cases.

**Acceptance.**
- All scripts run; their JSON outputs match the previous JSON files
  (`diff` is empty or limited to whitespace).
- `pytest` adds at least 4 new unit tests.

**Output.**
- Commit: `refactor(metrics): consolidate harness helpers into kv_quant/metrics`
- Files: 4 new + N updates.

---

## Reporting back

When you finish a task:

1. Land the commit on `main` (no feature branches per ARLE convention,
   this repo follows the same).
2. Push to `origin/main`.
3. In the response, name: (a) the commit SHA, (b) the three-number row
   for the new experiment, (c) one surprise or non-obvious gotcha.

If a task gets stuck (two good-faith attempts failed), stop. Write a
note under `docs/lessons/` describing what was tried and where the
falsification came from. Do not hack around the failure.

## What comes after T6

M3 (eviction) and M4 (weight quant) start in parallel. New execution
plans will be drafted once M1 + M2 have at least one landed experiment
each. The pattern is the same: harness exists → swap the lever →
report the three numbers.
