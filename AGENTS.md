# kv-quant — Agent Contract

For any AI agent (Codex, Claude, etc.) working in this repo.

## Mission

Anything that shrinks the storage cost of LLM inference is in scope:
quantization (KV / weights / activations), lossless and lossy compression,
eviction, sparsification, deduplication, tier-aware layouts. Methods
compose; a typical landed experiment stacks two or three axes.

## The bar — three numbers, every time

A method is not "supported" until one model on one shape on one SKU has
all three numbers reported together, reproducible from a single command:

1. **Correctness** vs an FP16/BF16 reference (logits MSE, output-token
   agreement, PPL, downstream task score). Lossless modes verify
   bit-equal roundtrip.
2. **Footprint** — peak HBM, on-disk bytes, bytes/token in persisted KV.
3. **Performance** — tokens/s and TFLOPs on a named SKU. A method that
   compresses well but halves throughput gets reported as such; never
   conflated with one that holds throughput.

If a number is not applicable, say why explicitly. No silent omissions.

## SOLID — first principle (inherited from ARLE)

- **Inference ≠ evidence.** Source survey, grep, callgraph reasoning are
  hypotheses. Evidence = real benches, real roundtrip checks, controlled
  comparison.
- **One variable per experiment.** Changing four things at once and
  reading the result is meaningless. Isolate.
- **Wall-clock framing is ground truth.** "X% of a narrow NVTX window"
  is not the same as "X% of end-to-end time". When framings disagree,
  take the more conservative one.
- **80% SOLID is not enough.** Either deepen to ≥95% or explicitly
  declare "deferred — accepting uncertainty" and move on. Never silent.

## Execution phases

For any non-trivial change:

| Phase | Exit |
|-------|------|
| **Explore** — read scope.md and adjacent experiments, grep prior art | You can name every file you will touch. |
| **Plan** — outline approach; >5 files or irreversible → stop, flag | Approach is written down and approved. |
| **Implement** — only what the plan covers; check prior art first | Diff runs `pytest`. |
| **Verify** — generate three numbers; commit raw data and writeup | `docs/experiments/YYYY-MM-DD-*.md` lands with the three numbers. |
| **Reflect** — bug after >1 attempt → write a note under `docs/lessons/` | Lesson logged. |

Skip rules: trivial → Implement + Verify; pure exploration questions →
Explore only.

## Editing rules

- **Preserve.** Never delete content not explicitly in scope.
- **Simple > clever.** Prefer deletion-style refactors. Collapse
  duplicate helpers. One canonical flow per concept.
- **No half-states.** Finish a refactor unit or revert it. Never leave
  parallel old + new paths.
- **Approach-first for >3 files.** Outline and confirm before coding.

## Verification — the harness

`experiments/kv_mlx_baseline.py` is the canonical harness. New methods
swap one piece (cache factory, codec, eviction policy, kernel) without
touching the timing / agreement / byte-counting code. If a new method
needs the harness changed structurally, add a sibling
`experiments/<axis>_<name>.py` rather than forking.

For non-MLX (CUDA) work — Triton / cudarc kernels: separate harness
that reports the same three numbers; do not retrofit the MLX harness.

## Filesystem layout

```
kv-quant/
├── AGENTS.md            this file
├── README.md            project intro + scope table
├── pyproject.toml
├── kv_quant/            library code (factories, codecs, metric helpers)
├── experiments/         standalone scripts; one file per harness
├── tests/               pytest — smoke + roundtrip + correctness fixtures
└── docs/
    ├── ROADMAP.md       milestone view, ordered by dependency
    ├── scope.md         per-axis plan
    ├── plans/           dated execution plans (Codex briefs etc.)
    ├── experiments/     dated experiment writeups (one per result)
    │   └── data/        raw JSON outputs
    └── lessons/         post-bug notes (one per non-trivial failure)
```

## Commit style

`<type>(<scope>): <subject>`. Scopes: `kv`, `weight`, `act`, `compress`,
`evict`, `sparse`, `share`, `tier`, `kernel`, `harness`, `docs`.

Commit small tranches. A docs change is a commit; a script change is a
commit; an experiment writeup with its raw data is a commit. Do not
fold three concerns into one diff.

## Out of scope

- Training-time quantization (QAT).
- Serving-runtime work — that is `agent-infer` (ARLE).
- Tier simulation — that is `kvcache-sim`.
