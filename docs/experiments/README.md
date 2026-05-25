# Experiments

One file per experiment. Filename: `YYYY-MM-DD-<axis>-<short-slug>.md`
(axes: `kv`, `weight`, `act`, `compress`, `evict`, `sparse`, `share`,
`tier`, `kernel`; use the dominant axis if an experiment stacks several).

Every entry has these sections, in this order:

```
# <Title>

## Goal
<one paragraph — what question this answers>

## Setup
- Model + revision
- SKU (GPU model, driver, CUDA / ROCm / Metal version)
- Shapes (batch, seq, ctx)
- Reference baseline (the FP16/BF16 thing we compare against)

## Method
<algorithm + any non-default hyperparameters; cite the paper>

## Results
- Correctness: <number, vs reference>
- Footprint: <bytes, vs reference>
- Performance: <tokens/s or TFLOPs, vs reference>

## Repro
`<one command that regenerates the result>`

## Notes
<gotchas, surprises, what would falsify this>
```

No entry lands without all three numbers (correctness, footprint, performance)
filled in. If one is genuinely not applicable, say so explicitly and explain
why.
