"""KV-cache quantization baseline on MLX (Apple Silicon).

Goal: establish the harness — load a small MLX-format Qwen, generate greedy
output with FP-baseline and several quantized KV configs, and report
correctness · footprint · perf together.

Uses mlx-lm's built-in `QuantizedKVCache` so this is an off-the-shelf
measurement. Custom KV-quant schemes (e.g. KIVI-style per-token V quant,
float-aware codecs on the bitstream, layer-delta) replace the per-step
quantization path; the harness stays identical.

Note: Qwen3.5 is hybrid (SSM + attention). We use `make_prompt_cache(model)`
to get the correctly-typed per-layer cache list and let mlx-lm's
`maybe_quantize_kv_cache` swap only the attention slots in-place.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

DEFAULT_MODEL = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
DEFAULT_MAX_NEW = 64

PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number.",
    "Explain CRDTs in two sentences.",
    "Sort the list [3, 1, 4, 1, 5, 9, 2, 6] in ascending order and show the result.",
    "What is the time complexity of merge sort, and why?",
    "Convert 27 degrees Celsius to Fahrenheit. Show the formula.",
    "Name three classical sorting algorithms and one property that distinguishes each.",
    "Write a one-line shell command to count lines in every .py file under cwd.",
    "Give a concise definition of overfitting in machine learning.",
    "Return the SQL to select the five highest-paid employees from table `emp`.",
    "What does INT4 weight quantization typically trade off, and why is it worth it?",
    "用一句话总结量子计算与经典计算的核心区别。",
    "请给出一个能体现 Python 列表推导式的简短示例。",
    "解释 LLM 推理中 KV cache 的作用，控制在两句话内。",
    "用一句话比较 INT8 和 INT4 量化在显存占用上的差异。",
    "请翻译为英文：模型量化既能降低显存，也能影响精度。",
    "If a process holds an O(n^2) algorithm on 10,000 items, roughly how many ops?",
    "What does 'attention sink' refer to in long-context LLM serving?",
    "List two reasons to compress KV cache before persisting it to NVMe.",
    "Write a C function signature for a saxpy kernel.",
    "Briefly describe the role of group size in groupwise weight quantization.",
]


@dataclass
class CacheSpec:
    name: str
    kv_bits: int | None  # None => FP baseline (no quantization)
    kv_group_size: int


CONFIGS: list[CacheSpec] = [
    CacheSpec("baseline_fp", None, 64),
    CacheSpec("int8_g64", 8, 64),
    CacheSpec("int4_g64", 4, 64),
    CacheSpec("int4_g32", 4, 32),
]


def argmax_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _walk_bytes(obj) -> int:
    if isinstance(obj, mx.array):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(_walk_bytes(x) for x in obj)
    return 0


def cache_nbytes(cache_list) -> int:
    # mlx_lm 0.31.2 QuantizedKVCache.nbytes references an unimported tree_reduce;
    # walk the `.state` tuple ourselves for robustness across versions and cache types.
    total = 0
    for c in cache_list:
        state = getattr(c, "state", None)
        total += _walk_bytes(state)
    return total


def run_one(model, tokenizer, prompt: str, spec: CacheSpec, max_new: int) -> dict:
    cache = make_prompt_cache(model)
    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = mx.array(prompt_ids)

    tokens: list[int] = []
    ttft: float | None = None
    t0 = time.perf_counter()
    step = generate_step(
        prompt_arr,
        model,
        max_tokens=max_new,
        sampler=argmax_sampler,
        prompt_cache=cache,
        kv_bits=spec.kv_bits,
        kv_group_size=spec.kv_group_size,
        quantized_kv_start=0,
    )
    for tok, _ in step:
        tok_id = int(tok.item() if hasattr(tok, "item") else tok)
        if ttft is None:
            ttft = time.perf_counter() - t0
        tokens.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
    total = time.perf_counter() - t0

    decode_tokens = max(len(tokens) - 1, 0)
    decode_secs = max(total - (ttft or 0.0), 1e-9)
    return {
        "prompt_tokens": len(prompt_ids),
        "output_tokens": tokens,
        "ttft_ms": (ttft or 0.0) * 1000.0,
        "total_ms": total * 1000.0,
        "decode_tokps": decode_tokens / decode_secs,
        "cache_bytes": cache_nbytes(cache),
        "cache_types": [type(c).__name__ for c in cache],
    }


def compare_tokens(reference: list[int], candidate: list[int]) -> dict:
    n = min(len(reference), len(candidate))
    matches = sum(1 for i in range(n) if reference[i] == candidate[i])
    divergence = next((i for i in range(n) if reference[i] != candidate[i]), n)
    return {
        "ref_len": len(reference),
        "cand_len": len(candidate),
        "matched_prefix": divergence,
        "agreement": matches / max(n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1] / "docs/experiments/data/kv_mlx_baseline.json"
        ),
    )
    ap.add_argument("--limit-prompts", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    prompts = PROMPTS if args.limit_prompts == 0 else PROMPTS[: args.limit_prompts]

    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    print(f"  vocab={tokenizer.vocab_size}", flush=True)

    # warmup
    _ = run_one(model, tokenizer, "warmup.", CONFIGS[0], max_new=8)

    results: dict = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(prompts),
        "configs": {},
    }
    per_prompt_baseline: list[dict] = []

    for spec in CONFIGS:
        print(f"\n=== {spec.name} ===", flush=True)
        per_prompt: list[dict] = []
        for i, prompt in enumerate(prompts):
            r = run_one(model, tokenizer, prompt, spec, args.max_new_tokens)
            per_prompt.append(r)
            if spec.name == "baseline_fp":
                per_prompt_baseline.append(r)
            ref_tokens = per_prompt_baseline[i]["output_tokens"]
            cmp = compare_tokens(ref_tokens, r["output_tokens"])
            print(
                f"  [{i+1:2d}/{len(prompts)}] "
                f"agree={cmp['agreement']*100:5.1f}%  "
                f"div@{cmp['matched_prefix']:>2d}/{cmp['ref_len']:<2d}  "
                f"ttft={r['ttft_ms']:5.0f}ms  "
                f"decode={r['decode_tokps']:6.1f} tok/s  "
                f"cache={r['cache_bytes']/1024:7.1f} KiB",
                flush=True,
            )
        agreements = [
            compare_tokens(
                per_prompt_baseline[i]["output_tokens"], per_prompt[i]["output_tokens"]
            )["agreement"]
            for i in range(len(prompts))
        ]
        results["configs"][spec.name] = {
            "kv_bits": spec.kv_bits,
            "kv_group_size": spec.kv_group_size,
            "per_prompt": per_prompt,
            "agg": {
                "mean_agreement": sum(agreements) / len(agreements),
                "perfect_match_rate": sum(1 for a in agreements if a == 1.0) / len(agreements),
                "mean_ttft_ms": sum(r["ttft_ms"] for r in per_prompt) / len(per_prompt),
                "mean_decode_tokps": sum(r["decode_tokps"] for r in per_prompt) / len(per_prompt),
                "mean_cache_bytes": sum(r["cache_bytes"] for r in per_prompt) / len(per_prompt),
            },
        }

    print("\n--- summary ---")
    base = results["configs"]["baseline_fp"]["agg"]
    header = f"{'config':<14} {'agree':>7} {'perfect':>8} {'ttft':>8} {'decode':>11} {'cache':>10} {'vs FP':>8}"
    print(header)
    for spec in CONFIGS:
        a = results["configs"][spec.name]["agg"]
        ratio = a["mean_cache_bytes"] / base["mean_cache_bytes"] if base["mean_cache_bytes"] else 0
        print(
            f"{spec.name:<14} "
            f"{a['mean_agreement']*100:6.1f}% "
            f"{a['perfect_match_rate']*100:7.1f}% "
            f"{a['mean_ttft_ms']:6.0f}ms "
            f"{a['mean_decode_tokps']:8.1f}t/s "
            f"{a['mean_cache_bytes']/1024:8.1f}KiB "
            f"{ratio:7.2f}x"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
