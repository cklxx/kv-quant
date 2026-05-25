"""Lossless compression sweep for persisted MLX KV-cache dumps."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from experiments.kv_mlx_baseline import PROMPTS, argmax_sampler, compare_tokens
from kv_quant.storage import BrotliCodec, Lz4Codec, RawCodec, ZstdCodec, dump_prompt_cache

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_OUT = "kv_dump_compress_qwen3_5_9b_3x64.json"


@dataclass(frozen=True)
class KvSpec:
    name: str
    kv_bits: int | None
    kv_group_size: int = 64


KV_SPECS = [
    KvSpec("fp", None),
    KvSpec("int8_g64", 8),
    KvSpec("int4_g64", 4),
]


def codec_suite():
    return [
        RawCodec(),
        Lz4Codec(),
        ZstdCodec(1),
        ZstdCodec(3),
        ZstdCodec(9),
        BrotliCodec(5),
    ]


def run_generation(model, tokenizer, prompt: str, spec: KvSpec, max_new: int) -> dict:
    cache = make_prompt_cache(model)
    prompt_ids = tokenizer.encode(prompt)
    prompt_arr = mx.array(prompt_ids)

    tokens: list[int] = []
    t0 = time.perf_counter()
    ttft: float | None = None
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
    mx.eval([c.state for c in cache])

    dump = dump_prompt_cache(cache)
    decode_tokens = max(len(tokens) - 1, 0)
    decode_secs = max(total - (ttft or 0.0), 1e-9)
    return {
        "prompt_tokens": len(prompt_ids),
        "output_tokens": tokens,
        "total_cache_tokens": len(prompt_ids) + len(tokens),
        "decode_tokps": decode_tokens / decode_secs,
        "dump": dump,
        "dump_bytes": len(dump),
    }


def timed_codec(codec, blob: bytes, repeat: int) -> dict:
    encoded = codec.encode(blob)
    if codec.decode(encoded) != blob:
        raise ValueError(f"{codec.name} failed byte-equal roundtrip")

    encode_secs = []
    decode_secs = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        enc = codec.encode(blob)
        encode_secs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dec = codec.decode(enc)
        decode_secs.append(time.perf_counter() - t0)
        if dec != blob:
            raise ValueError(f"{codec.name} failed byte-equal roundtrip")

    raw_mb = len(blob) / 1_000_000

    def mbps(samples: list[float]) -> list[float]:
        return [raw_mb / max(s, 1e-12) for s in samples]

    encode_mbps = mbps(encode_secs)
    decode_mbps = mbps(decode_secs)
    return {
        "codec": codec.name,
        "raw_bytes": len(blob),
        "encoded_bytes": len(encoded),
        "compression_ratio": len(encoded) / len(blob),
        "roundtrip_byte_equal": True,
        "encode_mbps_median": statistics.median(encode_mbps),
        "encode_mbps_p95": sorted(encode_mbps)[int(0.95 * (len(encode_mbps) - 1))],
        "decode_mbps_median": statistics.median(decode_mbps),
        "decode_mbps_p95": sorted(decode_mbps)[int(0.95 * (len(decode_mbps) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-prompts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "docs/experiments/data" / DEFAULT_OUT),
    )
    args = parser.parse_args()

    prompts = PROMPTS if args.limit_prompts == 0 else PROMPTS[: args.limit_prompts]

    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)

    results = {
        "model": args.model,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "kv": {},
    }

    baseline_outputs: list[list[int]] = []
    for spec in KV_SPECS:
        print(f"\n=== {spec.name} ===", flush=True)
        runs = []
        blobs = []
        for i, prompt in enumerate(prompts):
            run = run_generation(model, tokenizer, prompt, spec, args.max_new_tokens)
            if spec.kv_bits is None:
                baseline_outputs.append(run["output_tokens"])
            cmp = compare_tokens(baseline_outputs[i], run["output_tokens"])
            runs.append(
                {
                    "prompt_tokens": run["prompt_tokens"],
                    "output_tokens": run["output_tokens"],
                    "total_cache_tokens": run["total_cache_tokens"],
                    "dump_bytes": run["dump_bytes"],
                    "decode_tokps": run["decode_tokps"],
                    "agreement_vs_fp": cmp["agreement"],
                    "matched_prefix": cmp["matched_prefix"],
                }
            )
            blobs.append(run["dump"])
            print(
                f"  [{i + 1}/{len(prompts)}] agree={cmp['agreement'] * 100:5.1f}% "
                f"dump={run['dump_bytes'] / 1024 / 1024:7.1f} MiB "
                f"decode={run['decode_tokps']:6.1f} tok/s",
                flush=True,
            )

        blob = b"".join(blobs)
        total_tokens = sum(r["total_cache_tokens"] for r in runs)
        bytes_per_token = len(blob) / total_tokens
        codec_rows = []
        for codec in codec_suite():
            row = timed_codec(codec, blob, args.repeat)
            row["encoded_bytes_per_token"] = row["encoded_bytes"] / total_tokens
            raw_mb_per_token = bytes_per_token / 1_000_000
            row["theoretical_encode_tokps"] = row["encode_mbps_median"] / raw_mb_per_token
            row["theoretical_decode_tokps"] = row["decode_mbps_median"] / raw_mb_per_token
            codec_rows.append(row)
            print(
                f"  {codec.name:<7} ratio={row['compression_ratio']:.3f} "
                f"enc={row['encode_mbps_median']:8.1f} MB/s "
                f"dec={row['decode_mbps_median']:8.1f} MB/s "
                f"theory_dec={row['theoretical_decode_tokps']:8.0f} tok/s",
                flush=True,
            )

        results["kv"][spec.name] = {
            "kv_bits": spec.kv_bits,
            "kv_group_size": spec.kv_group_size,
            "n_cache_tokens": total_tokens,
            "raw_dump_bytes": len(blob),
            "raw_dump_bytes_per_token": bytes_per_token,
            "mean_generation_decode_tokps": statistics.mean(r["decode_tokps"] for r in runs),
            "mean_agreement_vs_fp": statistics.mean(r["agreement_vs_fp"] for r in runs),
            "per_prompt": runs,
            "codecs": codec_rows,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
