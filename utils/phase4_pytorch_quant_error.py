#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List


def load_profile(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Profile JSON must be a list of layer records")
        return data

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "node_id": int(row.get("node_id", 0)),
                    "node_name": row.get("node_name", ""),
                    "node_type": row.get("node_type", ""),
                    "comm_size": int(float(row.get("comm_size", 0) or 0)),
                    "quantized_event_ratio": float(row.get("quantized_event_ratio", 0.0) or 0.0),
                    "byte_reduction_ratio": float(row.get("byte_reduction_ratio", 0.0) or 0.0),
                }
            )
        return rows


def keep_ratio_to_bits(keep_ratio: float) -> int:
    keep_ratio = max(0.05, min(1.0, keep_ratio))
    bits = int(round(32.0 * keep_ratio))
    return max(4, min(16, bits))


def tensor_length_from_comm_size(comm_size: int, min_len: int, max_len: int) -> int:
    if comm_size <= 0:
        return min_len
    est = int(comm_size / 4)
    return max(min_len, min(max_len, est))


def fake_quantize_symmetric(x, bits: int):
    if bits >= 32:
        return x

    qmax = float((1 << (bits - 1)) - 1)
    max_abs = x.abs().max()
    scale = max_abs / qmax
    if float(scale) == 0.0:
        return x

    q = (x / scale).round().clamp(-qmax, qmax)
    return q * scale


def cosine_similarity(a, b) -> float:
    a_norm = a.norm().item()
    b_norm = b.norm().item()
    if a_norm == 0.0 or b_norm == 0.0:
        return 1.0
    return float((a @ b).item() / (a_norm * b_norm))


def evaluate_layer(torch, layer: Dict, trials: int, seed: int, min_len: int, max_len: int) -> Dict:
    gen = torch.Generator().manual_seed(seed + int(layer.get("node_id", 0)))

    qprob = max(0.0, min(1.0, float(layer.get("quantized_event_ratio", 0.0))))
    byte_reduction_ratio = max(0.0, min(0.99, float(layer.get("byte_reduction_ratio", 0.0))))
    keep_ratio = 1.0 - byte_reduction_ratio
    bits = keep_ratio_to_bits(keep_ratio)

    comm_size = int(layer.get("comm_size", 0))
    length = tensor_length_from_comm_size(comm_size, min_len, max_len)

    x = torch.randn(length, generator=gen)

    mse_values = []
    cos_values = []
    quantized_trials = 0

    for _ in range(trials):
        do_quant = bool(torch.rand(1, generator=gen).item() < qprob)
        if do_quant:
            y = fake_quantize_symmetric(x, bits)
            quantized_trials += 1
        else:
            y = x

        mse = torch.mean((x - y) ** 2).item()
        cos = cosine_similarity(x, y)
        mse_values.append(float(mse))
        cos_values.append(float(cos))

    def mean_std(values: List[float]):
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mu = sum(values) / n
        var = sum((v - mu) ** 2 for v in values) / n
        return mu, math.sqrt(var)

    mse_mean, mse_std = mean_std(mse_values)
    cos_mean, cos_std = mean_std(cos_values)

    return {
        "node_id": int(layer.get("node_id", 0)),
        "node_name": str(layer.get("node_name", "")),
        "node_type": str(layer.get("node_type", "")),
        "comm_size": comm_size,
        "tensor_len": length,
        "quantized_event_ratio": qprob,
        "byte_reduction_ratio": byte_reduction_ratio,
        "keep_ratio": keep_ratio,
        "approx_bits": bits,
        "quantized_trials": quantized_trials,
        "trials": trials,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approximate per-layer quantization impact in PyTorch using phase-4 layer profile"
    )
    parser.add_argument(
        "--profile",
        default="results/phase4/layer_quant_profile.json",
        help="Path to layer quantization profile JSON/CSV",
    )
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo trials per layer")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--min-tensor-len", type=int, default=1024, help="Minimum synthetic tensor length")
    parser.add_argument("--max-tensor-len", type=int, default=65536, help="Maximum synthetic tensor length")
    parser.add_argument(
        "--output-json",
        default="results/phase4/pytorch_quant_error.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-csv",
        default="results/phase4/pytorch_quant_error.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for this script. Install with: pip install torch"
        ) from exc

    profile_path = Path(args.profile)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    layers = load_profile(profile_path)

    results = [
        evaluate_layer(
            torch=torch,
            layer=layer,
            trials=args.trials,
            seed=args.seed,
            min_len=args.min_tensor_len,
            max_len=args.max_tensor_len,
        )
        for layer in layers
    ]

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "node_id",
        "node_name",
        "node_type",
        "comm_size",
        "tensor_len",
        "quantized_event_ratio",
        "byte_reduction_ratio",
        "keep_ratio",
        "approx_bits",
        "quantized_trials",
        "trials",
        "mse_mean",
        "mse_std",
        "cosine_mean",
        "cosine_std",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_comm = sum(max(1, int(r["comm_size"])) for r in results)
    weighted_mse = sum(r["mse_mean"] * max(1, int(r["comm_size"])) for r in results) / total_comm
    weighted_cos = sum(r["cosine_mean"] * max(1, int(r["comm_size"])) for r in results) / total_comm

    print("=== Phase 4 PyTorch Quantization Approximation ===")
    print(f"profile={profile_path}")
    print(f"layers={len(results)}")
    print(f"output_json={out_json}")
    print(f"output_csv={out_csv}")
    print(f"weighted_mse={weighted_mse:.8f}")
    print(f"weighted_cosine={weighted_cos:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
