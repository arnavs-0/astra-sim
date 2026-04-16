#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


KEY_COLUMNS = [
    "original_bytes",
    "payload_bytes",
    "metadata_bytes",
    "effective_bytes",
    "quantized",
    "congestion_score",
    "policy",
    "congestion_metric",
    "applied_ratio",
    "metadata_mode",
    "node_id",
    "comm_tag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Astra-Sim quantization event CSV output."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="results/phase3/quantization_events.csv",
        help="Path to quantization_events.csv",
    )
    parser.add_argument(
        "--show-rows",
        type=int,
        default=5,
        help="Number of suspicious rows to print",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def suspicious_reasons(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    original = as_int(row, "original_bytes")
    payload = as_int(row, "payload_bytes")
    metadata = as_int(row, "metadata_bytes")
    effective = as_int(row, "effective_bytes")
    quantized = as_int(row, "quantized")
    applied_ratio = as_float(row, "applied_ratio")

    if effective != payload + metadata:
        reasons.append("effective != payload + metadata")
    if quantized == 1 and effective >= original:
        reasons.append("quantized row is not smaller than original")
    if quantized == 0 and effective != original:
        reasons.append("unquantized row differs from original")
    if applied_ratio <= 0.0 or applied_ratio > 1.0:
        reasons.append("applied_ratio outside (0, 1]")
    if payload > original:
        reasons.append("payload larger than original")
    return reasons


def summarize(rows: list[dict[str, str]], show_rows: int) -> str:
    total_rows = len(rows)
    if total_rows == 0:
        return "No rows found in CSV."

    original_total = sum(as_int(r, "original_bytes") for r in rows)
    payload_total = sum(as_int(r, "payload_bytes") for r in rows)
    metadata_total = sum(as_int(r, "metadata_bytes") for r in rows)
    effective_total = sum(as_int(r, "effective_bytes") for r in rows)
    quantized_rows = [r for r in rows if as_int(r, "quantized") == 1]
    unquantized_rows = total_rows - len(quantized_rows)
    ratio_values = [as_float(r, "applied_ratio") for r in rows]
    quantized_ratio_values = [as_float(r, "applied_ratio") for r in quantized_rows]
    scores = [as_float(r, "congestion_score") for r in rows]

    suspicious: list[tuple[dict[str, str], list[str]]] = []
    for row in rows:
        reasons = suspicious_reasons(row)
        if reasons:
            suspicious.append((row, reasons))

    payload_savings = original_total - payload_total
    net_savings = original_total - effective_total
    metadata_share = (metadata_total / effective_total) if effective_total else 0.0
    quantized_fraction = len(quantized_rows) / total_rows
    avg_ratio_all = mean(ratio_values)
    avg_ratio_quantized = mean(quantized_ratio_values) if quantized_ratio_values else 1.0
    avg_score = mean(scores)

    lines: list[str] = []
    lines.append(f"CSV: rows={total_rows}")
    lines.append(
        "Traffic: "
        f"original={original_total} payload={payload_total} metadata={metadata_total} effective={effective_total}"
    )
    lines.append(
        "Savings: "
        f"payload_only={payload_savings} ({(payload_savings / original_total):.2%}) "
        f"net={net_savings} ({(net_savings / original_total):.2%})"
    )
    lines.append(
        "Compression: "
        f"quantized_rows={len(quantized_rows)} unquantized_rows={unquantized_rows} "
        f"quantized_fraction={quantized_fraction:.2%}"
    )
    lines.append(
        "Ratios: "
        f"avg_applied_ratio_all={avg_ratio_all:.4f} "
        f"avg_applied_ratio_quantized={avg_ratio_quantized:.4f}"
    )
    lines.append(
        "Overhead: "
        f"metadata_share_of_effective={metadata_share:.2%} avg_congestion_score={avg_score:.4f}"
    )

    policy_counts: dict[str, int] = {}
    metric_counts: dict[str, int] = {}
    metadata_counts: dict[str, int] = {}
    for row in rows:
        policy_counts[row["policy"]] = policy_counts.get(row["policy"], 0) + 1
        metric_counts[row["congestion_metric"]] = metric_counts.get(row["congestion_metric"], 0) + 1
        metadata_counts[row["metadata_mode"]] = metadata_counts.get(row["metadata_mode"], 0) + 1

    lines.append("Policies: " + ", ".join(f"{k}={v}" for k, v in sorted(policy_counts.items())))
    lines.append("Metrics: " + ", ".join(f"{k}={v}" for k, v in sorted(metric_counts.items())))
    lines.append("Metadata modes: " + ", ".join(f"{k}={v}" for k, v in sorted(metadata_counts.items())))

    lines.append(f"Invariant violations: {len(suspicious)}")
    if suspicious:
        lines.append("Suspicious rows:")
        for idx, (row, reasons) in enumerate(suspicious[:show_rows], start=1):
            compact = ", ".join(f"{k}={row.get(k, '')}" for k in KEY_COLUMNS if k in row)
            lines.append(f"  {idx}. reasons={'; '.join(reasons)} | {compact}")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    path = Path(args.csv_path)
    if not path.exists():
        print(f"CSV not found: {path}")
        return 1

    rows = load_rows(path)
    print(summarize(rows, args.show_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
