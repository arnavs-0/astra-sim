#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

WALL_RE = re.compile(r"sys\[(\d+)\],\s+Wall time:\s+(\d+)")
COMM_RE = re.compile(r"sys\[(\d+)\],\s+Comm time:\s+(\d+)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_log(path: Path) -> dict:
    wall = {}
    comm = {}
    quant = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = WALL_RE.search(line)
            if m:
                wall[int(m.group(1))] = int(m.group(2))
                continue

            m = COMM_RE.search(line)
            if m:
                comm[int(m.group(1))] = int(m.group(2))
                continue

            if "[quantization]" in line:
                for key, value in KV_RE.findall(line):
                    quant[key] = value

    if not wall:
        raise ValueError(f"No wall-time metrics found in {path}")

    avg_wall = sum(wall.values()) / len(wall)
    avg_comm = (sum(comm.values()) / len(comm)) if comm else float("nan")

    return {
        "path": str(path),
        "wall_by_rank": wall,
        "comm_by_rank": comm,
        "avg_wall": avg_wall,
        "avg_comm": avg_comm,
        "quant": quant,
    }


def write_csv(output: Path, baseline: dict, candidate: dict, speedup: float, throughput_gain: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run",
            "avg_wall_cycles",
            "avg_comm_cycles",
            "speedup_vs_baseline",
            "throughput_gain_vs_baseline",
            "quantized_chunk_ratio",
            "byte_reduction_ratio",
            "queue_above_threshold_ratio",
            "time_above_threshold_ns",
        ])

        for run_name, run_data, run_speedup, run_tgain in (
            ("baseline", baseline, 1.0, 0.0),
            ("candidate", candidate, speedup, throughput_gain),
        ):
            quant = run_data["quant"]
            writer.writerow([
                run_name,
                f"{run_data['avg_wall']:.2f}",
                f"{run_data['avg_comm']:.2f}",
                f"{run_speedup:.6f}",
                f"{run_tgain:.6f}",
                quant.get("quantized_chunk_ratio", ""),
                quant.get("byte_reduction_ratio", ""),
                quant.get("queue_above_threshold_ratio", ""),
                quant.get("time_above_threshold_ns", ""),
            ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract phase-1 Astra-sim metrics and compare baseline vs quantized run"
    )
    parser.add_argument("--baseline-log", required=True, help="Path to baseline (full precision) stdout log")
    parser.add_argument("--candidate-log", required=True, help="Path to candidate (quantized) stdout log")
    parser.add_argument("--output-csv", default="results/phase1/phase1_comparison.csv", help="Output CSV path")

    args = parser.parse_args()

    baseline = parse_log(Path(args.baseline_log))
    candidate = parse_log(Path(args.candidate_log))

    speedup = baseline["avg_wall"] / candidate["avg_wall"]
    throughput_gain = speedup - 1.0

    write_csv(Path(args.output_csv), baseline, candidate, speedup, throughput_gain)

    print("=== Phase 1 Baseline Comparison ===")
    print(f"baseline_log={baseline['path']}")
    print(f"candidate_log={candidate['path']}")
    print(f"avg_wall_baseline={baseline['avg_wall']:.2f}")
    print(f"avg_wall_candidate={candidate['avg_wall']:.2f}")
    print(f"speedup={speedup:.6f}x")
    print(f"throughput_gain={throughput_gain * 100.0:.2f}%")

    cand_quant = candidate["quant"]
    if cand_quant:
        print(f"candidate_quantized_chunk_ratio={cand_quant.get('quantized_chunk_ratio', 'NA')}")
        print(f"candidate_byte_reduction_ratio={cand_quant.get('byte_reduction_ratio', 'NA')}")
        print(f"candidate_queue_above_threshold_ratio={cand_quant.get('queue_above_threshold_ratio', 'NA')}")
        print(f"candidate_time_above_threshold_ns={cand_quant.get('time_above_threshold_ns', 'NA')}")

    print(f"comparison_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
