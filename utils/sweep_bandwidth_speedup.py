#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

WALL_RE = re.compile(r"sys\[(\d+)\],\s+Wall time:\s+(\d+)")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def parse_avg_wall_and_quant(log_path: Path) -> Tuple[float, Dict[str, str]]:
    wall = {}
    quant = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = WALL_RE.search(line)
            if m:
                wall[int(m.group(1))] = int(m.group(2))
                continue
            if "[quantization]" in line:
                for key, value in KV_RE.findall(line):
                    quant[key] = value

    if not wall:
        raise RuntimeError(f"No wall-time metrics found in {log_path}")

    return (sum(wall.values()) / len(wall), quant)


def write_network_yaml(path: Path, npus: int, bandwidth: float, latency: float, enabled: bool, ratio: float, threshold: int) -> None:
    lines = [
        "topology: [ Ring ]",
        f"npus_count: [ {npus} ]",
        f"bandwidth: [ {bandwidth:.6g} ]  # GB/s",
        f"latency: [ {latency:.6g} ]  # ns",
        "quantization:",
        f"  enabled: {'true' if enabled else 'false'}",
        f"  ratio: {ratio:.6g}",
        f"  queue_threshold: {threshold}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one(astra_bin: Path, workload: Path, system: Path, remote: Path, network: Path, out_log: Path) -> None:
    cmd = [
        str(astra_bin),
        f"--workload-configuration={workload}",
        f"--system-configuration={system}",
        f"--remote-memory-configuration={remote}",
        f"--network-configuration={network}",
    ]
    with out_log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def maybe_plot(output_dir: Path, rows: List[dict]) -> str:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return "matplotlib-not-available"

    bandwidths = [float(r["bandwidth_gbps"]) for r in rows]
    speedups = [float(r["speedup_vs_baseline"]) for r in rows]

    plt.figure(figsize=(7.0, 4.0))
    plt.plot(bandwidths, speedups, marker="o", linewidth=2)
    plt.xlabel("NoC link bandwidth (GB/s)")
    plt.ylabel("Speedup (quantized / baseline)")
    plt.title("Quantization Benefit vs NoC Bandwidth")
    plt.grid(True, alpha=0.3)

    png_path = output_dir / "speedup_vs_bandwidth.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return str(png_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep NoC bandwidth and measure quantization speedup")
    parser.add_argument("--bandwidths", default="10,20,30,40,50", help="Comma-separated GB/s values")
    parser.add_argument("--latency", type=float, default=500.0, help="Link latency in ns")
    parser.add_argument("--npus", type=int, default=8, help="Number of NPUs in Ring topology")
    parser.add_argument("--quant-ratio", type=float, default=0.25, help="Quantized effective size ratio")
    parser.add_argument("--output-dir", default="results/phase1_sweep", help="Directory for logs/results")
    parser.add_argument(
        "--astra-bin",
        default="build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
        help="Path to AstraSim analytical congestion-aware binary",
    )
    parser.add_argument(
        "--workload",
        default="examples/workload/microbenchmarks/all_reduce/8npus_1MB/all_reduce",
        help="Workload path prefix",
    )
    parser.add_argument(
        "--system",
        default="examples/system/native_collectives/Ring_4chunks.json",
        help="System config path",
    )
    parser.add_argument(
        "--remote-memory",
        default="examples/remote_memory/analytical/no_memory_expansion.json",
        help="Remote memory config path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    networks_dir = output_dir / "generated_networks"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    networks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    astra_bin = Path(args.astra_bin)
    workload = Path(args.workload)
    system = Path(args.system)
    remote = Path(args.remote_memory)

    if not astra_bin.exists():
        raise FileNotFoundError(f"Astra binary not found at {astra_bin}")

    bandwidths = [float(x.strip()) for x in args.bandwidths.split(",") if x.strip()]
    rows: List[dict] = []

    for bw in bandwidths:
        bw_tag = str(bw).replace(".", "p")
        net_fp = networks_dir / f"ring{args.npus}_bw{bw_tag}_fp.yml"
        net_q = networks_dir / f"ring{args.npus}_bw{bw_tag}_q.yml"
        log_fp = logs_dir / f"bw{bw_tag}_fp.log"
        log_q = logs_dir / f"bw{bw_tag}_q.log"

        write_network_yaml(
            path=net_fp,
            npus=args.npus,
            bandwidth=bw,
            latency=args.latency,
            enabled=False,
            ratio=1.0,
            threshold=18446744073709551615,
        )
        write_network_yaml(
            path=net_q,
            npus=args.npus,
            bandwidth=bw,
            latency=args.latency,
            enabled=True,
            ratio=args.quant_ratio,
            threshold=0,
        )

        run_one(astra_bin, workload, system, remote, net_fp, log_fp)
        run_one(astra_bin, workload, system, remote, net_q, log_q)

        avg_wall_fp, _ = parse_avg_wall_and_quant(log_fp)
        avg_wall_q, quant = parse_avg_wall_and_quant(log_q)
        speedup = avg_wall_fp / avg_wall_q
        throughput_gain = speedup - 1.0

        rows.append(
            {
                "bandwidth_gbps": f"{bw:.6g}",
                "avg_wall_baseline_cycles": f"{avg_wall_fp:.2f}",
                "avg_wall_quantized_cycles": f"{avg_wall_q:.2f}",
                "speedup_vs_baseline": f"{speedup:.6f}",
                "throughput_gain_vs_baseline": f"{throughput_gain:.6f}",
                "byte_reduction_ratio": quant.get("byte_reduction_ratio", ""),
                "quantized_chunk_ratio": quant.get("quantized_chunk_ratio", ""),
                "queue_above_threshold_ratio": quant.get("queue_above_threshold_ratio", ""),
            }
        )

    csv_path = output_dir / "speedup_vs_bandwidth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bandwidth_gbps",
                "avg_wall_baseline_cycles",
                "avg_wall_quantized_cycles",
                "speedup_vs_baseline",
                "throughput_gain_vs_baseline",
                "byte_reduction_ratio",
                "quantized_chunk_ratio",
                "queue_above_threshold_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_result = maybe_plot(output_dir, rows)

    print("=== Bandwidth Sweep Complete ===")
    print(f"results_csv={csv_path}")
    if plot_result == "matplotlib-not-available":
        print("plot_png=not-generated (matplotlib not available)")
    else:
        print(f"plot_png={plot_result}")
    print("rows=")
    for row in rows:
        print(
            f"  bw={row['bandwidth_gbps']}GB/s speedup={row['speedup_vs_baseline']}x "
            f"throughput_gain={float(row['throughput_gain_vs_baseline']) * 100.0:.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
