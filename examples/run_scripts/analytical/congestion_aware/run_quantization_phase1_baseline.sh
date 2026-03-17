#!/bin/bash
set -euo pipefail

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"
RESULT_DIR="${PROJECT_DIR:?}/results/phase1"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="${EXAMPLE_DIR:?}/workload/microbenchmarks/all_reduce/8npus_1MB/all_reduce"
SYSTEM="${EXAMPLE_DIR:?}/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"
NETWORK_FP="${EXAMPLE_DIR:?}/network/analytical/Ring_8npus_full_precision.yml"
NETWORK_Q="${EXAMPLE_DIR:?}/network/analytical/Ring_8npus_always_quantized.yml"
COMPARE_SCRIPT="${PROJECT_DIR:?}/utils/extract_phase1_metrics.py"

mkdir -p "${RESULT_DIR:?}"

echo "[ASTRA-sim] Building analytical backend..."
"${PROJECT_DIR:?}"/build/astra_analytical/build.sh

echo "[ASTRA-sim] Running full-precision baseline..."
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD:?}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK_FP:?}" \
    > "${RESULT_DIR:?}/baseline_fp.log" 2>&1

echo "[ASTRA-sim] Running always-quantized baseline..."
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD:?}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK_Q:?}" \
    > "${RESULT_DIR:?}/always_quantized.log" 2>&1

echo "[ASTRA-sim] Comparing outputs..."
python3 "${COMPARE_SCRIPT:?}" \
    --baseline-log "${RESULT_DIR:?}/baseline_fp.log" \
    --candidate-log "${RESULT_DIR:?}/always_quantized.log" \
    --output-csv "${RESULT_DIR:?}/phase1_comparison.csv"

echo "[ASTRA-sim] Done. See ${RESULT_DIR:?} for logs and summary CSV."
