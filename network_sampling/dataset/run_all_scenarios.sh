#!/bin/bash -l
# Submit all MATSim scenarios as a throttled SLURM job array.
#
# Usage (after generation):
#   bash run_all_scenarios.sh "$DATASET" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" [max_parallel]
# Example:
#   bash run_all_scenarios.sh "$DATASET" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" 3
#
# Internally, SLURM invokes the same script using the --worker entrypoint.

set -euo pipefail

if [[ "${1:-}" == "--worker" ]]; then
    shift
    if [[ "$#" -ne 5 ]]; then
        echo "Internal usage: --worker MANIFEST CONFIG POPULATION JAR PYTHON" >&2
        exit 2
    fi

    MANIFEST="$1"
    CONFIG="$2"
    POPULATION="$3"
    JAR="$4"
    PYTHON="$5"

    : "${SLURM_ARRAY_TASK_ID:?This entrypoint must run in a SLURM array task}"
    SCENARIO="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
    if [[ -z "$SCENARIO" || ! -f "$SCENARIO/network.xml.gz" || ! -f "$SCENARIO/transit_schedule.xml.gz" ]]; then
        echo "Missing/invalid scenario for task $SLURM_ARRAY_TASK_ID: $SCENARIO" >&2
        exit 1
    fi

    # Match the environment used by the successful single-scenario run.
    module purge
    module load EESSI/2026.06
    module load Java/25.36

    echo "[$(date -Is)] START scenario=$SCENARIO task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
    "$PYTHON" -m network_sampling.dataset.run_one \
        --scenario "$SCENARIO" \
        --base-config "$CONFIG" \
        --population "$POPULATION" \
        --jar "$JAR" \
        --threads 24 \
        --heap 96G \
        --reset-heap 24G \
        --last-iteration 5
    echo "[$(date -Is)] DONE scenario=$SCENARIO"
    exit 0
fi

if [[ "$#" -lt 5 || "$#" -gt 6 ]]; then
    echo "Usage: bash $0 DATASET CONFIG POPULATION JAR PYTHON [max_parallel]" >&2
    exit 2
fi

DATASET="$(readlink -f "$1")"
CONFIG="$(readlink -f "$2")"
POPULATION="$(readlink -f "$3")"
JAR="$(readlink -f "$4")"
PYTHON="$5"

if [[ "$PYTHON" != */* ]]; then
    # Python command such as "python" or "python3"
    PYTHON="$(command -v "$PYTHON")"
elif [[ "$PYTHON" != /* ]]; then
    # Relative path such as ".venv/bin/python"
    PYTHON="$PWD/$PYTHON"
fi
MAX_PARALLEL="${6:-3}"

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "max_parallel must be a positive integer (got '$MAX_PARALLEL')" >&2
    exit 2
fi
for f in "$CONFIG" "$POPULATION" "$JAR" "$PYTHON"; do
    if [[ ! -f "$f" ]]; then
        echo "File not found: $f" >&2
        exit 2
    fi
done
if [[ ! -d "$DATASET" ]]; then
    echo "Dataset directory not found: $DATASET" >&2
    exit 2
fi

mkdir -p "$DATASET/slurm_logs"
MANIFEST="$(mktemp "$DATASET/scenario_manifest.XXXXXXXX.txt")"

# Find all direct children with prepared network + transit schedule, independent
# of strategy name (cluster, random, corridor, or future strategies).
while IFS= read -r network_path; do
    scenario_dir="${network_path%/network.xml.gz}"
    if [[ -f "$scenario_dir/transit_schedule.xml.gz" ]]; then
        printf '%s\n' "$scenario_dir" >> "$MANIFEST"
    fi
done < <(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name 'network.xml.gz' | LC_ALL=C sort)

N="$(wc -l < "$MANIFEST")"
if (( N == 0 )); then
    echo "No complete scenario folders (network.xml.gz + transit_schedule.xml.gz) in $DATASET" >&2
    rm -f "$MANIFEST"
    exit 1
fi

SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
echo "Found $N scenarios; submitting an array with at most $MAX_PARALLEL running simultaneously."
echo "Scenario manifest: $MANIFEST"

sbatch \
    --partition=mech-cst-mov.cpu.q \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=24 \
    --mem=128G \
    --time=08:00:00 \
    --job-name=matsim-dataset \
    --output="$DATASET/slurm_logs/%A_%a.out" \
    --error="$DATASET/slurm_logs/%A_%a.err" \
    --array="0-$((N - 1))%$MAX_PARALLEL" \
    "$SCRIPT" --worker "$MANIFEST" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON"
