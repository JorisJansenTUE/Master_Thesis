#!/bin/bash -l
# SLURM array launcher: one task per prepared scenario, throttled concurrency.
# Run AFTER generate.py has finished.
#
# bash network_sampling/dataset/run_all_scenarios.sh \
#   "$DATASET" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" [max_parallel] [--delete-raw] [--no-compact]
#
# Default: compact both original MATSim statistics after each successful run,
# but KEEP raw_output. --delete-raw enables destructive cleanup only after
# both target files are independently verified.
set -euo pipefail


if [[ "${1:-}" == "--worker" ]]; then
    shift
    if [[ "$#" -lt 5 ]]; then
        echo "Internal: --worker MANIFEST CONFIG POPULATION JAR PYTHON [flags]" >&2
        exit 2
    fi
    MANIFEST="$1"; CONFIG="$2"; POPULATION="$3"; JAR="$4"; PYTHON="$5"
    shift 5
    FLAGS=("$@")
    : "${SLURM_ARRAY_TASK_ID:?Must run as SLURM array worker}"
    SCENARIO="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
    if [[ -z "$SCENARIO" || ! -f "$SCENARIO/scenario.json" || \
          ! -f "$SCENARIO/network.xml.gz" || \
          ! -f "$SCENARIO/transit_schedule.xml.gz" || \
          ! -f "$SCENARIO/link_mapping.csv.gz" ]]; then
        echo "Invalid scenario at task $SLURM_ARRAY_TASK_ID: $SCENARIO" >&2
        exit 1
    fi

    module purge
    module load EESSI/2026.06
    module load Java/25.36

    cd /vast.mnt/home/20201733/Master_Thesis
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"

    echo "[$(date -Is)] START task=$SLURM_ARRAY_TASK_ID scenario=$SCENARIO host=$(hostname)"
    "$PYTHON" -m network_sampling.dataset.run_one \
        --scenario "$SCENARIO" \
        --base-config "$CONFIG" \
        --population "$POPULATION" \
        --jar "$JAR" \
        --threads 2 \
        --heap 4G \
        --initial-heap 1G \
        --reset-heap 3G \
        --last-iteration 100 \
        --linkstats-interval 25 \
        --linkstats-average-iterations 5 \
        "${FLAGS[@]}"
    echo "[$(date -Is)] DONE scenario=$SCENARIO"
    exit 0
fi

if (( $# < 5 )); then
    echo "Usage: bash $0 DATASET CONFIG POPULATION JAR PYTHON [max_parallel] [--delete-raw] [--no-compact]" >&2
    exit 2
fi
DATASET="$(readlink -f "$1")"
CONFIG="$(readlink -f "$2")"
POPULATION="$(readlink -f "$3")"
JAR="$(readlink -f "$4")"
PYTHON="$5"
shift 5

# Bare command names are resolved with PATH. A venv path is NOT passed through
# readlink -f, which can dereference python symlinks to the system interpreter.
if [[ "$PYTHON" != */* ]]; then
    PYTHON="$(command -v "$PYTHON")" || {
        echo "Python command not found in PATH" >&2; exit 2;
    }
elif [[ "$PYTHON" != /* ]]; then
    PYTHON="$PWD/$PYTHON"
fi

MAX_PARALLEL=3
if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_PARALLEL="$1"
    shift
fi
if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "max_parallel must be a positive integer" >&2
    exit 2
fi
FLAGS=()
for option in "$@"; do
    case "$option" in
        --delete-raw|--no-compact) FLAGS+=("$option") ;;
        *) echo "Unknown option: $option" >&2; exit 2 ;;
    esac
done
if [[ " ${FLAGS[*]} " == *" --delete-raw "* && " ${FLAGS[*]} " == *" --no-compact "* ]]; then
    echo "--delete-raw and --no-compact are incompatible" >&2
    exit 2
fi
for file in "$CONFIG" "$POPULATION" "$JAR" "$PYTHON"; do
    if [[ ! -f "$file" ]]; then
        echo "File not found: $file" >&2
        exit 2
    fi
done
if [[ ! -f "$DATASET/dataset_manifest.json" || \
      ! -f "$DATASET/street_segments.csv.gz" ]]; then
    echo "Incomplete generated dataset: $DATASET" >&2
    exit 2
fi

mkdir -p "$DATASET/slurm_logs"
MANIFEST="$(mktemp "$DATASET/scenario_manifest.XXXXXXXX.txt")"
while IFS= read -r file; do
    directory="${file%/scenario.json}"
    if [[ -f "$directory/network.xml.gz" && \
          -f "$directory/transit_schedule.xml.gz" && \
          -f "$directory/link_mapping.csv.gz" && \
          -f "$directory/interventions.json" ]]; then
        printf '%s\n' "$directory" >> "$MANIFEST"
    fi
done < <(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name scenario.json | LC_ALL=C sort)
N="$(wc -l < "$MANIFEST")"
if (( N == 0 )); then
    rm -f "$MANIFEST"
    echo "No prepared scenarios found in $DATASET" >&2
    exit 1
fi

echo "Submitting $N scenarios, at most $MAX_PARALLEL simultaneous jobs."
echo "Manifest: $MANIFEST"
echo "Compaction: $([[ " ${FLAGS[*]} " == *" --no-compact "* ]] && echo disabled || echo enabled)"
echo "Delete raw outputs: $([[ " ${FLAGS[*]} " == *" --delete-raw "* ]] && echo YES || echo NO)"
SCRIPT="/vast.mnt/home/20201733/Master_Thesis/network_sampling/dataset/run_all_scenarios.sh"
# Split into arrays smaller than the cluster's MaxArraySize=1001.
# Each array uses its own manifest with local task indices starting at 0.
split -l 1000 -d -a 3 \
    "$MANIFEST" \
    "${MANIFEST%.txt}.part."

PREVIOUS_JOB_ID=""

for PART in "${MANIFEST%.txt}.part."*; do
    PART_N="$(wc -l < "$PART")"

    DEP_ARGS=()
    if [[ -n "$PREVIOUS_JOB_ID" ]]; then
        DEP_ARGS=(--dependency="afterany:$PREVIOUS_JOB_ID")
    fi

    JOB_ID="$(sbatch --parsable \
        --partition=mech-cst-mov.cpu.q \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=2 \
        --mem=6G \
        --time=00:30:00 \
        --job-name=matsim-dataset \
        --output="$DATASET/slurm_logs/%A_%a.out" \
        --error="$DATASET/slurm_logs/%A_%a.err" \
        --array="0-$((PART_N - 1))%$MAX_PARALLEL" \
        "${DEP_ARGS[@]}" \
        "$SCRIPT" --worker "$PART" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" "${FLAGS[@]}")"

    PREVIOUS_JOB_ID="${JOB_ID%%;*}"

    echo "Submitted $PART_N scenarios as array $PREVIOUS_JOB_ID"
done
