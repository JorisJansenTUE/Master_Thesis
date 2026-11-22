#!/bin/bash -l
#SBATCH --job-name=ga-overnight
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=ga-overnight_%j.out
#SBATCH --error=ga-overnight_%j.err

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/hpc_config.sh"

module purge
module load Umbrella/2024
module load Python/3.10.13
module load EESSI/2026.06
module load Java/25.36

cd "$HPC_REPO"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }
for f in "$WARM_START_FILE" "$REFERENCE_FILE" "$HISTORICAL_BASELINE/network.xml.gz" \
         "$HISTORICAL_BASELINE/transit_schedule.xml.gz" "$HISTORICAL_BASELINE/run_status.json" \
         "$MATSIM_CONFIG" "$MATSIM_POPULATION" "$MATSIM_JAR"; do
    [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
for f in origins.csv jobs.csv pt_origin_connectors.csv pt_job_connectors.csv; do
    [[ -s "$SPATIAL_DIR/$f" ]] || { echo "Missing accessibility input: $SPATIAL_DIR/$f" >&2; exit 1; }
done
[[ -s "${REFERENCE_FILE%.json}_attachments.json" ]] || {
    echo "Missing baseline attachment mask next to $REFERENCE_FILE" >&2; exit 1;
}

mkdir -p "$GA_DATASET"
exec 9>"$GA_DATASET/.overnight_controller.lock"
flock -n 9 || { echo "Another GA controller is already operating in $GA_DATASET" >&2; exit 1; }

echo "Worktree:       $HPC_REPO"
echo "GA output:      $GA_DATASET"
echo "Warm start:     $WARM_START_FILE"
echo "Parallel MATSim: $GA_MAX_PARALLEL | Budget: $GA_BUDGET_M m"

"$PYTHON_BIN" -m optimization.hpc_overnight \
  --repo "$HPC_REPO" \
  --python "$PYTHON_BIN" \
  --dataset "$GA_DATASET" \
  --baseline "$HISTORICAL_BASELINE" \
  --warm-start "$WARM_START_FILE" \
  --spatial-dir "$SPATIAL_DIR" \
  --reference "$REFERENCE_FILE" \
  --config "$MATSIM_CONFIG" \
  --population-file "$MATSIM_POPULATION" \
  --jar "$MATSIM_JAR" \
  --budget "$GA_BUDGET_M" \
  --population "$GA_POPULATION" \
  --generations "$GA_GENERATIONS" \
  --seed "$GA_SEED" \
  --elites "$GA_ELITES" \
  --immigrants "$GA_IMMIGRANTS" \
  --max-parallel "$GA_MAX_PARALLEL" \
  --delete-raw "$GA_DELETE_RAW" \
  --penalize-failed "$GA_PENALIZE_FAILED" \
  --stop-margin-min "$GA_STOP_MARGIN_MIN" \
  --poll-sec "$GA_POLL_SEC" \
  --dry-run "$GA_DRY_RUN"
