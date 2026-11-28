#!/bin/bash -l
#SBATCH --job-name=ga-trial08-overnight
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis_trial08/slurm_logs/ga-trial08-overnight_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis_trial08/slurm_logs/ga-trial08-overnight_%j.err

set -Eeuo pipefail
source "$SLURM_SUBMIT_DIR/hpc/GA_optimization/hpc_config.sh"
module purge
module load Umbrella/2024
module load Python/3.10.13
module load EESSI/2026.06
module load Java/25.36

#temporary adds for test set
export GA_DATASET="/home/20201733/locarno_ETH_Model/GA_Optimization/ga_trial08_20min_park0_50km_seed123_test"
export GA_SEED=123
export GA_GENERATIONS=50
export GA_NO_WARM_START=1
export WARM_START_FILE="$GA_DATASET/warm_start.json"

cd "$HPC_REPO"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }
for f in "$REFERENCE_FILE" "$HISTORICAL_BASELINE/network.xml.gz" \
         "$HISTORICAL_BASELINE/transit_schedule.xml.gz" "$HISTORICAL_BASELINE/run_status.json" \
         "$MATSIM_CONFIG" "$MATSIM_POPULATION" "$MATSIM_JAR"; do
    [[ -s "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
if [[ "${GA_NO_WARM_START:-0}" != 1 ]]; then
    [[ -s "$WARM_START_FILE" ]] || { echo "Missing warm start: $WARM_START_FILE" >&2; exit 1; }
    "$PYTHON_BIN" - "$WARM_START_FILE" "$WARM_EXCLUDE_TEST_SPLIT" <<'PY_SPLIT'
import json, sys, hashlib
from pathlib import Path
warm, split = map(Path, sys.argv[1:])
a = json.loads(warm.read_text())
assert a['excluded_test_split_sha256'] == hashlib.sha256(split.read_bytes()).hexdigest(), 'Frozen GNN split changed'
manifest = split.parent / 'preparation_metadata' / 'source_manifest.csv'
if manifest.is_file():
    assert a['excluded_source_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest(), 'Frozen GNN source manifest changed'
PY_SPLIT
fi
for f in origins.csv jobs.csv pt_origin_connectors.csv pt_job_connectors.csv population_by_hexagon.csv; do
    [[ -s "$SPATIAL_DIR/$f" ]] || { echo "Missing accessibility input: $SPATIAL_DIR/$f" >&2; exit 1; }
done
[[ -s "${REFERENCE_FILE%.json}_attachments.json" ]] || {
    echo "Missing baseline attachment mask next to $REFERENCE_FILE" >&2; exit 1;
}

if (( GA_FITNESS_WORKERS < 1 || GA_FITNESS_WORKERS > ${SLURM_CPUS_PER_TASK:-1} )); then
    echo "GA_FITNESS_WORKERS exceeds allocated CPUs or is invalid" >&2; exit 1
fi
mkdir -p "$GA_DATASET"
exec 9>"$GA_DATASET/.overnight_controller.lock"
flock -n 9 || { echo "Another GA controller is already operating in $GA_DATASET" >&2; exit 1; }

echo "Worktree:       $HPC_REPO"
echo "GA output:      $GA_DATASET"
echo "Warm start:     $WARM_START_FILE"
echo "Parallel MATSim: $GA_MAX_PARALLEL | Parallel accessibility: $GA_FITNESS_WORKERS | Budget: $GA_BUDGET_M m"
echo "Population: $GA_POPULATION | Generations: $GA_GENERATIONS (including generation 0)"
echo "MATSim last iteration: $GA_LAST_ITERATION | Linkstats: $GA_LINKSTATS_INTERVAL/$GA_LINKSTATS_AVERAGE_ITERATIONS"
# Reject inconsistent MATSim settings before submitting hundreds of simulations.
"$PYTHON_BIN" - "$HISTORICAL_BASELINE/run_status.json" <<'PY_BASELINE'
import json, os, sys
from pathlib import Path
status = json.loads(Path(sys.argv[1]).read_text())
expected = {
    'last_iteration': int(os.environ['GA_LAST_ITERATION']),
    'linkstats_interval': int(os.environ['GA_LINKSTATS_INTERVAL']),
    'linkstats_average_iterations': int(os.environ['GA_LINKSTATS_AVERAGE_ITERATIONS']),
}
for name, value in expected.items():
    actual = status.get(name)
    if actual is None:
        print(f'WARNING: baseline run_status.json does not record {name}; confirm settings manually')
    elif int(actual) != value:
        raise SystemExit(f'Baseline {name}={actual} differs from GA worker {value}; change GA settings before running')
PY_BASELINE

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
  --fitness-workers "$GA_FITNESS_WORKERS" \
  --delete-raw "$GA_DELETE_RAW" \
  --penalize-failed "$GA_PENALIZE_FAILED" \
  --stop-margin-min "$GA_STOP_MARGIN_MIN" \
  --poll-sec "$GA_POLL_SEC" \
  --dry-run "$GA_DRY_RUN"
