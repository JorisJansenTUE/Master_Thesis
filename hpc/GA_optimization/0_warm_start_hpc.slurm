#!/bin/bash -l
#SBATCH --job-name=ga-warmstart
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=ga-warmstart_%j.out
#SBATCH --error=ga-warmstart_%j.err

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/hpc_config.sh"

module purge
module load Umbrella/2024
module load Python/3.10.13

cd "$HPC_REPO"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$HISTORICAL_BASELINE/network.xml.gz" ]] || { echo "Missing baseline network" >&2; exit 1; }
[[ -s "$HISTORICAL_BASELINE/transit_schedule.xml.gz" ]] || { echo "Missing baseline schedule" >&2; exit 1; }
[[ -s "$HISTORICAL_BASELINE/targets/linkstats.txt.gz" ]] || { echo "Missing compacted baseline linkstats" >&2; exit 1; }
for f in origins.csv jobs.csv pt_origin_connectors.csv pt_job_connectors.csv; do
    [[ -s "$SPATIAL_DIR/$f" ]] || { echo "Missing accessibility input: $SPATIAL_DIR/$f" >&2; exit 1; }
done

mkdir -p "$(dirname -- "$WARM_START_FILE")" "$(dirname -- "$REFERENCE_FILE")"
if [[ -s "$WARM_START_FILE" && "$WARM_REBUILD" != 1 ]]; then
    # In particular, never silently reuse a different budget (a previous pitfall).
    "$PYTHON_BIN" - "$WARM_START_FILE" "$GA_BUDGET_M" "$REFERENCE_FILE" \
        "$HISTORICAL_BASELINE/scenario.json" <<'PY_CHECK'
import hashlib, json, sys
from pathlib import Path
from optimization.fitness import FITNESS_VERSION
path, budget, reference, baseline = map(Path, sys.argv[1:])
data = json.loads(path.read_text())
source = json.loads(baseline.read_text())
mask = reference.with_name(reference.stem + "_attachments.json")
assert float(data["budget_m"]) == float(str(budget)), "Existing warm start uses another budget; set WARM_REBUILD=1"
assert data["fitness_version"] == FITNESS_VERSION, "Existing warm start uses another fitness version; set WARM_REBUILD=1"
assert data["baseline_hashes"] == source["baseline_hashes"], "Existing warm start uses another baseline; set WARM_REBUILD=1"
for file, field in ((reference, "reference_sha256"), (mask, "baseline_attachment_sha256")):
    assert file.is_file(), f"Missing original reference {file}; set WARM_REBUILD=1"
    assert hashlib.sha256(file.read_bytes()).hexdigest() == data[field], f"Existing warm-start {field} differs; set WARM_REBUILD=1"
PY_CHECK
    echo "Warm start already exists and matches baseline/budget/fitness: $WARM_START_FILE"
    exit 0
fi

echo "Worktree:         $HPC_REPO"
echo "Historical data:  $HISTORICAL_DATASET"
echo "Spatial data:     $SPATIAL_DIR"
echo "Baseline reference: $REFERENCE_FILE"
echo "Warm start output: $WARM_START_FILE"
echo "Budget: $GA_BUDGET_M m | Count: $WARM_COUNT | Limit: $WARM_LIMIT"

ARGS=(
  --dataset "$HISTORICAL_DATASET"
  --output "$WARM_START_FILE"
  --spatial-dir "$SPATIAL_DIR"
  --reference "$REFERENCE_FILE"
  --budget "$GA_BUDGET_M"
  --count "$WARM_COUNT"
  --max-overlap "$WARM_MAX_OVERLAP"
  --limit "$WARM_LIMIT"
  --workers "${SLURM_CPUS_PER_TASK:-1}"
)
if [[ "$WARM_EXISTING_ONLY" == 1 ]]; then ARGS+=(--existing-only); fi
"$PYTHON_BIN" -m optimization.warm_start "${ARGS[@]}"
echo "Warm-start preparation finished: $WARM_START_FILE"
