#!/bin/bash -l
#SBATCH --job-name=locarno-fitness-20m
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/fitness_20min_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/fitness_20min_%j.err
set -Eeuo pipefail

# Requires the revised gnn_surrogate/evaluation/evaluate_fitness.py.
# The script checks the evaluator version before starting any scenario work.
# Reuses GNN test predictions saved by 05_evaluate_gnn.slurm.

PROJECT_ROOT="${PROJECT_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
source "$PROJECT_ROOT/hpc/GA_optimization/hpc_config.sh"
PROJECT_ROOT="$HPC_REPO"
GNN_DATA="${GNN_DATA:-$BASE/gnn_training_tt_delay}"
RUN_TAG="${RUN_TAG:-tt_delay_001}"
EVALUATION="${EVALUATION:-$GNN_DATA/runs/$RUN_TAG/evaluation_test}"
DATASET="${DATASET:-$HISTORICAL_DATASET}"
SPATIAL="${SPATIAL:-$SPATIAL_DIR}"
# Keep the opportunity dataset/directory from hpc_config.sh;
# change the reference filename only (no legacy 15-minute reference).
REFERENCE="${REFERENCE:-$(dirname "$REFERENCE_FILE")/sufficiency_reference_20min_park5.json}"
OUTPUT="${OUTPUT:-$GNN_DATA/runs/$RUN_TAG/fitness_test_20min_park5_squared_carconstraint}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
LIMIT="${LIMIT:-0}"
FLOOR="${FLOOR:-positive}"
PLOTS="${PLOTS:-1}"
RESUME="${RESUME:-0}"  # start clean; enable only for interruptions of this new run

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.10.13}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing GA Python: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing test splits" >&2; exit 1; }
[[ -s "$GNN_DATA/gnn_static/dual_nodes.csv.gz" ]] || { echo "Missing GNN dual graph" >&2; exit 1; }
[[ -s "$DATASET/baseline_00000/targets/linkstats.txt.gz" ]] || { echo "Missing baseline linkstats" >&2; exit 1; }
[[ -s "$SPATIAL/origins.csv" ]] || { echo "Missing accessibility spatial inputs" >&2; exit 1; }
[[ -s "$REFERENCE" ]] || { echo "Missing NEW baseline reference: $REFERENCE" >&2; echo "Run optimization.inspect_fitness on the HPC baseline first." >&2; exit 1; }
ATTACHMENTS="${REFERENCE%.json}_attachments.json"
[[ -s "$ATTACHMENTS" ]] || { echo "Missing NEW baseline eligibility mask: $ATTACHMENTS" >&2; echo "Run optimization.inspect_fitness on the HPC baseline first." >&2; exit 1; }

if [[ "$EVALUATION" != *.zip ]]; then
    [[ -s "$EVALUATION/summary.json" ]] || { echo "Missing test evaluation summary" >&2; exit 1; }
    [[ -d "$EVALUATION/predictions" ]] || { echo "Missing saved test predictions" >&2; exit 1; }
else
    [[ -s "$EVALUATION" ]] || { echo "Missing evaluation ZIP" >&2; exit 1; }
fi

# Verify the reference is not the old 15-minute/unparked one.
"$PYTHON_BIN" - "$REFERENCE" "$ATTACHMENTS" <<'PY'
import json, sys
from pathlib import Path
ref, mask = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
assert ref.get('version') == 2, 'Unexpected baseline reference version'
assert ref.get('car_parking_penalty_s') == 300.0, 'Wrong parking allowance'
assert set(ref.get('mean_car_jobs', {})) == {'jobs_20min'}, 'Wrong time cutoff'
assert ref.get('weighting') == 'unweighted_hexagons', 'Population weighting must remain off'
assert mask.get('version') == 2, 'Unexpected baseline eligibility mask version'
assert mask.get('eligibility_policy') == 'fixed_baseline_any_mode_attached', 'Wrong eligibility policy'
assert mask.get('eligible_origins'), 'Empty eligible origin set'
print('Validated 20-minute/5-minute-parking reference and fixed origin mask')
PY

# Fail closed: old screening evaluators can run with no errors but report old fitness.
"$PYTHON_BIN" - <<'PY'
from gnn_surrogate.evaluation import evaluate_fitness as evaluator
expected = 'squared_sustainable_car_constraint_v1'
actual = getattr(evaluator, 'FITNESS_EVALUATION_VERSION', None)
if actual != expected:
    raise SystemExit(
        f'Screening evaluator is not certified for new fitness: {actual!r} != {expected!r}. '
        'Update gnn_surrogate/evaluation/evaluate_fitness.py before submitting this job.'
    )
PY

if [[ "$RESUME" != 1 && -e "$OUTPUT" ]]; then
    echo "Output already exists. Set a new OUTPUT or use RESUME=1 for this exact definition: $OUTPUT" >&2
    exit 1
fi

ARGS=(
    --gnn-dir "$GNN_DATA"
    --evaluation "$EVALUATION"
    --dataset "$DATASET"
    --spatial-dir "$SPATIAL"
    --reference "$REFERENCE"
    --out "$OUTPUT"
    --workers "$WORKERS"
    --limit "$LIMIT"
    --floor "$FLOOR"
)
[[ "$PLOTS" == 1 ]] && ARGS+=(--plots)
[[ "$RESUME" == 1 ]] && ARGS+=(--resume)

# No --initialize-baseline: the new GA baseline reference and origin mask
# must already exist, generated with the revised optimization.inspect_fitness.
echo "$(date -Is) Screening $RUN_TAG; output=$OUTPUT; reference=$REFERENCE"
srun "$PYTHON_BIN" -u -m gnn_surrogate.evaluation.evaluate_fitness "${ARGS[@]}"
echo "$(date -Is) Finished: $OUTPUT/ranking_metrics.json"
