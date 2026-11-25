#!/bin/bash -l
#SBATCH --job-name=locarno-GA-ranking
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/GA_ranking_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/GA_ranking_%j.err
set -Eeuo pipefail

# End-to-end OLD frozen GNN evaluation on new GA scenarios:
#   1. Create a source-aware external GA split, verify graph and checkpoint
#   2. Run original evaluate.py to generate predicted road travel times
#   3. Use unchanged 07 fitness/routing functions for constrained ranking
# Original 05 and 07 stay untouched. No MATSim simulations are run.

GA_ROOT="${GA_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
GNN_CODE_ROOT="${GNN_CODE_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
source "$GA_ROOT/hpc/GA_optimization/hpc_config.sh"
BASE="${BASE:-/vast.mnt/home/20201733/locarno_ETH_Model}"
OLD_GNN="${OLD_GNN:-$BASE/gnn_training_tt_delay}"
PREPARED_GNN="${PREPARED_GNN:-$BASE/gnn_training_tt_delay_expanded_v2}"
MODEL_TAG="${MODEL_TAG:-tt_delay_001}"
CHECKPOINT="${CHECKPOINT:-$OLD_GNN/runs/$MODEL_TAG/best.pt}"
GNN_VENV="${GNN_VENV:-$GA_ROOT/.venv-gnn}"
GNN_PYTHON="$GNN_VENV/bin/python"
HISTORICAL="${HISTORICAL:-$HISTORICAL_DATASET}"
SPATIAL="${SPATIAL:-$SPATIAL_DIR}"
REFERENCE="${REFERENCE:-$(dirname "$REFERENCE_FILE")/sufficiency_reference_20min_park5.json}"
GROUP="${GROUP:-ga_spatial_no_warm}" # or ga_spatial_warm, ga_nonspatial_old
LIMIT="${LIMIT:-0}"                   # 10 for pilot; 0 = entire group
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
FLOOR="${FLOOR:-freeflow}"
PLOTS="${PLOTS:-1}"
RESUME="${RESUME:-0}"                 # resume completed predictions or partial ranking
OUTPUT="${OUTPUT:-$OLD_GNN/runs/$MODEL_TAG/GA_ranking_${GROUP}_${FLOOR}}"
SCRIPTS="$GA_ROOT/hpc/ranking_test"

module purge
module load Umbrella/2024
module load "${GNN_PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 MPLBACKEND=Agg

[[ -x "$GNN_PYTHON" ]] || { echo "Missing GNN venv: $GNN_PYTHON" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing GA venv: $PYTHON_BIN" >&2; exit 1; }
[[ -s "$CHECKPOINT" ]] || { echo "Missing ORIGINAL GNN checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -s "$PREPARED_GNN/preparation_report.json" ]] || { echo "Expanded preprocessing not complete" >&2; exit 1; }
[[ -s "$PREPARED_GNN/preparation_metadata/source_manifest.csv" ]] || { echo "Missing expanded source manifest" >&2; exit 1; }
[[ -s "$PREPARED_GNN/splits.json" ]] || { echo "Missing expanded splits" >&2; exit 1; }
[[ -s "$REFERENCE" && -s "${REFERENCE%.json}_attachments.json" ]] || { echo "Missing fixed baseline fitness reference/mask" >&2; exit 1; }
[[ -s "$SPATIAL/origins.csv" ]] || { echo "Missing spatial inputs" >&2; exit 1; }
for f in prepare_ga_inference.py ga_ranking_external.py; do
    [[ -s "$SCRIPTS/$f" ]] || { echo "Missing companion $SCRIPTS/$f" >&2; exit 1; }
done
if [[ "$RESUME" != 1 && -e "$OUTPUT" ]]; then
    echo "OUTPUT exists; use a fresh OUTPUT or RESUME=1: $OUTPUT" >&2; exit 1;
fi

# === STAGE 1: verify frozen model inputs, install external test split ===
cd "$GNN_CODE_ROOT"
export PYTHONPATH="$GNN_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$GNN_PYTHON" -u "$SCRIPTS/prepare_ga_inference.py" \
    --old-gnn "$OLD_GNN" --prepared-gnn "$PREPARED_GNN" \
    --checkpoint "$CHECKPOINT" --group "$GROUP" --limit "$LIMIT" --out "$OUTPUT"

INFERENCE="$OUTPUT/inference"
if [[ ! -s "$INFERENCE/summary.json" ]]; then
    if [[ -e "$INFERENCE" ]]; then
        echo "Partial inference folder found: $INFERENCE" >&2
        echo "Do not mix partial predictions. Use a fresh OUTPUT or rename/remove ONLY that inference folder." >&2
        exit 1
    fi
    echo "$(date -Is) OLD frozen checkpoint inference: group=$GROUP, limit=$LIMIT"
    cd "$GNN_CODE_ROOT"
    "$GNN_PYTHON" -u -m gnn_surrogate.evaluation.evaluate \
        --gnn-dir "$PREPARED_GNN" \
        --splits "$OUTPUT/external_test_splits.json" \
        --checkpoint "$CHECKPOINT" \
        --split test --out "$INFERENCE" --device cpu \
        --no-plots --save-predictions
fi
# Exact predictions and provenance MUST exist before accessibility scoring.
"$GNN_PYTHON" -u "$SCRIPTS/prepare_ga_inference.py" \
    --old-gnn "$OLD_GNN" --prepared-gnn "$PREPARED_GNN" \
    --checkpoint "$CHECKPOINT" --group "$GROUP" --limit "$LIMIT" --out "$OUTPUT" \
    --verify-predictions

# === STAGE 2: unchanged 07 accessibility/scoring and GA ranking ===
module purge
module load Umbrella/2024
module load "${GA_PYTHON_MODULE:-Python/3.10.13}"
cd "$GA_ROOT"
export PYTHONPATH="$GA_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
[[ -s "$GA_ROOT/gnn_surrogate/evaluation/evaluate_fitness.py" ]] || {
    echo "07-style evaluator missing in GA_ROOT=$GA_ROOT" >&2; exit 1;
}
"$PYTHON_BIN" - <<'PY'
from gnn_surrogate.evaluation import evaluate_fitness as e
assert e.FITNESS_EVALUATION_VERSION == 'squared_sustainable_car_constraint_v1', (
    'Unexpected fitness evaluator version'
)
PY
if [[ -e "$OUTPUT/ranking" && "$RESUME" != 1 ]]; then
    echo "Ranking output already exists; set RESUME=1 or choose a new OUTPUT" >&2; exit 1
fi
ARGS=(--old-gnn "$OLD_GNN" --prepared-gnn "$PREPARED_GNN"
      --evaluation "$INFERENCE" --historical "$HISTORICAL"
      --spatial "$SPATIAL" --reference "$REFERENCE"
      --out "$OUTPUT/ranking" --group "$GROUP" --workers "$WORKERS"
      --limit "$LIMIT" --floor "$FLOOR")
[[ "$PLOTS" == 1 ]] && ARGS+=(--plots)
[[ "$RESUME" == 1 && -e "$OUTPUT/ranking/evaluation_config.json" ]] && ARGS+=(--resume)
echo "$(date -Is) Accessibility and fitness ranking: group=$GROUP workers=$WORKERS"
"$PYTHON_BIN" -u "$SCRIPTS/ga_ranking_external.py" "${ARGS[@]}"
echo "$(date -Is) COMPLETE: $OUTPUT/ranking/ranking_metrics.json"
