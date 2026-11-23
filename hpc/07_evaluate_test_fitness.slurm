#!/bin/bash -l
#SBATCH --job-name=locarno-fitness-test
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/fitness_test_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/fitness_test_%j.err
set -Eeuo pipefail

# Run AFTER 05_evaluate_gnn.slurm has saved --save-predictions for split=test.
# No MATSim simulation, model inference or optimisation is performed here.
# Use the GA's established environment, spatial inputs and fixed references.
PROJECT_ROOT="${PROJECT_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
source "$PROJECT_ROOT/hpc/GA_optimization/hpc_config.sh"
PROJECT_ROOT="$HPC_REPO"
GNN_DATA="${GNN_DATA:-$BASE/gnn_training_tt_delay}"
RUN_TAG="${RUN_TAG:-tt_delay_001}"
EVALUATION="${EVALUATION:-$GNN_DATA/runs/$RUN_TAG/evaluation_test}"
DATASET="${DATASET:-$HISTORICAL_DATASET}"
REFERENCE="${REFERENCE:-$REFERENCE_FILE}"
OUTPUT="${OUTPUT:-$GNN_DATA/runs/$RUN_TAG/fitness_test}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
LIMIT="${LIMIT:-0}"
FLOOR="${FLOOR:-positive}"
PLOTS="${PLOTS:-1}"
INIT_REFERENCE="${INIT_REFERENCE:-1}"
RESUME="${RESUME:-1}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.10.13}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing GA Python: $PYTHON_BIN" >&2; exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing test splits" >&2; exit 1; }
[[ -s "$GNN_DATA/gnn_static/dual_nodes.csv.gz" ]] || { echo "Missing GNN dual graph" >&2; exit 1; }
[[ -s "$DATASET/baseline_00000/targets/linkstats.txt.gz" ]] || { echo "Missing baseline linkstats" >&2; exit 1; }
[[ -s "$SPATIAL_DIR/origins.csv" ]] || { echo "Missing accessibility spatial data" >&2; exit 1; }
if [[ "$EVALUATION" != *.zip ]]; then
    [[ -s "$EVALUATION/summary.json" ]] || { echo "Missing test evaluation summary" >&2; exit 1; }
    [[ -d "$EVALUATION/predictions" ]] || { echo "Missing test predictions" >&2; exit 1; }
else
    [[ -s "$EVALUATION" ]] || { echo "Missing evaluation ZIP" >&2; exit 1; }
fi

ARGS=(
    --gnn-dir "$GNN_DATA"
    --evaluation "$EVALUATION"
    --dataset "$DATASET"
    --spatial-dir "$SPATIAL_DIR"
    --reference "$REFERENCE"
    --out "$OUTPUT"
    --workers "$WORKERS"
    --limit "$LIMIT"
    --floor "$FLOOR"
)
[[ "$PLOTS" == 1 ]] && ARGS+=(--plots)
[[ "$INIT_REFERENCE" == 1 ]] && ARGS+=(--initialize-baseline)
[[ "$RESUME" == 1 ]] && ARGS+=(--resume)

echo "$(date -Is) Evaluate test fitness for $RUN_TAG (workers=$WORKERS, limit=$LIMIT, floor=$FLOOR)"
srun "$PYTHON_BIN" -u -m gnn_surrogate.evaluation.evaluate_fitness "${ARGS[@]}"
echo "$(date -Is) Finished: $OUTPUT/ranking_metrics.json"
