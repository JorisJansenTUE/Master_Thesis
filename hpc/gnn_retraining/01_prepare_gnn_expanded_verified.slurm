#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-expanded
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_expanded_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_expanded_%j.err
set -Eeuo pipefail

# All THREE files must reside together. Only GNN_DATA is persistent.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/hpc/gnn_retraining}"
PROJECT_ROOT="${PROJECT_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
BASE="${BASE:-/vast.mnt/home/20201733/locarno_ETH_Model}"
HISTORICAL_DATASET="${HISTORICAL_DATASET:-$BASE/training_dataset_v4}"
ORIGINAL_GNN="${ORIGINAL_GNN:-$BASE/gnn_training_tt_delay}"
WARM_GA_DIR="${WARM_GA_DIR:-$BASE/GA_Optimization/ga_20min_park5_15km_run01}"
NO_WARM_GA_DIR="${NO_WARM_GA_DIR:-$BASE/GA_Optimization/ga_20min_park5_15km_seed1234_no_warm}"
OLD_GA_ROOT="${OLD_GA_ROOT:-$BASE/GA_Optimization}"
INCLUDE_OLD_GA="${INCLUDE_OLD_GA:-1}"
GNN_DATA="${GNN_DATA:-$BASE/gnn_training_tt_delay_expanded_v2}"
PEAK_HOUR="${PEAK_HOUR:-8}"
SPLIT_SEED="${SPLIT_SEED:-42}"
WARM_VAL_FRACTION="${WARM_VAL_FRACTION:-0.15}"
VENV="${VENV:-/vast.mnt/home/20201733/Master_Thesis/.venv-gnn}"
WORKERS="${PREP_WORKERS:-12}"
PREP_RESUME="${PREP_RESUME:-0}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
[[ -x "$VENV/bin/python" ]] || { echo "Missing GNN Python: $VENV/bin/python" >&2; exit 1; }
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

[[ -s "$ORIGINAL_GNN/splits.json" ]] || { echo "Original splits missing" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/gnn_surrogate/dataset/prepare_hpc.py" ]] || { echo "prepare_hpc.py missing" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/stage_expanded_gnn_v3.py" && -f "$SCRIPT_DIR/install_gnn_splits_v3.py" ]] || { echo "Companion scripts missing" >&2; exit 1; }
[[ -d "$HISTORICAL_DATASET" && -d "$WARM_GA_DIR" && -d "$NO_WARM_GA_DIR" ]] || { echo "Missing historical/warm/no-warm source" >&2; exit 1; }
if [[ "$INCLUDE_OLD_GA" == 1 ]]; then
  [[ -d "$OLD_GA_ROOT" ]] || { echo "Missing old GA root: $OLD_GA_ROOT" >&2; exit 1; }
fi
if [[ -e "$GNN_DATA" && "$PREP_RESUME" != 1 ]]; then
  echo "GNN output exists: $GNN_DATA (set PREP_RESUME=1 for exact-source continuation)" >&2; exit 1
fi

# Temporary symlinks only, NO copies of networks, linkstats or plans.
STAGING_ROOT="$(mktemp -d "$BASE/.gnn_stage.${SLURM_JOB_ID:-manual}.XXXXXXXX")"
trap 'rm -rf -- "$STAGING_ROOT"' EXIT
STAGED="$STAGING_ROOT/scenarios"

ARGS=(--historical "$HISTORICAL_DATASET" --original-splits "$ORIGINAL_GNN/splits.json"
      --warm-ga "$WARM_GA_DIR" --no-warm-ga "$NO_WARM_GA_DIR"
      --out "$STAGED" --seed "$SPLIT_SEED" --warm-val-fraction "$WARM_VAL_FRACTION")
if [[ "$INCLUDE_OLD_GA" == 1 ]]; then
    ARGS+=(--old-ga-root "$OLD_GA_ROOT")
else
    ARGS+=(--skip-old)
fi

echo "$(date -Is) Stage source scenario symlinks and source-aware split plan"
python -u "$SCRIPT_DIR/stage_expanded_gnn_v3.py" "${ARGS[@]}"

echo "$(date -Is) Install train/val/test splits BEFORE preprocessing"
PREP_SPLIT_ARGS=()
[[ "$PREP_RESUME" == 1 ]] && PREP_SPLIT_ARGS+=(--resume)
python -u "$SCRIPT_DIR/install_gnn_splits_v3.py" --staged "$STAGED" --gnn-out "$GNN_DATA" --phase before "${PREP_SPLIT_ARGS[@]}"

echo "$(date -Is) Preprocess all included train, validation and test tensors"
python -u -m gnn_surrogate.dataset.prepare_hpc \
  --dataset "$STAGED" --out "$GNN_DATA" --workers "$WORKERS" \
  --seed "$SPLIT_SEED" --hour "$PEAK_HOUR"

echo "$(date -Is) Verify completion and strict split separation"
python -u "$SCRIPT_DIR/install_gnn_splits_v3.py" --staged "$STAGED" --gnn-out "$GNN_DATA" --phase after

echo "$(date -Is) PASS: $GNN_DATA ready for GNN training. Raw MATSim sources unchanged."
