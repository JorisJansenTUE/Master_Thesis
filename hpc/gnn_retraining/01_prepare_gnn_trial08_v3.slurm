#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-trial08-v3
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_trial08_v3_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_trial08_v3_%j.err
set -Eeuo pipefail

# Source-aware extension of the earlier expanded dataset. Parent split membership
# is frozen; both seed42 50-km GA runs are TRAIN. A different-seed GA is optional
# TEST. We do NOT infer independent validation from related seed42 trajectories.
SCRIPT_DIR="/vast.mnt/home/20201733/Master_Thesis/hpc/gnn_retraining"
PROJECT_ROOT="${PROJECT_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
BASE="${BASE:-/vast.mnt/home/20201733/locarno_ETH_Model}"
PARENT_GNN="${PARENT_GNN:-$BASE/gnn_training_tt_delay_expanded_v2}"
HISTORICAL="${HISTORICAL:-$BASE/training_dataset_v4}"
GA50_A="${GA50_A:-$BASE/GA_Optimization/ga_trial08_20min_park0_50km_seed42}"
GA50_B="${GA50_B:-$BASE/GA_Optimization/ga_trial08_20min_park0_50km_seed42_explore}"
TEST_GA_DIR="${TEST_GA_DIR:-}"  # optional, completed independent-seed GA, NEVER a seed42 continuation
GNN_DATA="${GNN_DATA:-$BASE/gnn_training_tt_delay_expanded_v3}"
VENV="${VENV:-/vast.mnt/home/20201733/Master_Thesis/.venv-gnn}"
PREP_WORKERS="${PREP_WORKERS:-16}"
PREP_RESUME="${PREP_RESUME:-0}"
PEAK_HOUR="${PEAK_HOUR:-8}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MIN_PER_RUN="${MIN_PER_RUN:-1}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
[[ -x "$VENV/bin/python" ]] || { echo "Missing GNN Python: $VENV/bin/python" >&2; exit 1; }
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

for file in "$SCRIPT_DIR/stage_expanded_gnn_v3.py" \
            "$SCRIPT_DIR/stage_add_ga_v4.py" \
            "$SCRIPT_DIR/install_gnn_splits_v3.py" \
            "$PROJECT_ROOT/gnn_surrogate/dataset/prepare_hpc.py" \
            "$PARENT_GNN/splits.json" \
            "$PARENT_GNN/preparation_metadata/source_manifest.csv"; do
  [[ -s "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done
for dir in "$HISTORICAL" "$GA50_A" "$GA50_B"; do
  [[ -d "$dir" ]] || { echo "Missing directory: $dir" >&2; exit 1; }
done
if [[ -n "$TEST_GA_DIR" ]]; then
  [[ -d "$TEST_GA_DIR" ]] || { echo "Missing test GA directory: $TEST_GA_DIR" >&2; exit 1; }
fi
if [[ -e "$GNN_DATA" && "$PREP_RESUME" != 1 ]]; then
  echo "Output exists: $GNN_DATA (set PREP_RESUME=1 only for EXACT same split and sources)" >&2
  exit 1
fi

STAGING_ROOT="$(mktemp -d "$BASE/.gnn_trial08_stage.${SLURM_JOB_ID:-manual}.XXXXXXXX")"
trap 'rm -rf -- "$STAGING_ROOT"' EXIT
STAGED="$STAGING_ROOT/scenarios"
args=(--parent-gnn "$PARENT_GNN" --historical "$HISTORICAL" --out "$STAGED"
      --seed "$SPLIT_SEED" --min-per-run "$MIN_PER_RUN"
      --train-ga "ga_trial08_seed42=$GA50_A"
      --train-ga "ga_trial08_seed42_explore=$GA50_B")
if [[ -n "$TEST_GA_DIR" ]]; then
  args+=(--test-ga "ga_trial08_new_seed_test=$TEST_GA_DIR")
fi

echo "$(date -Is) Stage + verify fixed split and new GA scenarios"
python -u "$SCRIPT_DIR/stage_add_ga_v4.py" "${args[@]}"

echo "$(date -Is) Install expanded explicit split BEFORE preprocessing"
resume_args=()
[[ "$PREP_RESUME" == 1 ]] && resume_args+=(--resume)
python -u "$SCRIPT_DIR/install_gnn_splits_v3.py" --staged "$STAGED" --gnn-out "$GNN_DATA" --phase before "${resume_args[@]}"

echo "$(date -Is) Prepare scenario tensors"
python -u -m gnn_surrogate.dataset.prepare_hpc --dataset "$STAGED" --out "$GNN_DATA" \
    --workers "$PREP_WORKERS" --seed "$SPLIT_SEED" --hour "$PEAK_HOUR"

echo "$(date -Is) Verify all intended tensors and split membership"
python -u "$SCRIPT_DIR/install_gnn_splits_v3.py" --staged "$STAGED" --gnn-out "$GNN_DATA" --phase after

echo "$(date -Is) READY: $GNN_DATA. Training must use this new directory and fresh checkpoint tag."
