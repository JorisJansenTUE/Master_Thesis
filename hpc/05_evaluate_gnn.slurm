#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-eval
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_eval_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_eval_%j.err
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
GNN_DATA=/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_v3
VENV="${VENV:-$PROJECT_ROOT/.venv-gnn}"
# Use the original training run folder containing best.pt.
RUN_TAG="${RUN_TAG:-fullgraph_100Ep_resume_001}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"  # val, test, or train
CHECKPOINT="${CHECKPOINT:-$GNN_DATA/runs/$RUN_TAG/best.pt}"
OUTPUT="${OUTPUT:-$GNN_DATA/runs/$RUN_TAG/evaluation_$EVAL_SPLIT}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"   # 1: save compressed per-scenario arrays
PLOTS="${PLOTS:-1}"                         # 0: do not require matplotlib

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1

[[ -f "$CHECKPOINT" ]] || { echo "Missing best.pt: $CHECKPOINT" >&2; exit 1; }
[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing dataset splits" >&2; exit 1; }
if [[ "$PLOTS" == 1 ]]; then
    python -c 'import matplotlib' || {
        echo 'Matplotlib missing. Install into .venv-gnn: python -m pip install matplotlib==3.9.4' >&2
        exit 1
    }
fi

ARGS=(--gnn-dir "$GNN_DATA" --splits "$GNN_DATA/splits.json"
      --checkpoint "$CHECKPOINT" --split "$EVAL_SPLIT"
      --out "$OUTPUT" --device cpu)
if [[ "$SAVE_PREDICTIONS" == 1 ]]; then ARGS+=(--save-predictions); fi
if [[ "$PLOTS" != 1 ]]; then ARGS+=(--no-plots); fi

echo "$(date -Is) Evaluate $RUN_TAG, split=$EVAL_SPLIT, checkpoint=$CHECKPOINT"
srun python -u -m gnn_surrogate.evaluation.evaluate "${ARGS[@]}"
echo "$(date -Is) Finished: $OUTPUT/summary.json"
