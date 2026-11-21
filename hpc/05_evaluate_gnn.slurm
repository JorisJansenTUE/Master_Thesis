#!/bin/bash -l
#SBATCH --job-name=locarno-delay-eval
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis_delay/slurm_logs/gnn_delay_eval_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis_delay/slurm_logs/gnn_delay_eval_%j.err
set -Eeuo pipefail

# Source files are read from the delay experiment worktree.
PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis_delay
GNN_DATA="${GNN_DATA:-/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_tt_delay}"
VENV="${VENV:-/vast.mnt/home/20201733/Master_Thesis/.venv-gnn}"
RUN_TAG="${RUN_TAG:-tt_delay_001}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
CHECKPOINT="${CHECKPOINT:-$GNN_DATA/runs/$RUN_TAG/best.pt}"
OUTPUT="${OUTPUT:-$GNN_DATA/runs/$RUN_TAG/evaluation_$EVAL_SPLIT}"
PLOTS="${PLOTS:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
HISTORY="${HISTORY:-$GNN_DATA/runs/$RUN_TAG/history.csv}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing dataset splits" >&2; exit 1; }
if [[ "$PLOTS" == 1 ]]; then
    python -c 'import matplotlib' || { echo 'Missing matplotlib in venv' >&2; exit 1; }
fi

ARGS=(--gnn-dir "$GNN_DATA" --splits "$GNN_DATA/splits.json"
      --checkpoint "$CHECKPOINT" --split "$EVAL_SPLIT"
      --out "$OUTPUT" --device cpu --history "$HISTORY")
if [[ "$PLOTS" != 1 ]]; then ARGS+=(--no-plots); fi
if [[ "$SAVE_PREDICTIONS" == 1 ]]; then ARGS+=(--save-predictions); fi

echo "$(date -Is) Delay-model evaluation: run=$RUN_TAG split=$EVAL_SPLIT"
srun python -u -m gnn_surrogate.evaluation.evaluate "${ARGS[@]}"
echo "$(date -Is) Evaluation finished: $OUTPUT/summary.json"
