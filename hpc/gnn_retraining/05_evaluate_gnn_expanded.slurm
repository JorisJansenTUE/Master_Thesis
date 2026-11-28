#!/bin/bash -l
#SBATCH --job-name=locarno-delay-eval
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_retrain_eval_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_retrain_eval_%j.err
set -Eeuo pipefail

# Python process-parallel evaluation; all workers run within this one SLURM job.
# Specify RUN_TAG explicitly to prevent accidentally evaluating the old model.
PROJECT_ROOT="${PROJECT_ROOT:-/vast.mnt/home/20201733/Master_Thesis}"
BASE="${BASE:-/vast.mnt/home/20201733/locarno_ETH_Model}"
GNN_DATA="${GNN_DATA:-$BASE/gnn_training_tt_delay_expanded_v2}"
VENV="${VENV:-/vast.mnt/home/20201733/Master_Thesis/.venv-gnn}"
: "${RUN_TAG:?Set RUN_TAG to the retrained model run directory under GNN_DATA/runs/}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-$GNN_DATA/runs/$RUN_TAG/best.pt}"
OUTPUT="${OUTPUT:-$GNN_DATA/runs/$RUN_TAG/evaluation_${EVAL_SPLIT}_parallel}"
HISTORY="${HISTORY:-$GNN_DATA/runs/$RUN_TAG/history.csv}"
WORKERS="${WORKERS:-8}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-2}"
PLOTS="${PLOTS:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
RESUME="${RESUME:-0}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

[[ -x "$VENV/bin/python" ]] || { echo "Missing GNN Python: $VENV/bin/python" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing dataset split manifest" >&2; exit 1; }
[[ -s "$PROJECT_ROOT/gnn_surrogate/evaluation/evaluate.py" ]] || { echo "Missing evaluation source in $PROJECT_ROOT" >&2; exit 1; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$THREADS_PER_WORKER" =~ ^[1-9][0-9]*$ ]] || {
  echo 'WORKERS and THREADS_PER_WORKER must be positive integers' >&2; exit 1;
}
if (( WORKERS * THREADS_PER_WORKER > SLURM_CPUS_PER_TASK )); then
  echo "Oversubscribed: $WORKERS x $THREADS_PER_WORKER > $SLURM_CPUS_PER_TASK" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$PROJECT_ROOT"

ARGS=(--gnn-dir "$GNN_DATA" --splits "$GNN_DATA/splits.json"
      --checkpoint "$CHECKPOINT" --split "$EVAL_SPLIT"
      --out "$OUTPUT" --device cpu --history "$HISTORY"
      --workers "$WORKERS" --threads-per-worker "$THREADS_PER_WORKER")
[[ "$PLOTS" == 1 ]] || ARGS+=(--no-plots)
[[ "$SAVE_PREDICTIONS" == 1 ]] && ARGS+=(--save-predictions)
[[ "$RESUME" == 1 ]] && ARGS+=(--resume)

"$VENV/bin/python" - <<'PY'
import inspect
from gnn_surrogate.evaluation import evaluate
assert '--workers' in inspect.getsource(evaluate.main), 'Parallel evaluator not installed in PROJECT_ROOT'
PY

echo "$(date -Is) Start $RUN_TAG; split=$EVAL_SPLIT; workers=$WORKERS; threads=$THREADS_PER_WORKER; resume=$RESUME"
srun "$VENV/bin/python" -u -m gnn_surrogate.evaluation.evaluate "${ARGS[@]}"
echo "$(date -Is) Complete: $OUTPUT/summary.json"
