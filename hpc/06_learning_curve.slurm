#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-curve
#SBATCH --partition=tue.gpu2.q,tue.gpu3.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_curve_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_curve_%j.err
set -Eeuo pipefail
trap 'echo "[$(date -Is)] Error on line $LINENO" >&2' ERR

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
GNN_DATA=/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_v3
VENV="$PROJECT_ROOT/.venv-gnn"
REFERENCE="$GNN_DATA/runs/fullgraph_100Ep_resume_001/best.pt"
CURVE_TAG="${CURVE_TAG:-learning_curve_001}"
OUT="$GNN_DATA/runs/$CURVE_TAG"

module purge
module load Umbrella/2024
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

[[ -f "$GNN_DATA/splits.json" ]] || { echo 'Missing splits.json' >&2; exit 1; }
[[ -f "$REFERENCE" ]] || { echo "Missing reference checkpoint $REFERENCE" >&2; exit 1; }
mkdir -p "$OUT"

python - <<'PY'
import torch, torch_geometric
assert torch.cuda.is_available(), 'No CUDA GPU allocated'
print('Torch:', torch.__version__, 'PyG:', torch_geometric.__version__, flush=True)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

printf '%s Starting learning curve: 150 / 350 / 750 / 1100; full reference=1423\n' "$(date -Is)"
srun python -u -m gnn_surrogate.training.learning_curve \
  --gnn-dir "$GNN_DATA" \
  --splits "$GNN_DATA/splits.json" \
  --reference "$REFERENCE" \
  --out "$OUT" \
  --sizes 150 350 750 1100 \
  --epochs 100 \
  --patience 10 \
  --subset-seed 2026 \
  --model-seed 42
printf '%s Completed learning curve: %s\n' "$(date -Is)" "$OUT/learning_curve.csv"
