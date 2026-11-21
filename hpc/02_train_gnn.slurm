#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-train-delay
#SBATCH --partition=tue.gpu2.q,tue.gpu3.q
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_train_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_train_%j.err
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
GNN_DATA="${GNN_DATA:-/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_tt_delay}"
VENV="${VENV:-$PROJECT_ROOT/.venv-gnn}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_TAG="${RUN_TAG:-tt_delay_001}"
RUN_DIR="$GNN_DATA/runs/$RUN_TAG"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
# PyTorch comes from the GNN venv, not the site-wide PyTorch module.
module list
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS=1

[[ -s "$GNN_DATA/splits.json" && -s "$GNN_DATA/preparation_report.json" ]] || {
    echo 'Dataset not ready: run hpc/01_prepare_dataset.slurm first.' >&2; exit 1;
}
[[ ! -e "$RUN_DIR" ]] || {
    echo "Run folder already exists (training refuses overwrite): $RUN_DIR" >&2; exit 1;
}

python - <<'PY'
import torch, torch_geometric
assert torch.cuda.is_available(), 'No CUDA GPU allocated! Check the selected Slurm GPU partition.'
print('PyTorch', torch.__version__, 'PyG', torch_geometric.__version__)
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, 'VRAM GiB:', round(p.total_memory / 2**30, 2))
if p.total_memory < 20 * 2**30:
    print('WARNING: full 6,508-node global attention may exceed this GPU memory!')
PY
nvidia-smi || true

echo "$(date -Is) Training on full graph; run=$RUN_TAG epochs=$EPOCHS batch=$BATCH_SIZE"
srun python -u -m gnn_surrogate.training.train \
    --gnn-dir "$GNN_DATA" \
    --splits "$GNN_DATA/splits.json" \
    --out "$RUN_DIR" \
    --max-nodes 0 \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr 0.0001 \
    --patience 15

echo "$(date -Is) Finished: $RUN_DIR/summary.json"
