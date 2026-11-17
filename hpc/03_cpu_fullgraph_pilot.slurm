#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-cpu-full
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_cpu_full_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_cpu_full_%j.err
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
GNN_DATA=/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_v3
VENV="${VENV:-$PROJECT_ROOT/.venv-gnn}"
RUN_TAG="${RUN_TAG:-cpu_full_epoch_001}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "Invalid RUN_TAG" >&2; exit 1; }
RUN_DIR="$GNN_DATA/runs/$RUN_TAG"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS=1

[[ -s "$GNN_DATA/splits.json" ]] || { echo "Missing $GNN_DATA/splits.json" >&2; exit 1; }
[[ -s "$GNN_DATA/preparation_report.json" ]] || { echo "Missing preparation report" >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "Run folder exists; choose a new RUN_TAG: $RUN_DIR" >&2; exit 1; }

python - <<'PY'
import json
import os
from pathlib import Path
import torch
import torch_geometric
splits = json.loads(Path('/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_v3/splits.json').read_text())
print('Full dataset split sizes:', {k: len(splits[k]) for k in ('train', 'val', 'test')}, flush=True)
print('PyTorch:', torch.__version__, 'PyG:', torch_geometric.__version__, 'device: CPU', flush=True)
PY

echo "$(date -Is) Starting 1 full CPU epoch (all train/val/test scenarios): $RUN_TAG"
srun python -u -m gnn_surrogate.training.train \
    --gnn-dir "$GNN_DATA" \
    --splits "$GNN_DATA/splits.json" \
    --out "$RUN_DIR" \
    --max-nodes 0 \
    --device cpu \
    --epochs 1 \
    --batch-size 1 \
    --lr 0.0001 \
    --patience 10

echo "$(date -Is) Finished CPU epoch: $RUN_DIR/summary.json"
