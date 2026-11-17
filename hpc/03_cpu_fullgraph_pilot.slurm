#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-cpu-pilot
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_cpu_pilot_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/gnn_cpu_pilot_%j.err
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
GNN_DATA=/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_v3
VENV="${VENV:-$PROJECT_ROOT/.venv-gnn}"
RUN_TAG="${RUN_TAG:-cpu_fullgraph_pilot_001}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "Invalid RUN_TAG" >&2; exit 1; }
RUN_DIR="$GNN_DATA/runs/$RUN_TAG"
PILOT_SPLITS="$GNN_DATA/splits_${RUN_TAG}.json"

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
[[ ! -e "$RUN_DIR" ]] || { echo "Run folder exists, choose a new RUN_TAG: $RUN_DIR" >&2; exit 1; }

# Preserve the original 80/10/10 split. Sample ONLY within each partition.
# These tiny subsets are for a technical pilot, not a meaningful evaluation.
export GNN_DATA PILOT_SPLITS
python - <<'PY'
import json
import os
import random
from pathlib import Path

root = Path(os.environ['GNN_DATA'])
original = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
rng = random.Random(42)
counts = {'train': 8, 'val': 2, 'test': 2}
subset = {'seed': 42, 'scenario_count': sum(counts.values()),
          'note': 'CPU pilot only. Original split membership preserved. Not for model evaluation.'}
for part, n in counts.items():
    candidates = original[part]
    if len(candidates) < n:
        raise ValueError(f'Need {n} {part} cases; only {len(candidates)} available')
    subset[part] = rng.sample(sorted(candidates), n)
all_ids = subset['train'] + subset['val'] + subset['test']
if len(set(all_ids)) != len(all_ids):
    raise ValueError('Original splits overlap')
path = Path(os.environ['PILOT_SPLITS'])
if path.exists() and json.loads(path.read_text(encoding='utf-8')) != subset:
    raise ValueError(f'Refusing to replace different existing pilot split {path}')
path.write_text(json.dumps(subset, indent=2) + '\n', encoding='utf-8')
print(f'CPU pilot split: {path}; 8 train / 2 val / 2 test')
PY

python - <<'PY'
import torch, torch_geometric
print('PyTorch:', torch.__version__, 'PyG:', torch_geometric.__version__, 'device: CPU')
PY

echo "$(date -Is) Starting full 6508-node CPU pilot: $RUN_TAG"
srun python -u -m gnn_surrogate.training.train \
    --gnn-dir "$GNN_DATA" \
    --splits "$PILOT_SPLITS" \
    --out "$RUN_DIR" \
    --max-nodes 0 \
    --device cpu \
    --epochs 1 \
    --batch-size 1 \
    --lr 0.0001 \
    --patience 10

echo "$(date -Is) Finished CPU pilot: $RUN_DIR/summary.json"
