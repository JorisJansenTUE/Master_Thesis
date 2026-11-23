#!/bin/bash -l
#SBATCH --job-name=locarno-gnn-prep
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis_delay/slurm_logs/gnn_prep_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis_delay/slurm_logs/gnn_prep_%j.err
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis_delay
DATASET="${DATASET:-/vast.mnt/home/20201733/locarno_ETH_Model/training_dataset_v4}"
GNN_DATA="${GNN_DATA:-/vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_tt_delay}"
PEAK_HOUR="${PEAK_HOUR:-8}"
VENV="${VENV:-/vast.mnt/home/20201733/Master_Thesis/.venv-gnn}"
WORKERS="${PREP_WORKERS:-12}"

module purge
module load Umbrella/2024
module load "${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
# PyTorch comes from the GNN venv, not the site-wide PyTorch module.
module list
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "$(date -Is) CPU preprocessing: $DATASET -> $GNN_DATA"
python -m gnn_surrogate.dataset.prepare_hpc \
    --dataset "$DATASET" \
    --out "$GNN_DATA" \
    --workers "$WORKERS" \
    --seed 42 \
    --hour "$PEAK_HOUR"

echo "$(date -Is) Preprocessing completed. Splits: $GNN_DATA/splits.json"
