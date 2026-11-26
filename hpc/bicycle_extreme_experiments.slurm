#!/usr/bin/env bash
#SBATCH --job-name=bicycle-extremes
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/%x_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/%x_%j.err
set -euo pipefail

module purge
module load Umbrella/2024
module load Python/3.10.13

REPO="${BIKE_EXP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${BIKE_EXP_PYTHON:-$REPO/.venv_hpc/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -u -m optimization.bicycle_extreme_diagnostics "$@"
