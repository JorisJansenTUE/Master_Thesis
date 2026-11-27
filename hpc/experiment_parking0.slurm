#!/usr/bin/env bash
#SBATCH --job-name=parking0-low
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=parking0_%j.out
#SBATCH --error=parking0_%j.err
set -euo pipefail

module purge
module load Umbrella/2024
module load Python/3.10.13

REPO="${FITEXP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${FITEXP_PYTHON:-$REPO/.venv_hpc/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ -x "$PYTHON_BIN" ]] || { echo "Python environment not found: $PYTHON_BIN" >&2; exit 1; }
echo "Python: $($PYTHON_BIN --version 2>&1)"
echo "Experiment C: recompute CAR ONLY (0 parking); reuse Trial 01 bike/PT caches"
"$PYTHON_BIN" -u -m optimization.experiment_parking0 \
  --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
