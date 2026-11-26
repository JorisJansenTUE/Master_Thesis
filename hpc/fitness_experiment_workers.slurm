#!/usr/bin/env bash
#SBATCH --job-name=statpop-fitness
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail

module purge
module load Umbrella/2024
module load Python/3.10.13

# Submit from Master_Thesis root, or set FITEXP_REPO explicitly.
# SLURM_SUBMIT_DIR is stable; BASH_SOURCE is not reliable inside a SLURM spool.
REPO="${FITEXP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${FITEXP_PYTHON:-$REPO/.venv_hpc/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PYTHON_BIN" -u -m optimization.fitness_experiment_batch run \
  --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
