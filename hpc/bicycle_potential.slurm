#!/usr/bin/env bash
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --time=24:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/%x_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/%x_%j.err
set -euo pipefail

module purge
module load Umbrella/2024
module load Python/3.11.3-GCCcore-12.3.0

REPO="${BIKE_POT_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${BIKE_POT_PYTHON:-$REPO/.venv_hpc/bin/python}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Java needed for the MATSim routing preflight, not a MATSim simulation.
command -v java >/dev/null || { echo "Java missing from PATH. Load your Java module." >&2; exit 2; }
"$PYTHON_BIN" -u -m optimization.bicycle_budget_potential "$@" --workers "${SLURM_CPUS_PER_TASK:-1}"
