#!/bin/bash
#SBATCH --job-name=bike_access_screen
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/20201733/Master_Thesis_GA/slurm_logs/slurm-bike-budget-%j.out
#SBATCH --error=/home/20201733/Master_Thesis_GA/slurm_logs/slurm-bike-budget-%j.err

set -euo pipefail

REPO="/home/20201733/Master_Thesis_GA"
cd "$REPO"
source "$REPO/hpc/GA_optimization/hpc_config.sh"
export BASE SPATIAL_DIR

# IMPORTANT: Use the facilities-based reference and spatial_baseline files
# that match one another. Override BIKE_REFERENCE_FILE here when necessary.
export BIKE_REFERENCE_FILE="${BIKE_REFERENCE_FILE:-$REFERENCE_FILE}"
export BIKE_EXPERIMENT_OUT="${BIKE_EXPERIMENT_OUT:-$BASE/Bike_Accessibility_Budgets}"
export BIKE_DATASET_MANIFEST="${BIKE_DATASET_MANIFEST:-$HISTORICAL_DATASET/dataset_manifest.json}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" -u -m accessibility.inspect_bike_budget_experiment \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
