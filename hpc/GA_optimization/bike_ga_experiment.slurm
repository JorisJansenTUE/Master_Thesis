#!/bin/bash
#SBATCH --job-name=bike_access_ga
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/20201733/Master_Thesis_GA/slurm_logs/slurm-bike-ga-%j.out
#SBATCH --error=/home/20201733/Master_Thesis_GA/slurm_logs/slurm-bike-ga-%j.err
set -euo pipefail
REPO="/home/20201733/Master_Thesis_GA"
cd "$REPO"
source "$REPO/hpc/GA_optimization/hpc_config.sh"
export BASE SPATIAL_DIR
export BIKE_REFERENCE_FILE="${BIKE_REFERENCE_FILE:-$REFERENCE_FILE}"
export BIKE_DATASET_MANIFEST="${BIKE_DATASET_MANIFEST:-$HISTORICAL_DATASET/dataset_manifest.json}"
export BIKE_GA_OUT="${BIKE_GA_OUT:-$BASE/Bike_Accessibility_GA_AllBike_operator_v2}"
export BIKE_GA_SEED="${BIKE_GA_SEED:-20261122}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON_BIN" -u -m accessibility.inspect_bike_ga_experiment \
  --workers "${SLURM_CPUS_PER_TASK:-24}"
