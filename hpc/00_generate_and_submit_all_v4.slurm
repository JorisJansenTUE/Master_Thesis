#!/usr/bin/env bash
#SBATCH --job-name=training_v4
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=training_v4_%j.out
#SBATCH --error=training_v4_%j.err

set -Eeuo pipefail

module purge
module load Umbrella/2024
module load Python/3.10.13
module load EESSI/2026.06
module load Java/25.36

BASE="/home/20201733/locarno_ETH_Model"
PY_REPO="/home/20201733/Master_Thesis"

JAR="/home/20201733/eqasim-java-Fork/switzerland/target/switzerland.jar"
NETWORK="$BASE/rebuilds/530517/reprepared_531334/06_final_scenario/network_with_widths.xml.gz"
SCHEDULE="$BASE/rebuilds/530517/reprepared_531334/06_final_scenario/locarno_transit_schedule.xml.gz"
POPULATION="$BASE/rebuilds/530517/reprepared_531334/baseline_100_768160/simulation_output/output_plans.xml.gz"
CONFIG="$BASE/rebuilds/530517/reprepared_531334/baseline_100_768160/simulation_output/output_config.xml"

# New directory: do not overwrite the already simulated dataset_v3.
DATASET="$BASE/training_dataset_v4"
PYTHON="$PY_REPO/.venv_hpc/bin/python"

for file in "$JAR" "$NETWORK" "$SCHEDULE" "$POPULATION" "$CONFIG" "$PYTHON"; do
    [[ -s "$file" ]] || { echo "Missing file: $file" >&2; exit 1; }
done

cd "$PY_REPO"
export PYTHONPATH="$PY_REPO:${PYTHONPATH:-}"

echo "STEP 1: Generating baseline and 667 scenarios per strategy ($SLURM_CPUS_PER_TASK workers)"
"$PYTHON" -m network_sampling.dataset.generate \
    --network "$NETWORK" \
    --schedule "$SCHEDULE" \
    --out "$DATASET" \
    --samples 2000 \
    --strategies baseline cluster random corridor \
    --randomize-parameters \
    --parameter-seed 42 \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "STEP 2: Submitting MATSim scenario arrays"
# IMPORTANT: the installed run_all_scenarios.sh must support >1000
# scenarios, --delete-raw, and your intended per-task resources/settings.
bash network_sampling/dataset/run_all_scenarios.sh \
    "$DATASET" "$CONFIG" "$POPULATION" "$JAR" "$PYTHON" \
    80 --delete-raw

echo "All MATSim scenarios submitted"
