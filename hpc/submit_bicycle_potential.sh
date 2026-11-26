#!/usr/bin/env bash
# Submit FOUR jobs: multi-order greedy first, then 15/50/150 km GAs in parallel.
# Usage: cd /vast.mnt/home/20201733/Master_Thesis; bash hpc/submit_bicycle_potential.sh
set -euo pipefail

REPO="${BIKE_POT_REPO:-$(pwd -P)}"
BASE="${BASE:-/vast.mnt/home/20201733/locarno_ETH_Model}"
TRIAL="${TRIAL:-$BASE/fitness_experiments/trial_03}"
SPATIAL="${SPATIAL:-$BASE/spatial_baseline}"
POPULATION="${POPULATION:-$SPATIAL/population_by_hexagon.csv}"
OUT="${OUT:-$BASE/fitness_experiments/bicycle_budget_potential_01}"
MATSIM_JAR="${MATSIM_JAR:-/home/20201733/eqasim-java-Fork/switzerland/target/switzerland.jar}"
GA_CPUS="${GA_CPUS:-64}"
GA_MEM="${GA_MEM:-192G}"
GA_TIME="${GA_TIME:-24:00:00}"
GREEDY_CPUS="${GREEDY_CPUS:-4}"
GREEDY_MEM="${GREEDY_MEM:-48G}"
GREEDY_TIME="${GREEDY_TIME:-12:00:00}"
POP_SIZE="${POP_SIZE:-100}"
GENERATIONS="${GENERATIONS:-80}"
MAX_GENERATIONS="${MAX_GENERATIONS:-$GENERATIONS}"
JAVA_OPTIONS=(--matsim-jar "$MATSIM_JAR")

for f in "$TRIAL/experiment.json" "$SPATIAL/origins.csv" "$SPATIAL/jobs.csv" "$POPULATION" \
         "$REPO/optimization/bicycle_budget_potential.py" "$REPO/optimization/bicycle_extreme_diagnostics.py" \
         "$REPO/hpc/bicycle_potential.slurm"; do
  if [[ ! -f "$f" ]]; then echo "MISSING REQUIRED INPUT: $f" >&2; exit 2; fi
done
if [[ ! -f "$MATSIM_JAR" ]]; then
  echo "MATSim Switzerland JAR not found at: $MATSIM_JAR" >&2
  echo "Set MATSIM_JAR to its actual path. (Avoid skipping Java for a feasibility experiment.)" >&2
  exit 2
fi
mkdir -p "$OUT/logs"
COMMON=(--trial "$TRIAL" --spatial "$SPATIAL" --population "$POPULATION" --out "$OUT")

GREEDY_ID=$(sbatch --parsable --job-name=bike-greedy --cpus-per-task="$GREEDY_CPUS" \
  --mem="$GREEDY_MEM" --time="$GREEDY_TIME" \
  --output="$OUT/logs/%x_%j.out" --error="$OUT/logs/%x_%j.err" \
  --export="ALL,BIKE_POT_REPO=$REPO" \
  "$REPO/hpc/bicycle_potential.slurm" greedy "${COMMON[@]}" "${JAVA_OPTIONS[@]}")
echo "Submitted greedy: $GREEDY_ID (four ordering strategies, $GREEDY_CPUS parallel workers)"

for BUDGET in 15 50 150; do
  ID=$(sbatch --parsable --job-name="bike-${BUDGET}km" \
    --dependency="afterok:${GREEDY_ID}" --cpus-per-task="$GA_CPUS" \
    --mem="$GA_MEM" --time="$GA_TIME" \
    --output="$OUT/logs/%x_%j.out" --error="$OUT/logs/%x_%j.err" \
    --export="ALL,BIKE_POT_REPO=$REPO" \
    "$REPO/hpc/bicycle_potential.slurm" ga "${COMMON[@]}" "${JAVA_OPTIONS[@]}" \
    --budget-km "$BUDGET" --population-size "$POP_SIZE" \
    --generations "$GENERATIONS" --max-generations "$MAX_GENERATIONS")
  echo "Submitted GA ${BUDGET}km: $ID (after greedy, $GA_CPUS workers)"
done
printf 'Outputs: %s\n' "$OUT"
