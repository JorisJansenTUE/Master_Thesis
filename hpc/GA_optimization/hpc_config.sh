#!/usr/bin/env bash
# Edit this ONE file for the Movement HPC. All paths may also be overridden
# with environment variables before calling sbatch --export=ALL.
# No data are placed in the Git worktree by default.

# Resolve this worktree, not the old Master_Thesis checkout.
HPC_REPO="${HPC_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
BASE="${BASE:-/home/20201733/locarno_ETH_Model}"

# Completed historical MATSim simulations, including baseline_00000.
HISTORICAL_DATASET="${HISTORICAL_DATASET:-$BASE/training_dataset_v4}"
HISTORICAL_BASELINE="${HISTORICAL_BASELINE:-$HISTORICAL_DATASET/baseline_00000}"

# New optimisation results, separate from the historical dataset and Git tree.
# Isolate the new run from prior 15-minute/linear-fitness GA checkpoints.
GA_DATASET="${GA_DATASET:-$BASE/GA_Optimization/ga_trial08_20min_park0_50km_seed42_explore}"
WARM_START_FILE="${WARM_START_FILE:-$GA_DATASET/warm_start.json}"
# New Trial 08 reference; legacy reference and accessibility caches are untouched.
REFERENCE_FILE="${REFERENCE_FILE:-$BASE/GA_Optimization/references/sufficiency_reference_20min_park0_low_statpop.json}"

# Transfer spatial inputs produced by the local accessibility package here:
# origins.csv, jobs.csv, pt_origin_connectors.csv, pt_job_connectors.csv.
SPATIAL_DIR="${SPATIAL_DIR:-$BASE/spatial_baseline}"

# Existing MATSim inputs. MUST match the historical baseline source hashes.
MATSIM_CONFIG="${MATSIM_CONFIG:-$BASE/rebuilds/530517/reprepared_531334/06_final_scenario/locarno_config.xml}"
MATSIM_POPULATION="${MATSIM_POPULATION:-$BASE/rebuilds/530517/reprepared_531334/simulation_output/output_plans.xml.gz}"
MATSIM_JAR="${MATSIM_JAR:-/home/20201733/eqasim-java-Fork/switzerland/target/switzerland.jar}"

# Reuse the existing working venv, even if this worktree has no .venv_hpc.
PYTHON_BIN="${PYTHON_BIN:-/home/20201733/Master_Thesis/.venv_hpc/bin/python}"

# Identical optimisation and warm-start budget (prevents the budget mismatch).
GA_BUDGET_M="${GA_BUDGET_M:-50000}"
WARM_COUNT="${WARM_COUNT:-8}"
WARM_MAX_OVERLAP="${WARM_MAX_OVERLAP:-0.70}"
# 0 = all historical candidates; set e.g. 20 for a quick pilot.
WARM_LIMIT="${WARM_LIMIT:-0}"
# 1 = score only already generated accessibility files (no routing).
WARM_EXISTING_ONLY="${WARM_EXISTING_ONLY:-0}"
# 1 = explicitly rebuild warm_start.json if it exists.
WARM_REBUILD="${WARM_REBUILD:-0}"
# Prevent the GA warm start from selecting scenarios in the frozen GNN test set.
# Specify another splits.json if your GNN data directory differs.
WARM_EXCLUDE_TEST_SPLIT="${WARM_EXCLUDE_TEST_SPLIT:-$BASE/gnn_training_tt_delay_expanded_v2/splits.json}"

TRIAL08_DIR="${TRIAL08_DIR:-$BASE/fitness_experiments/trial_08_parking0}"
BIKE_POTENTIAL_DIR="${BIKE_POTENTIAL_DIR:-$BASE/fitness_experiments/bicycle_budget_potential_01}"
WARM_SHORTLIST="${WARM_SHORTLIST:-80}"
WARM_BIKE_COUNT="${WARM_BIKE_COUNT:-8}"
GA_FITNESS_WORKERS="${GA_FITNESS_WORKERS:-16}"
GA_POPULATION="${GA_POPULATION:-80}"
# GA generations, INCLUDING generation zero; not MATSim's 100 simulation iterations.
GA_GENERATIONS="${GA_GENERATIONS:-80}"
GA_SEED="${GA_SEED:-42}"
GA_ELITES="${GA_ELITES:-2}"
GA_IMMIGRANTS="${GA_IMMIGRANTS:-0.20}"
GA_MAX_PARALLEL="${GA_MAX_PARALLEL:-80}"
# 1 = remove bulky raw MATSim files after successful, checked compaction.
GA_DELETE_RAW="${GA_DELETE_RAW:-1}"
# 1 = explicitly assign -1 fitness to failed MATSim jobs and continue.
# Unreported/missing completions still stop the orchestrator for investigation.
GA_PENALIZE_FAILED="${GA_PENALIZE_FAILED:-1}"
# Stop submitting before the actual SLURM walltime; active arrays may continue.
GA_STOP_MARGIN_MIN="${GA_STOP_MARGIN_MIN:-45}"
GA_POLL_SEC="${GA_POLL_SEC:-30}"
# 1 = validate settings and preview init / submission without starting MATSim.
GA_DRY_RUN="${GA_DRY_RUN:-0}"

# Keep exactly comparable MATSim settings to historical baseline.
GA_LAST_ITERATION="${GA_LAST_ITERATION:-100}"
GA_LINKSTATS_INTERVAL="${GA_LINKSTATS_INTERVAL:-25}"
GA_LINKSTATS_AVERAGE_ITERATIONS="${GA_LINKSTATS_AVERAGE_ITERATIONS:-5}"
export GA_LAST_ITERATION GA_LINKSTATS_INTERVAL GA_LINKSTATS_AVERAGE_ITERATIONS

export PYTHONPATH="$HPC_REPO${PYTHONPATH:+:$PYTHONPATH}"

# Avoid nested BLAS thread oversubscription in independent routing workers.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
