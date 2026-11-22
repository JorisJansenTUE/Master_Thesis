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
GA_DATASET="${GA_DATASET:-$BASE/GA_Optimization}"
WARM_START_FILE="${WARM_START_FILE:-$GA_DATASET/warm_start.json}"
REFERENCE_FILE="${REFERENCE_FILE:-$GA_DATASET/references/sufficiency_reference_15min.json}"

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
GA_BUDGET_M="${GA_BUDGET_M:-15000}"
WARM_COUNT="${WARM_COUNT:-16}"
WARM_MAX_OVERLAP="${WARM_MAX_OVERLAP:-0.70}"
# 0 = all historical candidates; set e.g. 20 for a quick pilot.
WARM_LIMIT="${WARM_LIMIT:-0}"
# 1 = score only already generated accessibility files (no routing).
WARM_EXISTING_ONLY="${WARM_EXISTING_ONLY:-0}"
# 1 = explicitly rebuild warm_start.json if it exists.
WARM_REBUILD="${WARM_REBUILD:-0}"

GA_POPULATION="${GA_POPULATION:-80}"
# Upper bound over ALL nights. The eight-hour SLURM walltime is a separate cap.
GA_GENERATIONS="${GA_GENERATIONS:-100}"
GA_SEED="${GA_SEED:-42}"
GA_ELITES="${GA_ELITES:-2}"
GA_IMMIGRANTS="${GA_IMMIGRANTS:-0.10}"
GA_MAX_PARALLEL="${GA_MAX_PARALLEL:-80}"
# 1 = remove bulky raw MATSim files after successful, checked compaction.
GA_DELETE_RAW="${GA_DELETE_RAW:-1}"
# 1 = explicitly assign -1 fitness to failed MATSim jobs and continue.
# Unreported/missing completions still stop the orchestrator for investigation.
GA_PENALIZE_FAILED="${GA_PENALIZE_FAILED:-1}"
# Leave time before SLURM's eight-hour hard limit for current arrays to finish.
GA_STOP_MARGIN_MIN="${GA_STOP_MARGIN_MIN:-45}"
GA_POLL_SEC="${GA_POLL_SEC:-30}"
# 1 = validate settings and preview init / submission without starting MATSim.
GA_DRY_RUN="${GA_DRY_RUN:-0}"

export PYTHONPATH="$HPC_REPO${PYTHONPATH:+:$PYTHONPATH}"
