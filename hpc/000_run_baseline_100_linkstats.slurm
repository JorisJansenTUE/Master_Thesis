#!/usr/bin/env bash
#SBATCH --job-name=locarno_baseline100
#SBATCH --partition=mech-cst-mov.cpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --output=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/baseline_100_%j.out
#SBATCH --error=/vast.mnt/home/20201733/Master_Thesis/slurm_logs/baseline_100_%j.err

set -Eeuo pipefail

JAVA_REPO="/home/20201733/eqasim-java-Fork"
SCENARIO_DIR="/home/20201733/locarno_ETH_Model/rebuilds/530517/reprepared_531334/06_final_scenario"

CONFIG="$SCENARIO_DIR/locarno_config.xml"
NETWORK="$SCENARIO_DIR/network_with_widths.xml.gz"
POPULATION="$SCENARIO_DIR/locarno_population.xml.gz"
FACILITIES="$SCENARIO_DIR/locarno_facilities.xml.gz"
PT_SCHEDULE="$SCENARIO_DIR/locarno_transit_schedule.xml.gz"
PT_VEHICLES="$SCENARIO_DIR/locarno_transit_vehicles.xml.gz"

BUILD_JAR="${BUILD_JAR:-1}"  # 1 = clean compile, 0 = reuse existing JAR
LINKSTATS_INTERVAL="${LINKSTATS_INTERVAL:-1}"
LINKSTATS_AVERAGING="${LINKSTATS_AVERAGING:-1}"
THREADS="${SLURM_CPUS_PER_TASK:-24}"
HEAP="200G"
OUTPUT_DIR="$(dirname "$SCENARIO_DIR")/baseline_100_${SLURM_JOB_ID}/simulation_output"

log() { echo "[$(date '+%F %T')] $*"; }
require_file() { [[ -s "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }

# Java 25 + Maven 3.9.6
module purge
module load EESSI/2026.06
module load Java/25.36

export MAVEN_HOME="$HOME/software/apache-maven-3.9.6"
export PATH="$MAVEN_HOME/bin:$PATH"

export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
export PATH="$JAVA_HOME/bin:$PATH"
export eqasim_configurator="org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"

if [[ "$BUILD_JAR" == 1 ]]; then
    log "Build Switzerland JAR (Java 25 / Maven 3.9.6)"
    (cd "$JAVA_REPO" && mvn -B -pl switzerland -am clean package -Pstandalone -DskipTests -T 8)
fi

JAR="$JAVA_REPO/switzerland/target/switzerland.jar"
require_file "$JAR"
log "Using JAR: $JAR"
ls -lh "$JAR"

for f in "$CONFIG" "$NETWORK" "$POPULATION" "$FACILITIES" "$PT_SCHEDULE" "$PT_VEHICLES"; do
    require_file "$f"
done

log "Running 100 iterations; linkstats interval=$LINKSTATS_INTERVAL, averaging=$LINKSTATS_AVERAGING"
log "Output: $OUTPUT_DIR"
cd "$SCENARIO_DIR"
java -Xmx"$HEAP" -XX:ActiveProcessorCount="$THREADS" -cp "$JAR" \
    org.eqasim.switzerland.ch_cmdp.RunSimulation \
    --config-path "$CONFIG" \
    --config:network.inputNetworkFile "$NETWORK" \
    --config:plans.inputPlansFile "$POPULATION" \
    --config:facilities.inputFacilitiesFile "$FACILITIES" \
    --config:transit.transitScheduleFile "$PT_SCHEDULE" \
    --config:transit.vehiclesFile "$PT_VEHICLES" \
    --config:controller.firstIteration 0 \
    --config:controller.lastIteration 100 \
    --config:controller.outputDirectory "$OUTPUT_DIR" \
    --config:linkStats.writeLinkStatsInterval "$LINKSTATS_INTERVAL" \
    --config:linkStats.averageLinkStatsOverIterations "$LINKSTATS_AVERAGING"

log "DONE: $OUTPUT_DIR"
log "Effective config: $OUTPUT_DIR/output_config.xml"
