"""Edit *this* file to define ONE STATPOP-weighted accessibility fitness experiment.

This is an offline rescoring experiment. It does NOT modify active GA fitness,
MATSim scenarios, their cached accessibility, or GNN training targets.

Thresholds are fractions of the STATPOP-weighted baseline mean CAR jobs at the
same time cutoff (0.50 = Martens low; 1.0 = high). Resident weights are FIXED:
all inhabited STATPOP hexagons are included, even if network-disconnected.

All settings are snapshotted at 'prepare' and every SLURM worker uses the frozen
snapshot. Change this file and submit to a NEW output directory for each trial.
"""

# -------- Main F(X): each mode / time weight is multiplied and normalized --------
TIME_WEIGHTS = {20.0: 1.0}                  # e.g. {15.0: 1.0, 20.0: 2.0, 30.0: 1.0}
MODE_WEIGHTS = {"car": 1.0, "bike": 1.0, "pt": 1.0}
SUFFICIENCY_FRACTION = 1                 # 0.50 = 'low'; 1.0 = 'high'; other >0 possible
SHORTFALL_EXPONENT = 2.0                   # Martens-like squared shortfall (2.0)

# -------- Optional car constraint, independent from F(X) --------
CAR_CONSTRAINT_ENABLED = True
CAR_CONSTRAINT_TIME_MIN = 20.0
CAR_CONSTRAINT_SUFFICIENCY_FRACTION = 1  # can differ from objective threshold
CAR_CONSTRAINT_TOLERANCE_PP = 5.0          # minimum = baseline car sufficiency share - 5 pp

# -------- Routing/cache provenance: not population-weight toggles --------
# These must match the accessibility_by_hexagon caches; changing requires
# --compute-missing --force-recalculate (and suitable MATSim scenario input).
CAR_PARKING_PENALTY_S = 300.0
HOUR_START = 8
LINKSTATS_STATISTIC = "avg"
BIKE_SPEED_M_S = 4.17
PT_TRANSFER_PENALTY_S = 300.0
ORIGIN_SNAP_M = 500.0
JOB_SNAP_M = 300.0
