"""One opt-in switch for STATPOP resident weighting across accessibility and fitness.

False preserves historical, unweighted output and reference semantics.
True requires spatial_baseline/population_by_hexagon.csv and selects a separate
resident-weighted baseline reference and population-aware reports.
"""

POPULATION_WEIGHTED = True
