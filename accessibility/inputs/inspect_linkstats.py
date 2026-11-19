"""Inspect a MATSim linkstats file locally. Configure paths below, then run.

    python scripts/inspect_linkstats.py

No command-line arguments, installed packages, or MATSim network are needed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean, median




from preprocessing.utils import PROJECT_ROOT
from accessibility.inputs.linkstats import LinkTravelTimes  


# -----------------------------------------------------------------------------
# LOCAL SETTINGS: 
# -----------------------------------------------------------------------------
LINKSTATS_FILE = PROJECT_ROOT / "data" / "Simulated_Runs" / "baseline_00000" / "200.linkstats.txt.gz"
HOUR_START = 8                # 8 means 08:00-09:00
STATISTIC = "avg"            # 'min', 'avg', or 'max'
AVERAGED_ITERATIONS = 8       # Metadata only; not inferred from file
SHOW_EXAMPLES = 8
# -----------------------------------------------------------------------------


def percentile(values: list[float], percentage: float) -> float:
    """Linearly interpolated percentile, only used for inspection output."""
    data = sorted(values)
    if not data:
        raise ValueError("Cannot calculate percentile of empty data")
    pos = (len(data) - 1) * percentage / 100
    lower, upper = int(pos), min(int(pos) + 1, len(data) - 1)
    frac = pos - lower
    return data[lower] * (1 - frac) + data[upper] * frac


def main() -> None:
    snapshot = LinkTravelTimes.from_linkstats(
        LINKSTATS_FILE,
        hour_start=HOUR_START,
        statistic=STATISTIC,
        averaged_iterations=AVERAGED_ITERATIONS,
    )

    print("=" * 65)
    print("MATSim linkstats inspection")
    print("=" * 65)
    print(f"File:                  {snapshot.source_path}")
    print(f"Selected column:       {snapshot.travel_time_column}")
    print(f"Iteration aggregation: {snapshot.averaged_iterations or 'unknown'} (metadata)")
    print(f"Total unique links:    {snapshot.row_count:,}")
    print(f"Valid travel times:    {snapshot.valid_count:,}")
    print(f"Invalid travel times:  {len(snapshot.invalid_travel_times):,}")
    print("Mode restrictions:     NOT CHECKED (network loading is Step 2)")

    print("\nInvalid travel-time reasons:")
    reasons = Counter(item.reason for item in snapshot.invalid_travel_times.values())
    if reasons:
        for reason, count in reasons.most_common():
            print(f"  {reason:24s} {count:>7,}")
        for link_id, value in list(snapshot.invalid_travel_times.items())[:SHOW_EXAMPLES]:
            print(f"    {link_id}: {value.raw_value!r} ({value.reason})")
    else:
        print("  None")

    times = list(snapshot.travel_times_s.values())
    if times:
        print("\nTravel times, valid links (seconds):")
        print(f"  mean={mean(times):.2f}, median={median(times):.2f}, "
              f"p95={percentile(times, 95):.2f}, max={max(times):.2f}")

    comparisons = [
        (link_id, tt / snapshot.free_flow_times_s[link_id])
        for link_id, tt in snapshot.travel_times_s.items()
        if link_id in snapshot.free_flow_times_s
    ]
    if comparisons:
        ratios = [ratio for _, ratio in comparisons]
        above_2x = sum(ratio > 2 for ratio in ratios)
        below_1x = sum(ratio < 0.999 for ratio in ratios)
        print("\nCompared with free-flow time (valid length and freespeed):")
        print(f"  Comparable links:   {len(ratios):,}")
        print(f"  Median TT / FF:     {median(ratios):.3f}")
        print(f"  p95 TT / FF:        {percentile(ratios, 95):.3f}")
        print(f"  TT > 2x free-flow:  {above_2x:,}")
        print(f"  TT < free-flow:     {below_1x:,}")
        print("  Highest travel-time ratios:")
        for link_id, ratio in sorted(comparisons, key=lambda item: item[1], reverse=True)[:SHOW_EXAMPLES]:
            print(f"    {link_id:>16s}  {ratio:8.2f}x  "
                  f"TT={snapshot.travel_times_s[link_id]:.2f} s  "
                  f"FF={snapshot.free_flow_times_s[link_id]:.2f} s")

    if snapshot.hourly_volumes:
        print("\nRecorded link volumes (not a car-mode filter):")
        print(f"  Available volumes:  {len(snapshot.hourly_volumes):,}")
        print(f"  Positive volume:    {sum(v > 0 for v in snapshot.hourly_volumes.values()):,}")

    print("\nNext step: load the MATSim network to identify car-allowed links.")


if __name__ == "__main__":
    main()
