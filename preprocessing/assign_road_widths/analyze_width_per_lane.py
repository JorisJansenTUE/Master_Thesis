#!/usr/bin/env python3
"""Evaluate cadastral carriageway width per lane from an enriched OSM XML file.

Supports .osm and .osm.gz (not PBF).

Example:
  python analyze_width_per_lane.py --osm 'Locarno_with_widths.osm.gz' \
      --outdir width_lane_analysis

Reliability matches WidthResultFixer:
  unreliable = n_samples <= 1 OR (n_samples <= 5 AND p90-p10 >= 2 m)

Widths are always taken from width:carriageway:raw, NEVER the fallback-influenced
width:carriageway:estimated.  Two cohorts are generated:
  explicit: reliable cadastral measurements with valid OSM 'lanes' tags
  all:      reliable cadastral measurements with width:carriageway:lanes_used

By default, the reference set uses method=cadastral_estimate.  An additional
--include-overridden-raw option admits other methods (except piazza_exclusion)
when their raw measurement meets the same reliability thresholds; useful for
checking primary_oneway_fallback separately.
"""
import argparse
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def float_tag(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def positive_int(value):
    try:
        n = int(value)
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def is_oneway(value):
    return str(value).strip().lower() in {'yes', 'true', '1', '-1'}


def read_osm(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    rows = []
    with opener(path, 'rb') as fh:
        for _, elem in ET.iterparse(fh, events=('end',)):
            if elem.tag.rsplit('}', 1)[-1] != 'way':
                continue
            tags = {c.get('k'): c.get('v') for c in elem
                    if c.tag.rsplit('}', 1)[-1] == 'tag'}
            if 'width:carriageway:method' in tags:
                get = lambda key: tags.get('width:carriageway:' + key)
                rows.append({
                    'way_id': int(elem.get('id')),
                    'highway': tags.get('highway', ''),
                    'name': tags.get('name', ''),
                    'oneway': tags.get('oneway', ''),
                    'lanes_tag': tags.get('lanes'),
                    'lanes_used_tag': get('lanes_used'),
                    'method': get('method'),
                    'optimizable': get('optimizable'),
                    'raw_width_m': float_tag(get('raw')),
                    'final_width_m': float_tag(get('estimated')),
                    'carriageway_n': float_tag(get('n_samples')),
                    'p10_m': float_tag(get('p10')),
                    'p90_m': float_tag(get('p90')),
                })
            elem.clear()
    if not rows:
        raise ValueError('No ways with width:carriageway:method found. Use the enriched OSM XML file.')
    return pd.DataFrame(rows)


def summarize(group):
    w = group['width_per_lane_m'].to_numpy(dtype=float)
    err = group['fallback_error_m_per_lane'].to_numpy(dtype=float)
    err_total = group['fallback_error_m_total'].to_numpy(dtype=float)
    return pd.Series({
        'n': len(w),
        'mean_m_per_lane': np.mean(w),
        'median_m_per_lane': np.median(w),
        'std_m_per_lane': np.std(w, ddof=1) if len(w) > 1 else np.nan,
        'p10_m_per_lane': np.percentile(w, 10),
        'p25_m_per_lane': np.percentile(w, 25),
        'p75_m_per_lane': np.percentile(w, 75),
        'p90_m_per_lane': np.percentile(w, 90),
        'fallback_mae_m_per_lane': np.mean(np.abs(err)),
        'fallback_bias_m_per_lane': np.mean(err),
        'fallback_mae_m_total': np.mean(np.abs(err_total)),
        'fallback_within_0_5m_pct': 100 * np.mean(np.abs(err) <= 0.5),
        'fallback_within_1m_pct': 100 * np.mean(np.abs(err) <= 1.0),
    })


def write_cohort(data, outdir, suffix, min_group_n, normal, wide):
    if data.empty:
        print(f'No eligible ways in cohort {suffix}; no statistics written.')
        return
    by_highway = data.groupby('highway').apply(summarize).reset_index()
    by_highway_lanes = data.groupby(['highway', 'lane_count']).apply(summarize).reset_index()
    for table in (by_highway, by_highway_lanes):
        table['small_sample'] = table['n'] < min_group_n
    by_highway.sort_values('n', ascending=False).to_csv(
        outdir / f'statistics_by_highway_{suffix}.csv', index=False)
    by_highway_lanes.sort_values(['highway', 'lane_count']).to_csv(
        outdir / f'statistics_by_highway_and_lanes_{suffix}.csv', index=False)

    comparisons = []
    for highway, group in data.groupby('highway'):
        for candidate in sorted({normal, wide}):
            per_lane_error = candidate - group['width_per_lane_m']
            total_error = candidate * group['lane_count'] - group['raw_width_m']
            comparisons.append({
                'highway': highway, 'candidate_m_per_lane': candidate,
                'n': len(group), 'mae_m_per_lane': per_lane_error.abs().mean(),
                'bias_m_per_lane': per_lane_error.mean(),
                'mae_m_total': total_error.abs().mean(),
                'pct_within_0_5m': 100 * per_lane_error.abs().le(0.5).mean(),
                'pct_within_1m': 100 * per_lane_error.abs().le(1.0).mean(),
            })
    pd.DataFrame(comparisons).to_csv(
        outdir / f'compare_3m_vs_4_5m_{suffix}.csv', index=False)
    print(f'\n{suffix.upper()} cohort: {len(data)} ways')
    print(by_highway[['highway', 'n', 'median_m_per_lane', 'p25_m_per_lane',
                      'p75_m_per_lane', 'fallback_mae_m_per_lane']]
          .sort_values('n', ascending=False).round(2).to_string(index=False))

def plot_width_per_lane(
    df,
    output_path,
    min_group_n=10,
    explicit_only=True,
):
    """
    Boxplot of cadastral width per lane by OSM highway type.

    Uses only reliable cadastral measurements.
    Optionally restricts to explicitly tagged lane counts.
    """

    data = df.copy()

    # Only reliable cadastral estimates
    if explicit_only:
        data = data[data["eligible_explicit"]].copy()
    else:
        data = data[data["eligible_all"]].copy()

    # Select highway types with enough observations
    counts = data.groupby("highway").size()
    valid_types = counts[counts >= min_group_n].index

    data = data[data["highway"].isin(valid_types)]

    if data.empty:
        print("No valid data for width-per-lane plot.")
        return

    # Sort categories by median width per lane
    highway_types = (
        data.groupby("highway")["width_per_lane_m"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    values = [
        data.loc[
            data["highway"] == highway,
            "width_per_lane_m"
        ].to_numpy()
        for highway in highway_types
    ]

    fig, ax = plt.subplots(
        figsize=(max(9, len(highway_types) * 1.3), 6)
    )

    ax.boxplot(
        values,
        tick_labels=highway_types,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#8CB9D9"},
        medianprops={"color": "#173F5F", "linewidth": 2},
    )

    # Current fallback width for each highway type
    for i, highway in enumerate(highway_types, start=1):
        fallback = 4.5 if highway == "trunk" else 3.0

        ax.scatter(
            i,
            fallback,
            color="red",
            marker="D",
            s=45,
            zorder=4,
            label="Current fallback" if i == 1 else None,
        )

    # Reference lines
    ax.axhline(3.0, color="red", linestyle="--", alpha=0.35)
    ax.axhline(4.5, color="darkred", linestyle="--", alpha=0.35)

    # Sample counts
    for i, highway in enumerate(highway_types, start=1):
        n = len(values[i - 1])
        ax.text(
            i, -0.08, f"n={n}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=9,
        )

    ax.set_ylabel("Estimated carriageway width per lane (m)")
    ax.set_xlabel("OSM highway type")
    ax.set_title(
        "Width per lane by road type\n"
        + ("Explicit OSM lanes" if explicit_only else "All lanes used")
    )

    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--osm', type=Path, required=True, help='Enriched OSM XML (.osm/.osm.gz)')
    parser.add_argument('--outdir', type=Path, default=Path('width_lane_analysis'))
    parser.add_argument('--max-unreliable-samples', type=int, default=5)
    parser.add_argument('--min-unreliable-spread-m', type=float, default=2.0)
    parser.add_argument('--normal-lane-width-m', type=float, default=3.0)
    parser.add_argument('--wide-lane-width-m', type=float, default=4.5)
    parser.add_argument('--min-group-n', type=int, default=10)
    parser.add_argument('--include-overridden-raw', action='store_true',
                        help='Also analyze reliable raw widths from non-cadastral methods, except piazza')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = read_osm(args.osm)
    data['spread_m'] = data['p90_m'] - data['p10_m']
    data['osm_lanes'] = data['lanes_tag'].map(positive_int)
    data['lane_count'] = data['lanes_used_tag'].map(positive_int)
    data['explicit_lanes'] = data['osm_lanes'].notna()
    data['lane_source'] = np.where(data['explicit_lanes'], 'osm_lanes',
                           np.where(data['oneway'].map(is_oneway), 'assumed_oneway_1',
                                    'assumed_default_2'))
    # Confirm that stored lanes_used agrees with explicit OSM lanes where both exist.
    data['lane_count_mismatch'] = (data['explicit_lanes'] & data['lane_count'].notna()
                                   & (data['osm_lanes'] != data['lane_count']))
    n = data['carriageway_n'].fillna(0)
    data['unreliable_by_fixer_thresholds'] = (
        n.le(1) | (n.le(args.max_unreliable_samples)
                   & data['spread_m'].notna()
                   & data['spread_m'].ge(args.min_unreliable_spread_m)))
    if args.include_overridden_raw:
        admissible = data['method'].ne('piazza_exclusion')
    else:
        admissible = data['method'].eq('cadastral_estimate')
    data['reliable_geometry'] = (admissible & data['raw_width_m'].gt(0)
                                 & ~data['unreliable_by_fixer_thresholds'])
    data['eligible_all'] = data['reliable_geometry'] & data['lane_count'].notna()
    data['eligible_explicit'] = (data['eligible_all'] & data['explicit_lanes']
                                 & ~data['lane_count_mismatch'])
    data['width_per_lane_m'] = data['raw_width_m'] / data['lane_count']
    is_wide = (data['highway'].eq('trunk') |
               (data['highway'].eq('primary') & data['oneway'].map(is_oneway)))
    data['fallback_m_per_lane'] = np.where(is_wide, args.wide_lane_width_m,
                                          args.normal_lane_width_m)
    data['fallback_total_width_m'] = data['lane_count'] * data['fallback_m_per_lane']
    data['fallback_error_m_per_lane'] = data['fallback_m_per_lane'] - data['width_per_lane_m']
    data['fallback_error_m_total'] = data['fallback_total_width_m'] - data['raw_width_m']
    data.to_csv(args.outdir / 'way_width_lane_diagnostics.csv', index=False)

    print('Enriched OSM ways:', len(data))
    print('Valid stored lanes_used:', data['lane_count'].notna().sum())
    print('Explicit valid OSM lanes:', data['explicit_lanes'].sum())
    print('Stored/explicit lane-count disagreements:', data['lane_count_mismatch'].sum())
    print('Reliable cadastral raw widths:', data['reliable_geometry'].sum())
    write_cohort(data[data['eligible_explicit']], args.outdir, 'explicit',
                 args.min_group_n, args.normal_lane_width_m, args.wide_lane_width_m)
    write_cohort(data[data['eligible_all']], args.outdir, 'all',
                 args.min_group_n, args.normal_lane_width_m, args.wide_lane_width_m)
    print(f'\nOutputs: {args.outdir.resolve()}')

    plot_width_per_lane(
        data,
        args.outdir / "width_per_lane_by_highway_explicit.png",
        min_group_n=args.min_group_n,
        explicit_only=True,
    )

    plot_width_per_lane(
        data,
        args.outdir / "width_per_lane_by_highway_all.png",
        min_group_n=args.min_group_n,
        explicit_only=False,
    )

if __name__ == '__main__':
    main()
