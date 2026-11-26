"""Compare MATSim scenario mode shares against one baseline.

Requires: pandas, numpy, matplotlib
Run on the HPC: python analyze_modal_shifts_hpc.py
Optional: set LOCARNO_ROOT to override the data root (e.g. for testing).
"""
from pathlib import Path
import warnings
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ============================== SETTINGS ==============================
# HPC paths recovered from the Locarno project discussions.
ROOT = Path(os.environ.get('LOCARNO_ROOT', '/vast.mnt/home/20201733/locarno_ETH_Model'))
BASELINE = ROOT / 'training_dataset_v4' / 'baseline_00000'
GA_ROOT = ROOT / 'GA_Optimization'
VANILLA = 'Vanilla GA'
WARM = 'Spatial GA – warm start'
NO_WARM = 'Spatial GA – no warm start'
GROUPS = {
    'Training set': [ROOT / 'training_dataset_v4'],
    VANILLA: [GA_ROOT],
    WARM: [GA_ROOT / 'ga_20min_park5_15km_run01'],
    NO_WARM: [GA_ROOT / 'ga_20min_park5_15km_seed1234_no_warm'],
}
# Match direct scenario folders, not the named spatial GA run directories.
VANILLA_SCENARIO = re.compile(r'ga_[0-9a-fA-F]{16}')
# Previously identified best solutions; selection is not based on modal shifts.
BEST_RUNS = {
    WARM: 'ga_044ded2d27389dc7',
    NO_WARM: 'ga_eecdb4e965e4fda5',
}
OUTPUT = ROOT / 'modal_shift_analysis'
ITERATION = None  # None = last available iteration; or set e.g. 100 for all.
# Group corresponding loop categories; no renormalization is performed.
MODE_GROUPS = {
    'Car': ('car', 'car_loop'),
    'Bike': ('bike', 'bike_loop'),
    'PT': ('pt',),
    'Walk': ('walk', 'walk_loop'),
    'Car passenger': ('car_passenger', 'car_passenger_loop'),
}
# ======================================================================


def read_shares(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={c: 'iteration' for c in df.columns if c.lower() == 'iteration'})
    if 'iteration' not in df or df.empty:
        raise ValueError('Missing iteration column or empty file')
    df['iteration'] = pd.to_numeric(df['iteration'], errors='raise')
    iteration = df['iteration'].max() if ITERATION is None else ITERATION
    rows = df.loc[df['iteration'] == iteration]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one row for iteration {iteration}')
    shares = pd.to_numeric(rows.iloc[0].drop('iteration'), errors='raise').astype(float)
    if not np.isfinite(shares).all() or (shares < 0).any():
        raise ValueError('Non-finite or negative mode share')
    if not np.isclose(shares.sum(), 1, atol=0.002, rtol=0):
        raise ValueError(f'Mode shares sum to {shares.sum():.5f}, not 1')
    return int(iteration), shares


def make_record(path, group, baseline, baseline_iteration):
    iteration, shares = read_shares(path)
    # Fail explicitly instead of assuming missing mode columns mean zero.
    if set(shares.index) != set(baseline.index):
        raise ValueError(f'Mode columns differ from baseline: {set(shares.index) ^ set(baseline.index)}')
    delta = (shares - baseline) * 100
    record = {'group': group, 'scenario': path.parent.parent.name,
              'path': str(path.resolve()), 'iteration': iteration,
              'baseline_iteration': baseline_iteration}
    for mode in baseline.index:
        record[f'raw_share_pct_{mode}'] = shares[mode] * 100
        record[f'raw_delta_pp_{mode}'] = delta[mode]
    for mode, columns in MODE_GROUPS.items():
        record[f'share_pct_{mode}'] = shares.reindex(columns, fill_value=0).sum() * 100
        record[f'delta_pp_{mode}'] = delta.reindex(columns, fill_value=0).sum()
    record['delta_pp_Bike + PT'] = record['delta_pp_Bike'] + record['delta_pp_PT']
    record['delta_pp_other_modes'] = delta.sum() - sum(record[f'delta_pp_{m}'] for m in MODE_GROUPS)
    return record


def save(fig, name):
    fig.tight_layout(rect=(0, .04, 1, 1) if name == '01_modal_shift_distributions' else None)
    for ext in ('png', 'svg'):
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def boxplot(data, colors):
    groups = list(data['group'].unique())
    modes = list(MODE_GROUPS)
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(modes) * 0.32 * len(groups))))
    width = 0.78 / len(groups)
    for j, group in enumerate(groups):
        subset = data[data['group'] == group]
        positions = np.arange(len(modes)) + (j - (len(groups)-1)/2) * width
        bp = ax.boxplot([subset[f'delta_pp_{m}'] for m in modes],
                        positions=positions, widths=width*0.85, vert=False,
                        patch_artist=True, whis=(5, 95), showfliers=False,
                        manage_ticks=False, medianprops={'color': '#202020', 'linewidth': 1.5})
        for box in bp['boxes']:
            box.set(facecolor=colors[group], alpha=0.55)
    ax.set_yticks(range(len(modes)), modes)
    ax.invert_yaxis()
    ax.axvline(0, color='0.35', linestyle='--', linewidth=1)
    ax.set_xlabel('Change from baseline (percentage points)')
    ax.set_title('Distribution of modal share changes')
    ax.grid(axis='x', alpha=0.2)
    ax.legend(handles=[Patch(facecolor=colors[g], alpha=.55, label=f'{g} (n={sum(data.group == g):,})')
                       for g in groups], loc='best', fontsize=9)
    fig.text(.5, .01, 'Boxes: 25th–75th percentile; whiskers: 5th–95th percentile.', ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.15)
    save(fig, '01_modal_shift_distributions')


def scatter(data, colors):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for group, subset in data.groupby('group', sort=False):
        ax.scatter(subset['delta_pp_Car'], subset['delta_pp_Bike + PT'],
                   s=12, alpha=.3, color=colors[group], edgecolors='none',
                   rasterized=True, label=f'{group} (n={len(subset):,})')
    ax.axhline(0, color='0.4', linestyle='--', linewidth=.9)
    ax.axvline(0, color='0.4', linestyle='--', linewidth=.9)
    ax.scatter(0, 0, c='black', marker='x', s=60, zorder=5, label='Baseline')
    ax.set(xlabel='Car share change (percentage points)',
           ylabel='Bike + PT share change (percentage points)',
           title='Car share change versus bicycle and PT share change')
    ax.grid(alpha=.15)
    ax.legend(fontsize=9)
    save(fig, '02_car_vs_bike_pt_shift')


def best_plot(best, colors):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    modes = list(MODE_GROUPS)
    width = .78 / len(best)
    for j, (_, row) in enumerate(best.iterrows()):
        y = np.arange(len(modes)) + (j - (len(best)-1)/2)*width
        vals = [row[f'delta_pp_{m}'] for m in modes]
        bars = ax.barh(y, vals, height=width*.85, color=colors[row['group']], label=row['group'])
        ax.bar_label(bars, labels=[f'{v:+.3f}' for v in vals], padding=3, fontsize=8)
    ax.set_yticks(range(len(modes)), modes)
    ax.invert_yaxis()
    ax.axvline(0, color='0.35', linestyle='--', linewidth=1)
    ax.set(xlabel='Change from baseline (percentage points)', title='Modal shifts in selected best GA solutions')
    ax.margins(x=.2)
    ax.grid(axis='x', alpha=.2)
    ax.legend(fontsize=9)
    save(fig, '03_best_ga_modal_shifts')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base_file = BASELINE / 'targets' / 'modestats.csv'
    base_iter, base = read_shares(base_file)
    for mode, columns in MODE_GROUPS.items():
        if columns[0] not in base:
            raise ValueError(f'Missing baseline mode: {columns[0]}')
    pd.DataFrame({'mode': base.index, 'share_pct': base.values*100,
                  'iteration': base_iter}).to_csv(OUTPUT / 'baseline_mode_shares.csv', index=False)
    records, errors = [], []
    seen = {}
    def add(path, group, destination):
        try:
            destination.append(make_record(path, group, base, base_iter))
        except (OSError, ValueError, TypeError) as e:
            errors.append({'group': group, 'path': str(path), 'reason': str(e)})
    for group, roots in GROUPS.items():
        before = len(records)
        for root in roots:
            if not root.is_dir():
                errors.append({'group': group, 'path': str(root), 'reason': 'Root directory not found'})
                continue
            if group == VANILLA:
                # Never recurse into spatial GA runs from the common parent.
                candidates = sorted(
                    d / 'targets' / 'modestats.csv'
                    for d in root.iterdir()
                    if d.is_dir() and VANILLA_SCENARIO.fullmatch(d.name)
                )
            else:
                candidates = sorted(root.rglob('targets/modestats.csv'))
            print(f'  {root}: {len(candidates):,} candidate files', flush=True)
            for path in candidates:
                if path.resolve() == base_file.resolve() or path.parent.parent.name.startswith('baseline_'):
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    if seen[resolved] != group:
                        errors.append({'group': group, 'path': str(path), 'reason': f'Overlapping roots: already assigned to {seen[resolved]}'})
                    continue
                seen[resolved] = group
                add(path, group, records)
        print(f'{group}: {len(records)-before:,} valid scenarios', flush=True)
    best_records = []
    for group, scenario in BEST_RUNS.items():
        matches = [r for r in records if r['group'] == group and r['scenario'] == scenario]
        if len(matches) == 1:
            best_records.append(matches[0])
        else:
            errors.append({'group': group, 'path': scenario,
                           'reason': f'Expected one valid best scenario, found {len(matches)}'})
    pd.DataFrame(errors, columns=['group', 'path', 'reason']).to_csv(OUTPUT / 'skipped_files.csv', index=False)
    if not records:
        raise RuntimeError(f'No valid scenarios found. Check {OUTPUT / "skipped_files.csv"}')
    data = pd.DataFrame(records)
    data.to_csv(OUTPUT / 'scenario_modal_shifts.csv', index=False)
    iterations = data.groupby(['group', 'iteration']).size().rename('n').reset_index()
    iterations.to_csv(OUTPUT / 'iteration_counts.csv', index=False)
    if (data.iteration != base_iter).any() or any(r['iteration'] != base_iter for r in best_records):
        warnings.warn('Some scenario iterations differ from the baseline. Consider setting ITERATION to a common available iteration.')
    summary = []
    for group, subset in data.groupby('group', sort=False):
        for mode in [*MODE_GROUPS, 'Bike + PT', 'other_modes']:
            s = subset[f'delta_pp_{mode}']
            summary.append({'group': group, 'mode': mode, 'n': len(s),
                            'mean_pp': s.mean(), 'median_pp': s.median(),
                            'mean_absolute_pp': s.abs().mean(), 'std_pp': s.std(),
                            'p05_pp': s.quantile(.05), 'p95_pp': s.quantile(.95),
                            'min_pp': s.min(), 'max_pp': s.max(),
                            'pct_positive': 100*(s > 1e-9).mean()})
    pd.DataFrame(summary).to_csv(OUTPUT / 'summary_by_group.csv', index=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    names = list(dict.fromkeys([*GROUPS, *BEST_RUNS]))
    palette = plt.get_cmap('tab10')
    colors = {name: palette(i % 10) for i, name in enumerate(names)}
    boxplot(data, colors)
    scatter(data, colors)
    if best_records:
        best = pd.DataFrame(best_records)
        best.to_csv(OUTPUT / 'best_ga_modal_shifts.csv', index=False)
        best_plot(best, colors)
    else:
        for name in ('best_ga_modal_shifts.csv', '03_best_ga_modal_shifts.png', '03_best_ga_modal_shifts.svg'):
            (OUTPUT / name).unlink(missing_ok=True)
    print(f'Baseline iteration: {base_iter}; skipped/warnings: {len(errors)}')
    if errors:
        print(f'Inspect {OUTPUT / "skipped_files.csv"} for details.')
    print(iterations.to_string(index=False))
    print(f'Output: {OUTPUT.resolve()}')


if __name__ == '__main__':
    main()
