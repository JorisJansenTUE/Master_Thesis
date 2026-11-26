"""Compare 20-minute sufficiency for the baseline and the best two GA runs.

Put this file in Master_Thesis/accessibility/calculations/ and run:
    python -m accessibility.calculations.plot_best_ga_sufficiency

This is a plot of the share of *attached* origin hexagons meeting the
sufficiency threshold, not the continuous squared-shortfall GA fitness or
its fixed-baseline-eligible car-constraint share.
"""
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# Paths are resolved relative to the repo, not your shell's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CSV = PROJECT_ROOT / 'data/accessibility/base_mapping/new_threshold/results_baseline_00000/sufficiency_summary.csv'
WARM_CSV = PROJECT_ROOT / 'data/optimization/best_runs/ga_warm_best/accessibility/sufficiency_summary.csv'
NO_WARM_CSV = PROJECT_ROOT / 'data/optimization/best_runs/ga_no_warm_best/accessibility/sufficiency_summary.csv'
OUTFILE = PROJECT_ROOT / 'data/optimization/best_runs/ga_best_sufficiency_20min_comparison.png'
MINUTES = 20

SCENARIOS = {
    'Baseline': BASELINE_CSV,
    'Warm-start GA': WARM_CSV,
    'No-warm-start GA': NO_WARM_CSV,
}
MODES = ('car', 'bike', 'pt')
MODE_LABELS = {'car': 'Car', 'bike': 'Bicycle', 'pt': 'Public transport'}
COLORS = ('#2E6699', '#D46E33')
SHAPES = ('o', 'D')


def load_data():
    records = []
    reference = None
    n_origins_by_mode = {}
    for label, path in SCENARIOS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {label} summary: {path}\n'
                'Check the downloaded accessibility/ folder or edit the paths at the top.'
            )
        df = pd.read_csv(path)
        required = {
            'mode', 'time_min', 'threshold_low_jobs', 'threshold_high_jobs',
            'pct_sufficient_low', 'pct_sufficient_high', 'n_attached', 'n_origins',
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{path} is missing columns: {sorted(missing)}; got {list(df.columns)}')
        df['mode'] = df['mode'].astype(str).str.lower().str.strip()
        for mode in MODES:
            sel = df.loc[df['mode'].eq(mode) & np.isclose(df['time_min'].astype(float), MINUTES)]
            if len(sel) != 1:
                raise ValueError(f'{label}: expected exactly one row for {mode}, {MINUTES} min; got {len(sel)}')
            r = sel.iloc[0]
            thresholds = np.array([r['threshold_low_jobs'], r['threshold_high_jobs']], dtype=float)
            if not np.all(np.isfinite(thresholds)):
                raise ValueError(f'{label} has nonfinite sufficiency thresholds')
            if reference is None:
                reference = thresholds
            elif not np.allclose(reference, thresholds, rtol=1e-8, atol=1e-6):
                raise ValueError(
                    f'Incompatible sufficiency thresholds in {label}: {thresholds} vs baseline {reference}. '
                    'The GA runs and baseline must use the SAME STATENT jobs and fixed reference; '
                    'do not compare a facilities baseline against STATENT GA runs.'
                )
            if 'car_parking_penalty_s' in r.index and not np.isclose(float(r['car_parking_penalty_s']), 300):
                raise ValueError(f'{label} does not use 5-minute car parking penalty')
            n_origins = int(r['n_origins'])
            if mode not in n_origins_by_mode:
                n_origins_by_mode[mode] = n_origins
            elif n_origins_by_mode[mode] != n_origins:
                raise ValueError(f'Origin counts differ for {mode}: {n_origins_by_mode[mode]} vs {n_origins} ({label})')
            for level in ('low', 'high'):
                val = float(r[f'pct_sufficient_{level}'])
                if not 0 <= val <= 100:
                    raise ValueError(f'Invalid percentage in {label}/{mode}/{level}: {val}')
                records.append({
                    'scenario': label, 'source_csv': str(path), 'mode': mode,
                    'level': level.capitalize(), 'pct_sufficient': val,
                    'n_attached': int(r['n_attached']), 'n_origins': n_origins,
                    'threshold_jobs': float(r[f'threshold_{level}_jobs']),
                })
    return pd.DataFrame(records)


def main():
    data = load_data()
    baseline = data[data.scenario.eq('Baseline')].set_index(['level', 'mode'])['pct_sufficient']
    data['baseline_pct'] = [float(baseline.loc[(r.level, r.mode)]) for r in data.itertuples(index=False)]
    data['delta_pp'] = data.pct_sufficient - data.baseline_pct
    selected = data[data.scenario.ne('Baseline')]

    maximum = float(selected.delta_pp.abs().max())
    axis_limit = max(7.0, np.ceil((maximum + 4.0) / 2.0) * 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharex=True)
    y_positions = {'car': 2, 'bike': 1, 'pt': 0}
    offsets = (0.15, -0.15)

    for ax, (level, heading) in zip(axes, (
        ('Low', 'LOW SUFFICIENCY  ·  50% of baseline car mean'),
        ('High', 'HIGH SUFFICIENCY  ·  100% of baseline car mean'),
    )):
        ax.set_title(heading, fontsize=11, fontweight='bold', pad=17)
        ax.axvline(0, color='#374151', lw=1.5, zorder=1)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-0.60, 2.60)
        ax.set_yticks([y_positions[m] for m in MODES])
        ax.set_yticklabels([f'{MODE_LABELS[m]}\nBaseline: {baseline.loc[(level,m)]:.1f}%'
                            for m in MODES], fontsize=10)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.grid(axis='x', color='#e3e7ed', lw=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:+.0f}' if x else '0'))
        ax.set_xlabel('Change from baseline (percentage points)', fontsize=10, labelpad=10)
        for idx, label in enumerate(('Warm-start GA', 'No-warm-start GA')):
            for mode in MODES:
                r = selected.loc[selected.scenario.eq(label) & selected.level.eq(level)
                                 & selected['mode'].eq(mode)].iloc[0]
                delta = float(r.delta_pp)
                y = y_positions[mode] + offsets[idx]
                ax.plot([0, delta], [y, y], color=COLORS[idx], lw=2.5,
                        alpha=0.8, solid_capstyle='round', zorder=2)
                ax.scatter(delta, y, color=COLORS[idx], marker=SHAPES[idx],
                           s=86, edgecolors='white', linewidths=0.8, zorder=3)
                sign = 1 if delta >= 0 else -1
                ax.annotate(f'{delta:+.1f} pp', (delta, y), xytext=(sign*8, 0),
                            textcoords='offset points', ha='left' if sign > 0 else 'right',
                            va='center', color=COLORS[idx], fontsize=9, fontweight='semibold')

    fig.suptitle(f'{MINUTES}-minute STATENT job accessibility — sufficiency change',
                 y=0.99, fontsize=16, weight='bold')
    handles = [Line2D([0], [0], color=COLORS[i], marker=SHAPES[i], lw=2.5,
                      markersize=8, label=label)
               for i, label in enumerate(('Warm-start GA', 'No-warm-start GA'))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.935),
               frameon=False, ncol=2, fontsize=11)

    attached = data.groupby('mode')['n_attached'].nunique()
    if (attached > 1).any():
        print('WARNING: n_attached differs between scenarios for modes:',
              list(attached[attached > 1].index))
        note = 'Percentages use each scenario’s attached origins; denominators may differ. Not the GA car constraint.'
    else:
        note = 'Percentage of attached origins; not the continuous GA fitness or fixed-mask car constraint.'
    fig.text(0.5, 0.035, note, ha='center', fontsize=9, color='#5b6572')
    fig.subplots_adjust(left=0.17, right=0.985, top=0.80, bottom=0.20, wspace=0.36)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFILE, dpi=240, facecolor='white')
    fig.savefig(OUTFILE.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    csv_path = OUTFILE.with_name(OUTFILE.stem + '_values.csv')
    data.to_csv(csv_path, index=False)
    print('Saved:', OUTFILE)
    print('Saved:', OUTFILE.with_suffix('.svg'))
    print('Saved:', csv_path)
    print('\nChanges from baseline in percentage points:')
    print(selected.pivot_table(index=['mode', 'level'], columns='scenario', values='delta_pp'))

if __name__ == '__main__':
    main()
