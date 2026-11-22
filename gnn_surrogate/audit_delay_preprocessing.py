"""Pre-flight audit of a preprocessed congestion-delay GNN dataset.

Run from the thesis repository root (with venv activated):

 python audit_delay_preprocessing.py \
   --gnn-dir /vast.mnt/home/20201733/locarno_ETH_Model/gnn_training_tt_delay_v2 \
   --dataset /vast.mnt/home/20201733/locarno_ETH_Model/training_dataset_v4 \
   --scenario cluster_00051

Use --sample-count 12 instead of --scenario to test multiple scenarios from
all three sampling strategies, when available. Never modifies files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_surrogate.preprocessing.targets.target_tensors import (
    _eligible_links,
    build_baseline_travel_times,
    build_delta_targets,
)
from gnn_surrogate.preprocessing.targets.eligibility import mask_from_inputs

TARGET_TYPE = "road_congestion_delay_change"


def get_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_allclose(name: str, actual: np.ndarray, expected: np.ndarray,
                   mask: np.ndarray | None = None, atol: float = .002) -> float:
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: shape mismatch {actual.shape} != {expected.shape}")
    if mask is None:
        mask = np.ones(actual.shape, dtype=bool)
    diff = np.abs(actual.astype(np.float64)[mask] - expected.astype(np.float64)[mask])
    maximum = float(np.max(diff)) if diff.size else 0.0
    if maximum > atol or not np.isfinite(maximum):
        raise ValueError(f"{name}: maximum absolute difference {maximum:.6g} > {atol}")
    return maximum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gnn-dir', required=True, type=Path)
    ap.add_argument('--dataset', required=True, type=Path)
    ap.add_argument('--scenario', action='append', default=[],
                    help='Scenario ID (repeatable). Otherwise sample from splits')
    ap.add_argument('--sample-count', type=int, default=15,
                    help='Minimum number of scenarios to sample across strategy/budget groups')
    args = ap.parse_args()
    out, root = args.gnn_dir, args.dataset
    graph = out / 'gnn_static'
    nodes = pd.read_csv(graph / 'dual_nodes.csv.gz', dtype={
        'segment_id': str, 'from_node': str, 'to_node': str, 'baseline_link_ids': str})
    with np.load(graph / 'baseline_tt.npz', allow_pickle=False) as cache:
        required = {'baseline_tt_s', 'baseline_freeflow_s', 'present_mask', 'hour'}
        if missing := required - set(cache.files):
            raise ValueError(f'Missing baseline cache keys: {sorted(missing)}')
        base_tt, base_ff, base_mask = [cache[key].copy() for key in
                                      ('baseline_tt_s', 'baseline_freeflow_s', 'present_mask')]
        hour = int(cache['hour'])
    recomputed_tt, recomputed_mask, recomputed_ff = build_baseline_travel_times(
        nodes, root / 'baseline_00000', hour)
    if not np.array_equal(base_mask, recomputed_mask):
        raise ValueError('Baseline cache eligibility differs from baseline network')
    check_allclose('baseline cached travel time', base_tt, recomputed_tt, atol=.001)
    check_allclose('baseline cached free-flow time', base_ff, recomputed_ff, atol=.001)
    print(f'BASELINE OK: hour={hour}, nodes={len(nodes)}, eligible={base_mask.sum()}')

    fixed_meta = get_json(out / 'gnn_inputs/input_metadata.json')
    with np.load(out / 'gnn_inputs/fixed_inputs.npz', allow_pickle=False) as arrays:
        x_static = arrays['x_static'].copy()
        hourly = arrays['baseline_hourly_raw'].copy()
    if fixed_meta['travel_time_hour'] != hour or hourly.shape[1] <= hour:
        raise ValueError('Baseline flow input hour differs from selected target hour')
    print('FIXED INPUTS OK:', x_static.shape, 'peak-hour flows', hourly[:, hour, :].shape)

    splits = get_json(out / 'splits.json')
    all_ids = splits['train'] + splits['val'] + splits['test']
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Split scenario IDs overlap')
    if args.scenario:
        ids = args.scenario
    else:
        # Audit multiple budgets within each strategy. Scenario discovery does
        # not imply a particular distribution of generated budgets.
        groups = {}
        budget_edges = [3000, 7000, 15000, 30000, 50000]
        for sid in sorted(all_ids):
            info = get_json(root / sid / 'scenario.json')
            strategy = info.get('strategy', sid.split('_')[0])
            budget = float(info.get('budget_used_m', 0.0))
            bucket = int(np.searchsorted(budget_edges, budget, side='left'))
            groups.setdefault((strategy, bucket), []).append(sid)
        print('Sampling groups (strategy, budget bin):',
              {str(k): len(v) for k, v in sorted(groups.items())})
        ids = []
        rng = np.random.default_rng(2026)
        # At least one scenario per observed group, even if there are more
        # groups than --sample-count.
        for group in groups.values():
            ids.extend(rng.choice(group, size=1, replace=False).tolist())
        if len(ids) < args.sample_count:
            remaining = sorted(set(all_ids) - set(ids))
            ids.extend(rng.choice(remaining,
                                  size=min(len(remaining), args.sample_count-len(ids)),
                                  replace=False).tolist())
    print('SPLITS:', {s:len(splits[s]) for s in ('train','val','test')})
    baseline_run = get_json(root / 'baseline_00000' / 'run_status.json')
    sim_keys = ('last_iteration', 'linkstats_interval', 'linkstats_average_iterations')
    print('Baseline MATSim output settings:', {k: baseline_run.get(k) for k in sim_keys})
    for sid in ids:
        meta = get_json(out / 'targets' / sid / 'target_metadata.json')
        if meta.get('target_type') != TARGET_TYPE or meta.get('hour') != hour:
            raise ValueError(f'{sid}: incorrect target type/hour {meta}')
        with np.load(out / 'targets' / sid / 'target_delta.npz', allow_pickle=False) as f:
            y, mask = f['y_delta'].copy(), f['valid_mask'].copy()
        expected_y, expected_mask, _ = build_delta_targets(
            nodes, root / sid, base_tt, base_mask, base_ff, hour)
        if not np.array_equal(mask, expected_mask):
            raise ValueError(f'{sid}: saved mask differs from network-derived mask')
        max_target_error = check_allclose(f'{sid} delay delta', y, expected_y, atol=.002)
        if np.any(y[~mask] != 0):
            raise ValueError(f'{sid}: nonzero ineligible target')
        x = np.load(out / 'dynamic' / sid / 'x_dynamic.npy', allow_pickle=False)
        dyn_meta = get_json(out / 'dynamic' / sid / 'dynamic_metadata.json')
        inferred = mask_from_inputs(x_static, x, fixed_meta['static_feature_names'],
                                    dyn_meta['feature_names'])
        if not np.array_equal(inferred, mask):
            raise ValueError(f'{sid}: model-input mask differs from target mask')
        scenario_links = _eligible_links(nodes, root / sid, hour)
        ff_from_stats = scenario_links.freeflow_s.to_numpy(dtype=np.float64)
        i = scenario_links.node_id.to_numpy(dtype=int)
        speed_change_col = dyn_meta['feature_names'].index('car_freespeed_change_fraction')
        speed_fraction = x[i, speed_change_col].astype(np.float64)
        reconstructed_ff = base_ff[i, 0].astype(np.float64) / (1.0 + speed_fraction)
        max_recon_error = np.max(np.abs(ff_from_stats - reconstructed_ff))
        if not np.isfinite(max_recon_error) or max_recon_error > .02:
            raise ValueError(f'{sid}: scenario free-flow reconstruction differs from linkstats by '
                             f'{max_recon_error:.4f} s. Inspect speedFactor handling.')
        info = get_json(root / sid / 'scenario.json')
        run = get_json(root / sid / 'run_status.json')
        if any(run.get(k) != baseline_run.get(k) for k in sim_keys):
            print(f'WARNING {sid}: MATSim output settings differ from baseline:',
                  {k: run.get(k) for k in sim_keys})
        print(f'PASS {sid:20s} strategy={info.get("strategy", "?"):9s} '
              f'budget={info.get("budget_used_m", float("nan"))/1000:6.2f}km '
              f'eligible={int(mask.sum()):4d} '
              f'Δdelay max diff={max_target_error:.6f}s '
              f'FF reconstruction max diff={max_recon_error:.6f}s')
    print(f'PASS: checked {len(ids)} scenarios and all key baseline/delta/mask/reconstruction assumptions')
    print('IMPORTANT: This tests preprocessing consistency, not MATSim simulation validity or intervention sampling quality.')


if __name__ == '__main__':
    main()
