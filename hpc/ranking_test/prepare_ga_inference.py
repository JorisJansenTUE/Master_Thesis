#!/usr/bin/env python3
"""Create/validate an external GA evaluation split for the frozen delay GNN.

Uses prepared tensors (never copies MATSim files). The checkpoint remains in
OLD_GNN; the expanded dataset is only used as inputs. The external split's
`test` group is solely the requested GA provenance group, even when its members
belong to train/val in the *expanded* model's splits. This is evaluation of the
ORIGINAL checkpoint, not a claim that every scenario is held out from retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

GROUPS = ('ga_spatial_no_warm', 'ga_spatial_warm', 'ga_nonspatial_old')
CHECK_KEYS = ('baseline_network_sha256', 'street_segments_sha256',
              'baseline_flows_sha256', 'baseline_linkstats_sha256',
              'travel_time_hour', 'static_features', 'target_representation',
              'baseline_flow_window')


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_fixed_inputs(old: Path, new: Path):
    old_prov = read_json(old / 'fixed_provenance.json')
    new_prov = read_json(new / 'fixed_provenance.json')
    for k in CHECK_KEYS:
        if k not in old_prov or old_prov[k] != new_prov.get(k):
            raise ValueError(f'Original vs expanded fixed provenance mismatch {k}: '
                             f'{old_prov.get(k)!r} != {new_prov.get(k)!r}')
    for f in ('gnn_static/dual_nodes.csv.gz',):
        left = pd.read_csv(old / f)
        right = pd.read_csv(new / f)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    for rel in ('gnn_static/dual_edge_index.npy',):
        a, b = np.load(old / rel, allow_pickle=False), np.load(new / rel, allow_pickle=False)
        if not np.array_equal(a,b):
            raise ValueError(f'Original vs expanded graph mismatch {rel}')
    for rel in ('gnn_inputs/fixed_inputs.npz', 'gnn_static/baseline_tt.npz'):
        with np.load(old / rel, allow_pickle=False) as a, np.load(new / rel, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError(f'Original vs expanded tensor keys mismatch {rel}')
            for k in a.files:
                if not np.array_equal(a[k], b[k]):
                    raise ValueError(f'Original vs expanded tensor mismatch {rel}:{k}')
    for rel in ('gnn_inputs/input_metadata.json',):
        a, b = read_json(old / rel), read_json(new / rel)
        for k in ('static_feature_names', 'node_count', 'hours', 'travel_time_hour'):
            if a.get(k) != b.get(k):
                raise ValueError(f'Original vs expanded input metadata mismatch {rel}:{k}')
    return old_prov


def chosen_rows(prepared: Path, group: str, limit: int):
    manifest = prepared / 'preparation_metadata/source_manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline='', encoding='utf-8') as f:
        allrows = list(csv.DictReader(f))
    rows = sorted((r for r in allrows if r['group'] == group),key=lambda r:r['scenario'])
    if not rows:
        raise ValueError(f'No {group} scenarios found in {manifest}')
    if limit:
        rows = rows[:limit]
    original = read_json(prepared / 'splits.json')
    id_roles = {sid: role for role in ('train', 'val', 'test') for sid in original[role]}
    for r in rows:
        sid = r['scenario']
        if sid not in id_roles or id_roles[sid] != r['split']:
            raise ValueError(f'Invalid split for {sid}: {id_roles.get(sid)} vs manifest {r["split"]}')
        if group == 'ga_spatial_no_warm' and r['split'] != 'test':
            raise ValueError(f'Independent GA test contaminated: {sid} assigned to {r["split"]}')
        if not (prepared / 'dynamic' / sid / 'x_dynamic.npy').is_file():
            raise FileNotFoundError(f'Missing GNN dynamic tensor for {sid}')
        if not (prepared / 'targets' / sid / 'target_delta.npz').is_file():
            raise FileNotFoundError(f'Missing GNN target tensor for {sid}')
        if not Path(r['source']).is_dir():
            raise FileNotFoundError(f'Missing MATSim source for {sid}: {r["source"]}')
    ids = [r['scenario'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate GA scenario names')
    return ids, original, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-gnn', type=Path, required=True)
    p.add_argument('--prepared-gnn', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--out', type=Path, required=True,
                   help='Result root; creates external_test_splits.json and inference_provenance.json')
    p.add_argument('--verify-predictions', action='store_true')
    args=p.parse_args()
    if args.limit < 0:
        p.error('limit must be >= 0')
    old, prepared, ckpt, out = args.old_gnn, args.prepared_gnn, args.checkpoint, args.out
    if not ckpt.is_file():
        raise FileNotFoundError(f'Frozen checkpoint missing: {ckpt}')
    provenance = verify_fixed_inputs(old,prepared)
    ids, original_splits, manifest = chosen_rows(prepared,args.group,args.limit)
    selected = set(ids)
    train = next((x for x in original_splits['train'] if x not in selected),None)
    val = next((x for x in original_splits['val'] if x not in selected and x != train),None)
    if train is None or val is None:
        raise ValueError('Cannot choose disjoint nonempty train and val placeholders')
    # evaluate.py only loads the designated `test` split; the one-element
    # train and val lists exist solely to satisfy its native split validator.
    split = {'train':[train], 'val':[val], 'test':ids,
             'scenario_count':len(ids)+2}
    descriptor = {
        'description':'Frozen original checkpoint on external GA scenarios',
        'group':args.group, 'limit':args.limit,
        'old_gnn':str(old.resolve()), 'prepared_gnn':str(prepared.resolve()),
        'checkpoint':str(ckpt.resolve()), 'checkpoint_sha256':sha(ckpt),
        'source_manifest_sha256':sha(manifest),
        'expanded_splits_sha256':sha(prepared / 'splits.json'),
        'fixed_baseline_network_sha256':provenance['baseline_network_sha256'],
        'hour':provenance['travel_time_hour'], 'scenario_ids':ids,
        'n_scenarios':len(ids)
    }
    out.mkdir(parents=True,exist_ok=True)
    splits_path=out / 'external_test_splits.json'
    desc_path=out / 'inference_provenance.json'
    for path, obj in ((splits_path,split),(desc_path,descriptor)):
        if path.is_file():
            if read_json(path)!=obj:
                raise ValueError(f'Existing {path} describes different scenarios/checkpoint. Use NEW OUTPUT.')
        else:
            path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    if args.verify_predictions:
        infer = out / 'inference'
        summary = read_json(infer / 'summary.json')
        if (summary.get('split')!='test' or summary.get('scenarios')!=len(ids)
                or summary.get('target_type')!='road_congestion_delay_change'
                or summary.get('hour')!=descriptor['hour']):
            raise ValueError('Inference summary is inconsistent with chosen group and target')
        with (infer / 'scenario_metrics.csv').open(newline='',encoding='utf-8') as f:
            recorded = [r['scenario'] for r in csv.DictReader(f)]
        if set(recorded)!=set(ids) or len(recorded)!=len(ids):
            raise ValueError('Inference scenario_metrics.csv does not cover exactly the GA group')
        missing = [sid for sid in ids if not (infer / 'predictions' / (sid + '.npz')).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} saved predictions missing: {missing[:5]}')
    print(f'PASS: {args.group} scenarios={len(ids)}; identical original/expanded '
          f'graph and fixed tensors; frozen checkpoint SHA={descriptor["checkpoint_sha256"][:12]}',flush=True)
    print('External test split:',splits_path,flush=True)

if __name__ == '__main__':
    main()
