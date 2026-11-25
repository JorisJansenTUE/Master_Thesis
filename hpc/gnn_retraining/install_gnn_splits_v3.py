#!/usr/bin/env python3
"""Install explicit source-aware splits BEFORE running prepare_hpc; verify AFTER."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from gnn_surrogate.dataset.build_splits import build_split_manifest
from gnn_surrogate.dataset.dataset import load_splits

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--staged',required=True,type=Path)
    p.add_argument('--gnn-out',required=True,type=Path)
    p.add_argument('--phase',required=True,choices=('before','after'))
    p.add_argument('--resume',action='store_true',help='Require an exact match with an existing prepared directory')
    a=p.parse_args()
    plan=json.loads((a.staged/'split_plan.json').read_text())
    roles=plan['splits']
    all_ids=sorted(sum((roles[k] for k in ('train','val','test')),[]))
    if len(all_ids)!=len(set(all_ids)):raise ValueError('Overlapping splits')
    manifest={r['scenario']:r for r in csv.DictReader((a.staged/'source_manifest.csv').open())}
    if set(manifest)!=set(all_ids):raise ValueError('Manifest and explicit splits disagree')
    destination=a.gnn_out/'splits.json'
    if a.phase=='before':
        if a.gnn_out.exists():
            if not a.resume:
                raise FileExistsError(f'GNN output already exists: {a.gnn_out}; set PREP_RESUME=1 to resume')
            if not destination.is_file():
                raise ValueError('Resume requested but existing splits.json is missing')
            existing=load_splits(destination)
            for key in ('train','val','test'):
                if set(existing[key])!=set(roles[key]):
                    raise ValueError(f'Resume split differs: {key}; use new GNN_DATA')
            old_manifest=a.gnn_out/'preparation_metadata'/'source_manifest.csv'
            if not old_manifest.is_file() or old_manifest.read_bytes()!=(a.staged/'source_manifest.csv').read_bytes():
                raise ValueError('Resume scenario sources differ; use new GNN_DATA')
            print('Resume: existing source manifest and splits match',flush=True)
            return
        a.gnn_out.mkdir(parents=True)
        # Preserve native split metadata/schema; replace only its members.
        splits=build_split_manifest(all_ids,seed=plan['seed'])
        for key in ('train','val','test'):
            splits[key]=roles[key]
        if 'counts' in splits and isinstance(splits['counts'],dict):
            for key in ('train','val','test'): splits['counts'][key]=len(roles[key])
        destination.write_text(json.dumps(splits,indent=2)+'\n')
        (a.gnn_out/'preparation_metadata').mkdir(exist_ok=True)
        import shutil
        shutil.copy2(a.staged/'source_manifest.csv',a.gnn_out/'preparation_metadata'/'source_manifest.csv')
        # Store count+policy, not redundant full split plan (splits.json already contains IDs).
        audit={k:v for k,v in plan.items() if k!='splits'}
        (a.gnn_out/'preparation_metadata'/'source_plan.json').write_text(json.dumps(audit,indent=2)+'\n')
    else:
        actual=load_splits(destination)
        for key in ('train','val','test'):
            if set(actual[key])!=set(roles[key]):raise ValueError(f'Split mismatch: {key}')
            if not all((a.gnn_out/'dynamic'/sid/'x_dynamic.npy').is_file() and
                       (a.gnn_out/'targets'/sid/'target_delta.npz').is_file() for sid in roles[key]):
                raise ValueError(f'Missing prepared tensors in split {key}')
        report=json.loads((a.gnn_out/'preparation_report.json').read_text())
        if set(report['processed_scenarios'])!=set(all_ids):raise ValueError('Not all intended samples preprocessed')
        print('VERIFIED: fixed source-aware splits and complete GNN tensors',
              {key:len(roles[key]) for key in ('train','val','test')},flush=True)

if __name__=='__main__':main()
