#!/usr/bin/env python3
"""Read-only audit of scenario-level baseline hashes in GA experiment directories.

Checks actual scenario.json values, not just optimization_state.json. Does not
change or copy any MATSim files. Standard-library only.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'_read_error': str(exc)}


def summarize(title, root, expected, limit=4):
    root=Path(root)
    print(f'\n===== {title}: {root} =====', flush=True)
    if not root.is_dir():
        print('ERROR: directory missing')
        return
    dirs=sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith('ga_')
                 and (p/'scenario.json').is_file()), key=lambda p:p.name)
    print('scenario directories:', len(dirs))
    groups=defaultdict(list)
    file_counts=Counter()
    for d in dirs:
        meta=read_json(d/'scenario.json')
        h=meta.get('baseline_hashes')
        key=json.dumps(h, sort_keys=True, default=str)
        groups[key].append(d.name)
        if not (d/'targets/output_detailed_link_statistics.csv.gz').is_file():
            file_counts['missing flow statistics']+=1
        if not (d/'targets/linkstats.txt.gz').is_file():
            file_counts['missing linkstats']+=1
        if not (d/'targets/target_summary.json').is_file():
            file_counts['missing target_summary']+=1
    print('other file issues:', dict(file_counts))
    print('distinct scenario-level baseline_hashes:', len(groups))
    for key, ids in sorted(groups.items(), key=lambda x:-len(x[1]))[:limit]:
        found=json.loads(key)
        print(f'\n  {len(ids)} scenarios; examples: {", ".join(ids[:3])}')
        print('  scenario.json baseline_hashes:', json.dumps(found, sort_keys=True))
        print('  exact dictionary match:', found == expected)
        if isinstance(found,dict):
            for field in sorted(set(expected)|set(found)):
                if found.get(field)!=expected.get(field):
                    print(f'  DIFFERENT {field}: expected={expected.get(field)!r} actual={found.get(field)!r}')
        else:
            print('  Not a dictionary: investigate schema')
    state=root/'optimization_state.json'
    if state.is_file():
        sh=read_json(state).get('baseline_hashes')
        print('\noptimization_state.json baseline hashes:', json.dumps(sh, sort_keys=True))
        print('state matches historical:',sh==expected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--warm', type=Path, required=True)
    p.add_argument('--no-warm', type=Path, required=True)
    p.add_argument('--old',type=Path, help='Optional root with direct ga_* directories')
    a=p.parse_args()
    expected=read_json(a.historical/'baseline_00000/scenario.json').get('baseline_hashes')
    if not expected:
        raise SystemExit('Historical baseline hashes not found')
    print('Historical baseline hashes:',json.dumps(expected,sort_keys=True))
    summarize('WARM',a.warm,expected)
    summarize('NO WARM',a.no_warm,expected)
    if a.old: summarize('OLD GA (DIRECT)',a.old,expected)

if __name__=='__main__':
    main()
