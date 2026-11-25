#!/usr/bin/env python3
"""Diagnose why stage_expanded_gnn_v3 reports 0 usable GA scenarios.

Read-only; uses the exact validation and fingerprinting logic from the existing
stage_expanded_gnn_v3.py instead of inventing different eligibility criteria.
"""
from __future__ import annotations
import argparse
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage-script', required=True, type=Path)
    p.add_argument('--historical', required=True, type=Path)
    p.add_argument('--warm-ga', required=True, type=Path)
    p.add_argument('--no-warm-ga', type=Path)
    p.add_argument('--old-ga-root', type=Path)
    p.add_argument('--examples', type=int, default=4)
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location('ga_stager', a.stage_script)
    if spec is None or spec.loader is None:
        raise SystemExit('Cannot import stage script')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = a.historical / 'baseline_00000' / 'scenario.json'
    hashes = module.readj(baseline).get('baseline_hashes')
    if not hashes:
        raise SystemExit(f'No baseline_hashes in {baseline}')
    print('Historical baseline hashes:', hashes, flush=True)
    entries = [('WARM', a.warm_ga)]
    if a.no_warm_ga: entries.append(('NO_WARM', a.no_warm_ga))
    if a.old_ga_root: entries.append(('OLD_DIRECT', a.old_ga_root))
    for label, root in entries:
        root = root.resolve()
        print(f'\n===== {label}: {root} =====', flush=True)
        if not root.is_dir():
            print('DIRECTORY DOES NOT EXIST', flush=True)
            continue
        children = sorted(x for x in root.iterdir() if x.is_dir())
        print('Top-level subdirectories:', len(children), flush=True)
        print('First 12 names:', [x.name for x in children[:12]], flush=True)
        candidates = module.direct_ga_scenarios(root)
        print('Discovered immediate ga_* scenario folders:', len(candidates), flush=True)
        if not candidates:
            nested = []
            for subdir in children[:40]:
                nested_count = len([x for x in subdir.iterdir()
                                    if x.is_dir() and x.name.startswith('ga_')
                                    and (x / 'scenario.json').is_file()])
                if nested_count:
                    nested.append((str(subdir), nested_count))
            if nested:
                print('NOTE: scenarios found ONE LEVEL LOWER:', nested[:10], flush=True)
            print('No scenarios discovered. Check the supplied root path and names.', flush=True)
            continue
        counts = Counter()
        examples = defaultdict(list)
        ok = []
        for folder in candidates:
            reason = module.eligible(folder, hashes) or 'USABLE'
            counts[reason] += 1
            if len(examples[reason]) < a.examples:
                examples[reason].append(folder.name)
            if reason == 'USABLE':
                ok.append(folder)
        print('Eligibility breakdown (before deduplication):', flush=True)
        for reason, n in counts.most_common():
            print(f'  {n:5d}  {reason}', flush=True)
            print('         examples:', ', '.join(examples[reason]), flush=True)
        if ok:
            fingerprints = Counter()
            errors=[]
            for folder in ok:
                try: fingerprints[module.fingerprint(folder)] += 1
                except Exception as e: errors.append((folder.name, str(e)))
            print(f'Usable folders: {len(ok)}, unique configuration fingerprints: {len(fingerprints)}', flush=True)
            if errors: print('Fingerprint errors:', errors[:a.examples], flush=True)
        else:
            print('NO USABLE SCENARIOS. The dominant rejection reason above identifies the mismatch.', flush=True)

if __name__ == '__main__':
    main()
