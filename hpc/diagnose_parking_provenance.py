#!/usr/bin/env python3
"""Classify old fitness experiment's failed caches by VERIFIED parking provenance.

Usage:
  python diagnose_parking_provenance.py /path/to/fitness_experiments/trial_01

Reads metadata and SHA hashes only; never edits source data.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def classify(entry):
    p=Path(entry['source'])
    if not p.is_file():
        return 'source accessibility missing'
    folder=p.parent
    summary = next((x for x in (
        folder/'sufficiency_summary_statpop.csv',
        folder/'sufficiency_summary.csv',
        folder/'sufficiency_summary.csv.gz') if x.is_file()), None)
    if not summary:
        return 'summary missing'
    metadata = [p for p in (folder/'fitness_statpop.json',folder/'fitness.json') if p.is_file()]
    if not metadata:
        return 'no fitness metadata'
    statuses=[]
    for fname in metadata:
        try:
            obj=json.loads(fname.read_text(encoding='utf-8'))
            park=obj.get('accessibility_parameters',{}).get('car_parking_penalty_s')
            if park is None:
                statuses.append('metadata has no parking value')
                continue
            if abs(float(park)-300)>1e-8:
                return f'parking differs: {park} s'
            if obj.get('summary_sha256')==hash_file(summary) and obj.get('hexagon_sha256')==hash_file(p):
                return 'verified 300s + matching file hashes'
            statuses.append('300s metadata but missing/mismatched hashes')
        except Exception as exc:
            statuses.append('invalid metadata: '+type(exc).__name__)
    return '; '.join(sorted(set(statuses)))


def main():
    if len(sys.argv)!=2:
        sys.exit('Usage: python diagnose_parking_provenance.py /path/to/experiment')
    root=Path(sys.argv[1])/'chunks'
    errors=[]
    for f in sorted(root.glob('chunk_*.json')):
        content=json.loads(f.read_text(encoding='utf-8'))
        errors.extend(content.get('errors',[]))
    failed=[x for x in errors if 'car-parking provenance' in x.get('error','')]
    counts=collections.Counter(); examples={}
    for e in failed:
        status=classify(e)
        counts[status]+=1
        examples.setdefault(status,e['scenario'])
    print('All scenario errors:',len(errors),'| parking-provenance errors:',len(failed))
    for category,count in counts.most_common():
        print(f'{count:5}  {category} | e.g. {examples[category]}')
    print('\nDo not use --ignore-cache-provenance unless you independently verified')
    print('the source accessibility files were computed with the chosen 300 s car allowance.')

if __name__=='__main__':
    main()
