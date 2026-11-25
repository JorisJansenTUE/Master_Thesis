#!/usr/bin/env python3
"""Read-only staging of historical + GA MATSim scenarios for one expanded GNN dataset.

Every scenario in the staged directory is a symlink to the existing source.
Splits are deterministic, source-aware and written to split_plan.json.  The
actual splits.json is installed *before* prepare_hpc by install_gnn_splits_v3.py.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

FLOW = Path('targets/output_detailed_link_statistics.csv.gz')
LINKSTATS = Path('targets/linkstats.txt.gz')
REQUIRED = ['scenario.json', 'interventions.json', 'network.xml.gz', 'link_mapping.csv.gz',
            str(FLOW), str(LINKSTATS), 'run_status.json', 'targets/target_summary.json']

def readj(p: Path):
    with p.open(encoding='utf-8') as f: return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def completed_baseline_hashes(baseline: Path) -> dict:
    """Hashes of the *files* from which optimization.prepare_scenario builds GA scenarios."""
    return {
        'network': sha256_file(baseline / 'network.xml.gz'),
        'schedule': sha256_file(baseline / 'transit_schedule.xml.gz'),
    }

def get_split(d, key):
    aliases = {'train':['train','training','train_ids','train_scenarios'],
               'val':['val','validation','valid','val_ids','validation_ids'],
               'test':['test','testing','test_ids','test_scenarios']}
    for src in (d.get('splits'), d):
        if isinstance(src, dict):
            for k in aliases[key]:
                if k in src:
                    assert isinstance(src[k], list), (key,k)
                    return set(src[k])
    raise ValueError(f'No {key} split found; keys={list(d)}')

def eligible(scenario: Path, baseline_hashes: dict, is_baseline=False,
             completed_hashes: dict | None = None):
    for f in REQUIRED:
        p=scenario/f
        if not p.is_file() or not p.stat().st_size:
            return f'missing/empty {f}'
    try:
        status=readj(scenario/'run_status.json')
        if status.get('state') not in ('completed','complete'):
            return f"run_status.state={status.get('state')!r}"
        if status.get('postprocessing',{}).get('state') != 'completed':
            return 'postprocessing incomplete'
        meta=readj(scenario/'scenario.json')
        if meta.get('id') != scenario.name:
            return 'scenario ID mismatch'
        ints=readj(scenario/'interventions.json')
        if not isinstance(ints,list):return 'interventions is not a list'
        if not is_baseline:
            allowed={'bicycle_street','protected_bike_lane','dedicated_bus_lane','destination_only'}
            unknown={i.get('intervention_type') for i in ints if isinstance(i,dict)}-allowed
            if unknown:return f'unsupported intervention types {sorted(map(str,unknown))}'
        if is_baseline:
            if ints or meta.get('strategy')!='baseline':return 'bad baseline metadata'
        elif not ints or meta.get('strategy')=='baseline':
            return 'empty/non-intervention scenario'
        # Historical scenarios record the original source hashes. The GA
        # scenario generator instead hashes the network/schedule in the
        # completed baseline_00000 folder. Accept this second identity only
        # when it is *derived from those actual files* (never a hardcoded bypass).
        observed = meta.get('baseline_hashes')
        if baseline_hashes and observed != baseline_hashes and observed != completed_hashes:
            return 'different baseline hashes (neither source nor completed baseline files)' 
        summary=readj(scenario/'targets/target_summary.json')
        if summary.get('flow_file') != str(FLOW):return 'target_summary wrong flow file'
        if summary.get('flow',{}).get('rows',0)<=0:return 'target_summary no flow rows'
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as e:
        return f'bad metadata: {e}'
    return None

def fingerprint(p:Path):
    interventions=readj(p/'interventions.json')
    # Normalize ordering of independent interventions and target link ID lists.
    norm=[]
    for item in interventions:
        if not isinstance(item,dict):raise ValueError(f'Invalid intervention: {p}')
        obj=dict(item)
        if isinstance(obj.get('target_link_ids'),list):
            obj['target_link_ids']=sorted(str(x) for x in obj['target_link_ids'])
        norm.append(json.dumps(obj,sort_keys=True,separators=(',',':')))
    canonical='\n'.join(sorted(norm))
    return hashlib.sha256(canonical.encode()).hexdigest()

def direct_ga_scenarios(root:Path):
    # IMPORTANT: nonspatial GA scenarios reside directly in GA_Optimization,
    # alongside nested experiment folders. Do not traverse those folders.
    if not root.is_dir():raise FileNotFoundError(root)
    return sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith('ga_')
                   and (p/'scenario.json').is_file()),key=lambda p:p.name)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--historical',type=Path,required=True)
    a.add_argument('--original-splits',type=Path,required=True)
    a.add_argument('--warm-ga',type=Path,required=True)
    a.add_argument('--no-warm-ga',type=Path,required=True)
    a.add_argument('--old-ga-root',type=Path,default=None)
    a.add_argument('--skip-old',action='store_true')
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--warm-val-fraction',type=float,default=0.15)
    a.add_argument('--min-warm',type=int,default=1000)
    a.add_argument('--min-no-warm',type=int,default=1000)
    a.add_argument('--min-old',type=int,default=100)
    a.add_argument('--min-historical',type=int,default=100)
    args=a.parse_args()
    if not 0 <= args.warm_val_fraction <= 0.5:raise ValueError('warm-val-fraction out of range')
    hist=args.historical.resolve(strict=True)
    warm=args.warm_ga.resolve(strict=True)
    no_warm=args.no_warm_ga.resolve(strict=True)
    if args.out.exists():raise FileExistsError(args.out)
    original=readj(args.original_splits)
    original_roles={k:get_split(original,k) for k in ('train','val','test')}
    if any(original_roles[x]&original_roles[y] for x,y in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Original split IDs overlap')
    baseline=hist/'baseline_00000'
    baseline_meta=readj(baseline/'scenario.json')
    baseline_hashes=baseline_meta.get('baseline_hashes')
    if not baseline_hashes:raise ValueError('Historical baseline has no baseline_hashes')
    reason=eligible(baseline,baseline_hashes,is_baseline=True)
    if reason:raise RuntimeError(f'Baseline unusable: {reason}')
    completed_hashes = completed_baseline_hashes(baseline)
    print('Historical source baseline hashes:', json.dumps(baseline_hashes, sort_keys=True), flush=True)
    print('Completed baseline file SHA-256:', json.dumps(completed_hashes, sort_keys=True), flush=True)
    segments=hist/'street_segments.csv.gz'
    if not segments.is_file() or not segments.stat().st_size:raise FileNotFoundError(segments)

    counters=Counter()
    candidates=[] # (priority, path, group, split)
    for role in ('test','val','train'):
        for sid in sorted(original_roles[role]):
            if sid=='baseline_00000':continue
            p=hist/sid
            if not p.is_dir():raise FileNotFoundError(f'Original {role} missing: {p}')
            reason=eligible(p,baseline_hashes,completed_hashes=completed_hashes)
            if reason:raise RuntimeError(f'Original {role} invalid {sid}: {reason}')
            # Original test/val take priority over GA duplicates.
            candidates.append(({'test':0,'val':1,'train':3}[role],p,f'historical_{role}',role))

    def add_ga(root, group, priority, role):
        dirs=direct_ga_scenarios(root)
        counters[group+'_discovered']=len(dirs)
        for p in dirs:
            reason=eligible(p,baseline_hashes,completed_hashes=completed_hashes)
            if reason:
                counters[group+'_excluded_'+reason]+=1
                continue
            candidates.append((priority,p,group,role))

    add_ga(no_warm,'ga_spatial_no_warm',2,'test')
    add_ga(warm,'ga_spatial_warm',4,'train')
    if not args.skip_old:
        oldroot=(args.old_ga_root or warm.parent).resolve(strict=True)
        add_ga(oldroot,'ga_nonspatial_old',5,'train')

    records=[]; seen_ids={};seen_fps={}
    for priority,p,group,role in sorted(candidates,key=lambda x:(x[0],x[1].name)):
        sid=p.name
        fp=fingerprint(p)
        if sid in seen_ids:
            if seen_ids[sid]['fingerprint'] != fp:
                raise ValueError(f'Conflicting scenario ID {sid} across sources: {seen_ids[sid]["source"]} / {p}')
            counters[group+'_duplicate_id']+=1
            continue
        if fp in seen_fps:
            counters[group+'_duplicate_configuration']+=1
            continue
        r={'scenario':sid,'group':group,'split':role,'source':str(p.resolve()),
           'fingerprint':fp,'baseline_network_hash':baseline_hashes.get('network','')}
        records.append(r);seen_ids[sid]=r;seen_fps[fp]=sid
        counters[group+'_included']+=1

    warm_records=[r for r in records if r['group']=='ga_spatial_warm']
    rng=random.Random(args.seed)
    rng.shuffle(warm_records)
    num_warm_val=round(args.warm_val_fraction*len(warm_records))
    for r in warm_records[:num_warm_val]:r['split']='val'
    counters['ga_spatial_warm_val']=num_warm_val

    for group,min_count in [('historical_train',args.min_historical),('ga_spatial_warm',args.min_warm),
                            ('ga_spatial_no_warm',args.min_no_warm)]:
        if counters[group+'_included']<min_count:
            causes = {k:v for k,v in counters.items() if k.startswith(group+'_excluded_')}
            raise ValueError(f'{group}: only {counters[group+"_included"]} usable, '
                             f'expected >= {min_count}. Exclusion reasons: {causes}')
    if not args.skip_old and counters['ga_nonspatial_old_included']<args.min_old:
        raise ValueError(f"Old GA root: {counters['ga_nonspatial_old_included']} usable scenarios; check old root, completion metadata or pass --skip-old")

    roles={key:sorted(r['scenario'] for r in records if r['split']==key) for key in ('train','val','test')}
    if not all(roles.values()):raise ValueError('Empty train/val/test split')
    if len(roles['val'])<200:raise ValueError('Too few validation scenarios')
    if len(roles['test'])<args.min_no_warm:raise ValueError('Too few test scenarios')
    args.out.mkdir(parents=True)
    (args.out/'baseline_00000').symlink_to(baseline,target_is_directory=True)
    (args.out/'street_segments.csv.gz').symlink_to(segments)
    for r in records:
        (args.out/r['scenario']).symlink_to(r['source'],target_is_directory=True)
    manifest=args.out/'source_manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scenario','group','split','source','fingerprint','baseline_network_hash'])
        writer.writeheader();writer.writerows(sorted(records,key=lambda r:r['scenario']))
    plan={'policy':'historical original train/val/test + warm GA 15% val + no-warm GA test + old root train',
          'seed':args.seed,'warm_val_fraction':args.warm_val_fraction,
          'splits':roles,'counts':dict(counters),
          'original_split_counts':{k:len(v) for k,v in original_roles.items()},
          'paths':{'historical':str(hist),'original_splits':str(args.original_splits),
                   'warm_ga':str(warm),'no_warm_ga':str(no_warm),
                   'old_ga_root':None if args.skip_old else str((args.old_ga_root or warm.parent).resolve())}}
    (args.out/'split_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'staged':len(records),'split_sizes':{k:len(v) for k,v in roles.items()},
                      'included':{k:v for k,v in counters.items() if k.endswith('_included')},
                      'excluded':{k:v for k,v in counters.items() if 'excluded' in k}},indent=2),flush=True)

if __name__=='__main__':main()
