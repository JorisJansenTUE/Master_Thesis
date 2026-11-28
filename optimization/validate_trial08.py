"""Reproduce baseline and best historical Trial 08 with the production evaluator."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from optimization.fitness_trial08 import read_json
from optimization.warm_start import _evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial-dir','baseline','spatial-dir','reference'):
        p.add_argument('--'+name, type=Path, required=True)
    args=p.parse_args()
    trial=read_json(args.trial_dir/'experiment.json')
    base=_evaluate(args.baseline,args.spatial_dir,args.reference,False)
    if abs(base['fitness']-0.89555067)>1e-8 or abs(base['fitness']-trial['baseline_score']['raw_fitness'])>1e-8:
        raise ValueError(f"Baseline score mismatch: {base['fitness']:.10f}")
    for name in ('car','bike','pt'):
        expected = next(c['score'] for c in trial['baseline_score']['components'] if c['mode']==name)
        actual = next(c['mode_score'] for c in base['components'] if c['mode']==name)
        if abs(actual-expected)>1e-8:
            raise ValueError(f'Baseline {name} score mismatch: {actual} vs {expected}')
    if abs(base['threshold_jobs']-trial['common_low_threshold_jobs'])>1e-6:
        raise ValueError(f"Baseline threshold mismatch: {base['threshold_jobs']}")
    rows=pd.read_csv(args.trial_dir/'fitness_scores.csv.gz')
    best=rows.loc[rows['fitness'].idxmax()]
    if best['scenario']!='ga_6ba45611ecf7426d' or abs(best['fitness']-0.90387663)>1e-8:
        raise ValueError('Trial 08 historical best differs from agreed benchmark')
    source=Path(best['source'])
    scenario=source.parent.parent if source.parent.name=='accessibility' else source.parent
    score=_evaluate(scenario,args.spatial_dir,args.reference,False)
    for name in ('car','bike','pt'):
        actual=next(c['mode_score'] for c in score['components'] if c['mode']==name)
        if abs(actual-float(best[f'{name}_score']))>1e-8:
            raise ValueError(f'Best scenario {name} score mismatch: {actual} vs {best[f"{name}_score"]}')
    if abs(score['fitness']-float(best['fitness']))>1e-8:
        raise ValueError('Historical best fitness mismatch')
    print(f"PASS Trial 08: baseline={base['fitness']:.8f}, best={score['fitness']:.8f}, threshold={base['threshold_jobs']:.8f}", flush=True)

if __name__=='__main__':
    main()
