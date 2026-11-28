"""Production Trial 08 fitness; never writes legacy accessibility caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.calculations.evaluate import calculate_accessibility
from accessibility.calculations.population_metrics import load_population, weight_sha256
from accessibility.calculations.sufficiency import SufficiencyReference, assess_sufficiency
from optimization.fitness import _write_json, _write_csv, _load_hexagons

FITNESS_VERSION = 'trial08_statpop_multimodal_park0_low_v1'
OUTPUT_DIR = 'accessibility_trial08'
MODES = ('car', 'bike', 'pt')
THRESHOLDS = (20.0,)
ROUTING = dict(hour_start=8, linkstats_statistic='avg', bike_speed_m_s=4.17,
               pt_transfer_penalty_s=300.0, origin_snap_m=500.0, job_snap_m=300.0,
               car_parking_penalty_s=0.0, population_weighted=True)
SETTINGS = dict(mode_weights=dict.fromkeys(MODES, 1.0), time_weights={'20': 1.0},
                sufficiency_level='low', threshold_fraction=0.5, shortfall_exponent=2,
                car_constraint_enabled=False, **ROUTING)
SPATIAL_FILES = ('origins.csv', 'jobs.csv', 'pt_origin_connectors.csv',
                 'pt_job_connectors.csv', 'population_by_hexagon.csv')
SCENARIO_FILES = ('network.xml.gz', 'transit_schedule.xml.gz', 'targets/linkstats.txt.gz')


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def spatial_hashes(spatial):
    return {f: sha(Path(spatial)/f) for f in SPATIAL_FILES}


def scenario_hashes(scenario):
    return {f: sha(Path(scenario)/f) for f in SCENARIO_FILES}


def mask_path(reference):
    # Keep existing controller's mask filename contract; contents now record
    # the fixed population/origin universe and full baseline provenance.
    return reference.with_name(reference.stem + '_attachments.json')


def validate_reference(reference_file, spatial):
    reference_file = Path(reference_file)
    ref = SufficiencyReference.load(reference_file)
    mask = read_json(mask_path(reference_file))
    if (not ref.population_weighted or ref.car_parking_penalty_s != 0
            or ref.low_fraction != .5 or ref.high_fraction != 1
            or set(ref.mean_car_jobs) != {CumulativeAccessibility.column_name(20)}
            or ref.population_sha256 != weight_sha256(spatial)
            or mask['fitness_version'] != FITNESS_VERSION or mask['settings'] != SETTINGS
            or mask['reference_sha256'] != sha(reference_file)
            or mask['spatial_sha256'] != spatial_hashes(spatial)):
        raise ValueError('Trial 08 baseline reference or spatial provenance mismatch')
    return ref, mask


def validate_rows(data, spatial):
    pop = load_population(spatial).set_index('hex_id')['population'].sort_index()
    if set(data['mode']) != set(MODES) or data.duplicated(['mode','hex_id']).any():
        raise ValueError('Expected exactly one row per mode/origin')
    if not pd.api.types.is_bool_dtype(data['attached']):
        raise ValueError('Attachment flags must be boolean')
    for mode in MODES:
        rows = data.loc[data['mode']==mode].set_index('hex_id').sort_index()
        if not rows.index.equals(pop.index) or not np.array_equal(rows['population'].to_numpy(), pop.to_numpy()):
            raise ValueError(f'Origin/population mismatch: {mode}')
    return pop


def score_rows(data, ref, mask, spatial):
    pop = validate_rows(data, spatial)
    scored, summary = assess_sufficiency(data, ref, THRESHOLDS)
    components = []
    for mode in MODES:
        row = summary.loc[summary['mode']==mode].iloc[0]
        current = set(data.loc[(data['mode']==mode)&data['attached'],'hex_id'])
        baseline = set(mask['attached_origins_by_mode'][mode])
        inhabited = set(pop[pop>0].index)
        shortfall = float(row['mean_shortfall_population_low'])
        components.append(dict(mode=mode, time_min=20.0, mode_score=1-shortfall,
            mean_squared_shortfall=shortfall, sufficient_share=float(row['pct_population_sufficient_low'])/100,
            population_total=float(row['population_total']), n_origins=len(pop),
            n_eligible_origins=len(inhabited), n_attached=int(row['n_attached']),
            n_newly_attached=len((current-baseline)&inhabited), n_lost_attachment=len((baseline-current)&inhabited),
            mean_accessible_jobs_population=float(row['mean_accessible_jobs_population']), weight=1.0))
    fitness = sum(c['mode_score'] for c in components)/3
    if not math.isfinite(fitness) or not 0 <= fitness <= 1:
        raise ValueError('Invalid fitness')
    # Legacy report consumers expect this dictionary; it is diagnostic only.
    car = components[0]['sufficient_share']
    score = dict(fitness_version=FITNESS_VERSION, fitness=fitness, raw_fitness=fitness,
        weighted_shortfall=1-fitness, feasible=True, provisional=False,
        mode_weights=SETTINGS['mode_weights'], time_weights=SETTINGS['time_weights'],
        sufficiency_level='low', unattached_policy='all_statpop_origins_fixed',
        n_eligible_origins=int((pop>0).sum()), n_excluded_all_mode_unattached=0,
        population_total=float(pop.sum()), components=components,
        car_constraint=dict(enabled=False, scenario_share=car, baseline_share=mask['baseline_car_sufficient_share'],
                            minimum_share=0.0, violation_pp=0.0, feasible=True))
    return scored, summary, score


def evaluate_scenario(scenario_dir, spatial_dir, baseline_reference_file, *, score_only=False):
    scenario, spatial, reference = map(Path,(scenario_dir,spatial_dir,baseline_reference_file))
    if reference.name == 'sufficiency_reference_20min_park5.json':
        raise ValueError('Refusing to overwrite legacy baseline reference')
    status = read_json(scenario/'run_status.json')
    if status.get('state', status.get('status')) not in ('complete','completed'):
        raise ValueError(f'Simulation not completed: {scenario}')
    out = scenario/OUTPUT_DIR
    hexfile, summaryfile, fitnessfile = (out/x for x in ('accessibility_by_hexagon.csv.gz','sufficiency_summary.csv','fitness.json'))
    inputs = scenario_hashes(scenario)
    is_baseline = scenario.name == 'baseline_00000'
    if reference.exists() or mask_path(reference).exists():
        ref, mask = validate_reference(reference, spatial)
        if is_baseline and mask['baseline_scenario_sha256'] != inputs:
            raise ValueError('Baseline MATSim inputs changed; use a new reference')
    elif is_baseline and not score_only:
        ref = mask = None
    else:
        raise FileNotFoundError('Evaluate baseline first to establish Trial 08 reference')
    provenance = dict(settings=SETTINGS, scenario_sha256=inputs, spatial_sha256=spatial_hashes(spatial))
    if score_only:
        cached = read_json(fitnessfile)
        if (cached.get('fitness_version') != FITNESS_VERSION or cached.get('provenance') != provenance
            or cached.get('reference_sha256') != sha(reference)
            or cached.get('baseline_attachment_sha256') != sha(mask_path(reference))
            or cached.get('hexagon_sha256') != sha(hexfile)
            or cached.get('summary_sha256') != sha(summaryfile)):
            raise ValueError('Incompatible Trial 08 cache')
        data = _load_hexagons(hexfile)
    else:
        data = calculate_accessibility(scenario, spatial, thresholds_min=THRESHOLDS, **ROUTING)
    validate_rows(data, spatial)
    if is_baseline:
        candidate = SufficiencyReference.from_baseline_car(data.loc[data['mode']=='car'], THRESHOLDS,
                    car_parking_penalty_s=0, population_weighted=True, population_sha256=weight_sha256(spatial))
        if ref is not None and candidate != ref:
            if not math.isclose(candidate.mean_car_jobs[CumulativeAccessibility.column_name(20)], ref.mean_car_jobs[CumulativeAccessibility.column_name(20)], rel_tol=1e-12):
                raise ValueError('Baseline reference changed')
        if ref is None:
            ref = candidate
            tmp = reference.with_name(reference.name + ".tmp")
            ref.save(tmp)
            os.replace(tmp, reference)
            _, baseline_summary = assess_sufficiency(data, ref, THRESHOLDS)
            mask = dict(fitness_version=FITNESS_VERSION, settings=SETTINGS, reference_sha256=sha(reference),
                        spatial_sha256=provenance['spatial_sha256'], baseline_scenario_sha256=inputs,
                        attached_origins_by_mode={m: sorted(data.loc[(data['mode']==m)&data['attached'],'hex_id'].tolist()) for m in MODES},
                        baseline_car_sufficient_share=float(baseline_summary.loc[baseline_summary['mode']=='car','pct_population_sufficient_low'].iloc[0])/100)
            _write_json(mask_path(reference), mask)
    scored, summary, result = score_rows(data, ref, mask, spatial)
    if not score_only:
        _write_csv(hexfile, scored, compression='gzip')
        _write_csv(summaryfile, summary)
    result.update(scenario=scenario.name, provenance=provenance, reference_file=str(reference),
        reference_sha256=sha(reference), baseline_attachment_file=str(mask_path(reference)),
        baseline_attachment_sha256=sha(mask_path(reference)), hexagon_file=str(hexfile), summary_file=str(summaryfile),
        hexagon_sha256=sha(hexfile), summary_sha256=sha(summaryfile), accessibility_parameters=ROUTING,
        threshold_jobs=ref.threshold_jobs(CumulativeAccessibility.column_name(20), 'low'))
    _write_json(fitnessfile,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--spatial-dir',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--score-only',action='store_true')
    p.add_argument('--expected-fitness',type=float)
    a=p.parse_args()
    score=evaluate_scenario(a.scenario,a.spatial_dir,a.reference,score_only=a.score_only)
    print(json.dumps({k:score[k] for k in ('fitness','threshold_jobs','components')},indent=2))
    if a.expected_fitness is not None and abs(score['fitness']-a.expected_fitness)>1e-8:
        raise ValueError('Trial 08 benchmark mismatch')

if __name__=='__main__':
    main()
