#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import approved saved reference models and predictions without model work."""

import argparse
import importlib.util
import shutil
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    'weekly',
    Path(__file__).with_name('weekly_check.py'))
weekly = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(weekly)
CHECKS = ('fit', 'prediction', 'repeat_prediction', 'partial_prediction',
          'reload', 'hidden_answers', 'chronology')


def original_bundle(root, expected):
    path = weekly.data_file(root, 'manifest.json', {'manifest.json': expected})
    value = weekly.read(path)
    if (value.get('schema') != 1 or value.get('mode') != 'reference'
            or value.get('status') != 'MEASURED'
            or value.get('physical_fits') != 11
            or len(value.get('cases', [])) != 11
            or set(value['cases']) != weekly.ROSTER):
        raise ValueError(
            'Import needs a complete approved eleven-case reference')
    files = weekly.inventory(root)
    files.pop('manifest.json')
    if files != value['files']:
        raise ValueError(
            'Legacy reference contains missing, changed, or unlisted evidence')
    return value


def coordinates(case, row):
    ids = row['entity_ids']
    if case['metric'].startswith('recall@'):
        value = dict(entity_ids=ids, ranked_ids=row['ranked_ids'],
                     values=row['ranked_scores'])
    else:
        values = row['predictions']
        values = [[v[0]] if isinstance(v, list) and len(v) == 1 else [v]
                  for v in values]
        value = dict(
            entity_ids=ids, values=values,
            decisions=[v[0] > .5 for v in values]
            if case['metric'] == 'auroc' else [None] * len(ids))
    weekly.compare_predictions(value, value, **case['tolerance'])
    if 'ranked_ids' in value:
        if (len(value['ranked_ids']) != len(ids)
                or any(not items or len(items) != len(scores)
                       or len(set(items)) != len(items)
                       for items, scores in zip(value['ranked_ids'],
                                                value['values'], strict=True))):
            raise ValueError(
                'Legacy reference ranking coordinates are incomplete')
    return value


def import_reference(legacy_root, legacy_sha256, manifest, manifest_sha256,
                     output, *, retained_p2=False):
    legacy = original_bundle(legacy_root, legacy_sha256)
    if dict(version=legacy['version'],
            wheel_sha256=legacy['wheel_sha256']) != manifest['reference']:
        raise ValueError(
            'Legacy reference wheel differs from the approved manifest')
    roster = {'P2-fixed'} if retained_p2 else weekly.ROSTER
    if len(manifest['cases']) != len(roster) or {
            c['id']
            for c in manifest['cases']
    } != roster:
        raise ValueError(
            'Import cannot shrink or duplicate the selected roster')
    if output.exists():
        raise ValueError('Use a fresh imported-reference directory')
    specs = {r['id']: r for r in legacy['case_specs']}
    if len(legacy['case_specs']) != 11 or set(specs) != weekly.ROSTER:
        raise ValueError('Legacy reference has mismatched case specifications')
    report = dict(
        format='kumo-engine-weekly-p2-reference-v1'
        if retained_p2 else 'kumo-engine-weekly-reference-v2', status='PASS',
        wheel=manifest['reference'], manifest_sha256=manifest_sha256,
        legacy_reference_sha256=legacy_sha256, new_physical_fits=0, cases={})
    fit_ids = set()
    for case in manifest['cases']:
        name = case['id']
        source = legacy_root / 'cases' / name
        original = weekly.read(source / 'weekly-case.json')
        row = weekly.read(source / 'result.json')
        spec = specs[name]
        arm = 'scrambled_answer_control' if case[
            'arm'] == 'control' else 'fixed_scale_workload' if case[
                'arm'] == 'fixed' else 'normal'
        if (original.get('schema') != 1 or original.get('status') != 'MEASURED'
                or original.get('mode') != 'reference'
                or original.get('case_id') != name
                or original.get('case_spec') != spec
                or original.get('result') != row
                or original.get('identity') != case.get('legacy_identity')
                or original['identity'] != spec['identity']
                or row.get('identity') != spec['identity']
                or original.get('physical_fits') != 1
                or row.get('status') != 'MEASURED' or row.get('seed') != 0
                or row.get('training_seed') != 0 or spec.get('seed') != 0
                or spec.get('condition') != arm
                or (spec.get('dataset'),
                    spec.get('metric')) != (case['dataset'], case['metric'])
                or row.get('metric') != case['metric']
                or row.get('query') != case['query']
                or row.get('prediction_query', row.get('query')) != case.get(
                    'prediction_query', case['query'])
                or any(row.get('checks', {}).get(k) != 'PASS' for k in CHECKS)
                or not row.get('fit_id') or row['fit_id'] in fit_ids):
            raise ValueError(
                'Legacy case input, fit, query, or completion differs')
        fit_ids.add(row['fit_id'])
        cuda = weekly.read(source / 'cuda-batches.json')
        if any(
                type(cuda.get(k)) is not int or cuda[k] <= 0
                for k in ('training', 'prediction')):
            raise ValueError('Legacy reference has no completed CUDA work')
        bundles = [
            path.parent
            for path in source.glob('*/saved-model/training-spec.json')
            if path.is_file()
        ]
        if len(bundles) != 1:
            raise ValueError(
                'Legacy case needs exactly one saved public model bundle')
        [saved] = bundles
        checkpoint = saved / 'model/checkpoint'
        checkpoint_sha256 = weekly.digest(checkpoint)
        if checkpoint_sha256 != row.get('expected_reload_checkpoint_sha256'):
            raise ValueError(
                'Saved reference checkpoint differs from the restored model')
        if not (saved / 'training-spec.json').is_file():
            raise ValueError(
                'Legacy reference lacks a loadable public model bundle')
        destination = output / 'cases' / name
        destination.mkdir(parents=True)
        shutil.copytree(saved, destination / 'model')
        if weekly.inventory(saved) != weekly.inventory(destination / 'model'):
            raise ValueError('Imported reference model bytes changed')
        weekly.save(destination / 'predictions.json', coordinates(case, row))
        weekly.save(destination / 'source-case.json', original)
        record = dict(
            status='PASS', physical_fits=1, fit_id=row['fit_id'],
            score=row['score'], checkpoint_sha256=checkpoint_sha256,
            source_model_path=str(saved.relative_to(source)),
            legacy_identity=spec['identity'],
            source_case_sha256=weekly.digest(source / 'weekly-case.json'))
        record['files'] = weekly.inventory(destination)
        report['cases'][name] = record
    scores = {name: row['score'] for name, row in report['cases'].items()}
    report['learning'] = (dict(status='NOT_APPLICABLE',
                               reason='Retained P2 has no quality answers')
                          if retained_p2 else weekly.check_learning(
                              scores, scores))
    if not retained_p2 and report['learning']['status'] != 'PASS':
        raise ValueError(
            'Legacy reference does not establish the required learning separation'
        )
    # Check the immutable source again after all copies. Historical code is never imported.
    original_bundle(legacy_root, legacy_sha256)
    weekly.save(output / 'reference.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-root', 'input-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('legacy-sha256', 'manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--retained-p2', action='store_true',
                        help='Import only the retained P2 model for B9')
    args = parser.parse_args()
    manifest = weekly.load_manifest(args.input_root, args.manifest_sha256,
                                    retained_p2=args.retained_p2)
    import_reference(args.legacy_root, args.legacy_sha256, manifest,
                     args.manifest_sha256, args.output,
                     retained_p2=args.retained_p2)


if __name__ == '__main__':
    main()
