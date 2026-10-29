#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack nightly outputs and compare them with a pinned, complete reference."""

import argparse
import ast
import copy
import hashlib
import json
import math
import re
import runpy
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

MODULES = Path(__file__).resolve().parents[1] / 'test'
sys.path.insert(0, str(MODULES))
NIGHTLY = runpy.run_path(str(Path(__file__).with_name('nightly_check.py')))
REFERENCE = {
    'version':
    '1.0.0',
    'wheel_sha256':
    '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'
}
LABELS = ('predictions', 'repeat-predictions', 'partial-predictions',
          'reloaded-predictions')
PLANS = ('requested-plan', 'selected-model-plan', 'resolved_model_plan',
         'generation-plan')
RAW_JSON = ('result', 'device', 'measurements', *PLANS, 'metadata',
            'runtime_manifest', 'feature_schema', 'trainer_config', 'encoders',
            'target_encoder')
RUNTIME = ('device', 'gpu', 'gpu_memory_bytes', 'cuda', 'torch', 'threads')
DELTA_FIELDS = ('repeat_max_abs_difference', 'partial_max_abs_difference',
                'reload_max_abs_difference')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def confined(root, name):
    if (root.is_symlink() or not isinstance(name, str) or not name
            or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name
            or any(part in ('.', '..') for part in name.split('/'))):
        raise ValueError('Evidence path must stay inside its directory')
    path = root
    for part in name.split('/'):
        path = path / part
        if path.is_symlink():
            raise ValueError('Evidence cannot contain symbolic links')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Evidence path escapes its directory')
    return path


def load_manifest(root, expected_sha256):
    manifest_path = confined(root, 'manifest.json')
    if (not re.fullmatch('[0-9a-f]{64}', expected_sha256)
            or digest(manifest_path) != expected_sha256):
        raise ValueError('The reference manifest fingerprint differs')
    manifest = read(manifest_path)
    if (manifest.get('schema') != 1 or manifest.get('reference') != REFERENCE
            or not isinstance(manifest.get('files'), dict)):
        raise ValueError('Use a pinned cache for the approved reference wheel')
    for name, expected in manifest['files'].items():
        path = confined(root, name)
        if (not re.fullmatch('[0-9a-f]{64}', expected) or not path.is_file()
                or digest(path) != expected):
            raise ValueError(f'Missing or changed reference file: {name}')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Reference evidence cannot contain symbolic links')
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest['files']):
        raise ValueError('The reference file roster differs from its manifest')
    return manifest


def key(row):
    return f'{row["task"]}-{row["seed"]}-{row["case"]}'


def require_roster(rows, *, reference_fit_ids=()):
    expected = set(NIGHTLY['case_roster'](NIGHTLY['TASKS'],
                                          ['normal', 'scrambled']))
    actual = [(row['task'], row['seed'], row['case']) for row in rows]
    if (len(actual) != 16 or set(actual) != expected or any(
            type(row['seed']) is not int
            or not isinstance(row.get('fit_id'), str) or not row['fit_id']
            for row in rows) or len({row['fit_id']
                                     for row in rows}) != 16):
        raise ValueError('Nightly requires all sixteen unique seed-zero cases')
    if {row['fit_id'] for row in rows}.intersection(reference_fit_ids):
        raise ValueError(
            'Candidate models must be fresh fits, separate from the reference')
    return actual


def validate_report(report):
    import wheel_gpu_learning as gpu

    expected = report.get('expected_runs')
    gpu.verify_prepared_report(report, expected_runs=expected)
    actual = [[row['task'], row['seed'], row['case']] for row in report['runs']]
    if (report.get('completed_runs') != actual or expected != actual
            or type(report.get('physical_fits')) is not int
            or report['physical_fits'] != len(actual)):
        raise ValueError('Completed cases or physical fit counts differ')


def verify_fit_ledger(directory, report):
    ledger = read(confined(directory, 'fits.json'))
    NIGHTLY['PR']['verify_fits'](ledger)
    if len(ledger) != report['physical_fits']:
        raise ValueError('The physical fit ledger differs')
    batches = read(confined(directory, 'cuda-batches.json'))
    if any(
            type(batches.get(stage)) is not int or batches[stage] <= 0
            for stage in ('training', 'prediction')):
        raise ValueError(
            'Completed training and prediction CUDA work is missing')


def dependency_changes(reference, candidate, declared):
    if not isinstance(reference, dict) or not reference or not isinstance(
            candidate, dict) or not candidate:
        raise ValueError('Both exact dependency profiles are required')
    actual = {
        name: dict(reference=reference.get(name), candidate=candidate.get(name))
        for name in reference.keys() | candidate.keys()
        if reference.get(name) != candidate.get(name)
    }
    if actual != declared:
        raise ValueError('Dependency changes are not declared exactly')
    return actual


def runtime_profile(report):
    runtime = {name: report['configuration'].get(name) for name in RUNTIME}
    if (any(value is None
            for value in runtime.values()) or runtime['device'] != 'cuda'
            or runtime['gpu'] != 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
            or type(runtime['gpu_memory_bytes']) is not int
            or runtime['gpu_memory_bytes'] <= 0
            or type(runtime['threads']) is not int or runtime['threads'] != 4):
        raise ValueError('The fixed GPU runtime profile is missing or differs')
    return runtime


def input_signature(row, directory, profile):
    from wheel_task_checks import fingerprint

    fields = ('task', 'seed', 'case', 'training_seed', 'task_type', 'metric',
              'data_sha256', 'stage_sha256', 'query', 'entity_ids', 'targets',
              'classes', 'baseline_predictions', 'control_eligible',
              'split_counts', 'entity_table', 'entity_time_column',
              'target_format', 'target_ids', 'catalog', 'excluded_ids',
              'baseline_ranked_ids', 'baseline_ranked_scores')
    return fingerprint(
        dict(row={name: row.get(name)
                  for name in fields}, profile=profile,
             plans={name: read(directory / f'{name}.json')
                    for name in PLANS}))


def quality_difference(candidate, reference):
    before, after = reference['score'], candidate['score']
    if (candidate['metric'] != reference['metric'] or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (before, after))):
        raise ValueError('Quality scores must use the same finite metric')
    difference = after - before
    return dict(
        candidate=after, reference=before, candidate_minus_reference=difference,
        quality_drop=difference
        if candidate['metric'] == 'mae' else -difference, policy='report_only')


def bundle_path(directory, row):
    parts = row['fit_id'].split('/')
    if len(parts) != 2 or any(not part or part in ('.', '..')
                              for part in parts):
        raise ValueError('The fitted model identity is malformed')
    return confined(
        directory,
        f'scratch/trainers/{parts[0]}/runs/{parts[1]}/published-model')


def local_artifact(value):
    from kumo_core.io.filesystem import local_path

    value = str(value)
    if urlsplit(value).scheme not in ('', 'file', 'local'):
        raise ValueError('Raw model evidence must use a local filesystem')
    return local_path(unquote(value) if value.startswith('file://') else value)


def read_prediction_inputs(root, row, artifact):
    import pandas as pd

    inputs = {}
    for label in LABELS:
        path = confined(
            root,
            artifact.get('prediction_inputs', {}).get(
                label,
                f'{artifact["directory"]}/prediction-inputs/{label}-inputs.parquet'
            ))
        frame = pd.read_parquet(path)
        required = ['ENTITY1'] if row['task'] == 'static_link' else [
            'ENTITY1', 'TIMESTAMP'
        ]
        allowed = ['ENTITY1', 'TIMESTAMP'] + (
            ['autoregression_labels'] if row['task'] == 'forecasting' else [])
        if row['task'] == 'forecasting':
            required = allowed
        ids = row['entity_ids'][:max(
            1,
            len(row['entity_ids']) //
            2)] if label == 'partial-predictions' else row['entity_ids']
        if (not set(required) <= set(frame.columns) <= set(allowed)
                or frame['ENTITY1'].tolist() != ids
                or frame['ENTITY1'].isna().any()
                or row['task'] != 'static_link' and
            (frame['TIMESTAMP'].isna().any()
             or frame['TIMESTAMP'].nunique() != 1)):
            raise ValueError(
                'Prediction inputs are incomplete or expose answers')
        if row['task'] == 'forecasting' and any(
                len(values) != 12 for values in frame['autoregression_labels']):
            raise ValueError('Forecast history must contain twelve values')
        inputs[label] = frame
    original = inputs['predictions'].set_index('ENTITY1')
    if any(not frame.equals(original.loc[frame['ENTITY1']].reset_index())
           for frame in inputs.values()):
        raise ValueError(
            'Prediction inputs changed history or timestamps across calls')
    anchor = None if row['task'] == 'static_link' else pd.Timestamp(
        original['TIMESTAMP'].iloc[0])
    return inputs, anchor


def validate_artifacts(root, row, artifact):
    import numpy as np
    import pandas as pd
    import wheel_gpu_learning as gpu
    import wheel_learning as binary
    import wheel_scale_learning as learning
    from wheel_task_checks import fingerprint

    from kumo.engine.trainer._spec import read_bundle_artifact_paths

    directory = confined(root, artifact['directory'])
    for name in RAW_JSON:
        confined(directory, f'{name}.json').read_bytes()
    if read(directory / 'result.json') != row:
        raise ValueError('Raw model result differs from its report')
    device = read(directory / 'device.json')
    required_device = {
        'device', 'observed_devices', 'consistency_measurements',
        'batch_retries_disabled', 'training_work', 'training_window',
        'expected_reload_checkpoint_sha256', 'reload_identity_checks'
    }
    if not required_device.issubset(device) or any(
            row.get(name) != value for name, value in device.items()):
        raise ValueError('Raw device evidence differs from its report')
    if read(directory / 'measurements.json') != row['operation_measurements']:
        raise ValueError('Raw operation measurements differ')
    selected = read(directory / 'selected-model-plan.json')
    if fingerprint(selected) != row['plan_sha256']:
        raise ValueError('Selected model plan fingerprint differs')
    binary.verify_plan(read(directory / 'requested-plan.json'), selected)
    learning.verify_runtime(row['task'],
                            read(directory / 'runtime_manifest.json'),
                            read(directory / 'feature_schema.json'),
                            read(directory / 'trainer_config.json'))
    bundle = confined(root, artifact['bundle'])
    for path in bundle.rglob('*'):
        confined(bundle, path.relative_to(bundle).as_posix())
    paths = read_bundle_artifact_paths(bundle)
    checkpoint = local_artifact(paths['selected_checkpoint'])
    if (digest(checkpoint) != row['expected_reload_checkpoint_sha256']
            or learning.selected_training_step(checkpoint) !=
            row['selected_checkpoint_global_step']):
        raise ValueError('Published checkpoint differs from the measured model')
    gpu.verify_reload_identity(row)
    inputs, anchor = read_prediction_inputs(root, row, artifact)
    truth = dict(entity_ids=row['entity_ids'], classes=row['classes'],
                 targets=row.get('targets'), anchor=anchor)
    if row['task'] == 'forecasting':
        truth['horizons'] = [
            anchor + pd.Timedelta(days=7 * i) for i in range(12)
        ]
    if row.get('target_format') == 'ranked_link_v1':
        validate_ranked_files(root, row, artifact, truth)
    else:
        if row['task'].endswith('link'):
            catalog = read(confined(root, artifact['catalog']))
            truth.update(
                catalog=catalog, known={
                    str(entity): [
                        item for item, eligible in zip(catalog, mask)
                        if not eligible
                    ]
                    for entity, mask in zip(row['entity_ids'],
                                            row.get('control_eligible', []))
                })
        matrices = {}
        for label in LABELS:
            ids = inputs[label]['ENTITY1'].tolist()
            selected_truth = learning.subset_truth(truth, ids, {
                entity: index
                for index, entity in enumerate(row['entity_ids'])
            })
            frame = pd.read_parquet(confined(directory, label + '.parquet'))
            frame = learning.normalize_prediction_time(
                row['task'], frame, selected_truth, row['entity_time_column'],
                require_anchor=True)
            matrices[label] = learning.prediction_matrix(
                row['task'], frame, selected_truth)
        if not np.array_equal(matrices['predictions'], row['predictions']):
            raise ValueError('Raw predictions differ from the saved matrix')
        for label, field in zip(LABELS[1:], DELTA_FIELDS):
            measured = gpu.measure_consistency(
                matrices['predictions'][:len(matrices[label])], matrices[label])
            if measured['max_abs_difference'] != row[field]:
                raise ValueError('Raw prediction agreement measurement differs')
        if not row['task'].endswith('link'):
            holdout = pd.read_parquet(confined(directory, 'holdout.parquet'))
            holdout = learning.normalize_prediction_time(
                row['task'], holdout, truth, row['entity_time_column'])
            learning.verify_holdout(row['task'], holdout, truth)
    return directory


def validate_ranked_files(root, row, artifact, truth):
    import numpy as np
    import pandas as pd
    import wheel_scale_learning as learning
    from wheel_measurements import measure_recommendations

    truth.update({name: row[name] for name in ('catalog', 'target_ids')})
    truth.update(metric=row['metric'], target_format='ranked_link_v1')
    if 'excluded_ids' in row:
        truth['excluded_ids'] = row['excluded_ids']
    calls = row.get('ranked_prediction_calls', [])
    observed = {}
    for call in calls:
        label = call['label']
        primary = label.removesuffix('-fallback')
        if primary not in LABELS or label in observed:
            raise ValueError('Unknown or repeated ranked prediction call')
        ids = row['entity_ids'][:max(
            1,
            len(row['entity_ids']) //
            2)] if primary == 'partial-predictions' else row['entity_ids']
        if label.endswith('-fallback'):
            ids = observed[primary]['fallback_ids']
        selected = learning.subset_truth(
            truth, ids,
            {entity: index
             for index, entity in enumerate(row['entity_ids'])})
        files = [
            confined(root, artifact['ranked_files'][name])
            for name in call['files']
        ]
        with tempfile.TemporaryDirectory(prefix='nightly-ranked-') as temporary:
            batches = (learning.normalize_prediction_time(
                row['task'], frame, selected, row['entity_time_column'],
                require_anchor=True) for frame in learning.ordered_batches(
                    [str(path) for path in files], Path(temporary)))
            values = learning.ranked_from_batches(batches, selected,
                                                  call['raw_limit'])
        if values['raw_rows'] != call['raw_rows'] or values[
                'fallback_ids'] != call['fallback_ids'] or len(
                    ids) != call['entities']:
            raise ValueError('Ranked output work differs from its raw receipt')
        observed[label] = values
        if label.endswith('-fallback'):
            positions = {
                entity: index
                for index, entity in enumerate(
                    row['entity_ids'][:len(observed[primary]['ranked_ids'])])
            }
            for index, entity in enumerate(ids):
                for field in ('ranked_ids', 'ranked_scores'):
                    observed[primary][field][
                        positions[entity]] = values[field][index]
    if not set(LABELS).issubset(observed) or any(
            values['fallback_ids'] and label + '-fallback' not in observed
            for label, values in observed.items() if label in LABELS):
        raise ValueError(
            'Ranked prediction output or required fallback is missing')
    if any(observed['predictions'][field] != row[field]
           for field in ('ranked_ids', 'ranked_scores')):
        raise ValueError('Ranked raw predictions differ from the saved result')
    for index, (label, field) in enumerate(zip(LABELS[1:], DELTA_FIELDS)):
        count = len(observed[label]['ranked_ids'])

        def frame(values):
            lengths = [len(items) for items in values['ranked_ids'][:count]]
            return pd.DataFrame(
                dict(
                    ENTITY=np.repeat(row['entity_ids'][:count], lengths),
                    CLASS=[
                        item for items in values['ranked_ids'][:count]
                        for item in items
                    ], SCORE=[
                        score for scores in values['ranked_scores'][:count]
                        for score in scores
                    ]))

        measured = measure_recommendations(frame(observed['predictions']),
                                           frame(observed[label]))
        if measured['max_abs_difference'] != row[field] or any(
                row['consistency_measurements'][index].get(name) != value
                for name, value in measured.items()):
            raise ValueError(
                'Ranked raw prediction agreement measurement differs')


def verify_legacy_summary(state, row, result_sha256, descriptor_sha256):
    selected = [
        item for item in state.get('results', [])
        if item.get('fit_id') == row['fit_id']
    ]
    if (state.get('status') != 'PART_MEASURED' or state.get('trial') != 'R1'
            or state.get('task') != row['task']
            or state.get('condition') != row['case'] or len(selected) != 1
            or type(selected[0].get('seed')) is not int
            or any(selected[0].get(name) != row[name]
                   for name in ('seed', 'fit_id', 'score', 'baseline_score'))
            or selected[0].get('result_sha256') != result_sha256
            or selected[0].get('frozen_control', {}).get('descriptor_sha256') !=
            descriptor_sha256):
        raise ValueError(
            'The original study summary does not bind the selected raw model')


def legacy_report(root, descriptor):
    import wheel_gpu_learning as gpu
    from wheel_task_checks import fingerprint

    row = read(confined(root, descriptor['path']))
    legacy = descriptor['legacy']
    compatibility = read(confined(root, legacy['compatibility']))
    models = [
        model for model in compatibility.get('models', [])
        if model.get('task') == row.get('task')
    ]
    if (compatibility.get('status') !=
            'PASS_SELECTED_SOURCE_AND_PLAN_COMPATIBILITY'
            or row.get('task') not in ('binary', 'multiclass')
            or row.get('case') != 'learnable'
            or type(row.get('seed')) is not int or row['seed'] != 0
            or len(models) != 1 or legacy.get('validator_source_sha256') !=
            gpu.source_identity(prepared=True)):
        raise ValueError('The legacy source bridge is missing or differs')
    model = models[0]
    state_path, packet_path = (confined(root, legacy[name])
                               for name in ('producer_state',
                                            'producer_packet'))
    state, packet = read(state_path), read(packet_path)
    dependencies = read(confined(root, legacy['dependencies']))
    verify_legacy_summary(state, row, digest(confined(root,
                                                      descriptor['path'])),
                          model['frozen_normal_descriptor_sha256'])
    if (digest(confined(
            root, descriptor['path'])) != model['selected_result']['sha256']
            or digest(state_path) != model['source_state']['sha256']
            or digest(packet_path) !=
            model['actual_launch_worker_packet']['sha256']
            or state.get('environment') != model['environment']
            or {name: state['environment'].get(name)
                for name in REFERENCE} != REFERENCE
            or state.get('source_packet_sha256') != digest(packet_path)
            or state['environment']['dependencies_sha256'] !=
            fingerprint(dependencies)):
        raise ValueError(
            'Legacy result, producer, runtime or dependency identity differs')
    for name, expected in packet['files'].items():
        if digest(confined(packet_path.parent, name)) != expected:
            raise ValueError('A frozen legacy source file changed')
    for compared in model['helper_comparison']:
        name = compared['file']
        if digest(confined(packet_path.parent,
                           'helpers/' + name)) != compared['source']['sha256']:
            raise ValueError(
                'The compatibility receipt names different producer sources')
        if digest(MODULES / name) == compared['current']['sha256']:
            continue
        if name != 'wheel_gpu_learning.py':
            raise ValueError('A legacy learning or metric helper changed')
        before = ast.parse(
            confined(packet_path.parent, 'helpers/' + name).read_text())
        after = ast.parse((MODULES / name).read_text())
        for module in (before, after):
            module.body = [
                node for node in module.body
                if not isinstance(node, ast.FunctionDef)
                or node.name != 'verify_prepared_report'
            ]
        if ast.dump(before) != ast.dump(after):
            raise ValueError(
                'The legacy GPU helper changed outside the reviewed verifier')
    configuration = dict(legacy['configuration'], device='cuda',
                         seed_profile='nightly', ranked_links=False)
    if any(
            configuration.get(name) != state['environment'][name]
            for name in ('gpu', 'cuda', 'torch')):
        raise ValueError('The legacy hardware/runtime bridge differs')
    expected = [[row['task'], 0, 'learnable']]
    report = dict(status='MEASURED', group='all', seeds=[0], runs=[row],
                  expected_runs=expected, completed_runs=expected,
                  physical_fits=1, configuration=configuration,
                  gpu_source_sha256=gpu.source_identity(prepared=True),
                  **REFERENCE, dependencies=dependencies,
                  profiles={row['task']: model['source_profile']})
    diagnostic = dict(
        kind='validation_only', original_status=state['status'],
        raw_result_sha256=digest(confined(root, descriptor['path'])),
        original_packet_sha256=digest(packet_path),
        compatibility_sha256=digest(confined(root, legacy['compatibility'])),
        validator_source_sha256=report['gpu_source_sha256'], new_fits=0)
    source_hashes = {
        item['file']: item['source']['sha256']
        for item in model['helper_comparison']
    }
    diagnostic['producer_gpu_source_sha256_from_frozen_files'] = fingerprint(
        source_hashes)
    report['envelope_kind'] = 'legacy_validation_only'
    return report, diagnostic


def collect_reports(root, descriptors, *, legacy_allowed=False):
    rows, records, runtimes, dependencies, wheels, bridges = [], {}, [], [], [], []
    for descriptor in descriptors:
        if 'legacy' in descriptor:
            if not legacy_allowed:
                raise ValueError(
                    'A candidate cannot use a legacy source bridge')
            report, diagnostic = legacy_report(root, descriptor)
            bridges.append(diagnostic)
        else:
            path = confined(root, descriptor['path'])
            report = read(path)
            verify_fit_ledger(path.parent, report)
        validate_report(report)
        runtime = runtime_profile(report)
        runtimes.append(runtime)
        dependencies.append(report.get('dependencies'))
        wheels.append(
            {name: report.get(name)
             for name in ('version', 'wheel_sha256')})
        for row in report['runs']:
            name = key(row)
            if name in records:
                raise ValueError('A nightly task/condition is repeated')
            artifact = descriptor['artifacts'][name]
            directory = validate_artifacts(root, row, artifact)
            records[name] = dict(
                row=row,
                signature=input_signature(row, directory,
                                          report['profiles'][row['task']]))
            rows.append(row)
    require_roster(rows)
    if any(value != runtimes[0] for value in runtimes) or any(
            value != dependencies[0]
            for value in dependencies) or any(value != wheels[0]
                                              for value in wheels):
        raise ValueError(
            'Shards do not use one wheel, runtime and dependency profile')
    if (not isinstance(wheels[0]['version'], str) or
            not re.fullmatch('[0-9a-f]{64}', wheels[0]['wheel_sha256'] or '')):
        raise ValueError('The wheel identity is missing')
    return dict(records=records, runtime=runtimes[0],
                dependencies=dependencies[0], wheel=wheels[0], bridges=bridges)


def compare(candidate_paths, reference_root, reference_sha256, output):
    manifest = load_manifest(reference_root, reference_sha256)
    reference = collect_reports(reference_root, manifest['reports'],
                                legacy_allowed=True)
    if reference['wheel'] != REFERENCE:
        raise ValueError(
            'Stored reports do not use the approved reference wheel')
    # Candidate reports may come from separate task shards. Pack them without copying native caches.
    candidate = collect_candidate_reports(candidate_paths)
    require_roster([record['row'] for record in candidate['records'].values()],
                   reference_fit_ids=[
                       record['row']['fit_id']
                       for record in reference['records'].values()
                   ])
    if candidate['runtime'] != reference['runtime']:
        raise ValueError(
            'Candidate and reference hardware/runtime profiles differ')
    changes = dependency_changes(
        reference['dependencies'], candidate['dependencies'],
        manifest.get('declared_dependency_changes', {}))
    comparisons = []
    for name, record in candidate['records'].items():
        baseline = reference['records'][name]
        if record['signature'] != baseline['signature']:
            raise ValueError(
                f'Frozen data, population, query or model recipe differs: {name}'
            )
        comparisons.append(
            dict(case=name, metric=record['row']['metric'],
                 **quality_difference(record['row'], baseline['row'])))
    result = dict(status='MEASURED', cases=16, quality_policy='report_only',
                  release_approved=False,
                  reference_manifest_sha256=reference_sha256,
                  reference=reference['wheel'], candidate=candidate['wheel'],
                  runtime=reference['runtime'],
                  declared_dependency_changes=changes,
                  legacy_bridges=reference['bridges'], comparisons=comparisons)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'comparison.json', result)
    return result


def collect_candidate_reports(paths):
    # A common filesystem root permits bounded relative artifact maps across shards.
    import os

    root = Path(
        os.path.commonpath([str(path.parent.resolve()) for path in paths]))
    descriptors = []
    for path in paths:
        packed = path.parent / 'pack.json'
        if packed.is_file():
            packet = read(packed)
            if packet.get('status') != 'PACKED':
                raise ValueError(
                    'Incomplete candidate evidence cannot be compared')
            for name, expected in packet['files'].items():
                if digest(confined(path.parent, name)) != expected:
                    raise ValueError('A packed candidate artifact changed')
            if len(packet['reports']
                   ) != 1 or packet['reports'][0]['path'] != path.name:
                raise ValueError('The packed report descriptor differs')
            descriptor = copy.deepcopy(packet['reports'][0])
            prefix = path.parent.relative_to(root)
            descriptor['path'] = (prefix / descriptor['path']).as_posix()
            for artifact in descriptor['artifacts'].values():
                for field in ('directory', 'bundle', 'catalog'):
                    if field in artifact:
                        artifact[field] = (prefix / artifact[field]).as_posix()
                for field in ('prediction_inputs', 'ranked_files'):
                    artifact[field] = {
                        name: (prefix / value).as_posix()
                        for name, value in artifact.get(field, {}).items()
                    }
            descriptors.append(descriptor)
            continue
        report = read(path)
        artifacts = {}
        for row in report.get('runs', []):
            directory = path.parent / key(row)
            artifacts[key(row)] = dict(
                directory=directory.relative_to(root).as_posix(),
                bundle=bundle_path(directory, row).relative_to(root).as_posix(),
                ranked_files={
                    name: local_artifact(name).relative_to(root).as_posix()
                    for call in row.get('ranked_prediction_calls', [])
                    for name in call['files']
                })
        descriptors.append(
            dict(path=path.relative_to(root).as_posix(), artifacts=artifacts))
    return collect_reports(root, descriptors)


def pack(report_path, output, *, allow_incomplete=False):
    report = read(report_path)
    errors = []
    try:
        validate_report(report)
        verify_fit_ledger(report_path.parent, report)
    except (ValueError, OSError, KeyError) as error:
        if not allow_incomplete:
            raise
        errors.append(str(error))
    output.mkdir(parents=True, exist_ok=False)
    for path in report_path.parent.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log'):
            confined(report_path.parent, path.name)
            shutil.copyfile(path, output / path.name)
    if report_path.name != 'nightly-result.json':
        shutil.copyfile(report_path, output / 'nightly-result.json')
    artifacts = {}
    for row in report['runs']:
        name = key(row)
        source, target = report_path.parent / name, output / name
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.json', '.parquet'):
                confined(source, path.name)
                shutil.copyfile(path, target / path.name)
        for folder in ('prediction-inputs', 'prediction-outputs'):
            directory = confined(source, folder)
            if directory.exists():
                for path in directory.rglob('*'):
                    confined(directory, path.relative_to(directory).as_posix())
                shutil.copytree(directory, target / folder)
        bundle = bundle_path(source, row)
        packed_bundle = target / 'published-model'
        for path in bundle.rglob('*'):
            confined(bundle, path.relative_to(bundle).as_posix())
        shutil.copytree(bundle, packed_bundle)
        mapping = {}
        for call in row.get('ranked_prediction_calls', []):
            for filename in call['files']:
                path = local_artifact(filename)
                if not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError(
                        'Ranked raw files must belong to this model workflow')
                confined(source, path.relative_to(source).as_posix())
                if filename in mapping:
                    continue
                destination = target / 'ranked-raw' / str(
                    len(mapping)) / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                mapping[filename] = destination.relative_to(output).as_posix()
        artifact = dict(directory=name, bundle=f'{name}/published-model',
                        ranked_files=mapping)
        try:
            if row['task'].endswith(
                    'link') and row.get('target_format') != 'ranked_link_v1':
                raise ValueError(
                    'Dense link packing needs an independently frozen catalogue; use the admitted ranked profile'
                )
            validate_artifacts(output, row, artifact)
        except (ValueError, OSError, KeyError) as error:
            if not allow_incomplete:
                raise
            errors.append(f'{name}: {error}')
        artifacts[name] = artifact
    missing = []
    for task, seed, arm in report.get('expected_runs', []):
        if task not in NIGHTLY['TASKS'] or type(
                seed) is not int or seed != 0 or arm not in ('learnable',
                                                             'independent'):
            raise ValueError('The pending case roster differs')
        name = f'{task}-{seed}-{arm}'
        if name in artifacts:
            continue
        missing.append(name)
        source = confined(report_path.parent, name)
        if source.is_dir():
            for path in source.iterdir():
                if path.is_file() and path.suffix in ('.json', '.parquet'):
                    confined(source, path.name)
                    destination = output / name / path.name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(path, destination)
    descriptor = dict(path='nightly-result.json', artifacts=artifacts)
    files = {
        path.relative_to(output).as_posix(): digest(path)
        for path in output.rglob('*') if path.is_file()
    }
    result = dict(schema=1,
                  status='INCOMPLETE' if errors or missing else 'PACKED',
                  errors=errors, missing_cases=missing,
                  current_case=report.get('stage'), reports=[descriptor],
                  files=files)
    save(output / 'pack.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack')
    packing.add_argument('--report', type=Path, required=True)
    packing.add_argument('--output', type=Path, required=True)
    packing.add_argument('--allow-incomplete', action='store_true')
    comparing = commands.add_parser('compare')
    comparing.add_argument('--candidate', type=Path, nargs='+', required=True)
    comparing.add_argument('--reference', type=Path, required=True)
    comparing.add_argument('--reference-sha256', required=True)
    comparing.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.report.resolve(strict=True), args.output.resolve(),
             allow_incomplete=args.allow_incomplete)
    else:
        compare([path.resolve(strict=True) for path in args.candidate],
                args.reference.resolve(strict=True), args.reference_sha256,
                args.output.resolve())


if __name__ == '__main__':
    main()
