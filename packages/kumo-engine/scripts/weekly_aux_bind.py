#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a pinned auxiliary template to the selected runtime and retained P2."""

import copy
import runpy
import shutil
from pathlib import Path

WEEKLY = runpy.run_path(str(Path(__file__).with_name('weekly_check.py')))
read, save, digest, pinned = [
    WEEKLY[name] for name in ('read', 'save', 'digest', 'pinned')
]
RUNTIME_FIELDS = {
    'mode', 'expected_prefix', 'wheel', 'dependency_profile', 'host_lock'
}


def admit_bindings(bindings, host_lock, p2):
    if set(bindings) != {'P1', 'B9'}:
        raise ValueError('Bind both auxiliary checks before model work')
    for check, binding in bindings.items():
        if set(binding) != {'runner', 'template'}:
            raise ValueError('Bind exact auxiliary runners and templates')
        runner = pinned(binding['runner']['path'], binding['runner']['sha256'])
        spec = read(
            pinned(binding['template']['path'], binding['template']['sha256']))
        if (spec.get('schema') != 1 or spec.get('check') != check
                or spec.get('host_lock') != host_lock
                or spec.get('source_sha256') != digest(runner)):
            raise ValueError(
                'Auxiliary template source or host binding differs')
        for path, sha in spec['inputs'].items():
            pinned(path, sha)
        capacity = read(
            pinned(spec['capacity']['path'], spec['capacity']['sha256']))
        if (capacity.get('status') != 'ADMITTED'
                or capacity.get('check') != check
                or capacity.get('native_capacity_checked') is not True):
            raise ValueError('Auxiliary native capacity is not admitted')
        if check == 'B9':
            auxiliary = runpy.run_path(str(runner))
            auxiliary['verify_b9_capacity'](dict(spec,
                                                 population=p2['population']),
                                            capacity, p2)


def render(template, selected, retained=None):
    if (set(selected) != RUNTIME_FIELDS
            or set(selected['wheel']) != {'path', 'sha256', 'version'}
            or set(selected['dependency_profile']) != {'path', 'sha256'}
            or selected['host_lock'] != template['host_lock']
            or selected['dependency_profile']['sha256']
            != template['dependency_profile']['sha256']
            or selected['mode'] not in ('reference', 'candidate')
            or selected['mode'] == 'reference'
            and selected['wheel']['version'] != '1.0.0'):
        raise ValueError(
            'Auxiliary runtime fields differ from the admitted environment')
    spec = copy.deepcopy(template)
    spec.update(copy.deepcopy(selected))
    if spec['check'] == 'P1':
        if retained is not None:
            raise ValueError('P1 cannot select a retained model')
        return spec
    if (spec['check'] != 'B9' or retained is None
            or retained['case_spec']['workload'] != 'P2'
            or retained['mode'] != selected['mode']
            or retained['version'] != selected['wheel']['version']
            or retained['wheel_sha256'] != selected['wheel']['sha256']):
        raise ValueError('B9 requires the actual selected P2 model evidence')
    if spec.get('prediction_batch_size') != retained['prediction_batch_size']:
        raise ValueError(
            'B9 batch size differs from the saved native scorer configuration')
    worker = retained['case_spec']['worker']
    spec.update(
        retained_case=dict(path=retained['path'], sha256=retained['sha256']),
        retained_fit_id=retained['result']['fit_id'],
        checkpoint_sha256=retained['result']
        ['expected_reload_checkpoint_sha256'],
        population=copy.deepcopy(retained['case_spec']['population']),
        gpu_name=retained['device']['gpu'],
        model_bundle=retained['model_bundle'],
        model_files=copy.deepcopy(retained['model_files']))
    spec['worker']['model_helper'] = dict(path=worker['path'],
                                          sha256=worker['sha256'])
    for path, sha in ((retained['path'], retained['sha256']),
                      (worker['path'], worker['sha256'])):
        if path in spec['inputs'] and spec['inputs'][path] != sha:
            raise ValueError(
                'Actual P2 evidence would overwrite a frozen input')
        spec['inputs'][path] = sha
    return spec


def prediction_batch_size(bundle, files):
    config = read(
        pinned(bundle / 'model/config.json', files.get('model/config.json')))
    evaluation = config.get('eval_loader_kwargs') or {}
    if not isinstance(evaluation, dict):
        raise ValueError(
            'Saved native evaluation loader configuration is invalid')
    batch = evaluation.get('batch_size', (config.get('loader_kwargs')
                                          or {}).get('batch_size'))
    if type(batch) is not int or batch <= 0:
        raise ValueError('Saved native prediction batch size is invalid')
    return batch


def retained_evidence(path, mode):
    report = read(path)
    primary = report['case_spec']
    original = read(path.parent / 'input-manifest.json')
    WEEKLY['load_case'](path, primary, mode, original)
    [saved_files] = [
        path.parent / name for name in report['files']
        if Path(name).name == 'saved-model-files.json'
    ]
    bundle = saved_files.parent / 'saved-model'
    files = read(saved_files)
    if (WEEKLY['inventory'](bundle) != files
            or digest(bundle / 'model/checkpoint')
            != report['result']['expected_reload_checkpoint_sha256']):
        raise ValueError(
            'Retained P2 model bytes differ from the completed fit')
    return dict(report, path=str(path.resolve()), sha256=digest(path),
                model_bundle=str(bundle.resolve()), model_files=files,
                prediction_batch_size=prediction_batch_size(bundle, files))


def bind(binding, selected, output, retained_case=None):
    if set(binding) != {'runner', 'template'}:
        raise ValueError('Bind an exact auxiliary runner and template')
    runner = pinned(binding['runner']['path'], binding['runner']['sha256'])
    source = pinned(binding['template']['path'], binding['template']['sha256'])
    template = read(source)
    if template['source_sha256'] != digest(runner):
        raise ValueError('The auxiliary template selects another runner source')
    for name in ('wheel', 'dependency_profile'):
        pinned(selected[name]['path'], selected[name]['sha256'])
    retained = retained_evidence(retained_case,
                                 selected['mode']) if retained_case else None
    spec = render(template, selected, retained)
    auxiliary = runpy.run_path(str(runner))
    if spec['check'] == 'B9':
        auxiliary['verify_b9_capacity'](spec, read(spec['capacity']['path']),
                                        retained['case_spec'])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'input-template.json')
    save(output / 'binding.json', binding)
    manifest = output / 'manifest.json'
    save(manifest, spec)
    sha = digest(manifest)
    auxiliary['load_spec'](manifest, sha)
    save(
        output / 'binding-receipt.json',
        dict(template_sha256=binding['template']['sha256'],
             runner_sha256=binding['runner']['sha256'], manifest_sha256=sha,
             selected_runtime=selected,
             retained_case_sha256=retained['sha256'] if retained else None))
    return dict(runner=str(runner), manifest=str(manifest), manifest_sha256=sha)
