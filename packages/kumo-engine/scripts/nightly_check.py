#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run frozen 100M nightly cases. Input artifacts contain data, never code."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
from pathlib import Path, PurePosixPath
from unittest.mock import patch

HELPER = Path(__file__).with_name('wheel_gpu_smoke.py')
if not HELPER.is_file():
    HELPER = Path(__file__).resolve().parents[1] / 'test/wheel_gpu_smoke.py'
_SPEC = importlib.util.spec_from_file_location('nightly_smoke', HELPER)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)
STAGES = ('train', 'val', 'test')
PREDICTIONS = ('predict', 'repeat', 'partial', 'reload')
KEY_FAILURES = ('key_parse_failures', 'key_lookup_failures',
                'key_membership_failures', 'pk_duplicate_total')
REFERENCE_WHEEL = '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'
RESOURCE_FIELDS = ('elapsed_seconds', 'cpu_self_seconds',
                   'cpu_tree_pss_sampled_peak_bytes',
                   'cpu_tree_rss_sampled_peak_bytes',
                   'cuda_device_used_sampled_peak_bytes',
                   'torch_allocated_sampled_peak_bytes',
                   'torch_reserved_sampled_peak_bytes')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True,
                   allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def start_measurement(scope):
    """Reuse the original nightly process-tree and CUDA sampling protocol."""
    import psutil
    import torch

    started, cpu = time.monotonic(), time.process_time()
    result = dict(format='kumo-nightly-resources-v1', scope=scope,
                  sample_interval_seconds=.2, samples=0, cpu_read_errors=0,
                  cuda_read_errors=0)
    fields = RESOURCE_FIELDS[2:]
    result.update({key: None for key in RESOURCE_FIELDS})
    process, stopped = psutil.Process(), threading.Event()

    def sample():
        result['samples'] += 1
        rss, pss = 0, 0
        try:
            for child in [process, *process.children(recursive=True)]:
                try:
                    memory = child.memory_full_info()
                    rss += memory.rss
                    pss += memory.pss
                except psutil.NoSuchProcess:
                    continue
            for key, value in zip(fields[:2], (pss, rss), strict=True):
                result[key] = max(result[key] or 0, value)
        except (psutil.Error, AttributeError):
            result['cpu_read_errors'] += 1
        try:
            free, total = torch.cuda.mem_get_info()
            result[fields[2]] = max(result[fields[2]] or 0, total - free)
            memory = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            for key, value in zip(fields[3:], memory, strict=True):
                result[key] = max(result[key] or 0, value)
        except (RuntimeError, AssertionError, AttributeError, OSError):
            result['cuda_read_errors'] += 1

    def monitor():
        while not stopped.wait(result['sample_interval_seconds']):
            sample()

    sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def finish():
        stopped.set()
        thread.join()
        sample()
        result.update(elapsed_seconds=time.monotonic() - started,
                      cpu_self_seconds=time.process_time() - cpu)
        if result['cpu_read_errors']:
            result.update({key: None for key in fields[:2]})
        if result['cuda_read_errors']:
            result.update({key: None for key in fields[2:]})

    return result, finish


def validate_measurement(values):
    if (values.get('format') != 'kumo-nightly-resources-v1'
            or not values.get('scope') or values.get('samples', 0) < 2
            or values.get('cpu_read_errors') or values.get('cuda_read_errors')
            or any(type(values.get(key)) not in (int, float)
                   or not math.isfinite(values[key]) or values[key] < 0
                   for key in RESOURCE_FIELDS)):
        raise ValueError('Resource measurement is missing or invalid')


class TrainingWindow:
    """Reuse Group 5's completed-batch clock without changing the fit recipe."""

    def __init__(self,
                 batch_size,
                 cuda,
                 clock=time.perf_counter,
                 cpu_clock=time.process_time):
        self.batch_size, self.cuda = batch_size, cuda
        self.clock, self.cpu_clock = clock, cpu_clock
        self.completed, self.started, self.stop_sampling = 0, False, None
        self.result = dict(format='kumo-training-window-v1',
                           status='INCOMPLETE',
                           scope='training_batches_21_220',
                           reason='Training needs 220 completed batches',
                           warmup_batches=20,
                           measured_batches=0,
                           completed_batches=0,
                           batch_size=batch_size,
                           measured_examples=0,
                           measured_optimizer_updates=0,
                           seconds=None,
                           examples_per_second=None)

    def invalidate(self, reason):
        self.result.update(status='INVALID',
                           reason=reason,
                           examples_per_second=None)
        self.finish()

    def on_train_start(self, trainer, module):
        if self.started:
            self.invalidate('Training retry invalidates the window')
            raise ValueError(self.result['reason'])
        self.started = True
        self.device = trainer.strategy.root_device
        self.optimizer_count = len(trainer.optimizers)
        self.result['optimizer_count'] = self.optimizer_count
        if (self.device.type != 'cuda' or module.device.type != 'cuda'
                or type(self.batch_size) is not int or self.batch_size < 1
                or not self.optimizer_count or trainer.global_step != 0
                or trainer.accumulate_grad_batches != 1):
            self.invalidate('Timing requires fresh, unaccumulated CUDA work')
            raise ValueError(self.result['reason'])

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        from numbers import Integral

        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if getattr(getattr(loss, 'device', None), 'type', None) != 'cuda':
            self.invalidate('Completed training loss is not on CUDA')
            raise ValueError(self.result['reason'])
        self.completed += 1
        measured = min(200, max(0, self.completed - 20))
        self.result.update(completed_batches=self.completed,
                           measured_batches=measured)
        seed = (getattr(module, 'lhs_name', None)
                or getattr(module, 'entity_table_name', None)
                or getattr(batch, 'input_type', None))
        size = batch[seed].batch_size
        if (isinstance(size, bool) or not isinstance(size, Integral)
                or size != self.batch_size or trainer.global_step
                != self.completed * self.optimizer_count):
            self.invalidate('Batch size or completed optimizer work changed')
        if self.result['status'] == 'INVALID':
            return
        if self.completed == 20:
            self.cuda.synchronize(self.device)
            self.cuda.reset_peak_memory_stats(self.device)
            resources, self.stop_sampling = start_measurement(
                'training_window')
            self.result['resources'] = resources
            self.cpu_start, self.start = self.cpu_clock(), self.clock()
        elif self.completed == 220:
            self.cuda.synchronize(self.device)
            seconds = self.clock() - self.start
            self.result.update(
                status='MEASURED',
                reason=None,
                seconds=seconds,
                measured_examples=200 * self.batch_size,
                measured_optimizer_updates=200 * self.optimizer_count,
                main_cpu_seconds=self.cpu_clock() - self.cpu_start,
                gpu_allocated_peak_bytes=self.cuda.max_memory_allocated(
                    self.device),
                gpu_reserved_peak_bytes=self.cuda.max_memory_reserved(
                    self.device))
            self.result['examples_per_second'] = (200 * self.batch_size /
                                                  seconds
                                                  if seconds > 0 else None)
            self.finish()
            try:
                self.summary()
            except ValueError as error:
                self.invalidate(str(error))

    def finish(self):
        if self.stop_sampling is not None:
            stop, self.stop_sampling = self.stop_sampling, None
            stop()

    def summary(self):
        validate_training_window(self.result)
        return self.result


def validate_training_window(values):
    integer = lambda key: type(values.get(key)) is int
    if (values.get('format') != 'kumo-training-window-v1'
            or values.get('scope') != 'training_batches_21_220'
            or values.get('status') != 'MEASURED' or not all(
                integer(key)
                for key in ('warmup_batches', 'measured_batches',
                            'completed_batches', 'batch_size',
                            'measured_examples', 'optimizer_count',
                            'measured_optimizer_updates'))
            or values['warmup_batches'] != 20
            or values['measured_batches'] != 200
            or values['completed_batches'] < 220 or values['batch_size'] < 1
            or values['optimizer_count'] < 1
            or values['measured_examples'] != 200 * values['batch_size']
            or values['measured_optimizer_updates']
            != 200 * values['optimizer_count'] or any(
                type(values.get(key)) not in (int, float)
                or not math.isfinite(values[key]) or values[key] <= 0
                for key in ('seconds', 'examples_per_second',
                            'gpu_allocated_peak_bytes',
                            'gpu_reserved_peak_bytes')) or
            not math.isclose(values['examples_per_second'],
                             values['measured_examples'] / values['seconds'],
                             rel_tol=1e-9)):
        raise ValueError(
            'Training window work, timing or GPU peaks are incomplete')
    resources = values.get('resources', {})
    validate_measurement(resources)
    if (resources['scope'] != 'training_window'
            or any(resources[key] <= 0
                   for key in ('cpu_tree_pss_sampled_peak_bytes',
                               'cpu_tree_rss_sampled_peak_bytes'))):
        raise ValueError('Training window CPU memory or scope is missing')


def validate_case_measurements(row):
    values = row.get('measurements', {})
    validate_measurement(values)
    if values['scope'] != 'complete_case':
        raise ValueError('Case measurement scope differs')
    for name in ('training', 'predict', 'repeat', 'partial',
                 'reloaded_prediction'):
        operation = row.get('operation_measurements', {}).get(name, {})
        validate_measurement(operation)
        if (operation['scope'] != name or row.get('timings', {}).get(name)
                != operation['elapsed_seconds']):
            raise ValueError('Operation measurement scope or elapsed time differs')
    work = row.get('work', {})
    window = work.get('training_window', {})
    if window.get('status') == 'MEASURED':
        validate_training_window(window)
        if (window['completed_batches'] != work.get('training_batches')
                or window['batch_size'] != work.get('requested_batch_size')
                or work.get('optimizer_steps')
                != window['completed_batches'] * window['optimizer_count']):
            raise ValueError('Training window differs from completed fit work')


@contextmanager
def measure_operation(report, name):
    import torch

    torch.cuda.synchronize()
    values, finish = start_measurement(name)
    report.setdefault('operation_measurements', {})[name] = values
    try:
        yield
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            finish()
            report.setdefault('timings', {})[name] = values['elapsed_seconds']


def verify_artifacts(root, files, required=()):
    if not files or not set(required) <= set(files):
        raise ValueError(
            'Required retained artifacts are missing from the hash map')
    for name in files:
        safe_file(root, name, files)


def safe_file(root, name, files):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or relative.is_absolute() or '\\' in name
            or name != relative.as_posix()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Invalid input path: {name}')
    path = Path(root)
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Input paths must be regular: {name}')
    if (not path.is_file() or name not in files or digest(path) != files[name]):
        raise ValueError(f'Missing or changed frozen input: {name}')
    return path


def load_inputs(root, expected):
    root = Path(root).resolve()
    manifest = root / 'manifest.json'
    if manifest.is_symlink() or digest(manifest) != expected:
        raise ValueError('Input manifest checksum differs')
    spec = read(manifest)
    if spec.get('format') != 'kumo-engine-nightly-inputs-v1':
        raise ValueError('Use the admitted nightly data manifest')
    if not isinstance(spec.get('files'), dict) or not spec['files']:
        raise ValueError('Input manifest needs exact regular-file hashes')
    for name in spec['files']:
        if Path(name).suffix not in ('.json', '.parquet', '.bin', ''):
            raise ValueError('Input artifacts contain data and metadata only')
        safe_file(root, name, spec['files'])
    identities = [(c['task'], c['arm'], c['seed']) for c in spec['cases']]
    wanted = {(t, a, 0) for t in smoke.TASKS for a in ('normal', 'control')}
    if (len(identities) != 16 or set(identities) != wanted
            or any(type(c['seed']) is not int for c in spec['cases'])):
        raise ValueError(
            'Input manifest requires exactly sixteen seed-zero cases')
    for case in spec['cases']:
        graph = spec['graphs'][case['graph']]
        if (graph['profile'] != smoke.TASKS[case['task']]['graph']
                or graph['static'] is not (case['task'] == 'static_link')):
            raise ValueError('Task graph profile or static visibility differs')
        failures = case['native_expectations']['key_failures']
        if set(failures) != set(KEY_FAILURES) or any(
                type(v) is not int or v for v in failures.values()):
            raise ValueError(
                'Admitted native key failures must be exactly four integer zeros'
            )
    safe_file(root, spec['candidate_dependencies'], spec['files'])
    safe_file(root, spec['reference_dependencies'], spec['files'])
    for graph in spec['graphs'].values():
        safe_file(root, graph['schema'], spec['files'])
        for source in graph['sources'].values():
            path = root / source
            if source.startswith('/') or '..' in PurePosixPath(source).parts:
                raise ValueError('Graph sources must stay under the input root')
            paths = list(path.rglob('*')) if path.is_dir() else [path]
            if not paths or any(p.is_symlink() for p in paths):
                raise ValueError('Graph sources must be regular frozen files')
            for item in paths:
                if item.is_file() and str(
                        item.relative_to(root)) not in spec['files']:
                    raise ValueError('Graph contains an undeclared source file')
    return spec


def check_dependencies(packages, profile):
    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    if not profile or not all(
            packages.get(normalize(k)) == v for k, v in profile.items()):
        raise ValueError(
            'Installed dependencies differ from the selected profile')


def hardware_profile(runtime):
    name = runtime['gpu']
    if ('RTX PRO 6000' not in name or runtime['gpu_count'] != 1
            or runtime['architecture'] != 'x86_64'):
        raise ValueError('Use one authorized RTX PRO 6000 GPU')
    return dict(gpu_class='RTX PRO 6000')


def check_reference_runtime(reference, runtime, reference_only):
    observed = hardware_profile(runtime)
    if not reference_only:
        previous = reference.get('reference_runtime',
                                 {}).get('reference_hardware', {})
        gpu_class = previous.get('gpu_class')
        if gpu_class is None and 'RTX PRO 6000' in previous.get('gpu', ''):
            gpu_class = 'RTX PRO 6000'
        if observed['gpu_class'] != gpu_class:
            raise ValueError('Reference and candidate GPU classes differ')


def load_reference(args):
    reference = read(args.reference)
    if (digest(args.reference) != args.reference_sha256
            or reference.get('input_manifest_sha256') != args.input_sha256
            or reference.get('format') != 'kumo-engine-nightly-reference-v1'
            or reference.get('version') != '1.0.0'
            or reference.get('wheel_sha256') != REFERENCE_WHEEL):
        raise ValueError('Approved reference or input-manifest binding differs')
    return reference


def reference_cases(reference, manifest):
    references = {(v['task'], v['arm']): v for v in reference['cases']}
    admitted = {(v['task'], v['arm']): v for v in manifest['cases']}
    if (len(reference['cases']) != 16 or set(references) != set(admitted)
            or len({v['fit_id']
                    for v in reference['cases']}) != 16 or len(admitted) != 16):
        raise ValueError('Reference requires sixteen distinct admitted cases')
    for key, previous in references.items():
        case = admitted[key]
        if (type(previous['seed']) is not int or previous['seed'] != 0
                or previous['input_signature'] != input_signature(
                    manifest, case)
                or previous['metric'] != smoke.TASKS[key[0]]['metric']
                or not previous['fit_id'] or
                not re.fullmatch('[0-9a-f]{64}', previous['checkpoint_sha256'])
                or any(
                    type(previous[k]) not in
                    (int, float) or not math.isfinite(previous[k])
                    for k in ('score', 'baseline'))):
            raise ValueError(
                'Reference input, recipe or measurement identity differs')
    return references


def verify_runtime(version, profile, *, reference_only=False):
    import torch

    import kumo.engine as kumo

    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    packages = {
        normalize(d.metadata['Name']): d.version
        for d in importlib.metadata.distributions()
        if normalize(d.metadata['Name']) != 'kumo-engine'
    }
    if reference_only:
        check_dependencies(packages, profile)
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or sys.version_info[:2] != (3, 12)
            or importlib.metadata.version('kumo-engine') != version
            or kumo.__version__ != version
            or not Path(kumo.__file__).resolve().is_relative_to(
                Path(sys.prefix).resolve()) or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1):
        raise ValueError(
            'Use the installed Linux3.12 wheel and one actual CUDA GPU')
    torch.set_num_threads(4)
    properties = torch.cuda.get_device_properties(0)
    identifier = str(properties.uuid)
    uuid.UUID(identifier.removeprefix('GPU-'))
    if 'RTX PRO 6000' not in properties.name or properties.total_memory <= 0:
        raise ValueError(
            'CUDA work requires the authorized RTX PRO 6000 profile')
    driver = subprocess.check_output([
        'nvidia-smi', '--id=GPU-' + identifier.removeprefix('GPU-'),
        '--query-gpu=driver_version', '--format=csv,noheader'
    ], text=True).strip()
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', driver):
        raise ValueError('CUDA driver identity is missing or ambiguous')
    direct = json.loads(
        importlib.metadata.distribution('kumo-engine').read_text(
            'direct_url.json'))
    wheel_hash = direct['archive_info']['hashes']['sha256']
    if len(wheel_hash) != 64:
        raise ValueError('Installed wheel archive identity is missing')
    changes = {
        name: dict(profile=profile.get(name), installed=packages.get(name))
        for name in profile.keys() | packages.keys()
        if profile.get(name) != packages.get(name)
    }
    return dict(version=version, dependencies=packages,
                dependency_changes=changes, gpu=properties.name, gpu_count=1,
                driver=driver, architecture=platform.machine(),
                compute_capability=[properties.major,
                                    properties.minor], gpu_uuid=identifier,
                gpu_memory_bytes=int(properties.total_memory),
                torch=torch.__version__, cuda=torch.version.cuda, threads=4,
                import_path=kumo.__file__, wheel_sha256=wheel_hash,
                cpu_count=os.cpu_count(),
                system_memory_bytes=os.sysconf('SC_PAGE_SIZE') *
                os.sysconf('SC_PHYS_PAGES'),
                cache_protocol='hash-verified inputs; fresh native indexes; fixed case order')


def load_table(root, files, case):
    import pyarrow.parquet as pq
    from kumo_core.io.json import from_json_dict
    from kumo_core.model_plan import TrainingTableGenerationPlan

    import kumo.engine as kumo

    descriptor = safe_file(root, case['descriptor'], files)
    value = read(descriptor)
    task, arm = case['task'], case['arm']
    if (value['format'] != 'wheel-scale-control-v1' or value['case'] != task
            or value['arm'] != ('normal' if arm == 'normal' else 'control')
            or value['seed'] != (None if arm == 'normal' else 0)
            or value['query'] != smoke.TASKS[task]['query']
            or value['identity']['control_seed'] != value['seed']
            or value['train_table_path'] != 'tt.parquet'
            or value['manifest_path'] != 'manifest.json'):
        raise ValueError('Frozen task, arm, query or native paths differ')
    staging = descriptor.parent / 'staging'
    declared = {
        str(p.relative_to(staging))
        for p in staging.rglob('*') if p.is_file()
    }
    if declared != set(value['files']) or any(p.is_symlink()
                                              for p in staging.rglob('*')):
        raise ValueError('Frozen staging has missing, extra or linked files')
    for name, expected in value['files'].items():
        path = safe_file(root, str((staging / name).relative_to(root)), files)
        if digest(path) != expected:
            raise ValueError('Descriptor and artifact hashes differ')
    spec, manifest = read(staging / 'tt_ingest_spec.json'), read(
        staging / 'manifest.json')
    if spec['schema_version'] != 'diskgraph.tt_ingest_spec.v3':
        raise ValueError('Unexpected frozen native schema')
    if task == 'static_link':
        fact = spec.get('fact_pseudo_time')
        if (manifest['version'] != 4
                or value['native_split_kind'] != 'CustomLinkSplit'
                or manifest['split_kind'] != 'CustomLinkSplit'
                or fact != dict(table='purchases', entity_col='customer_id',
                                target_col='merchant_id')
                or manifest['fact_pseudo_time'] != dict(
                    path='fact_pseudo_time.bin',
                    format='tt_fact_pseudo_time_v1', node_type='purchases')
                or manifest['checksums']['fact_pseudo_time_sha256']
                != value['files']['fact_pseudo_time.bin']):
            raise ValueError('Static native routing or fact visibility differs')
    elif manifest['version'] != 3:
        raise ValueError('Unexpected frozen native version')
    stages = {v['key']: v for v in spec['split_stages']}
    if set(stages) != set(STAGES):
        raise ValueError('Frozen table needs all three native stages')
    for stage, entry in stages.items():
        rows = sum(
            pq.ParquetFile(staging / p).metadata.num_rows
            for p in entry['parts'])
        if type(case['stage_counts'][stage]) is not int or not rows or (
                rows != entry['row_count']
                or rows != case['stage_counts'][stage]):
            raise ValueError('Frozen native stage count differs')
    table = kumo.TrainingTable(
        query=value['query'], staging_dir=staging,
        train_table_path=staging / 'tt.parquet',
        manifest_path=staging / 'manifest.json',
        required_stages=tuple(value['required_stages']),
        generation_plan=from_json_dict(value['generation_plan'],
                                       TrainingTableGenerationPlan))
    return table, stages, value


def stage_frame(table, entry):
    import pandas as pd
    return pd.concat(
        [pd.read_parquet(table.staging_dir / p) for p in entry['parts']],
        ignore_index=True)


def check_inputs(frame, truth, *, partial=False, static=False):
    required = {'ENTITY1'} if static else {'ENTITY1', 'TIMESTAMP'}
    if static and 'TIMESTAMP' in frame:
        required.add('TIMESTAMP')
    if 'autoregression_labels' in truth:
        required.add('autoregression_labels')
    if set(frame) != required:
        raise ValueError(
            'Prediction inputs omit required anchors/history or include labels')
    if (frame.empty or frame.ENTITY1.isna().any() or not frame.ENTITY1.is_unique
            or not truth.ENTITY1.is_unique):
        raise ValueError('Prediction populations must be nonempty and unique')
    ids, expected = frame.ENTITY1.tolist(), truth.set_index('ENTITY1')
    if (not set(ids) <= set(expected.index)
            or not partial and set(ids) != set(expected.index)
            or partial and len(ids) >= len(expected)):
        raise ValueError('Full or partial prediction population differs')
    for column in frame.columns:
        if column != 'ENTITY1':
            if column not in expected or signature(
                    frame[column].astype(str).tolist()) != signature(
                        expected.loc[ids, column].astype(str).tolist()):
                raise ValueError(
                    'Prediction anchors or history differ from frozen rows')
    return ids


def check_training(state, selected_steps):
    epochs = state['training_epochs']
    if (state['core_fit_attempts'] != 1 or state['fits'] != 1
            or state['completed_fits'] != 1 or not 1 <= len(epochs) <= 10
            or any(e['batches'] != 20 or len(e['batch_sizes']) != 20 or any(
                type(n) is not int or n <= 0 for n in e['batch_sizes'])
                   for e in epochs)
            or state['training_batches'] != 20 * len(epochs)
            or type(state['optimizer_steps']) is not int
            or state['optimizer_steps'] <= 0 or type(selected_steps) is not int
            or not 0 < selected_steps <= state['optimizer_steps']):
        raise ValueError(
            'Fit count, fixed training work or optimizer updates differ')
    if len(epochs) < 10:
        stop = state.get('early_stopping', {})
        if (type(stop.get('stopped_epoch')) is not int
                or stop['stopped_epoch'] <= 0
                or type(stop.get('patience')) is not int or stop['patience'] < 0
                or stop.get('wait_count', 0) < stop['patience']):
            raise ValueError(
                'Short training lacks actual early-stopping evidence')


def check_recipe(case, directory):
    config = read(directory / 'trainer_config.json')
    if any(config['refit_cfg'][key] is not False
           for key in ('refitted_full', 'refitted_trainval')):
        raise ValueError('Nightly must not refit the selected model')


def stable_trainer(value):
    # Preserve the original reference's normalization of run-specific fields.
    return {key: item for key, item in value.items()
            if key not in ('root_dir', 'version', 'kumo_version', 'refit_cfg')}


def input_signature(manifest, case):
    return dict(descriptor=manifest['files'][case['descriptor']],
                prediction=manifest['files'][case['prediction_input']],
                partial=manifest['files'][case['partial_input']],
                requested_plan=case['requested_plan_sha256'])


def ranked_difference(before, after):
    added, removed, reordered, differences = 0, 0, 0, []
    for left, right, a, b in zip(before['ranked_ids'], after['ranked_ids'],
                                 before['ranked_scores'],
                                 after['ranked_scores'], strict=True):
        old, new = dict(zip(left, a,
                            strict=True)), dict(zip(right, b, strict=True))
        added += len(new.keys() - old.keys())
        removed += len(old.keys() - new.keys())
        reordered += int(left != right)
        differences.extend(
            abs(old[k] - new[k]) for k in old.keys() & new.keys())
    if added or removed or reordered:
        raise ValueError(
            'Prediction items or ranks changed after repeat, partial or reload')
    return dict(added_items=added, removed_items=removed,
                reordered_entities=reordered,
                max_abs_aligned_score_difference=max(differences, default=None),
                policy='report_only')


def ranked_rows(batches, ids, catalog, exclusions, k, raw_limit, *,
                anchor=None):
    """Certify top-K from a bounded, entity/score/key-ordered raw stream."""
    import pandas as pd
    if (not ids or len(set(ids)) != len(ids)
            or len(set(catalog)) != len(catalog) or len(exclusions) != len(ids)
            or type(raw_limit) is not int
            or not 0 < raw_limit <= min(500, len(catalog))):
        raise ValueError('Invalid ranked population or bounded output request')
    positions, members = {v: i for i, v in enumerate(ids)}, set(catalog)
    items, scores = [None] * len(ids), [None] * len(ids)
    current, pairs, count = None, [], 0

    def finish(entity):
        index = positions.get(entity)
        if index is None or items[index] is not None:
            raise ValueError('Repeated or foreign ranked entity')
        blocked = set(exclusions[index])
        if (len(blocked) != len(exclusions[index]) or not blocked <= members
                or len(pairs) != raw_limit or len({p[0]
                                                   for p in pairs}) != raw_limit
                or any(p[0] not in members or not math.isfinite(p[1])
                       for p in pairs)):
            raise ValueError(
                'Incomplete, duplicate or invalid ranked coordinates')
        wanted = min(k, len(catalog) - len(blocked))
        eligible = sorted((p for p in pairs if p[0] not in blocked),
                          key=lambda p: (-p[1], p[0]))[:wanted]
        if (not wanted or len(eligible) != wanted or raw_limit < len(catalog)
                and min(p[1] for p in pairs) >= eligible[-1][1]):
            raise ValueError(
                'Capped ranked output cannot certify the eligible cutoff')
        items[index], scores[index] = [p[0] for p in eligible
                                       ], [p[1] for p in eligible]

    for frame in batches:
        if not {'ENTITY', 'CLASS', 'SCORE'} <= set(frame):
            raise ValueError('Ranked columns are missing')
        for col in ('ENTITY', 'CLASS'):
            if (not pd.api.types.is_integer_dtype(frame[col])
                    or pd.api.types.is_bool_dtype(frame[col])
                    or frame[col].isna().any()):
                raise ValueError(
                    'Ranked IDs must retain their integer business type')
        if anchor is not None and (
                'ANCHOR_TIMESTAMP' not in frame or not pd.to_datetime(
                    frame.ANCHOR_TIMESTAMP).eq(pd.Timestamp(anchor)).all()):
            raise ValueError('Ranked prediction anchors differ')
        for entity, item, score in frame[['ENTITY', 'CLASS', 'SCORE'
                                          ]].itertuples(index=False, name=None):
            if current is not None and entity != current:
                finish(current)
                pairs = []
            current = entity
            pairs.append((int(item), float(score)))
            count += 1
            if len(pairs) > raw_limit:
                raise ValueError('Too many ranked rows for one entity')
    if current is not None:
        finish(current)
    if any(v is None for v in items):
        raise ValueError('Ranked output is missing entities')
    return dict(ranked_ids=items, ranked_scores=scores, raw_rows=count)


def ordered_batches(files, directory):
    import duckdb
    directory.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={
            'memory_limit': '256MB',
            'threads': 4,
            'temp_directory': str(directory)
    }) as db:
        for batch in db.read_parquet(list(files)).order(
                'ENTITY, SCORE DESC, CLASS ASC').fetch_record_batch(65536):
            yield batch.to_pandas()


def verify_installed_files(wheel, distribution):
    """Match installed files to the finished archive, except pip's rewritten RECORD."""
    from zipfile import ZipFile

    location = Path(distribution.locate_file('')).resolve()
    checked = 0
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/') or name.endswith('.dist-info/RECORD'):
                continue
            destination = Path(distribution.locate_file(name)).resolve()
            if (not destination.is_relative_to(location)
                    or not destination.is_file()
                    or hashlib.sha256(destination.read_bytes()).digest()
                    != hashlib.sha256(archive.read(name)).digest()):
                raise ValueError(f'Installed wheel member differs: {name}')
            checked += 1
    if not checked:
        raise ValueError('Wheel contains no installed files')
    return checked


def verify_checkpoint(model, path, expected_digest):
    """Compare the weights that Lightning actually restored, with no tolerance."""
    import numpy as np
    import pandas as pd
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint

    if path is None:
        raise ValueError('Reload did not select a checkpoint')
    path = local_path(str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError('Reload selected a different checkpoint')
    expected = load_kumo_checkpoint(str(path), map_location='cpu')['state_dict']
    actual = model.state_dict()
    tensors = 0

    def equal(left, right):
        nonlocal tensors
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if (left.dtype != right.dtype or left.shape != right.shape
                    or not torch.equal(left.detach().cpu(),
                                       right.detach().cpu())):
                raise ValueError(
                    'Restored model tensor differs from its checkpoint')
            tensors += 1
        elif type(left) is not type(right):
            raise ValueError('Restored model state types differ')
        elif isinstance(left, dict):
            if set(left) != set(right):
                raise ValueError('Restored model state keys differ')
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise ValueError('Restored model state lengths differ')
            for old, new in zip(left, right):
                equal(old, new)
        elif isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif left != right:
            raise ValueError('Restored model state values differ')

    equal(expected, actual)
    if not tensors:
        raise ValueError('Reload checked no model tensors')
    return tensors


def build_graph(spec, directory, *, static=False, sources=None):
    from kumoapi.graph import Edge
    from kumoapi.table import Column

    import kumo.engine as kumo

    files = kumo.FileConnector(root_dir=str(directory))
    tables = {}
    for name, definition in spec['tables'].items():
        columns = [Column(**column) for column in definition['cols']]
        table = kumo.Table.from_source_table(
            files[(sources
                   or {}).get(name, name)], [column.name for column in columns],
            primary_key=definition.get('pkey'),
            time_column=None if static else definition.get('time_col'),
            end_time_column=None if static else definition.get('end_time_col'),
            dtypes={column.name: column.dtype
                    for column in columns})
        tables[name] = replace(
            table, definition=replace(
                table.definition, cols=columns,
                time_col=None if static else definition.get('time_col'),
                end_time_col=None
                if static else definition.get('end_time_col')))
    graph = kumo.Graph(tables, [Edge(**edge) for edge in spec['edges']])
    graph.validate()
    return graph, graph.snapshot()


def compare_predictions(before, after, *, atol=0, before_decisions=None,
                        after_decisions=None):
    """Keep decisions exact and enforce score bounds outside explicit reporting."""
    import numpy as np

    if not math.isfinite(atol) or atol < 0:
        raise ValueError('Reload tolerance must be finite and nonnegative')
    left, right = np.asarray(before,
                             dtype=float), np.asarray(after, dtype=float)
    if (left.ndim != 2 or not left.size or left.shape != right.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError(
            'Reload comparison requires equal finite prediction shapes')
    difference = np.abs(left - right)
    result = dict(
        decisions_equal=bool(np.array_equal(before_decisions, after_decisions)),
        max_abs_difference=float(difference.max()),
        max_relative_difference=float(
            (difference / np.maximum(np.abs(left), 1e-12)).max()),
        within_tolerance=bool(np.allclose(left, right, atol=atol, rtol=0.)))
    if not result['decisions_equal']:
        raise ValueError('Prediction decisions changed after repeat or reload')
    result['policy'] = 'report_only'
    return result


@contextmanager
def observe_cuda(state):
    """Extend the shared public CUDA observer with nightly-only work proof."""
    from numbers import Integral

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    windows = []

    class Proof(pl.Callback):

        def on_train_start(self, trainer, module):
            if 'requested_batch_size' in state:
                if not hasattr(self, 'window'):
                    self.window = TrainingWindow(state['requested_batch_size'],
                                                 torch.cuda)
                    windows.append(self.window)
                    state['training_window'] = self.window.result
                self.window.on_train_start(trainer, module)

        def on_fit_start(self, trainer, module):
            if 'fit_ledger' in state:
                state['fit_ledger'].append(
                    dict(status='RUNNING', steps=0, cuda_batches=0))

        def on_train_epoch_start(self, trainer, module):
            if 'training_epochs' in state:
                state['training_epochs'].append(dict(
                    epoch=int(trainer.current_epoch), batches=0, batch_sizes=[]))

        def on_train_batch_end(self, trainer, module, values, batch,
                               batch_idx):
            if 'fit_ledger' in state:
                state['fit_ledger'][-1]['cuda_batches'] += 1
            if 'training_epochs' in state:
                seed = (getattr(module, 'lhs_name', None)
                        or getattr(module, 'entity_table_name', None)
                        or getattr(batch, 'input_type', None))
                size = batch[seed].batch_size
                if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
                    raise ValueError('Training batch has no valid seed-row count')
                state['training_epochs'][-1]['batch_sizes'].append(int(size))
                state['training_epochs'][-1]['batches'] += 1
            if hasattr(self, 'window'):
                self.window.on_train_batch_end(trainer, module, values, batch,
                                               batch_idx)

        def on_fit_end(self, trainer, module):
            if hasattr(self, 'window'):
                self.window.finish()
            if 'fit_ledger' in state:
                state['fit_ledger'][-1].update(status='PASS',
                                               steps=int(trainer.global_step))
            if 'training_epochs' in state:
                [stop] = [callback for callback in trainer.callbacks
                          if isinstance(callback, EarlyStopping)]
                state['early_stopping'] = dict(
                    stopped_epoch=int(stop.stopped_epoch),
                    wait_count=int(stop.wait_count), patience=int(stop.patience))

        def on_predict_start(self, trainer, module):
            if state['stage'] == 'reload' and 'reload_checks' in state:
                state['restored_tensors'] = verify_checkpoint(
                    module, trainer.ckpt_path, state['checkpoint_sha256'])
                state['reload_checks'] += 1

    with smoke.observe_cuda(state):
        base = pl.Trainer

        class ObservedTrainer(base):
            def __init__(self, *args, **kwargs):
                kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Proof()]
                super().__init__(*args, **kwargs)

        with patch.object(pl, 'Trainer', ObservedTrainer):
            try:
                yield
            finally:
                for window in windows:
                    window.finish()


def quality(task, truth, values, classes):
    import numpy as np
    from sklearn import metrics
    targets = truth.TARGET1.tolist()
    if task.endswith('link'):
        return float(
            np.mean([
                len(set(y) & set(p)) / len(set(y))
                for y, p in zip(targets, values['ranked_ids'], strict=True)
            ]))
    if task == 'binary':
        return float(metrics.roc_auc_score(targets, values[:, 1])) if len(
            set(targets)) == 2 else None
    if task == 'multiclass':
        return float(
            np.mean(np.asarray(classes)[values.argmax(axis=1)] == targets))
    if task in ('multilabel', 'multilabel_ranking'):
        labels = [
            set(v.split('|') if isinstance(v, str) else v) for v in targets
        ]
        y = np.asarray([[c in row for c in classes] for row in labels],
                       dtype=int)
        if task == 'multilabel':
            return float(
                metrics.average_precision_score(y, values, average='macro'))
        return float(
            np.mean([
                sum(y[i, np.lexsort((np.arange(len(classes)), -row))[:5]]) /
                sum(y[i]) if sum(y[i]) else 0 for i, row in enumerate(values)
            ]))
    target = np.asarray(truth.future_labels.tolist() if task ==
                        'forecasting' else targets)
    return float(np.mean(np.abs(target.reshape(values.shape) - values)))


def check_learning(rows):
    """Report normal gain and scrambled controls without unapproved bounds."""
    tasks = {}
    for task in sorted({row['task'] for row in rows}):
        pair = [row for row in rows if row['task'] == task]
        if len(pair) != 2 or {row['arm'] for row in pair} != {'normal', 'control'}:
            raise ValueError('Exactly one normal and control measurement required')
        normal, control = [next(row for row in pair if row['arm'] == arm)
                           for arm in ('normal', 'control')]
        for row in pair:
            validate_quality_measurement(task, row.get('score'))
            validate_quality_measurement(task, row.get('baseline'))
        lower = task in ('regression', 'forecasting')
        tasks[task] = dict(
            normal_score=normal['score'], control_score=control['score'],
            normal_baseline=normal['baseline'],
            control_baseline=control['baseline'],
            normal_gain=normal['baseline'] - normal['score'] if lower else
            normal['score'] - normal['baseline'],
            control_drift=abs(control['score'] - control['baseline']))
    if not tasks:
        raise ValueError('Learning measurements are missing')
    return dict(policy='report_only', tasks=tasks)


def validate_quality_measurement(task, score):
    if (type(score) not in (int, float) or not math.isfinite(score)
            or score < 0 or task not in ('regression', 'forecasting')
            and score > 1):
        raise ValueError('Quality measurement is missing or invalid')


def reporting_change(reference, candidate, margin, *, lower_is_better=True,
                     compatible=True):
    valid = lambda value: type(value) in (int, float) and math.isfinite(value)
    result = dict(reference=reference, candidate=candidate,
                  absolute_difference=candidate - reference
                  if valid(candidate) and valid(reference) else None,
                  percentage_change=None, relative_degradation_percent=None,
                  reporting_margin_percent=margin,
                  above_reporting_margin=None, policy='report_only')
    if (not compatible or not valid(reference) or reference <= 0
            or not valid(candidate) or candidate < 0):
        result['unavailable_reason'] = (
            'Inputs, configuration, runtime or measurement scope differ'
            if not compatible else 'Missing, zero or invalid reference/value')
        return result
    change = 100 * (candidate - reference) / reference
    if margin is None:
        result.update(percentage_change=change)
        return result
    previous, current = Decimal(str(reference)), Decimal(str(candidate))
    fraction = Decimal(str(margin)) / 100
    result.update(
        percentage_change=change,
        relative_degradation_percent=change if lower_is_better else -change,
        above_reporting_margin=bool(current > previous * (1 + fraction)
                                    if lower_is_better else
                                    current < previous * (1 - fraction)))
    return result


def measurement_changes(current, baseline, *, compatible=True):
    same_scope = (current.get('format') == baseline.get('format') ==
                  'kumo-nightly-resources-v1' and bool(current.get('scope'))
                  and current.get('scope') == baseline.get('scope')
                  and current.get('sample_interval_seconds') ==
                  baseline.get('sample_interval_seconds'))
    operation = current.get('scope') in ('training', 'training_window',
                                         'predict', 'repeat', 'partial',
                                         'reloaded_prediction')
    return {
        key: reporting_change(
            baseline.get(key), current.get(key),
            20 if key in RESOURCE_FIELDS[2:] or
            key == 'elapsed_seconds' and operation else None,
            compatible=compatible and same_scope)
        for key in RESOURCE_FIELDS
    }


def training_window_changes(current, baseline, *, compatible=True):
    reason = None if compatible else 'Inputs, configuration or runtime differ'
    try:
        validate_training_window(current)
        validate_training_window(baseline)
    except ValueError as error:
        compatible, reason = False, str(error)
    if any(
            current.get(key) != baseline.get(key)
            for key in ('batch_size', 'optimizer_count', 'scope')):
        compatible, reason = False, 'Training window profiles differ'
    return dict(eligible=compatible,
                unavailable_reason=reason if not compatible else None,
                time=reporting_change(baseline.get('seconds'),
                                      current.get('seconds'),
                                      20,
                                      compatible=compatible),
                throughput=reporting_change(
                    baseline.get('examples_per_second'),
                    current.get('examples_per_second'),
                    None,
                    lower_is_better=False,
                    compatible=compatible),
                resources=measurement_changes(current.get('resources', {}),
                                              baseline.get('resources', {}),
                                              compatible=compatible))


def training_work(row, trainer):
    work = row.get('work', {})
    batch = trainer.get('loader_kwargs', {}).get('batch_size')
    epochs = work.get('training_epochs', [])
    if (type(batch) is not int or batch <= 0 or not epochs
            or any(size != batch for epoch in epochs
                   for size in epoch.get('batch_sizes', []))):
        return None
    result = dict(requested_batch_size=batch, training_epochs=epochs,
                  optimizer_steps=work.get('optimizer_steps'),
                  selected_checkpoint_steps=row.get('selected_checkpoint_steps'),
                  early_stopping=work.get('early_stopping', {}))
    if 'training_window' in work:
        result['training_window_protocol'] = {
            key: work['training_window'].get(key)
            for key in ('format', 'scope', 'warmup_batches')
        }
    return result


def comparison_context(reference, report, previous, directory):
    prior = reference.get('reference_runtime', {})
    observed = prior.get('observed', {})
    dependency_signature = prior.get('reference_dependencies_signature')
    if dependency_signature is None and observed.get('dependencies'):
        dependency_signature = signature(observed['dependencies'])
    runtime = report['runtime']
    reasons = []
    if dependency_signature != signature(runtime['dependencies']):
        reasons.append('Dependency profile is missing or different')
    for key in ('gpu', 'gpu_count', 'architecture', 'driver', 'torch', 'cuda',
                'threads', 'cpu_count', 'system_memory_bytes', 'cache_protocol'):
        if observed.get(key) is None or observed[key] != runtime.get(key):
            reasons.append(f'Runtime comparison field is missing or different: {key}')
    actual = signature(stable_trainer(read(directory / 'trainer_config.json')))
    if actual != previous.get('trainer_config_sha256'):
        reasons.append('Effective trainer configuration is missing or different')
    current_work = training_work(read(directory / 'result.json'),
                                 read(directory / 'trainer_config.json'))
    if current_work is None or current_work != previous.get('training_work'):
        reasons.append('Fixed training work is missing, different or batch-reduced')
    return dict(compatible=not reasons, reasons=reasons,
                reference_trainer_config_sha256=previous.get('trainer_config_sha256'),
                candidate_trainer_config_sha256=actual)


@contextmanager
def native_calls(state):
    import diskgraph
    from kumo_core.train.trainer import Trainer
    original, ingest = Trainer.fit, diskgraph.execute_tt_ingest

    def fit(*args, **kwargs):
        state['core_fit_attempts'] += 1
        row = dict(status='RUNNING', attempt=state['core_fit_attempts'])
        state.setdefault('core_fit_ledger', []).append(row)
        save(state['ledger_path'], state['core_fit_ledger'])
        try:
            if state['core_fit_attempts'] != 1:
                raise ValueError(
                    'Automatic fit retries or refits are forbidden')
            result = original(*args, **kwargs)
            row.update(status='PASS', cuda_batches=state['training_batches'],
                       steps=state['optimizer_steps'])
            return result
        except BaseException:
            row['status'] = 'FAIL'
            raise
        finally:
            save(state['ledger_path'], state['core_fit_ledger'])

    def consume(*args, **kwargs):
        result = ingest(*args, **kwargs)
        failures = result['key_failures']
        if any(
                type(failures.get(k)) is not int or failures[k] != 0
                for k in ('parse_failures', 'lookup_failures',
                          'membership_failures')):
            raise ValueError(
                'Native training-table ingest has missing or failed keys')
        state['native_ingests'].append(
            dict(extension=result['tt_extension_path'], key_failures=failures))
        return result

    with patch.object(Trainer, 'fit',
                      fit), patch.object(diskgraph, 'execute_tt_ingest',
                                         consume):
        yield


def check_snapshot(snapshot, graph_spec):
    from kumo_core.diskgraph.metadata import load_diskgraph_schema
    stats = read(Path(snapshot.index_path) / 'ingest_stats.json')['stats']
    failures = {k: stats.get(k) for k in KEY_FAILURES}
    if (any(type(v) is not int or v != 0 for v in failures.values())
            or stats['node_counts'] != graph_spec['row_counts']):
        raise ValueError('Graph row counts or strict native key checks differ')
    schema = read(Path(graph_spec['input_root']) /
                  graph_spec['schema'])[graph_spec['profile']]
    if load_diskgraph_schema(str(snapshot.index_path)).pkey_dict != {
            name: table['pkey']
            for name, table in schema['tables'].items() if table.get('pkey')
    }:
        raise ValueError('Native graph primary-key roles differ')
    return dict(node_counts=stats['node_counts'], key_failures=failures)


def check_feature_schema(runtime, features, schema, *, static=False):
    expected_nodes = set(schema['tables'])
    extra = {'train_table'
             } if runtime['task']['task_type'] == 'forecasting' else set()
    if (not expected_nodes <= set(features['node_types'])
            or not set(features['node_types']) - expected_nodes <= extra):
        raise ValueError('Fitted graph is missing source tables')
    for name, table in schema['tables'].items():
        expected = {
            c['name']: (c['dtype'], c['stype'])
            for c in table['cols'] if c['stype'] != 'ID'
        }
        actual = features['node_types'][name]
        if ({
                c['column_name']: (c['dtype'], c['stype'])
                for c in actual['column_specs']
        } != expected or actual['time_column']
                != (None if static else table.get('time_col'))):
            raise ValueError('Fitted source feature types or time roles differ')
    if {
            tuple(e[k] for k in ('src_table', 'fkey', 'dst_table'))
            for e in schema['edges']
    } != {tuple(e)
          for e in runtime['graph']['edge_types']}:
        raise ValueError('Fitted source relationship roles differ')


def run_case(root, manifest, case, graph, snapshot, output):
    import numpy as np
    import pandas as pd
    import torch
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    output.mkdir(parents=True)
    task = case['task']
    started = time.monotonic()
    state = dict(stage='fit', fits=0, completed_fits=0, core_fit_attempts=0,
                 training_batches=0, optimizer_steps=0, training_epochs=[],
                 prediction_batches={s: 0
                                     for s in PREDICTIONS}, reload_checks=0,
                 restored_tensors=0, native_ingests=[], fit_ledger=[])
    report = dict(task=task, arm=case['arm'], seed=0, status='RUNNING',
                  work=state)
    state['ledger_path'] = str(output / 'core-fits.json')
    measurements, finish_measurement = start_measurement('complete_case')
    try:
        with measure_operation(report, 'preparation'):
            table, stages, descriptor = load_table(root, manifest['files'],
                                                   case)
            truth = stage_frame(table, stages['test'])
            full = pd.read_parquet(
                safe_file(root, case['prediction_input'], manifest['files']))
            partial = pd.read_parquet(
                safe_file(root, case['partial_input'], manifest['files']))
            ids = check_inputs(full, truth, static=task == 'static_link')
            partial_ids = check_inputs(partial, truth, partial=True,
                                       static=task == 'static_link')
            truth = truth.set_index('ENTITY1').loc[ids].reset_index()
            if task == 'forecasting' and any(
                    len(v) != 12
                    for col in ('autoregression_labels', 'future_labels')
                    for v in truth[col]):
                raise ValueError(
                    'Forecast inputs/answers require all twelve coordinates')
            query = kumo.PredictiveQuery(graph, descriptor['query'])
            query.validate()
            plan = smoke.make_plan(task)
            save(output / 'requested-plan.json', plan.as_json())
            [state['requested_batch_size']
             ] = plan.as_json()['optimization']['batch_size']
            save(output / 'generation-plan.json', descriptor['generation_plan'])
            report['input_signature'] = input_signature(manifest, case)
            classes = case.get('classes', [])
            catalog = []
            if task.endswith('link'):
                spec = case['catalog']
                catalog = pd.read_parquet(
                    safe_file(root, spec['path'], manifest['files']),
                    columns=[spec['column']])[spec['column']].tolist()
                if not catalog or len(set(catalog)) != len(catalog):
                    raise ValueError(
                        'Link catalog keys are missing or duplicated')
            anchor = None if task == 'static_link' else full.TIMESTAMP.iloc[0]
            if anchor is not None and not full.TIMESTAMP.eq(anchor).all():
                raise ValueError(
                    'Full prediction needs the fixed frozen test anchor')
            seed = lambda: (random.seed(0), np.random.seed(0),
                            torch.manual_seed(0))
            work = output.parent / 'work' / output.name
            kumo.configure(scratch_dir=work)
            seed()
        with observe_cuda(state), native_calls(state):
            trainer = kumo.Trainer(plan)
            torch.cuda.synchronize()
            with measure_operation(report, 'training'):
                fitted = trainer.fit(table, snapshot=snapshot)
                torch.cuda.synchronize()
            checkpoint = local_path(
                str(fitted.artifact_paths['selected_checkpoint']))
            state['checkpoint_sha256'] = digest(checkpoint)
            selected_steps = load_kumo_checkpoint(
                str(checkpoint), map_location='cpu')['global_step']
            check_training(state, selected_steps)
            [runtime_path] = trainer.scratch_dir.glob(
                'runs/*-fit-*/task_dataset/runtime_manifest.json')
            runtime = read(runtime_path)
            if runtime['task']['task_type'] != smoke.TASKS[task][
                    'task_type'] or {
                        name: v['row_count']
                        for name, v in runtime['stages'].items()
                    } != case['stage_counts']:
                raise ValueError('Fitted task or consumed stage counts differ')
            for name in ('runtime_manifest', 'metadata', 'feature_schema'):
                shutil.copyfile(runtime_path.with_name(name + '.json'),
                                output / (name + '.json'))
            for name in ('resolved_model_plan', 'trainer_config', 'encoders',
                         'target_encoder'):
                shutil.copyfile(fitted.artifact_paths[name],
                                output / (name + '.json'))
            [selected] = trainer.scratch_dir.glob(
                'runs/*-fit-*/models/*/automl/version_*/model_plan.json')
            shutil.copyfile(selected, output / 'selected-model-plan.json')
            check_recipe(case, output)
            graph_spec = manifest['graphs'][case['graph']]
            graph_schema = read(root /
                                graph_spec['schema'])[graph_spec['profile']]
            check_feature_schema(runtime, read(output / 'feature_schema.json'),
                                 graph_schema, static=task == 'static_link')
            saved = output / 'published-model'
            shutil.copytree(fitted.bundle_dir, saved)
            checkpoints = [
                p for p in saved.rglob('*')
                if p.is_file() and digest(p) == state['checkpoint_sha256']
            ]
            if not checkpoints:
                raise ValueError(
                    'Published model does not retain the selected checkpoint bytes'
                )
            report.update(
                fit_id=str(uuid.uuid4()) + ':' + str(fitted.bundle_dir),
                checkpoint_sha256=digest(checkpoint),
                checkpoint_artifact=str(checkpoints[0].relative_to(output)),
                selected_checkpoint_steps=selected_steps)
            time_column = read(
                runtime_path.with_name('feature_schema.json'))['node_types'][
                    runtime['table']['seed_entity_table']]['time_column']

            def predict(model, stage, frame):
                label = 'reloaded_prediction' if stage == 'reload' else stage
                with measure_operation(report, label):
                    seed()
                    torch.cuda.synchronize()
                    state['stage'] = stage
                    before = state['prediction_batches'][stage]
                    inputs = output / 'prediction-inputs' / (stage + '.parquet')
                    inputs.parent.mkdir(exist_ok=True)
                    frame.to_parquet(inputs, index=False)
                    pt = kumo.PredictionTable.from_source_table(
                        kumo.FileConnector(
                            root_dir=str(inputs.parent))[inputs.name], pq=query,
                        snapshot=snapshot,
                        plan=model.training_table_generation_plan)
                    selected = frame.ENTITY1.tolist()
                    options = dict(output_config=kumo.PredictionOutputConfig(
                        kumo.FileConnector(root_dir=str(output /
                                                        'prediction-outputs')),
                        output_paths={'predictions': stage
                                      }, extra_fields=[] if task ==
                        'static_link' else [MetadataField.ANCHOR_TIMESTAMP]))
                    if task.endswith('link'):
                        options['num_classes_to_return'] = min(
                            500, len(catalog))
                    elif classes:
                        options['num_classes_to_return'] = len(classes)
                    if task == 'binary':
                        options['binary_classification_threshold'] = .5
                    result = model.predict(pt, snapshot=snapshot, **options)
                    torch.cuda.synchronize()
                    if state['prediction_batches'][stage] <= before:
                        raise ValueError(
                            'Prediction completed no actual CUDA batches')
                    raw = output / 'prediction-outputs' / stage
                    raw.mkdir(parents=True, exist_ok=True)
                    retained = []
                    for index, source in enumerate(result.files):
                        destination = raw / f'raw-{index}.parquet'
                        if Path(source).resolve() != destination.resolve():
                            shutil.copyfile(local_path(source), destination)
                        retained.append(str(destination.relative_to(output)))
                    if task.endswith('link'):
                        selected_truth = truth.set_index(
                            'ENTITY1').loc[selected]
                        excluded = selected_truth.exclude_target_keys.tolist(
                        ) if task == 'static_link' else [[] for _ in selected]
                        values = ranked_rows(
                            ordered_batches(result.files,
                                            work / 'sort' / stage), selected,
                            catalog, excluded,
                            5 if task == 'static_link' else 10,
                            min(500, len(catalog)), anchor=anchor)
                        save(output / (stage + '-ranked.json'), values)
                    else:
                        frame = result.frame
                        if frame.ENTITY.dtype != full.ENTITY1.dtype:
                            raise ValueError(
                                'Prediction changed the original entity ID type'
                            )
                        frame.to_parquet(output / (stage + '.parquet'),
                                         index=False)
                        values = smoke.prediction_values(
                            task, frame, selected, classes=classes,
                            anchor=anchor, time_column=time_column
                            or 'TIMESTAMP')
                    decisions = (values['ranked_ids'] if task.endswith('link')
                                 else smoke.prediction_decisions(
                                     task, frame, selected, classes=classes))
                    report.setdefault('predictions', {})[stage] = dict(
                        entities=len(selected), rows=result.num_rows,
                        files=retained, original_files=list(result.files),
                        input_sha256=digest(inputs))
                    return values, decisions

            original, original_decisions = predict(trainer, 'predict', full)
            repeated, repeat_decisions = predict(trainer, 'repeat', full)
            subset, subset_decisions = predict(trainer, 'partial', partial)
            del trainer
            gc.collect()
            with measure_operation(report, 'model_load'):
                trainer = kumo.Trainer.load(saved)
                torch.cuda.synchronize()
            reloaded, reload_decisions = predict(trainer, 'reload', full)
            if state['reload_checks'] < 1 or state['restored_tensors'] <= 0:
                raise ValueError(
                    'Reload lacks an exact restored checkpoint check')
            lookup = {v: i for i, v in enumerate(ids)}
            positions = [lookup[v] for v in partial_ids]
            report['differences'] = {}
            for label, other, decisions, selected in [
                ('repeat', repeated, repeat_decisions, None),
                ('partial', subset, subset_decisions, positions),
                ('reload', reloaded, reload_decisions, None)
            ]:
                if task.endswith('link'):
                    before = original if selected is None else {
                        k: [original[k][i] for i in selected]
                        for k in ('ranked_ids', 'ranked_scores')
                    }
                    report['differences'][label] = ranked_difference(
                        before, other)
                else:
                    report['differences'][label] = compare_predictions(
                        original if selected is None else original[selected],
                        other, atol=0,
                        before_decisions=original_decisions
                        if selected is None or original_decisions is None else
                        original_decisions[selected], after_decisions=decisions)
            report.update(status='MEASURED',
                          physical_fits=state['core_fit_attempts'],
                          prediction_batches=state['prediction_batches'],
                          score=quality(task, truth, original, classes),
                          metric=smoke.TASKS[task]['metric'],
                          policy='report_only')
            check_training(state, selected_steps)
            load_table(root, manifest['files'], case)
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc(),
                      physical_fits=state['core_fit_attempts'])
        raise
    finally:
        try:
            report['artifacts'] = {}
            for path in output.rglob('*'):
                if path.is_file() and path != output / 'result.json':
                    report['artifacts'][str(
                        path.relative_to(output))] = digest(path)
        except BaseException as error:
            report[
                'artifact_preservation_error'] = f'{type(error).__name__}: {error}'
            if report['status'] != 'FAIL':
                report.update(status='FAIL',
                              error=report['artifact_preservation_error'],
                              traceback=traceback.format_exc())
                raise
        finally:
            finish_measurement()
            report['measurements'] = measurements
            measurement_error = None
            if report['status'] != 'FAIL':
                try:
                    validate_case_measurements(report)
                except ValueError as error:
                    measurement_error = error
                    report.update(status='FAIL', error=str(error),
                                  traceback=traceback.format_exc())
            report['seconds'] = time.monotonic() - started
            save(output / 'result.json', report)
            if measurement_error is not None:
                raise measurement_error

    return report


def check_results(rows):
    wanted = {(task, arm, 0)
              for task in smoke.TASKS
              for arm in ('normal', 'control')}
    if (len(rows) != 16 or {(r['task'], r['arm'], r['seed'])
                            for r in rows} != wanted
            or len({r['fit_id']
                    for r in rows}) != 16
            or any(r['status'] != 'MEASURED' or type(r['seed']) is not int
                   or type(r['physical_fits']) is not int or r['physical_fits']
                   != 1 or len(r['checkpoint_sha256']) != 64 or not all(
                       r['prediction_batches'].get(s, 0) > 0
                       for s in PREDICTIONS) for r in rows)):
        raise ValueError(
            'Comparison needs sixteen complete unique one-fit workflows')


def load_report(root, expected):
    paths = list(Path(root).rglob('nightly-result.json'))
    if len(paths) != 1:
        raise ValueError('Exactly one complete sixteen-case report is required')
    path = paths[0]
    report = read(path)
    if (report['format'] != 'kumo-engine-nightly-result-v1' or 'shard' in report
            or report['status'] != 'MEASURED'
            or report['input_manifest_sha256'] != expected
            or report['expected_runs'] != 16 or report['completed_runs'] != 16):
        raise ValueError(
            'The single report must complete all sixteen frozen cases')
    check_results(report['runs'])
    return path, report


def run(args):
    started = time.monotonic()
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    if output.exists() and any(p.name != 'setup' for p in output.iterdir()):
        raise ValueError('Use a fresh output directory')
    manifest = load_inputs(root, args.input_sha256)
    reference = load_reference(args)
    reference_cases(reference, manifest)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'input-metadata'
    metadata.mkdir()
    shutil.copyfile(root / 'manifest.json', metadata / 'manifest.json')
    for key in ('candidate_dependencies', 'reference_dependencies'):
        relative = manifest[key]
        (metadata / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, metadata / relative)
    profile = manifest['reference_dependencies' if args.
                       reference_only else 'candidate_dependencies']
    runtime = verify_runtime(args.expected_version, read(root / profile),
                             reference_only=args.reference_only)
    check_reference_runtime(reference, runtime, args.reference_only)
    check_install(output, runtime, reference_only=args.reference_only)
    report = dict(format='kumo-engine-nightly-result-v1', status='RUNNING',
                  input_manifest_sha256=args.input_sha256, runtime=runtime,
                  reference_sha256=args.reference_sha256,
                  reference_only=args.reference_only, expected_runs=16, runs=[],
                  helper_sha256=digest(HELPER), runner_sha256=digest(__file__))
    graphs = {}
    try:
        for case in manifest['cases']:
            key = case['graph']
            if key not in graphs:
                import kumo.engine as kumo
                spec = manifest['graphs'][key]
                with measure_operation(report, 'graph.' + key):
                    kumo.configure(scratch_dir=output / 'work' / 'graphs' / key)
                    graphs[key] = build_graph(
                        read(root / spec['schema'])[spec['profile']], root,
                        static=spec['static'], sources=spec['sources'])
                    report.setdefault('graphs', {})[key] = check_snapshot(
                        graphs[key][1], dict(spec, input_root=str(root)))
            row = run_case(root, manifest, case, *graphs[key],
                           output / (case['task'] + '-' + case['arm']))
            report['runs'].append(row)
            save(output / 'nightly-result.json', report)
        check_results(report['runs'])
        load_inputs(root, args.input_sha256)
        report['status'] = 'MEASURED'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.monotonic() - started,
                      completed_runs=len(report['runs']))
        save(output / 'nightly-result.json', report)


def check_install(root, runtime, *, reference_only=False):
    wheel, build = read(root / 'setup/wheel.json'), read(root /
                                                         'setup/build.json')
    if (type(wheel['installed_files']) is not int
            or wheel['installed_files'] <= 0
            or wheel['sha256'] != build['wheel_sha256']
            or wheel['sha256'] != runtime['wheel_sha256']
            or wheel['version'] != build['version']
            or wheel['version'] != runtime['version']):
        raise ValueError(
            'Selected source, installed wheel or archive identity differs')
    if reference_only and (runtime['version'] != '1.0.0'
                           or runtime['wheel_sha256'] != REFERENCE_WHEEL):
        raise ValueError(
            'Reference-only work requires the exact approved wheel')
    if build.get('format') == 'kumo-engine-published-reference-v1':
        harness = os.environ.get('GITHUB_SHA', '')
        if (not reference_only or not re.fullmatch('[0-9a-f]{40}', harness)
                or build.get('harness') != harness):
            raise ValueError(
                'Published reference receipt has no matching harness')
    elif any(not re.fullmatch('[0-9a-f]{40}', build.get(k, ''))
             for k in ('revision', 'tree')):
        raise ValueError('Selected source revision or tree identity differs')
    return build


def compare(args):
    manifest_path = args.input_root / 'manifest.json'
    if manifest_path.is_symlink() or digest(manifest_path) != args.input_sha256:
        raise ValueError('Copied input manifest checksum differs')
    manifest = read(manifest_path)
    reference = load_reference(args)
    references = reference_cases(reference, manifest)
    path, report = load_report(args.results_dir, args.input_sha256)
    rows = report['runs']
    if (report.get('reference_only') is not args.reference_only
            or report.get('reference_sha256') != args.reference_sha256):
        raise ValueError('Run used a different reference or measurement mode')
    if report['runner_sha256'] != digest(
            __file__) or report['helper_sha256'] != digest(HELPER):
        raise ValueError('Candidate report used different source helpers')
    expected = read(
        safe_file(
            args.input_root,
            manifest['reference_dependencies' if args.
                     reference_only else 'candidate_dependencies'],
            manifest['files']))
    admitted = {(v['task'], v['arm']): v for v in manifest['cases']}
    if args.reference_only:
        check_dependencies(report['runtime']['dependencies'], expected)
    check_reference_runtime(reference, report['runtime'], args.reference_only)
    check_install(path.parent, report['runtime'],
                  reference_only=args.reference_only)
    if digest(path.parent / 'reference.json') != args.reference_sha256:
        raise ValueError('Run did not retain the exact approved reference copy')
    if digest(path.parent /
              'input-metadata/manifest.json') != args.input_sha256:
        raise ValueError('Run did not retain the exact admitted input manifest')
    for row in report['runs']:
        validate_case_measurements(row)
        directory = path.parent / (row['task'] + '-' + row['arm'])
        if read(directory / 'result.json') != row:
            raise ValueError(
                'Aggregate case row differs from original result bytes')
        required = [
            row['checkpoint_artifact'],
            'requested-plan.json',
            'generation-plan.json',
            'selected-model-plan.json',
            'resolved_model_plan.json',
            'trainer_config.json',
            'feature_schema.json',
            'runtime_manifest.json',
            'metadata.json',
            'encoders.json',
            'target_encoder.json',
            'core-fits.json',
        ]
        for stage in PREDICTIONS:
            required.extend(row['predictions'].get(stage, {}).get('files', []))
            required.append(f'prediction-inputs/{stage}.parquet')
            required.append(stage + (
                '-ranked.json' if row['task'].endswith('link') else '.parquet'))
        verify_artifacts(directory, row['artifacts'], required=required)
        check_training(row['work'], row['selected_checkpoint_steps'])
        case = admitted[(row['task'], row['arm'])]
        if row['input_signature'] != input_signature(manifest, case):
            raise ValueError(
                'Candidate input signatures differ from the admitted manifest')
        check_recipe(case, directory)
        if row['artifacts'].get(
                row['checkpoint_artifact']) != row['checkpoint_sha256']:
            raise ValueError(
                'Published checkpoint artifact and restored checkpoint identity differ'
            )
        for key in ('core_fit_ledger', 'fit_ledger'):
            ledger = row['work'].get(key, [])
            if len(ledger) != 1 or ledger[0]['status'] != 'PASS' or ledger[0][
                    'steps'] <= 0 or ledger[0]['cuda_batches'] <= 0:
                raise ValueError(
                    'Core or Lightning physical-fit ledger is incomplete')
        if not all(row['predictions'].get(s, {}).get('files')
                   for s in PREDICTIONS):
            raise ValueError('Complete raw prediction artifacts are missing')
        if row['work']['restored_tensors'] <= 0 or row['work'][
                'reload_checks'] < 1:
            raise ValueError('Restored checkpoint state proof is missing')
    comparisons = []
    for row in rows:
        previous = references[(row['task'], row['arm'])]
        validate_quality_measurement(row['task'], row['score'])
        context = comparison_context(
            reference, report, previous,
            path.parent / (row['task'] + '-' + row['arm']))
        comparisons.append(
            dict(
                task=row['task'], arm=row['arm'], score=row['score'],
                baseline=previous['baseline'], metric=row['metric'],
                reference_score=None if args.reference_only else
                previous['score'], score_difference=None if args.reference_only
                or row['score'] is None else row['score'] - previous['score'],
                seconds=row['seconds'], differences=row['differences'],
                fit_id=row['fit_id'],
                checkpoint_sha256=row['checkpoint_sha256'],
                comparison_context=context,
                training_window=row['work'].get('training_window', {}),
                training_window_changes=training_window_changes(
                    row['work'].get('training_window', {}),
                    previous.get('training_window', {}),
                    compatible=context['compatible']),
                resource_changes=measurement_changes(
                    row['measurements'], previous.get('measurements', {}),
                    compatible=context['compatible']),
                operation_changes={
                    name: measurement_changes(
                        values,
                        previous.get('operation_measurements', {}).get(name, {}),
                        compatible=context['compatible'])
                    for name, values in row['operation_measurements'].items()
                },
                quality_change=reporting_change(
                    previous['score'], row['score'], 5,
                    lower_is_better=row['task'] in ('regression', 'forecasting'),
                    compatible=context['compatible'])
                if row['arm'] == 'normal' else None))
    learning = check_learning(comparisons)
    save(
        args.output_dir / 'comparison.json',
        dict(status='PASS', quality_policy='report_only',
             learning=learning, numerical_difference_policy='report_only',
             performance_policy='report_only',
             steady_state_training_status='MEASURED' if all(
                 row['training_window_changes']['eligible']
                 for row in comparisons) else 'INVALID',
             reference_only=args.reference_only,
             input_manifest_sha256=args.input_sha256,
             reference_sha256=args.reference_sha256, primary_physical_fits=16,
             seconds=report['seconds'], runtime=report['runtime'],
             cases=comparisons))
    if args.reference_only:
        cases = [
            dict(references[(row['task'], row['arm'])], score=row['score'],
                 fit_id=row['fit_id'],
                 checkpoint_sha256=row['checkpoint_sha256'],
                 seconds=row['seconds'], timings=row['timings'],
                 measurements=row['measurements'],
                 operation_measurements=row['operation_measurements'],
                 training_window=row['work'].get('training_window', {}),
                 training_work=training_work(row, read(
                     path.parent / (row['task'] + '-' + row['arm']) /
                     'trainer_config.json')),
                 trainer_config_sha256=signature(stable_trainer(read(
                     path.parent / (row['task'] + '-' + row['arm']) /
                     'trainer_config.json')))) for row in rows
        ]
        save(
            args.output_dir / 'reference.json',
            dict(
                format='kumo-engine-nightly-reference-v1', version='1.0.0',
                wheel_sha256=REFERENCE_WHEEL, quality_policy='report_only',
                input_manifest_sha256=args.input_sha256, cases=cases,
                predecessor_reference_sha256=args.reference_sha256,
                source_report_sha256=digest(path),
                runner_sha256=report['runner_sha256'],
                helper_sha256=report['helper_sha256'], reference_runtime=dict(
                    reference_hardware=hardware_profile(report['runtime']),
                    reference_dependencies_signature=signature(
                        report['runtime']['dependencies']),
                    observed=report['runtime'])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'compare'):
        p = commands.add_parser(name)
        p.add_argument('--input-root', type=Path, required=True)
        p.add_argument('--input-sha256', required=True)
        p.add_argument('--output-dir', type=Path, required=True)
        p.add_argument('--reference', type=Path, required=True)
        p.add_argument('--reference-sha256', required=True)
        p.add_argument('--reference-only', action='store_true')
        if name == 'run':
            p.add_argument('--expected-version', required=True)
        else:
            p.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    (run if args.command == 'run' else compare)(args)


if __name__ == '__main__':
    main()
