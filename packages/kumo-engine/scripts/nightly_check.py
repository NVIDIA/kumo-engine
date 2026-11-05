#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one frozen scale case or collect the declared nightly case roster."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

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
VARIANT_TASKS = dict(streaming='temporal_link', community='regression',
                     payments='binary', fraud='binary')


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


def sibling(name):
    """Load a checked-in script without depending on the caller's sys.path."""
    module_name = 'engine_' + name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).with_name(name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    return sys.modules[module_name]


def load_configuration(path):
    return sibling('nightly_job').configuration(Path(path).read_text())


def check_tier(configuration, args):
    tier = ('weekly' if configuration['format'] == 'kumo-engine-weekly-job-v2'
            else 'nightly')
    if args.tier != tier:
        raise ValueError('The requested tier differs from the configuration')
    if tier == 'weekly' and args.command == 'run':
        args.replay_reference = True


def scale_configuration(configuration, scale):
    value = configuration['scales'][scale]
    if value['inputs'] is None or value['input_manifest_sha256'] is None:
        raise ValueError(f'{scale} inputs are not frozen')
    return value


def expected_cases(configuration):
    cases = [
        dict(scale=scale, task=task, arm=arm, seed=0)
        for scale in configuration['scales'] for task in smoke.TASKS
        for arm in ('normal', 'control')
    ]
    cases += [
        dict(scale=scale, task=spec['task'], variant=name, arm=arm, seed=0)
        for scale in configuration['scales']
        for name, spec in configuration.get('variants', {}).items()
        for arm in ('normal', 'control')
    ]
    return cases


def case_name(case):
    return case.get('variant', case['task']) + '-' + case['arm']


def task_profile(configuration, case):
    if 'variant' not in case:
        return smoke.TASKS[case['task']]
    value = configuration.get('variants', {}).get(case['variant'])
    if not value or value['task'] != case['task']:
        raise ValueError('Workload variant has the wrong semantic task')
    return value


def bind_case_definition(case, configuration, descriptor):
    profile = task_profile(configuration, case)
    if descriptor.get('query') != profile['query']:
        raise ValueError('Frozen query differs from the workload definition')
    result = dict(case)
    for name in ('metric', 'top_k', 'raw_limit', 'min_hops', 'max_hops'):
        if name in profile:
            if name in result and result[name] != profile[name]:
                raise ValueError(
                    f'Frozen {name} differs from the workload definition')
            result[name] = profile[name]
    return result


def recipe_identity(configuration):
    caps = sibling('nightly_case').CAPS
    return signature(
        dict(caps, variants=configuration['variants']) if configuration.
        get('variants') else caps)


def runtime_records(path, configuration, *, mode):
    root = Path(path).resolve().parent
    records = {}
    for name in ('reference', 'policy', 'requirements'):
        binding = configuration[name]
        if 'path' not in binding:
            raise ValueError('Stage compact runtime records beside inputs.json')
        record = safe_file(root, binding['path'],
                           {binding['path']: binding['sha256']})
        records[name] = read(record)
        if mode == 'candidate' and records[name].get('status') != 'APPROVED':
            raise ValueError(f'Candidate work requires approved {name}')
    if mode == 'candidate':
        reference = records['reference']
        if (reference.get('reference_identity')
                != configuration['reference_identity']
                or reference.get('input_manifest_sha256') != {
                    scale: value['input_manifest_sha256']
                    for scale, value in configuration['scales'].items()
                } or reference.get('recipe_sha256')
                != recipe_identity(configuration)):
            raise ValueError(
                'Approved reference wheel, input, or recipe binding differs')
    return records


@contextmanager
def measure_operation(report, name):
    started = time.monotonic()
    try:
        yield
    finally:
        report.setdefault('timings', {})[name] = time.monotonic() - started


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
        raise ValueError(
            f'Frozen input is missing or its checksum differs: {name}')
    return path


def load_input_manifest(root, expected, scale='100M', configuration=None, *,
                        reference_only=False):
    """Validate the complete frozen identity without requiring payload bytes."""
    root = Path(root).resolve()
    manifest = root / 'manifest.json'
    if manifest.is_symlink() or digest(manifest) != expected:
        raise ValueError('Input manifest checksum differs')
    spec = read(manifest)
    if spec.get('format') not in ('kumo-engine-nightly-inputs-v1',
                                  'kumo-engine-nightly-inputs-v2'):
        raise ValueError('Use the admitted nightly data manifest')
    if spec.get('scale', '100M') != scale:
        raise ValueError('Frozen graph scale differs from the selected scale')
    if not isinstance(spec.get('files'), dict) or not spec['files']:
        raise ValueError('Input manifest needs exact regular-file hashes')
    transport = sibling('nightly_job')
    for name, checksum in spec['files'].items():
        if (transport.input_path(name).suffix
                not in ('.json', '.parquet', '.bin', '')
                or not isinstance(checksum, str)
                or not re.fullmatch('[0-9a-f]{64}', checksum)):
            raise ValueError('Input artifacts need data paths and exact hashes')
    configuration = configuration or dict(scales={scale: {}})
    for case in spec['cases']:
        if case.get('scale', scale) != scale:
            raise ValueError('A frozen case belongs to another scale')
        case['scale'] = scale
    identities = [
        sibling('nightly_gates').case_key(case) for case in spec['cases']
    ]
    wanted = {
        sibling('nightly_gates').case_key(case)
        for case in expected_cases(configuration) if case['scale'] == scale
    }
    if len(identities) != len(wanted) or set(identities) != wanted:
        raise ValueError(
            'Input manifest must contain every configured case at this scale')
    for case in spec['cases']:
        graph = spec['graphs'][case['graph']]
        if (graph['profile'] != task_profile(configuration, case)['graph']
                or graph['static'] is not (case['task'] == 'static_link')):
            raise ValueError('Task graph profile or static visibility differs')
        failures = case['native_expectations']['key_failures']
        if set(failures) != set(KEY_FAILURES) or any(
                type(v) is not int or v for v in failures.values()):
            raise ValueError(
                'Admitted native key failures must be exactly four integer zeros'
            )
    references = [spec['reference_dependencies']]
    for graph in spec['graphs'].values():
        references.append(graph['schema'])
        for source in graph['sources'].values():
            transport.input_path(source)
            if not any(name == source or name.startswith(source + '/')
                       for name in spec['files']):
                raise ValueError('Graph source has no declared frozen files')
    for case in spec['cases']:
        if reference_only:
            if case['arm'] == 'normal':
                references.append(case['descriptor'])
            if 'catalog' in case:
                references.append(case['catalog']['path'])
            continue
        references.extend(case[key]
                          for key in ('descriptor', 'prediction_input',
                                      'preparation_probe') if key in case)
        if 'catalog' in case:
            references.append(case['catalog']['path'])
        for key in ('normal_rows', 'control_rows'):
            values = case.get(key, [])
            if not isinstance(values, (str, list)):
                raise ValueError('Frozen answer payloads need declared paths')
            references.extend([values] if isinstance(values, str) else values)
    for name in references:
        transport.input_path(name)
        if name not in spec['files']:
            raise ValueError('Input manifest references an undeclared file')
    spec['scale'] = scale
    for case in spec['cases']:
        case['scale'] = scale
    return spec


def load_reference_inputs(root, spec, configuration, selected_case=None):
    """Admit the pinned inputs used for fresh reference generation."""
    cases = spec['cases']
    if selected_case is not None:
        chosen = [case for case in cases if case_name(case) == selected_case]
        if len(chosen) != 1:
            raise ValueError(
                'Select one configured case from the frozen roster')
        family = spec['graphs'][chosen[0]['graph']]['profile']
        cases = [
            case for case in cases
            if spec['graphs'][case['graph']]['profile'] == family
        ]
    selected_graphs = {case['graph'] for case in cases}
    required = {spec['reference_dependencies']}
    for case in cases:
        if case['arm'] == 'normal':
            required.add(case['descriptor'])
        else:
            sibling('nightly_controls').validate_recipe(
                case['task'], case.get('control_recipe'))
        if 'catalog' in case:
            required.add(case['catalog']['path'])
    for key in selected_graphs:
        graph = spec['graphs'][key]
        required.add(graph['schema'])
        for source in graph['sources'].values():
            relative = PurePosixPath(source)
            if (relative.is_absolute() or relative.as_posix() != source
                    or '\\' in source or any(part in ('', '.', '..')
                                             for part in source.split('/'))):
                raise ValueError('Graph sources must stay under the input root')
            wanted = {
                name
                for name in spec['files']
                if name == source or name.startswith(source + '/')
            }
            if not wanted:
                raise ValueError('Graph source has no declared frozen files')
            path = root / source
            paths = list(path.rglob('*')) if path.is_dir() else [path]
            if not paths or any(
                    item.is_symlink() or not (item.is_file() or item.is_dir())
                    for item in paths):
                raise ValueError('Graph sources must be regular frozen files')
            actual = {
                item.relative_to(root).as_posix()
                for item in paths if item.is_file()
            }
            if actual != wanted:
                raise ValueError(
                    'Graph contains missing or undeclared source files')
            required.update(wanted)
    for name in required:
        safe_file(root, name, spec['files'])
    for case in cases:
        if case['arm'] == 'normal':
            bind_case_definition(case, configuration,
                                 read(root / case['descriptor']))
    return spec


def load_inputs(root, expected, scale='100M', configuration=None, *,
                selected_case=None, reference_only=False):
    root = Path(root).resolve()
    spec = load_input_manifest(root, expected, scale, configuration,
                               reference_only=reference_only)
    if reference_only:
        return load_reference_inputs(root, spec, configuration
                                     or dict(scales={scale: {}}), selected_case)
    graphs = spec['graphs'].values()
    if selected_case is None:
        files = spec['files']
    else:
        transport = sibling('nightly_job')
        family = transport.workload_for_case(spec, selected_case)
        files = transport.workload_files(spec, family)
        graphs = [graph for graph in graphs if graph['profile'] == family]
    for name in files:
        safe_file(root, name, spec['files'])
    for graph in graphs:
        for source in graph['sources'].values():
            path = root / source
            paths = list(path.rglob('*')) if path.is_dir() else [path]
            if not paths or any(
                    p.is_symlink() or not (p.is_file() or p.is_dir())
                    for p in paths):
                raise ValueError('Graph sources must be regular frozen files')
            actual = {
                item.relative_to(root).as_posix()
                for item in paths if item.is_file()
            }
            wanted = {
                name
                for name in spec['files']
                if name == source or name.startswith(source + '/')
            }
            if actual != wanted:
                raise ValueError(
                    'Graph contains missing or undeclared source files')
    return spec


def check_dependencies(packages, profile):
    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    if not profile or not all(
            packages.get(normalize(k)) == v for k, v in profile.items()):
        raise ValueError(
            'Installed dependencies differ from the selected profile')


def verify_runtime(version, profile, *, reference_only=False):
    import torch

    import kumo.engine as kumo

    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
    packages = {
        normalize(d.metadata['Name']): d.version
        for d in importlib.metadata.distributions()
        if normalize(d.metadata['Name']) != 'kumo-engine'
    }
    if reference_only and profile is not None:
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
    properties = torch.cuda.get_device_properties(0)
    identifier = str(properties.uuid)
    uuid.UUID(identifier.removeprefix('GPU-'))
    if not properties.name or properties.total_memory <= 0:
        raise ValueError('CUDA GPU name or memory is missing')
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
    return dict(
        version=version, dependencies=packages, reproducibility=dict(
            protocol='product-defaults',
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(
            ),
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        gpu=properties.name, gpu_count=1, driver=driver,
        architecture=platform.machine(), compute_capability=[
            properties.major, properties.minor
        ], gpu_uuid=identifier, gpu_memory_bytes=int(properties.total_memory),
        torch=torch.__version__, cuda=torch.version.cuda,
        threads=torch.get_num_threads(), import_path=kumo.__file__,
        wheel_sha256=wheel_hash, cpu_count=os.cpu_count(),
        system_memory_bytes=os.sysconf('SC_PAGE_SIZE') *
        os.sysconf('SC_PHYS_PAGES'))


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
    ledger = state['fit_ledger']
    completed = [fit for fit in ledger if fit['status'] == 'PASS']
    if (not completed or type(state['fits']) is not int
            or state['fits'] != len(ledger)
            or state['completed_fits'] != len(completed)
            or state['optimizer_steps'] != sum(fit['steps']
                                               for fit in completed)
            or state['training_batches'] != sum(fit['cuda_batches']
                                                for fit in ledger)
            or type(selected_steps) is not int or selected_steps <= 0):
        raise ValueError(
            'Candidate counts, CUDA work or optimizer updates differ')
    for fit in ledger:
        epochs = fit['epochs']
        if (fit['status'] not in ('PASS', 'FAIL')
                or fit['cuda_batches'] != sum(e['batches'] for e in epochs)
                or any(e['batches'] != len(e['batch_sizes']) or any(
                    type(n) is not int or n <= 0 for n in e['batch_sizes'])
                       for e in epochs) or fit['status'] == 'PASS' and
            (type(fit['steps']) is not int or fit['steps'] <= 0
             or fit['cuda_batches'] <= 0 or not fit['validation'])):
            raise ValueError('Candidate training evidence is incomplete')
    if not any(
            fit.get('checkpoint_sha256') == state['checkpoint_sha256']
            and selected_steps <= fit['steps'] for fit in completed):
        raise ValueError(
            'Selected checkpoint has no completed candidate evidence')


def input_signature(manifest, case, *, prediction=None):
    return dict(
        descriptor=manifest['files'][case['descriptor']],
        prediction=(manifest['files'][case['prediction_input']]
                    if prediction is None else prediction),
        scale=case.get('scale', manifest.get('scale', '100M')),
        task=case['task'], arm=case['arm'], seed=case['seed'],
        **({
            'variant': case['variant']
        } if 'variant' in case else {}))


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


def check_snapshot(snapshot, graph_spec):
    from kumo_core.diskgraph.metadata import load_diskgraph_schema
    stats = read(Path(snapshot.index_path) / 'ingest_stats.json')['stats']
    schema = read(Path(graph_spec['input_root']) /
                  graph_spec['schema'])[graph_spec['profile']]
    # The frozen manifest binds nullable FK declarations to raw null counts.
    nullable = graph_spec.get('nullable_foreign_key_null_counts', {})
    foreign_keys = {
        f"fkey:{edge['src_table']}.{edge['fkey']}"
        for edge in schema['edges']
        if edge['fkey'] != schema['tables'][edge['src_table']].get('pkey')
        and any(column['name'] == edge['fkey']
                for column in schema['tables'][edge['src_table']]['cols'])
    }
    if not isinstance(nullable, dict) or any(
            key not in foreign_keys or type(value) is not int or value < 0
            for key, value in nullable.items()):
        raise ValueError('Invalid frozen nullable foreign-key null counts')
    contexts = stats.get('key_failures_by_context', {})
    if not isinstance(contexts, dict):
        raise ValueError('Native key failure contexts differ')
    context_counters = ('parse_failures', 'lookup_failures',
                        'membership_failures', 'non_null_parse_failures')
    for context, counts in contexts.items():
        if (context not in nullable or not isinstance(counts, dict) or any(
                type(counts.get(key)) is not int or counts[key] < 0
                for key in context_counters)
                or counts['parse_failures'] != nullable[context]
                or any(counts[key] for key in context_counters[1:])):
            raise ValueError('Native key failure contexts differ')
    native_failures = {key: stats.get(key) for key in KEY_FAILURES}
    expected_nulls = sum(nullable.values())
    if (any(
            type(value) is not int or value < 0
            for value in native_failures.values())
            or any(native_failures[key]
                   for key in KEY_FAILURES if key != 'key_parse_failures')
            or native_failures['key_parse_failures'] != expected_nulls
            or sum(counts['parse_failures']
                   for counts in contexts.values()) != expected_nulls
            or stats['node_counts'] != graph_spec['row_counts']):
        raise ValueError('Graph row counts or strict native key checks differ')
    if load_diskgraph_schema(str(snapshot.index_path)).pkey_dict != {
            name: table['pkey']
            for name, table in schema['tables'].items() if table.get('pkey')
    }:
        raise ValueError('Native graph primary-key roles differ')
    return dict(node_counts=stats['node_counts'],
                key_failures=dict(native_failures, key_parse_failures=0),
                native_key_failures=native_failures,
                native_key_failures_by_context=contexts,
                nullable_foreign_key_null_counts=nullable)


def check_install(root, runtime, *, reference_only=False, build_receipt=None,
                  reference_identity=None):
    if type(runtime.get(
            'installed_files')) is not int or runtime['installed_files'] <= 0:
        raise ValueError('Installed wheel files were not verified')
    if reference_only:
        if (not reference_identity or any(
                runtime.get(key) != reference_identity.get(key)
                for key in ('version', 'wheel_sha256'))):
            raise ValueError('Use the configured approved reference wheel')
        if build_receipt is None:
            return dict(format='reference-wheel', **reference_identity)
    if build_receipt is None:
        raise ValueError('Candidate work requires an authentic build receipt')
    build = read(build_receipt)
    if (type(runtime.get('installed_files')) is not int
            or runtime['installed_files'] <= 0
            or build['wheel_sha256'] != runtime['wheel_sha256']
            or build['version'] != runtime['version']):
        raise ValueError(
            'Selected source, installed wheel or archive identity differs')
    if any(not re.fullmatch('[0-9a-f]{40}', build.get(k, ''))
           for k in ('revision', 'tree')):
        raise ValueError('Selected source revision or tree identity differs')
    return build


def runner_identity():
    names = ('nightly_check.py', 'nightly_case.py', 'nightly_controls.py',
             'nightly_gates.py', 'nightly_predictions.py',
             'nightly_resources.py', 'nightly_job.py', 'nightly_build.py',
             'nightly_sources.py', 'nightly_evidence.py', 'nightly_host.py')
    if Path(__file__).with_name('weekly_check.py').is_file():
        names += ('weekly_check.py', )
    return signature(
        {name: digest(Path(__file__).with_name(name))
         for name in names} | {'wheel_gpu_smoke.py': digest(HELPER)})


def execution_identity(configuration, wheel):
    return dict(
        wheel_sha256=digest(wheel),
        reference_sha256=configuration['reference']['sha256'],
        policy_sha256=configuration['policy']['sha256'],
        requirements_sha256=configuration['requirements']['sha256'],
        runner_sha256=runner_identity(),
        recipe_sha256=recipe_identity(configuration), input_manifest_sha256={
            scale: item['input_manifest_sha256']
            for scale, item in configuration['scales'].items()
        })


def reference_case(reference, case):
    key = sibling('nightly_gates').case_key(case)
    rows = [
        row for row in reference.get('cases', [])
        if sibling('nightly_gates').case_key(row) == key
    ]
    if len(rows) != 1:
        raise ValueError('The case needs a complete approved reference row')
    for name in ('score', 'baseline'):
        value = rows[0].get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('The case needs a complete approved reference row')
    return rows[0]


def case_minimum(requirements, case):
    name = sibling('nightly_gates').case_label(case)
    value = requirements.get('cases', {}).get(name)
    if not isinstance(value, dict):
        raise ValueError(f'{name} needs reviewed resource minima')
    # Validate the compact record before using its time budget.
    sibling('nightly_resources').admit_resources(value, value)
    return value


def load_archive(configuration, configuration_path, reference_root):
    """Read the checksum-bound manifest; payloads are checked per selected case."""
    binding = configuration.get('archive')
    if binding is None:
        return None
    if reference_root is None:
        raise ValueError('Extract the bound reference archive before using it')
    root = Path(reference_root)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError('Use a regular absolute reference archive directory')
    if 'path' in binding:
        metadata_root = Path(configuration_path).resolve().parent
        path = safe_file(metadata_root, binding['path'],
                         {binding['path']: binding['sha256']})
    else:
        path = safe_file(root, 'manifest.json',
                         {'manifest.json': binding['sha256']})
    value = read(path)
    if (value.get('format') != 'kumo-engine-nightly-archive-v1'
            or value.get('reference_identity')
            != configuration['reference_identity']
            or value.get('input_manifest_sha256') != {
                scale: item['input_manifest_sha256']
                for scale, item in configuration['scales'].items()
            } or value.get('recipe_sha256') != recipe_identity(configuration)
            or not isinstance(value.get('files'), dict)
            or not isinstance(value.get('cases'), list)):
        raise ValueError(
            'Reference archive wheel, input, recipe, or schema differs')
    return value


def archived_case(archive, case, reference, root, *, replay=False):
    if archive is None:
        raise ValueError(
            'Candidate work needs archived training-table identities')
    key = sibling('nightly_gates').case_key(case)
    rows = [
        row for row in archive['cases']
        if sibling('nightly_gates').case_key(row) == key
    ]
    if len(rows) != 1 or not isinstance(rows[0].get('training_table'), dict):
        raise ValueError(
            'The archive needs one complete case and its stage identities')
    result = dict(reference, training_table=rows[0]['training_table'])
    if replay:
        payload = rows[0].get('replay')
        if not isinstance(payload, dict):
            raise ValueError(
                'The requested reference replay has no archived model')
        for name in ('population', 'predictions'):
            item = payload[name]
            safe_file(root, item['path'], archive['files'])
            if archive['files'][item['path']] != item['sha256']:
                raise ValueError(
                    'Replay payload binding differs from the archive')
        model = payload['model']
        name = model.get('path')
        files = model.get('files')
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/'))
                or not isinstance(files, dict) or not files):
            raise ValueError(
                'Replay needs a regular model directory and nonempty file map')
        directory = Path(root) / name
        if (directory.is_symlink() or not directory.is_dir()
                or any(path.is_symlink() for path in directory.rglob('*')) or {
                    path.relative_to(directory).as_posix()
                    for path in directory.rglob('*') if path.is_file()
                } != set(files)):
            raise ValueError(
                'Replay model contains missing, extra, or linked files')
        for name, checksum in model['files'].items():
            path = str(PurePosixPath(model['path']) / name)
            safe_file(root, path, archive['files'])
            if archive['files'][path] != checksum:
                raise ValueError(
                    'Replay model binding differs from the archive')
        result['replay'] = payload
    return result


def _worker(args, configuration, records, initial, scope):
    """Run all measured work after the supervisor places this process in scope."""
    resources_api = sibling('nightly_resources')
    resources_api.join_cgroup(scope)
    row = dict(initial)
    runtime = dict(gpu='pending', gpu_count=1)
    resources = None
    try:
        with resources_api.track_resources(args.output_dir, args.scratch,
                                           runtime, scope) as resources:
            scale = scale_configuration(configuration, args.scale)
            case = dict(initial)
            profile = None
            if args.mode == 'reference':
                header = load_input_manifest(args.input_root,
                                             scale['input_manifest_sha256'],
                                             args.scale, configuration,
                                             reference_only=True)
                profile = read(
                    safe_file(args.input_root, header['reference_dependencies'],
                              header['files']))
            version = (configuration['reference_identity']['version']
                       if args.mode == 'reference' else args.expected_version
                       or read(args.build_receipt)['version'])
            runtime.update(
                verify_runtime(version, profile,
                               reference_only=args.mode == 'reference'))
            runtime['installed_files'] = verify_installed_files(
                args.wheel, importlib.metadata.distribution('kumo-engine'))
            if digest(args.wheel) != runtime['wheel_sha256']:
                raise ValueError('Selected wheel and installed archive differ')
            build = check_install(
                None, runtime, reference_only=args.mode == 'reference',
                build_receipt=args.build_receipt,
                reference_identity=configuration['reference_identity'])
            capacity = resources_api.available_capacity(
                args.scratch, runtime, args.time_budget_seconds)
            previous = None
            if args.mode == 'candidate':
                resources_api.admit_resources(
                    case_minimum(records['requirements'], case), capacity)
                previous = reference_case(records['reference'], case)
                archive = load_archive(configuration, args.configuration,
                                       args.reference_root)
                previous = archived_case(archive, case, previous,
                                         args.reference_root)
                if args.replay_reference:
                    try:
                        previous = archived_case(archive, case, previous,
                                                 args.reference_root,
                                                 replay=True)
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        previous[
                            'replay_error'] = f'{type(error).__name__}: {error}'
            row.update(runtime=runtime, build=build,
                       admission=dict(capacity=capacity))
            save(args.output_dir / 'result.json', row)
            manifest = load_inputs(
                args.input_root, scale['input_manifest_sha256'], args.scale,
                configuration,
                selected_case=(args.case if args.mode == 'reference'
                               or 'workloads' in scale
                               or 'by_graph' in scale['inputs'] else None),
                reference_only=args.mode == 'reference')
            case = next(c for c in manifest['cases']
                        if args.case == case_name(c))
            if args.mode == 'reference':
                normal = sibling('nightly_case').normal_case(manifest, case)
                case = dict(case, descriptor=normal['descriptor'])
            case = bind_case_definition(
                case, configuration,
                read(
                    safe_file(args.input_root, case['descriptor'],
                              manifest['files'])))
            if args.mode == 'candidate' and args.replay_reference:
                case = dict(
                    case, **{
                        key: previous[key]
                        for key in ('replay', 'replay_error') if key in previous
                    })
            import kumo.engine as kumo
            graph_spec = manifest['graphs'][case['graph']]
            with measure_operation(row, 'graph'):
                kumo.configure(scratch_dir=args.scratch / 'graph')
                graph, snapshot = build_graph(
                    read(args.input_root /
                         graph_spec['schema'])[graph_spec['profile']],
                    args.input_root, static=graph_spec['static'],
                    sources=graph_spec['sources'])
                row['snapshot'] = check_snapshot(
                    snapshot, dict(graph_spec, input_root=str(args.input_root)))
            tolerance = records['policy']['predictions'][args.scale][case.get(
                'variant', case['task'])]
            result = sibling('nightly_case').run_case(
                args.input_root, manifest, case, graph, snapshot,
                args.output_dir / 'case', args.scratch, tolerance,
                helpers=SimpleNamespace(**globals()),
                measure_only=args.mode == 'reference', reference=previous,
                reference_root=args.reference_root,
                reference_destination=getattr(args, 'reference_output_s3',
                                              None),
                reference_context=row['bindings'])
            row.update(result)
            row.update({
                name: initial[name]
                for name in ('scale', 'task', 'arm', 'seed', 'execution_id',
                             'mode', 'variant') if name in initial
            })
            row['bindings'] = dict(initial['bindings'],
                                   plan_sha256=row['requested_plan_sha256'])
            limits = records['policy']['learning'][args.scale][case.get(
                'variant', case['task'])]
            row['learning'] = sibling('nightly_gates').case_learning(
                row, previous or dict(baseline=row['baseline']), limits,
                tier=args.tier)
            if row.get('training_coverage', {}).get('status') != 'PASS':
                raise ValueError(
                    'Actual model-trial or epoch coverage is insufficient')
            if args.mode == 'candidate' and any(
                    row.get(name, {}).get('status') != 'PASS'
                    for name in ('learning', 'prediction_checks')):
                raise ValueError('Learning or prediction checks failed')
            if row.get('prediction_checks',
                       {}).get('status') not in ('PASS', 'MEASURED'):
                raise ValueError(
                    'Prediction structure or consistency evidence is incomplete'
                )
            if row.get('reference_replay', {}).get('status') == 'FAIL':
                raise ValueError('Reference replay failed')
            row['status'] = 'MEASURED' if args.mode == 'reference' else 'PASS'
            save(args.output_dir / 'result.json', row)
    except BaseException as error:
        case_result = args.output_dir / 'case/result.json'
        if case_result.is_file():
            row.update(read(case_result))
        row.update(status='FAIL', error=f'{type(error).__name__}: {error}',
                   traceback=traceback.format_exc())
    finally:
        if resources is not None:
            resources['scope'] = dict(
                included=[
                    'Engine import', 'input checks', 'fresh snapshot',
                    'training-table generation', 'optional model replay',
                    'training', 'prediction checks', 'model reload',
                    'case artifacts'
                ], excluded=[
                    'suite wheel build', 'shared input download and extraction'
                ])
            resources.update(
                task=row['task'], arm=row['arm'], scale=args.scale,
                wheel_sha256=runtime.get('wheel_sha256'),
                input_manifest_sha256=configuration['scales'][args.scale]
                ['input_manifest_sha256'], harness_sha256=runner_identity())
            if 'variant' in row:
                resources['variant'] = row['variant']
            row['resources'] = resources
            if resources['errors']:
                row['status'] = 'FAIL'
                row['resource_error'] = 'Resource measurement is incomplete'
                row.setdefault('error', row['resource_error'])
            save(args.output_dir / 'resources.json', resources)
        save(args.output_dir / 'result.json', row)
    return row


def run(args):
    """Supervise exactly one independent case and retain failures before work starts."""
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_name, arm = args.case.rsplit('-', 1)
    task = VARIANT_TASKS.get(selected_name, selected_name)
    row = dict(
        format='kumo-engine-nightly-case-v2', status='RUNNING',
        scale=args.scale, task=task, arm=arm, seed=0, mode=args.mode,
        execution_id=os.environ.get('NIGHTLY_EXECUTION_ID',
                                    uuid.uuid4().hex),
        measurement_only=args.mode == 'reference')
    if selected_name in VARIANT_TASKS:
        row['variant'] = selected_name
    try:
        for field in ('input_root', 'scratch', 'output_dir'):
            setattr(args, field, getattr(args, field).resolve())
        if any(
                args.scratch.is_relative_to(p) or p.is_relative_to(args.scratch)
                for p in (args.input_root, args.output_dir)):
            raise ValueError('Keep scratch outside input and evidence roots')
        if (args.output_dir.is_relative_to(args.input_root)
                or args.input_root.is_relative_to(args.output_dir)):
            raise ValueError('Keep input and evidence roots separate')
        configuration = load_configuration(args.configuration)
        check_tier(configuration, args)
        if not any(
                case_name(case) == args.case and case['scale'] == args.scale
                for case in expected_cases(configuration)):
            raise ValueError(
                'Select a case declared in the nightly configuration')
        scale_configuration(configuration, args.scale)
        row['frozen_inputs'] = sibling('nightly_host').require_frozen_inputs(
            args.input_root)
        if getattr(args, 'require_local_ssd', False):
            row['storage'] = sibling('nightly_host').require_local_ssd(
                args.scratch)
        records = runtime_records(args.configuration, configuration,
                                  mode=args.mode)
        sibling('nightly_gates').validate_policy(records['policy'],
                                                 expected_cases(configuration))
        if args.mode == 'candidate':
            limits = records['policy']['learning'][args.scale][selected_name]
            tolerance = records['policy']['predictions'][
                args.scale][selected_name]
            if (any(value is None
                    for value in limits.values()) or set(tolerance) != {'atol'}
                    or type(tolerance['atol']) not in (int, float)
                    or not math.isfinite(tolerance['atol'])
                    or tolerance['atol'] < 0):
                raise ValueError(
                    'Candidate work needs reviewed learning and prediction calibration'
                )
            minimum = case_minimum(records['requirements'], row)
            args.time_budget_seconds = args.time_budget_seconds or minimum[
                'time_seconds']
            if args.build_receipt is None:
                raise ValueError('Candidate work requires a build receipt')
            reference_case(records['reference'], row)
        row['bindings'] = execution_identity(configuration, args.wheel)
        row['bindings']['input_manifest_sha256'] = configuration['scales'][
            args.scale]['input_manifest_sha256']
        scope = os.environ.get('NIGHTLY_CASE_CGROUP')
        if scope:
            args.scratch.mkdir(parents=True, exist_ok=False)
            return _worker(args, configuration, records, row,
                           scope)['status'] != 'FAIL'
        resources_api = sibling('nightly_resources')
        environment = dict(os.environ, NIGHTLY_EXECUTION_ID=row['execution_id'])
        started = time.monotonic()
        with resources_api.dedicated_cgroup(args.cgroup_root) as group:
            command = [
                sys.executable, '-B',
                str(Path(__file__).resolve()), *args.worker_arguments
            ]
            completed = False
            worker_error = None
            try:
                process = resources_api.run_with_deadline(
                    command, group, args.time_budget_seconds, env=environment)
                if process.returncode != 0:
                    raise ValueError(
                        f'Case worker exited with code {process.returncode}')
                completed = True
            except BaseException as error:
                worker_error = f'{type(error).__name__}: {error}'
                raise
            finally:
                if (args.output_dir / 'result.json').is_file():
                    row = read(args.output_dir / 'result.json')
                measured = resources_api.finalize_resources(
                    row.get('resources'), args.output_dir, args.scratch, group,
                    time.monotonic() - started, completed=completed
                    and row['status'] in ('PASS', 'MEASURED'),
                    error=worker_error)
                row['resources'] = measured
                if measured['status'] != 'MEASURED':
                    row['status'] = 'FAIL'
                    row.setdefault(
                        'error', worker_error
                        or 'Complete resource measurements are missing')
                    if worker_error:
                        row['supervisor_error'] = worker_error
                elif args.mode == 'reference' and row['status'] == 'MEASURED':
                    proposal = resources_api.resource_requirements(measured)
                    save(args.output_dir / 'requirements-proposal.json',
                         proposal)
                save(args.output_dir / 'resources.json', measured)
        if row['status'] not in ('PASS', 'MEASURED'):
            raise ValueError(
                'The case worker did not produce a complete result')
    except BaseException as error:
        detail = f'{type(error).__name__}: {error}'
        if row.get('error'):
            row['supervisor_error'] = detail
            row['supervisor_traceback'] = traceback.format_exc()
        else:
            row.update(error=detail, traceback=traceback.format_exc())
        row['status'] = 'FAIL'
    save(args.output_dir / 'result.json', row)
    return row['status'] in ('PASS', 'MEASURED')


def combine(args):
    output = args.output_dir / 'comparison.json'
    report = None
    try:
        configuration = load_configuration(args.configuration)
        check_tier(configuration, args)
        records = runtime_records(args.configuration, configuration,
                                  mode=args.mode)
        identity = execution_identity(configuration, args.wheel)
        report = sibling('nightly_gates').combine(
            args.results, expected_cases=expected_cases(configuration),
            reference=records['reference'], policy=records['policy'],
            identity=identity, output=output,
            measurement=args.mode == 'reference', tier=args.tier,
            canonical_executions=read(args.canonical_executions)
            if args.canonical_executions else None)
        if args.mode == 'reference':
            provenance = dict(
                status='PROPOSAL_NOT_APPROVED',
                reference_identity=configuration['reference_identity'],
                input_manifest_sha256=identity['input_manifest_sha256'],
                recipe_sha256=identity['recipe_sha256'])
            proposal = report.get('reference_proposal', dict(cases=[]))
            proposal.update(provenance,
                            format='kumo-engine-nightly-reference-v2')
            report['reference_proposal'] = proposal
            save(args.output_dir / 'reference-proposal.json', proposal)
            requirements = dict(provenance,
                                format='kumo-engine-nightly-requirements-v2',
                                headroom=1.25, cases={}, executions={})
            for row in report['cases'] + report['repeats']:
                measured = row.get('resources', {})
                if measured.get('status') != 'MEASURED':
                    continue
                minimum = sibling('nightly_resources').resource_requirements(
                    measured)['minimum']
                name = sibling('nightly_gates').case_label(row)
                previous = requirements['cases'].setdefault(name, minimum)
                requirements['cases'][name] = {
                    key: max(previous[key], value)
                    for key, value in minimum.items()
                }
                requirements['executions'].setdefault(name, []).append(
                    row['execution_id'])
            save(args.output_dir / 'requirements-proposal.json', requirements)
            archive = sibling('nightly_gates').archive_proposal(
                report['cases'],
                reference_identity=configuration['reference_identity'],
                input_manifest_sha256=identity['input_manifest_sha256'],
                recipe_sha256=identity['recipe_sha256'], campaign=report,
                expected_cases=expected_cases(configuration))
            save(args.output_dir / 'archive-proposal.json', archive)
            save(output, report)
    except (OSError, KeyError, TypeError, ValueError) as error:
        if report is None:
            report = dict(cases=[], errors=[])
        report.update(status='FAIL', qualification='NOT_QUALIFIED')
        report['errors'].append(f'{type(error).__name__}: {error}')
        save(output, report)
    return report['status'] in ('PASS', 'MEASURED')


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'combine'):
        command = commands.add_parser(name)
        command.add_argument('--configuration', type=Path, required=True)
        command.add_argument('--output-dir', type=Path, required=True)
        command.add_argument('--wheel', type=Path, required=True)
        command.add_argument('--mode', choices=('reference', 'candidate'),
                             required=True)
        command.add_argument('--tier', choices=('nightly', 'weekly'),
                             default='nightly')
        if name == 'run':
            command.add_argument('--scale',
                                 choices=('100M', '1B', '10B', '25B'),
                                 required=True)
            command.add_argument(
                '--case', choices=[
                    task + '-' + arm for task in (*smoke.TASKS, *VARIANT_TASKS)
                    for arm in ('normal', 'control')
                ], required=True)
            command.add_argument('--input-root', type=Path, required=True)
            command.add_argument('--scratch', type=Path, required=True)
            command.add_argument('--cgroup-root', type=Path, required=True)
            command.add_argument('--time-budget-seconds', type=int)
            command.add_argument('--build-receipt', type=Path)
            command.add_argument('--expected-version')
            command.add_argument('--reference-root', type=Path)
            command.add_argument('--replay-reference', action='store_true')
            command.add_argument('--reference-output-s3')
            command.set_defaults(require_local_ssd=True)
        else:
            command.add_argument('--results', type=Path, nargs='+',
                                 required=True)
            command.add_argument('--canonical-executions', type=Path)
    args = parser.parse_args(arguments)
    if args.command == 'run':
        if args.time_budget_seconds is not None and args.time_budget_seconds <= 0:
            parser.error('--time-budget-seconds must be positive')
        if args.mode == 'reference' and args.time_budget_seconds is None:
            parser.error('Reference measurement needs --time-budget-seconds')
        if args.mode == 'reference' and not args.reference_output_s3:
            parser.error('Reference creation needs --reference-output-s3')
        args.worker_arguments = arguments
    try:
        return 0 if (
            run(args) if args.command == 'run' else combine(args)) else 1
    except (OSError, TypeError, KeyError, ValueError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


# Keep shared helpers available to the case runner and focused CPU tests.
ordered_batches = sibling('nightly_predictions').ordered_batches
ranked_rows = sibling('nightly_predictions').ranked_rows
ranked_difference = sibling('nightly_predictions').ranked_difference
quality = sibling('nightly_predictions').quality
validate_quality_measurement = sibling('nightly_gates').validate_score
case_learning = sibling('nightly_gates').case_learning
check_learning = sibling('nightly_gates').check_learning
available_capacity = sibling('nightly_resources').available_capacity
admit_resources = sibling('nightly_resources').admit_resources
resource_requirements = sibling('nightly_resources').resource_requirements


def observe_cuda(state):
    return sibling('nightly_case').observe_cuda(
        state, helpers=SimpleNamespace(**globals()))


if __name__ == '__main__':
    raise SystemExit(main())
