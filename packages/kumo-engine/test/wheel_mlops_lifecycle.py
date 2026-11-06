#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the local MLOps lifecycle against an installed candidate wheel.

Run with -I in a clean virtual environment containing kumo-engine[mlflow].
Each invocation needs a new output directory; failed attempts are retained.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from contextlib import nullcontext
from functools import cache
from pathlib import Path

QUERY = 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH users.ID'
MODEL = 'kumo-local-lifecycle'
EXPERIMENT = 'kumo-local-lifecycle'
POPULATION = 60
PHASES = ('fit', 'score', 'evaluate', 'negative')
SCHEMA = ['ENTITY', 'TARGET_PRED', 'False_PROB', 'True_PROB']


class PrerequisiteError(RuntimeError):
    """A required execution environment is unavailable."""


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@cache
def gpu_helpers():
    """Reuse the existing eight-task runner's binary model recipe."""
    path = Path(__file__).with_name('wheel_gpu_smoke.py')
    spec = importlib.util.spec_from_file_location('engine_gpu_smoke', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the task helper at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def platform_helpers():
    path = Path(__file__).with_name('wheel_mlops_platform.py')
    spec = importlib.util.spec_from_file_location('mlops_platform', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the platform helper at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def platform_backend(root):
    if not (root / 'platform.json').is_file():
        return None
    return platform_helpers().backend(root)


def model_name(root):
    profile = platform_backend(root)
    return MODEL if profile is None else profile.model


def accelerator(root):
    profile = platform_backend(root)
    return 'cpu' if profile is None else profile.accelerator


def set_alias(root, cfg, version):
    profile = platform_backend(root)
    if profile is not None:
        return profile.set_alias(cfg, version)
    with cfg.tracking.client() as client:
        client.set_registered_model_alias(MODEL, 'champion', version)


def alias_version(root, cfg):
    profile = platform_backend(root)
    if profile is not None:
        return profile.alias(cfg)
    with cfg.tracking.client() as client:
        return str(client.get_model_version_by_alias(MODEL, 'champion').version)


def training_rows(root, table, name):
    import pandas as pd

    import kumo.engine as kumo

    directory = root / name
    directory.mkdir()
    table.export(
        kumo.TrainingTableOutputConfig(
            kumo.FileConnector(root_dir=str(directory)), 'rows/'))
    frame = pd.read_parquet(directory / 'rows')
    frame = frame.sort_values(list(frame)).reset_index(drop=True)
    payload = json.dumps(list(frame)).encode() + pd.util.hash_pandas_object(
        frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def check_install(wheel, expected_version, *, managed=False):
    """Bind execution to a non-editable wheel in an isolated environment."""
    import diskgraph
    import kumo_core

    import kumo.engine as kumo

    require(sys.platform == 'linux', 'This initial profile requires Linux CPU')
    if not managed:
        require(sys.prefix != sys.base_prefix,
                'Use a clean virtual environment')
        config = Path(sys.prefix) / 'pyvenv.cfg'
        require('include-system-site-packages = false' in config.read_text(),
                'System site packages are not allowed for qualification')
    dist = importlib.metadata.distribution('kumo-engine')
    require(dist.version == expected_version == kumo.__version__,
            'Installed Engine version differs from the selected wheel')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    wheel_hash = digest(wheel)
    require(
        direct.get('archive_info', {}).get('hashes',
                                           {}).get('sha256') == wheel_hash,
        'Install the exact candidate wheel with pip')
    require(not direct.get('dir_info', {}).get('editable'),
            'Editable installations are not qualification')
    imports = {}
    import diskgraph._c as native

    modules = (kumo, kumo_core, diskgraph, native)
    location = Path(dist.locate_file('')).resolve()
    files = {
        Path(dist.locate_file(entry)).resolve()
        for entry in (dist.files or [])
    }
    for module in modules:
        path = Path(module.__file__).resolve()
        if not managed:
            require(path.is_relative_to(Path(sys.prefix).resolve()),
                    f'{module.__name__} imports outside the environment')
        require(
            path.is_relative_to(location),
            f'{module.__name__} imports outside the wheel location: {path}')
        require(path in files,
                f'{module.__name__} is not supplied by the candidate wheel')
        verify_wheel_file(wheel, location, path)
        imports[module.__name__] = str(path)
    verify_wheel_file(wheel, location,
                      Path(diskgraph.__file__).parent / 'bin' / 'ingest')
    return dict(wheel=str(wheel), sha256=wheel_hash, version=dist.version,
                python=sys.version, executable=sys.executable, imports=imports)


def verify_wheel_file(wheel, location, path):
    relative = path.resolve().relative_to(location).as_posix()
    with zipfile.ZipFile(wheel) as archive:
        with archive.open(relative) as stream:
            expected = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(
        digest(path) == expected,
        f'Installed file differs from the selected wheel: {relative}')


def settings(root, *, register=False):
    profile = platform_backend(root)
    if profile is not None:
        return profile.settings(register)
    import mlflow

    import kumo.engine as kumo
    from kumo.engine.integrations.mlflow import MlflowTracking

    uri = f'sqlite:///{root / "tracking.db"}'
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return kumo.MLOps(
        str(root / 'store'), tracking=MlflowTracking(EXPERIMENT,
                                                     log_system_metrics=False),
        registry=(kumo.ModelRegistryLocation(MODEL) if register else None),
        on_error='raise')


def graph(root, cutoff):
    """Materialize only the events available at this snapshot's cutoff."""
    import numpy as np
    import pandas as pd
    from kumoapi.typing import Stype

    import kumo.engine as kumo

    data = root / f'data-{cutoff}'
    data.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({
        'ID': np.arange(POPULATION)
    }).to_parquet(data / 'users.parquet')
    events = pd.DataFrame({
        'EID':
        np.arange(1500),
        'USER_ID':
        rng.integers(0, POPULATION, 1500),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 1500), unit='D'),
    })
    events[events.TS < pd.Timestamp(cutoff)].to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    return kumo.Graph(
        tables={
            'users':
            kumo.Table.from_source_table(connector['users.parquet'],
                                         primary_key='ID'),
            'events':
            kumo.Table.from_source_table(connector['events.parquet'],
                                         primary_key='EID',
                                         time_column='TS').with_column(
                                             'USER_ID', stype=Stype.ID),
        }, edges=[
            kumo.Edge(src_table='events', fkey='USER_ID', dst_table='users')
        ])


def population(run):
    import numpy as np

    frame = run.frame.sort_values('ENTITY').reset_index(drop=True)
    require(list(frame) == SCHEMA, 'Default binary prediction schema changed')
    require(frame.ENTITY.tolist() == list(range(POPULATION)),
            'Prediction population is missing, duplicated, or foreign')
    require((run.requested_rows, run.scored_rows,
             run.skipped_rows) == (POPULATION, POPULATION, 0),
            'Incorrect population accounting')
    scores = frame[['False_PROB', 'True_PROB']].to_numpy(dtype=float)
    require(
        np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
        'Invalid prediction probabilities')
    require(np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0),
            'Prediction probabilities do not sum to one')
    require(np.array_equal(frame.TARGET_PRED, frame.True_PROB > 0.5),
            'Labels disagree with the configured threshold')
    require(
        len(run.manifest.prediction_table.anchors) == 1,
        'Expected one prediction time anchor')
    return frame


def prediction_tracking(cfg, run, expected, *, root=None):
    profile = platform_backend(root) if root is not None else None
    if profile is not None:
        return platform_helpers().prediction_record(profile, cfg, run, expected)
    require(run.tracking is not None, 'Prediction not tracked')
    with cfg.tracking.client() as client:
        record = client.get_run(run.tracking.run_id)
        require(record.info.status == 'FINISHED',
                'Prediction tracking unfinished')
        tags = record.data.tags
        require(
            all(
                tags.get(key) == value for key, value in {
                    'kumo.model_name': MODEL,
                    'kumo.model_version': expected['version'],
                    'kumo.bundle_id': expected['bundle'],
                    'kumo.prediction_id': run.id,
                }.items()),
            'Prediction tracking records the wrong model/run identity')


def fit(root):
    import kumo.engine as kumo

    cfg = settings(root, register=True)
    profile = platform_backend(root)
    with kumo.mlops(cfg):
        training_graph = graph(root, '2024-03-01')
        snapshot = training_graph.snapshot()
        query = kumo.PredictiveQuery(graph=training_graph, query=QUERY)
        table = query.generate_training_table(snapshot)
        rows_sha256 = training_rows(root, table, 'original-training-table')
        scored = graph(root, '2024-03-31').snapshot()
        versions = []
        for seed in (0, 1):
            plan = gpu_helpers().make_plan('binary')
            plan.training_job.manual_seed = seed
            plan.optimization.max_epochs = 2
            plan.optimization.min_steps_per_epoch = 2
            plan.optimization.max_steps_per_epoch = 2
            plan.optimization.batch_size = [32]
            trainer = kumo.Trainer(plan, accelerator=accelerator(root))
            result = trainer.fit(
                table, snapshot=snapshot, scoring_defaults=kumo.ScoringDefaults(
                    binary_classification_threshold=0.5))
            require(result.registered_model is not None, 'Model not registered')
            require(result.tracking is not None, 'Fit not tracked')
            parent = result.manifest.parents.training_table
            require(
                parent.id == table.id
                and parent.parents.snapshot.id == snapshot.id,
                'Training lineage is incorrect')
            reference = None
            if profile is not None:
                reference = platform_helpers().fit_record(profile, cfg, result)
                history = reference['history']
            else:
                with cfg.tracking.client() as client:
                    run = client.get_run(result.tracking.run_id)
                    require(run.info.status == 'FINISHED',
                            'Fit tracking unfinished')
                    history = {
                        name: [(p.step, p.value)
                               for p in client.get_metric_history(
                                   run.info.run_id, name)]
                        for name in ('train/loss', 'val/auroc')
                    }
                    require(
                        all(
                            len({p[0]
                                 for p in points}) >= 2
                            for points in history.values()),
                        'Expected at least two tracked epochs')
                    registered = client.get_model_version(
                        MODEL, result.registered_model.version)
                    require(registered.tags['kumo_bundle_id'] == result.id,
                            'Registry version points to another bundle')
            versions.append(
                dict(
                    version=result.registered_model.version, bundle=result.id,
                    run=result.tracking.run_id, history=history,
                    reference=reference, exported_artifacts=read(
                        result.bundle_dir /
                        'publication-manifest.json')['artifacts']))
        require(versions[0]['version'] != versions[1]['version'],
                'Expected two distinct registered versions')
        require(versions[0]['bundle'] != versions[1]['bundle'],
                'Expected two independently fitted bundles')
    set_alias(root, cfg, versions[0]['version'])
    return dict(training_snapshot=snapshot.id, scoring_snapshot=scored.id,
                training_table=table.id, training_rows_sha256=rows_sha256,
                model=model_name(root), versions=versions)


def load(root, *, version=None, alias=None):
    import kumo.engine as kumo

    return kumo.Trainer.load_from_connector(
        settings(root),
        location=kumo.ModelRegistryLocation(model_name(root), version=version,
                                            alias=alias),
        accelerator=accelerator(root))


def score(root):
    import pandas as pd

    import kumo.engine as kumo

    state = read(root / 'fit.json')['result']
    cfg = settings(root)
    snapshot = kumo.GraphSnapshot.load_from_connector(
        cfg, id=state['scoring_snapshot'])
    training = kumo.GraphSnapshot.load_from_connector(
        cfg, id=state['training_snapshot'])
    require(snapshot.id != training.id, 'Scoring must use a distinct snapshot')
    # Loading an exported TT gives it a new source identity, but preserves rows.
    restored = kumo.TrainingTable.load_from_connector(
        cfg, path=f'training_tables/{state["training_table"]}',
        pquery=kumo.PredictiveQuery(graph=training,
                                    query=QUERY), snapshot=training)
    require(restored.manifest.parents.snapshot.id == training.id,
            'Restored training table has the wrong snapshot')
    restored_hash = training_rows(root, restored, 'restored-training-table')
    require(restored_hash == state['training_rows_sha256'],
            'Restored training rows changed')
    # Also verify the artifact-root model copy, independently of the registry.
    for expected in state['versions']:
        stored = kumo.Trainer.load_from_connector(
            cfg, path=f'models/{expected["bundle"]}',
            accelerator=accelerator(root))
        require(stored.result.id == expected['bundle'],
                'Stored model copy changed')
    results = []
    for expected in state['versions']:
        trainer = load(root, version=expected['version'])
        require(trainer.result.id == expected['bundle'],
                'Explicit version loaded the wrong bundle')
        run = trainer.predict(snapshot, mlops=cfg)
        frame = population(run)
        lineage = run.manifest.parents
        require(
            lineage.model.id == expected['bundle'] and
            lineage.model.parents.training_table.id == state['training_table']
            and lineage.model.parents.training_table.parents.snapshot.id
            == training.id and lineage.snapshot.id == snapshot.id,
            'Training and scoring lineage are conflated')
        require(
            lineage.snapshot.parents.source_tables
            != training.manifest.parents.source_tables,
            'Scoring source-table lineage did not change')
        reference = prediction_tracking(cfg, run, expected, root=root)
        results.append(
            dict(prediction=run.id, bundle=expected['bundle'],
                 version=expected['version'], reference=reference))

    aliases = []
    for index, expected in enumerate(state['versions']):
        if index:
            set_alias(root, cfg, expected['version'])
        require(
            alias_version(root, cfg) == expected['version'],
            'Alias update did not resolve to the intended version')
        trainer = load(root, alias='champion')
        require(
            trainer.result.id == expected['bundle']
            and trainer.result.registered_model.version == expected['version'],
            'Alias loaded the wrong version')
        run = trainer.predict(snapshot, mlops=cfg)
        frame = population(run)
        reference = prediction_tracking(cfg, run, expected, root=root)
        aliases.append(
            dict(prediction=run.id, bundle=expected['bundle'],
                 version=expected['version'], reference=reference))

    # Outside an MLOps context, the same public predict call stays untracked.
    plain = trainer.predict(snapshot)
    baseline = population(plain)
    require(plain.tracking is None, 'MLOps leaked into an ordinary prediction')
    require(baseline.dtypes.equals(frame.dtypes),
            'MLOps changed the ordinary prediction schema')

    # Real public-API negative case: unknown IDs must be accounted for.
    inputs = root / 'unknown-inputs'
    inputs.mkdir()
    path = inputs / 'rows.parquet'
    pd.DataFrame({
        'ENTITY1': [0, 10**9],
        'TIMESTAMP':
        [pd.Timestamp(run.manifest.prediction_table.anchors[0])] * 2
    }).to_parquet(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        partial = trainer.predict(
            snapshot,
            entities=kumo.FileConnector(root_dir=str(inputs))[path.name],
            mlops=cfg)
    require(any('skipped 1 of 2' in str(w.message) for w in caught),
            'Missing diagnostic for unknown prediction ID')
    require((partial.requested_rows, partial.scored_rows, partial.skipped_rows)
            == (2, 1, 1) and partial.frame.ENTITY.tolist() == [0],
            'Unknown ID disappeared without correct accounting')
    require(partial.manifest.prediction_table.skipped_sample == [10**9],
            'Wrong skipped prediction ID')
    return dict(explicit=results, aliases=aliases, unknown_id='passed',
                schema_compatibility='passed')


def evaluate(root):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    import kumo.engine as kumo

    cfg = settings(root)
    fitted = read(root / 'fit.json')['result']
    scored = read(root / 'score.json')['result']
    labels = graph(root, '2024-05-01').snapshot(mlops=cfg)
    checks = []
    for expected in scored['aliases']:
        trainer = load(root, version=expected['version'])
        run = kumo.PredictionResult.load_from_connector(
            cfg, path=f'predictions/{expected["prediction"]}')
        frame = population(run)
        require(
            run.bundle_id == expected['bundle']
            and run.snapshot_id == fitted['scoring_snapshot'],
            'Saved prediction identity changed after moving the alias')
        drift = trainer.score_drift(run, previous=run, mlops=cfg)
        require((drift.prediction_id, drift.bundle_id)
                == (run.id, expected['bundle']) and drift.bundle_matches,
                'Drift refers to another prediction/model')
        require(not drift.reasons,
                f'Drift calculation incomplete: {drift.reasons}')
        require(drift.previous_prediction_id == run.id,
                'Drift compared the wrong previous prediction')
        require(
            all(value is not None and math.isfinite(value)
                for value in (drift.psi, drift.ks, drift.wasserstein)),
            f'Holdout drift is unavailable: {drift.reasons}')
        require((drift.previous_psi, drift.previous_ks,
                 drift.previous_wasserstein) == (0, 0, 0),
                'A saved prediction must have zero drift against itself')
        result = trainer.evaluate(run, labels_snapshot=labels, mlops=cfg)
        require(not result.reasons,
                f'Evaluation calculation incomplete: {result.reasons}')
        require((result.prediction_id, result.bundle_id,
                 result.snapshot_id) == (run.id, expected['bundle'], labels.id),
                'Evaluation refers to another prediction/model/label snapshot')
        require(
            result.matured and result.bundle_matches
            and result.rows == result.labelled == POPULATION
            and result.labelled_fraction == 1, f'Incomplete labels: {result}')
        events = pd.read_parquet(root / 'data-2024-05-01' / 'events.parquet')
        anchor = pd.Timestamp(run.manifest.prediction_table.anchors[0])
        if anchor.tzinfo is not None:
            anchor = anchor.tz_convert(None)
        active = set(events.loc[(events.TS > anchor)
                                & (events.TS <= anchor + pd.Timedelta(days=7)),
                                'USER_ID'])
        targets = frame.ENTITY.isin(active).to_numpy()
        require(len(set(targets)) == 2, 'Fixture needs both realized classes')
        actuals = pd.read_parquet(result.actuals).sort_values('ENTITY')
        require(
            actuals.ENTITY.tolist() == list(range(POPULATION))
            and np.array_equal(actuals.TARGET, targets),
            'Realized labels do not match the independent event-window join')
        require(np.array_equal(actuals.True_PROB, frame.True_PROB),
                'Evaluation changed the saved probabilities')
        require(
            math.isclose(result.metrics['auroc'],
                         roc_auc_score(targets, frame.True_PROB), abs_tol=1e-6,
                         rel_tol=0),
            'Realized AUROC differs from independent recomputation')
        profile = platform_backend(root)
        reference = None
        drift_reference = None
        if profile is not None:
            record = profile.record(cfg, result.tracking)
            require(
                record['params'].get('kumo.labels_snapshot_id') == labels.id
                and record['params'].get('kumo.prediction_id') == run.id,
                'Platform monitoring records another prediction/label snapshot')
            require(
                math.isclose(record['metrics'].get('realized/auroc', math.nan),
                             result.metrics['auroc'], rel_tol=0, abs_tol=1e-6),
                'Platform realized metric differs from the evaluated result')
            reference = record
            drift_reference = platform_helpers().drift_record(
                profile, cfg, run, drift)
        else:
            with cfg.tracking.client() as client:
                record = client.get_run(result.tracking.run_id)
                require(
                    record.data.tags['kumo.labels_snapshot_id'] == labels.id,
                    'Tracker records another label snapshot')
        checks.append(
            dict(prediction=run.id, bundle=run.bundle_id,
                 labels_snapshot=labels.id, auroc=result.metrics['auroc'],
                 psi=drift.psi, reference=reference,
                 drift_reference=drift_reference))
    return dict(monitoring=checks)


def negative(root):
    profile = platform_backend(root)
    if profile is not None:
        return profile.negative()
    import kumo.engine as kumo

    state = read(root / 'fit.json')['result']
    bundle = root / 'store' / 'models' / state['versions'][0]['bundle']
    broken = root / 'incomplete-model'
    shutil.copytree(bundle, broken)
    manifest = read(broken / 'publication-manifest.json')
    entry = next(item for item in manifest['artifacts']
                 if item['role'] == 'training_spec')
    missing = broken / entry['relative_path']
    missing.unlink()
    try:
        kumo.Trainer.load(broken, accelerator='cpu')
    except (ValueError, FileNotFoundError) as error:
        require(
            str(error).strip()
            and (missing.name in str(error) or 'missing' in str(error).lower()),
            f'Incomplete model diagnostic is not actionable: {error}')
        incomplete = str(error)
    else:
        raise AssertionError('An incomplete model was accepted')

    denied = root / 'denied'
    denied.mkdir(mode=0o500)
    try:
        if os.access(denied, os.W_OK):
            raise PrerequisiteError(
                'Permission case requires a non-root user without write access')
        try:
            kumo.MLOps(str(denied), on_error='raise')
        except PermissionError as error:
            require('artifact_root' in str(error) and 'write' in str(error),
                    f'Artifact access diagnostic is not actionable: {error}')
            permission = str(error)
        else:
            raise AssertionError('An unwritable artifact root was accepted')
    finally:
        denied.chmod(0o700)

    # A partially copied prediction export must not be treated as committed.
    scored = read(root / 'score.json')['result']['aliases'][0]
    exported = root / 'store' / 'predictions' / scored['prediction']
    interrupted = root / 'interrupted-prediction'
    shutil.copytree(exported, interrupted)
    markers = list(interrupted.rglob('_SUCCESS'))
    require(markers, 'Prediction export had no completion marker')
    for marker in markers:
        marker.unlink()
    try:
        kumo.PredictionResult.load_from_connector(
            kumo.FileConnector(root_dir=str(root)),
            path='interrupted-prediction')
    except (ValueError, FileNotFoundError) as error:
        require('_SUCCESS' in str(error) or 'complete' in str(error).lower(),
                f'Incomplete export diagnostic is not actionable: {error}')
        interruption = str(error)
    else:
        raise AssertionError('An uncommitted prediction export was accepted')
    return dict(incomplete_model=incomplete, denied_artifact_access=permission,
                incomplete_prediction=interruption)


def phase(args):
    started = time.monotonic()
    report = dict(status='failed', phase=args.phase)
    selected = 'cpu'
    profile = None
    try:
        profile = platform_backend(args.output_dir)
        selected = accelerator(args.output_dir)
        if profile is not None and profile.config['environment'] == 'managed':
            report['environment'] = check_install(args.wheel,
                                                  args.expected_version,
                                                  managed=True)
        else:
            report['environment'] = check_install(args.wheel,
                                                  args.expected_version)
        if profile is not None:
            report['platform'] = dict(platform=profile.config['platform'],
                                      model=profile.model,
                                      experiment=profile.experiment,
                                      artifact_root=profile.artifact_root,
                                      placement=profile.placement())
        import torch

        import kumo.engine as kumo

        torch.set_num_threads(2)
        kumo.configure(scratch_dir=args.output_dir / f'scratch-{args.phase}')
        state = dict(stage=args.phase, fits=0, completed_fits=0,
                     optimizer_steps=0, training_batches=0,
                     prediction_batches={args.phase: 0})
        if selected == 'gpu' and (not torch.cuda.is_available()
                                  or torch.cuda.device_count() != 1):
            raise PrerequisiteError(
                'GPU profile requires exactly one visible CUDA device')
        observer = gpu_helpers().observe_cuda(
            state) if selected == 'gpu' else nullcontext()
        with observer:
            report['result'] = globals()[args.phase](args.output_dir)
        if selected == 'gpu':
            report['cuda_work'] = state
            if args.phase == 'fit':
                require(
                    state['fits'] == state['completed_fits'] == 2
                    and state['training_batches'] > 0
                    and state['optimizer_steps'] > 0,
                    'Missing CUDA training work')
            if args.phase == 'score':
                require(state['prediction_batches'][args.phase] > 0,
                        'Missing CUDA prediction work')
        report['status'] = 'passed'
    except (ImportError, PermissionError, PrerequisiteError):
        report['status'] = 'blocked'
        report['error'] = traceback.format_exc()
    except Exception as error:
        if profile is not None and platform_helpers().access_error(error):
            report['status'] = 'blocked'
        report['error'] = traceback.format_exc()
    finally:
        report['seconds'] = time.monotonic() - started
        report['resources'] = dict(
            requested_device=selected, torch_threads=2,
            maxrss_self_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maxrss_terminated_children_kib=resource.getrusage(
                resource.RUSAGE_CHILDREN).ru_maxrss,
            scope='Linux process maxima, not simultaneous process-tree memory')
        write(args.output_dir / f'{args.phase}.json', report)
    return 0 if report['status'] == 'passed' else 1


def run(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    report = dict(
        status='blocked', command=sys.argv, phases={
            name: dict(status='blocked', reason='Prerequisite has not passed')
            for name in PHASES
        }, test_sha256=digest(Path(__file__)), required_gaps={
            'exported_epoch_history':
            dict(
                status='not implemented',
                reason='Full epoch history is not available in model exports.'),
        })
    started = time.monotonic()
    execution_started = False
    try:
        if getattr(args, 'platform_config', None) is not None:
            platform_helpers().prepare(root, args.platform_config)
            report['platform_helpers_sha256'] = digest(
                Path(__file__).with_name('wheel_mlops_platform.py'))
        profile = platform_backend(root)
        if profile is not None:
            report['required_gaps']['ui_review'] = {
                'status':
                'blocked',
                'reason':
                'Inspect platform UI against the recorded runs, metrics and versions',
            }
            if profile.config['execution_context'] == 'client':
                report['required_gaps']['native_platform_execution'] = {
                    'status':
                    'blocked',
                    'reason':
                    'Client execution does not qualify the managed runtime',
                }
        report['gpu_helpers_sha256'] = digest(
            Path(__file__).with_name('wheel_gpu_smoke.py'))
        if profile is not None and profile.config['environment'] == 'managed':
            report['environment'] = check_install(args.wheel,
                                                  args.expected_version,
                                                  managed=True)
        else:
            report['environment'] = check_install(args.wheel,
                                                  args.expected_version)
        with (root / 'pip-check.log').open('w') as log:
            subprocess.run(
                [sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        execution_started = True
        phases = (('negative', 'fit', 'score',
                   'evaluate') if profile is not None else PHASES)
        for name in phases:
            command = [
                sys.executable, '-I',
                str(Path(__file__).resolve()), '--output-dir',
                str(root), '--wheel',
                str(args.wheel), '--expected-version', args.expected_version,
                '--phase', name
            ]
            if profile is not None:
                command.append('--execute-platform')
            with (root / f'{name}.log').open('w') as log:
                try:
                    returncode = execute(command, root, log, args.phase_timeout)
                except subprocess.TimeoutExpired:
                    report['phases'][name] = dict(status='failed',
                                                  error='Phase timed out')
                    report['status'] = 'failed'
                    break
            receipt = root / f'{name}.json'
            report['phases'][name] = (read(receipt) if receipt.is_file() else
                                      dict(status='failed',
                                           error='Worker produced no receipt'))
            report['phases'][name]['returncode'] = returncode
            if returncode or report['phases'][name]['status'] != 'passed':
                status = ('blocked' if report['phases'][name]['status']
                          == 'blocked' else 'failed')
                report['phases'][name]['status'] = status
                report['status'] = status
                break
            # Recovery must not use the producer's objects, source files or caches.
            scratch = root / f'scratch-{name}'
            if scratch.exists():
                shutil.rmtree(scratch)
            require(not scratch.exists(), 'Producer scratch survived cleanup')
            if name == 'fit':
                for source in root.glob('data-*'):
                    shutil.rmtree(source)
                require(not list(root.glob('data-*')),
                        'Producer source files survived cleanup')
        else:
            report['status'] = ('blocked' if any(
                gap['status'] != 'passed'
                for gap in report['required_gaps'].values()) else 'passed')
    except Exception:
        if execution_started:
            report['status'] = 'failed'
        report['error'] = traceback.format_exc()
    finally:
        report['seconds'] = time.monotonic() - started
        write(root / 'summary.json', report)
        print(
            json.dumps(
                dict(status=report['status'],
                     summary=str(root / 'summary.json'))))
    return 0 if report['status'] == 'passed' else 1


def execute(command, cwd, log, timeout):
    """Bound each worker and its training/prediction descendants."""
    with subprocess.Popen(command, cwd=cwd, stdout=log,
                          stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
                # Kill descendants which outlived their parent.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--expected-version', required=True)
    parser.add_argument('--phase-timeout', type=int, default=600)
    parser.add_argument('--platform-config', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--execute-platform', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.platform_config is not None or
        (args.phase is not None and
         (args.output_dir / 'platform.json').is_file())
        ) and not args.execute_platform:
        parser.error(
            'Use the platform entrypoint (Snowflake notebook or platform --execute)'
        )
    if args.phase_timeout <= 0:
        parser.error('--phase-timeout must be positive')
    args.output_dir = args.output_dir.resolve()
    args.wheel = args.wheel.resolve(strict=True)
    return phase(args) if args.phase else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
