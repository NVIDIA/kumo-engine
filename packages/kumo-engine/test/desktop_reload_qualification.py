#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process save and reload qualification for one installed wheel.

The cases are the eight tasks of the GPU wheel smoke (wheel_gpu_smoke.py,
which must sit next to this file) on its PR172 fixture data. The controller
runs each task in two new interpreters. The producer trains, exports the
model and the graph snapshot, and saves baseline predictions. The controller
then deletes every producer scratch, cache, temp and raw-input directory.
The consumer loads only the durable exports, predicts with the same options
and must not train. The controller compares the two outputs with the shared
task checks.

Run it with the interpreter of a clean environment, outside the checkout:

    python -I desktop_reload_qualification.py run --wheel <wheel> \
        --device cpu --data-dir <pr172 inputs> \
        --graph-metadata <pr172 inputs>/graphs.json \
        --work-dir <dir> --output-dir <dir>

The harness supports POSIX hosts only: Linux, WSL2 and macOS.
"""

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager
from pathlib import Path


def _load_tasks():
    """Load the shared task definitions that sit next to this file."""
    path = Path(__file__).resolve().with_name('wheel_gpu_smoke.py')
    spec = importlib.util.spec_from_file_location('wheel_gpu_smoke', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The eight tasks of the GPU wheel smoke: one definition of each query,
# fixture split, model plan and prediction check for both checks.
tasks = _load_tasks()
CASES = tuple(tasks.TASKS)
DEVICES = ('cpu', 'gpu', 'mps')
TORCH_DEVICE = {'cpu': 'cpu', 'gpu': 'cuda', 'mps': 'mps'}
SEED = 0
# Artifact and device rejections do not depend on the task, so they run once
# per configuration. The Faiss check needs a retrieval task.
ARTIFACT_CASE = 'binary'
FAISS_CASE = 'static_link'
RANKING_CASES = ('multilabel_ranking', 'static_link', 'temporal_link')
RECEIPT_SCHEMA = 1
DESCENDANT_GRACE = 15.0
REQUIRED_ATOL = 1e-4
PYTHONS = ('3.11', '3.12', '3.13', '3.14')
# Supported desktop configurations on GitHub-hosted runners. Keep in sync
# with the `reload` job in .github/workflows/kumo-engine-platform-smoke.yml.
HOSTED_TARGETS = (
    ('linux-x86_64', 'cpu'),
    ('linux-x86_64', 'gpu'),
    ('linux-arm64', 'cpu'),
    ('windows-wsl', 'cpu'),
    ('windows-wsl', 'gpu'),
    ('macos-15', 'cpu'),
    ('macos-26', 'cpu'),
)
# Hosted macOS runners are VMs whose paravirtual GPU has no Metal 2.0, so
# PyTorch reports no MPS there. MPS runs on a bare-metal Apple Silicon Mac
# with `run` and joins the report through a second --evidence-dir. macOS 14
# is the supported floor, which hosted runners no longer offer, so its CPU
# row comes from a Mac too, on bare metal or in a VM. MPS needs macOS 15. On
# macOS 14 the unavailable_device check proves that MPS fails clearly.
EXTERNAL_TARGETS = (
    ('macos-14', 'cpu'),
    ('macos-15', 'mps'),
)
MPS_MIN_MACOS = 15
# Checks every configuration must carry, by case. Score drift must be caught
# on every task, artifact rejections once and missing Faiss on static link.
BASE_CHECKS = {
    ARTIFACT_CASE: ('model_file_corrupt', 'model_file_missing',
                    'snapshot_file_corrupt', 'snapshot_missing'),
    FAISS_CASE: ('faiss_missing', ),
}
# Checks a configuration must carry, by target and case. A macOS 14 row
# passes only with proof that MPS fails there with a clear error.
REQUIRED_CHECKS = {
    ('macos-14', 'cpu'): {
        ARTIFACT_CASE: ('unavailable_device_mps_old_macos', ),
    },
}
REQUIRED_TARGETS = HOSTED_TARGETS + EXTERNAL_TARGETS


class QualificationError(Exception):
    """A case did not meet the qualification contract."""


# ---------------------------------------------------------------------------
# Pure helpers: identity, paths and process control. No Engine imports.
# ---------------------------------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root):
    """Map every regular file under root to its SHA-256."""
    root = Path(root)
    if not root.is_dir():
        raise QualificationError(f'Durable export is missing: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise QualificationError(f'Durable export holds a symlink: {path}')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    if not result:
        raise QualificationError(f'Durable export is empty: {root}')
    return result


def require_same_tree(expected, root, label):
    actual = tree_hashes(root)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual)
                         if expected[name] != actual[name])
        raise QualificationError(
            f'{label} changed after the producer exited: missing={missing}, '
            f'extra={extra}, changed={changed}')


def require_disjoint(durable, ephemeral_roots):
    """Durable paths must not resolve inside any directory we delete."""
    durable = Path(durable).resolve()
    for root in ephemeral_roots:
        root = Path(root).resolve()
        if (durable == root or durable.is_relative_to(root)
                or root.is_relative_to(durable)):
            raise QualificationError(
                f'Durable directory {durable} overlaps cleanup root {root}')


def wheel_version(wheel):
    name = Path(wheel).name
    parts = name.split('-')
    if len(parts) < 5 or parts[0] != 'kumo_engine' or not name.endswith('.whl'):
        raise QualificationError(f'Not a kumo-engine wheel: {name}')
    return parts[1]


def verify_installed_wheel(wheel):
    """Require every packaged file to be installed with the wheel's hash."""
    roots = {Path(sysconfig.get_paths()[key]) for key in ('purelib', 'platlib')}
    with zipfile.ZipFile(wheel) as archive:
        [record] = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/RECORD') and name.count('/') == 1
        ]
        rows = list(csv.reader(archive.read(record).decode().splitlines()))
    checked = 0
    for row in rows:
        if len(row) < 2 or not row[1] or '.dist-info/' in row[0] or ('.data/'
                                                                     in row[0]):
            continue
        algorithm, expected = row[1].split('=', 1)
        if algorithm != 'sha256':
            raise QualificationError(f'Unexpected RECORD hash: {row[1]}')
        installed = [
            root / row[0] for root in roots if (root / row[0]).is_file()
        ]
        if not installed:
            raise QualificationError(f'Wheel file is not installed: {row[0]}')
        digest = base64.urlsafe_b64encode(
            bytes.fromhex(sha256_file(installed[0]))).rstrip(b'=').decode()
        if digest != expected:
            raise QualificationError(
                f'Installed file differs from the candidate wheel: {row[0]}')
        checked += 1
    if not checked:
        raise QualificationError('The candidate wheel lists no files')
    return checked


def run_worker(command, *, cwd, env, timeout, log_path):
    """Run one worker in a new session and require a clean, timely exit."""
    started = time.monotonic()
    with open(log_path, 'wb') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process.pid)
            process.wait()
            raise QualificationError(
                f'Worker exceeded {timeout} seconds: {command[2:4]}') from None
        except BaseException:
            # An interrupted controller must not leave a detached worker.
            _kill_group(process.pid)
            raise
    # Helpers such as the multiprocessing resource tracker exit shortly
    # after their parent. Anything still alive after the grace period is a
    # leaked process.
    deadline = time.monotonic() + DESCENDANT_GRACE
    while _group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(.2)
    if _group_alive(process.pid):
        listing = _group_listing(process.pid)
        _kill_group(process.pid)
        raise QualificationError(
            f'Worker left descendant processes running: {listing}')
    return dict(exit_code=code, seconds=round(time.monotonic() - started, 3))


def _group_listing(pgid):
    try:
        lines = subprocess.run(['ps', '-A', '-o', 'pid=,pgid=,command='],
                               capture_output=True, text=True,
                               timeout=10).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        return f'ps failed: {error}'
    return [
        line.strip()[:200] for line in lines
        if len(line.split()) > 2 and line.split()[1] == str(pgid)
    ]


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_receipt(path, *, phase, case, device):
    """Load a worker receipt and reject skips, failures and mismatches."""
    path = Path(path)
    if not path.is_file():
        raise QualificationError(f'{phase} receipt is missing: {path}')
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != RECEIPT_SCHEMA:
        raise QualificationError(f'{phase} receipt has an unknown schema')
    if receipt.get('phase') != phase or receipt.get('case') != case:
        raise QualificationError(f'{phase} receipt names another phase/case')
    if receipt.get('device') != device:
        raise QualificationError(f'{phase} receipt names another device')
    if receipt.get('status') != 'PASS':
        raise QualificationError(
            f'{phase} receipt status is {receipt.get("status")!r}: '
            f'{receipt.get("error")}')
    return receipt


def require_device_work(receipt, *, phase, device):
    """Require real batches on the requested device; reject CPU fallback."""
    work = receipt.get('work') or {}
    expected = TORCH_DEVICE[device]
    if work.get('device_types') != [expected]:
        raise QualificationError(
            f'{phase} ran on {work.get("device_types")}, not [{expected!r}]')
    if not work.get('prediction_batches'):
        raise QualificationError(f'{phase} completed no prediction batches')
    if phase == 'producer':
        if work.get('fits') != 1 or not work.get('training_batches') or (
                work.get('optimizer_steps', 0) <= 0):
            raise QualificationError('Producer completed no real training')
    else:
        if work.get('fits') or work.get('training_batches') or work.get(
                'optimizer_steps') or work.get('other_optimizer_steps'):
            raise QualificationError(
                'Consumer trained the model: '
                f'fits={work.get("fits")}, '
                f'training_batches={work.get("training_batches")}, '
                f'optimizer_steps={work.get("optimizer_steps")}')


def expected_retrieval_backend(device, system):
    if device == 'mps':
        return 'faiss-metal'
    if device == 'gpu':
        return 'faiss-cuda'
    return 'torch-bruteforce' if system == 'Darwin' else 'faiss-cpu'


def require_retrieval(receipt, *, phase, case, device):
    if not case.endswith('link'):
        return
    searches = (receipt.get('work') or {}).get('retrieval') or {}
    expected = expected_retrieval_backend(device, receipt.get('system'))
    predicted = searches.get('predict') or {}
    if set(predicted) != {expected} or predicted[expected] <= 0:
        raise QualificationError(
            f'{phase} retrieval used {predicted}, expected only {expected!r}')
    if device == 'mps':
        metal = receipt.get('metal') or {}
        if not (metal.get('mps_available') and metal.get('faiss_metal')
                and metal.get('faiss_num_gpus', 0) > 0
                and metal.get('mps_fallback') == '0'):
            raise QualificationError(
                f'{phase} has no Apple GPU evidence: {metal}')


# ---------------------------------------------------------------------------
# Prediction validation and comparison. Pandas and NumPy only.
# ---------------------------------------------------------------------------


def compare_frames(case, baseline, after, *, producer, atol):
    """Check both outputs with the shared task rules, then compare them.

    Entities, anchor timestamps, classes and forecast steps must match the
    saved inputs exactly. Scores must agree within atol. Chosen labels and
    ranks may change only between scores that tie within atol.
    """
    def aligned(frame):
        values = tasks.prediction_values(
            case, frame, producer['entities'], classes=producer['classes'],
            anchor=producer['anchor'], time_column=producer['time_column'])
        decisions = tasks.prediction_decisions(case, frame,
                                               producer['entities'],
                                               classes=producer['classes'])
        return values, decisions

    try:
        (before, chosen), (later, rechosen) = aligned(baseline), aligned(after)
        return tasks.compare_predictions(before, later, task=case, atol=atol,
                                         before_decisions=chosen,
                                         after_decisions=rechosen)
    except ValueError as error:
        raise QualificationError(str(error)) from None


# ---------------------------------------------------------------------------
# Worker side: fixtures, training, export, reload and observation.
# ---------------------------------------------------------------------------


def _block_imports():
    """Hide modules named in DRQ_BLOCK_IMPORTS to test missing dependencies."""
    names = {
        name
        for name in os.environ.get('DRQ_BLOCK_IMPORTS', '').split(',') if name
    }
    if not names:
        return

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in names:
                raise ModuleNotFoundError(f"No module named '{fullname}'",
                                          name=fullname)
            return None

    sys.meta_path.insert(0, Blocker())


def _seed():
    import random

    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def prediction_options(case, durable, output_dir, classes):
    """Score the saved inputs with the options of the shared GPU smoke."""
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    options = dict(entities=kumo.FileConnector(
        root_dir=str(durable))['inputs.parquet'])
    if classes:
        options['num_classes_to_return'] = len(classes)
    if case == 'binary':
        options['binary_classification_threshold'] = .5
    if case != 'static_link':
        options['output_config'] = kumo.PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(output_dir)),
            output_paths={'predictions': 'predictions'},
            extra_fields=[MetadataField.ANCHOR_TIMESTAMP])
    return options


def seed_time_column(trainer):
    """The time column of the fitted seed table, as forecasts name it."""
    [runtime] = trainer.scratch_dir.glob(
        'runs/*-fit-*/task_dataset/runtime_manifest.json')
    seed = json.loads(runtime.read_text())['table']['seed_entity_table']
    schema = json.loads(runtime.with_name('feature_schema.json').read_text())
    return schema['node_types'][seed]['time_column'] or 'TIMESTAMP'


@contextmanager
def observe(device, work):
    """Record real device work through public Lightning and Torch hooks."""
    from unittest.mock import patch

    import pytorch_lightning as pl
    import torch
    from kumo_core.modeling.retrieval import mips
    from torch.optim.optimizer import register_optimizer_step_post_hook

    expected = TORCH_DEVICE[device]
    seen = set()

    def tensors(values):
        if isinstance(values, torch.Tensor):
            return [values]
        if isinstance(values, dict):
            return [v for v in values.values() if isinstance(v, torch.Tensor)]
        if isinstance(values, (list, tuple)):
            return [v for v in values if isinstance(v, torch.Tensor)]
        return []

    def check(module, values, label):
        found = [value for value in tensors(values) if value.numel()]
        seen.add(module.device.type)
        seen.update(value.device.type for value in found)
        if module.device.type != expected or not found or any(
                value.device.type != expected for value in found):
            raise QualificationError(
                f'{label} did not run on {expected}: module='
                f'{module.device}, outputs='
                f'{[str(v.device) for v in found]}')

    class Proof(pl.Callback):
        def on_fit_start(self, trainer, module):
            work['fits'] += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, index):
            check(module, outputs, 'Training batch')
            work['training_batches'] += 1

        def on_predict_batch_end(self, trainer, module, outputs, batch, index,
                                 dataloader_index=0):
            check(module, outputs, 'Prediction batch')
            work['prediction_batches'] += 1

    class ObservedTrainer(pl.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Proof()]
            super().__init__(*args, **kwargs)

    def on_step(optimizer, _args, _kwargs):
        grads = [
            parameter.grad for group in optimizer.param_groups
            for parameter in group['params'] if parameter.grad is not None
        ]
        if grads and all(grad.device.type == expected for grad in grads):
            work['optimizer_steps'] += 1
            return
        # Post-hoc calibration fits its few parameters on the CPU during
        # validation. The callback above proves where each training batch ran.
        other = work.setdefault('other_optimizer_steps', {})
        name = type(optimizer).__name__
        other[name] = other.get(name, 0) + 1

    original = mips.MIPSFaiss._search

    def search(index, query, k, exclude):
        if index._cpu_index is not None:
            backend = 'torch-bruteforce'
        elif index._metal_index is not None:
            backend = 'faiss-metal'
        elif index._res is not None:
            backend = 'faiss-cuda'
        else:
            backend = 'faiss-cpu'
        stage = work['retrieval'].setdefault(work['stage'], {})
        stage[backend] = stage.get(backend, 0) + 1
        return original(index, query, k, exclude)

    hook = register_optimizer_step_post_hook(on_step)
    try:
        with patch.object(pl, 'Trainer', ObservedTrainer), patch.object(
                mips.MIPSFaiss, '_search', search):
            yield
    finally:
        hook.remove()
        work['device_types'] = sorted(seen)


def environment(device, expected_version):
    import torch

    import kumo.engine as kumo

    version = importlib.metadata.version('kumo-engine')
    location = Path(kumo.__file__).resolve()
    if version != expected_version or kumo.__version__ != version:
        raise QualificationError(
            f'Installed kumo-engine {version} is not candidate '
            f'{expected_version}')
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        raise QualificationError(f'kumo.engine imported from {location}')
    info = dict(kumo_engine=version, import_path=str(location),
                python=sys.version.split()[0], executable=sys.executable,
                platform=platform.platform(), machine=platform.machine(),
                system=platform.system(), torch=torch.__version__,
                torch_cuda=torch.version.cuda,
                cuda_available=torch.cuda.is_available(),
                mps_available=bool(torch.backends.mps.is_available()))
    try:
        import faiss
        info['faiss'] = dict(
            version=faiss.__version__, path=faiss.__file__,
            options=faiss.get_compile_options(), num_gpus=faiss.get_num_gpus()
            if hasattr(faiss, 'get_num_gpus') else None)
    except ImportError as error:
        info['faiss'] = dict(error=str(error))
    return info


def preflight(device, info):
    """A required device must exist; never fall back to CPU silently."""
    import torch

    if device == 'gpu' and not (torch.cuda.is_available()
                                and torch.cuda.device_count() >= 1):
        raise QualificationError(
            'Required CUDA device is unavailable: torch.cuda.is_available() '
            f'is {torch.cuda.is_available()}')
    if device == 'mps':
        if platform.system(
        ) != 'Darwin' or not torch.backends.mps.is_available():
            raise QualificationError(
                'Required MPS device is unavailable on this host')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
            raise QualificationError(
                'Strict MPS needs PYTORCH_ENABLE_MPS_FALLBACK=0')
    if device == 'gpu':
        info['gpu_name'] = torch.cuda.get_device_name(0)


def metal_evidence(info):
    import torch

    faiss = info.get('faiss') or {}
    return dict(mps_available=bool(torch.backends.mps.is_available()),
                mps_built=bool(torch.backends.mps.is_built()),
                faiss_metal='MAC_METAL' in (faiss.get('options') or ''),
                faiss_num_gpus=faiss.get('num_gpus') or 0,
                mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))


def produce(args):
    import kumo.engine as kumo

    durable, raw = Path(args.durable), Path(args.raw)
    spec = tasks.TASKS[args.case]
    work = dict(stage='fit', fits=0, training_batches=0, prediction_batches=0,
                optimizer_steps=0, retrieval={})
    receipt = dict(schema=RECEIPT_SCHEMA, phase='producer', case=args.case,
                   device=args.device, status='RUNNING', work=work)
    info = environment(args.device, args.expected_version)
    receipt.update(info)
    preflight(args.device, info)
    receipt.update(info)
    kumo.configure(scratch_dir=args.scratch)
    # Only producer-owned storage holds the source tables, and the controller
    # deletes it before the consumer starts.
    shutil.copytree(Path(args.data_dir) / spec['graph'], raw / spec['graph'])
    metadata = json.loads(Path(args.graph_metadata).read_text())
    _seed()
    prepared = tasks.prepare_task(args.case, raw, metadata, raw / 'task', {})
    shutil.copy2(prepared['input_file'], durable / 'inputs.parquet')
    snapshot, classes = prepared['snapshot'], prepared['classes']
    with observe(args.device, work):
        _seed()
        trainer = kumo.Trainer(tasks.make_plan(args.case),
                               accelerator=args.device)
        fitted = trainer.fit(prepared['training_table'], snapshot=snapshot)
        if not fitted.metrics:
            raise QualificationError('Training reported no metrics')
        files = kumo.FileConnector(root_dir=str(durable))
        published = fitted.export(kumo.ModelOutputConfig(files, path='model'))
        location = snapshot.export(
            kumo.SnapshotOutputConfig(files, path='snapshot'))
        work['stage'] = 'predict'
        _seed()
        before = work['prediction_batches']
        frame = trainer.predict(
            snapshot,
            **prediction_options(args.case, durable,
                                 Path(args.scratch) / 'out', classes)).frame
        if work['prediction_batches'] <= before:
            raise QualificationError('Baseline prediction ran no batches')
    frame.to_parquet(durable / 'baseline.parquet')
    anchor = prepared['anchor']
    receipt.update(
        status='PASS', metrics={
            key: float(value) if math.isfinite(value) else None
            for key, value in fitted.metrics.items()
        }, snapshot_id=snapshot.id, snapshot_location=location,
        model=dict(name=published.name, version=published.version,
                   uri=published.uri, metadata=dict(published.metadata)),
        query=spec['query'], task_type=spec['task_type'],
        entities=prepared['ids'], classes=classes,
        anchor=None if anchor is None else str(anchor),
        time_column=seed_time_column(trainer), rows=len(frame))
    if args.device == 'mps':
        receipt['metal'] = metal_evidence(info)
    return receipt


def consume(args):
    import kumo.engine as kumo

    durable = Path(args.durable)
    producer = json.loads(Path(args.producer_receipt).read_text())
    work = dict(stage='load', fits=0, training_batches=0, prediction_batches=0,
                optimizer_steps=0, retrieval={})
    receipt = dict(schema=RECEIPT_SCHEMA, phase='consumer', case=args.case,
                   device=args.device, status='RUNNING', work=work,
                   producer_receipt_sha256=sha256_file(args.producer_receipt))
    info = environment(args.device, args.expected_version)
    receipt.update(info)
    if not args.skip_preflight:
        preflight(args.device, info)
        receipt.update(info)
    kumo.configure(scratch_dir=args.scratch)
    with observe(args.device, work):
        files = kumo.FileConnector(root_dir=str(durable))
        trainer = kumo.Trainer.load_from_connector(files, path='model',
                                                   accelerator=args.device)
        snapshot = kumo.GraphSnapshot.load_from_connector(
            files, path=producer['snapshot_location'])
        if snapshot.id != producer['snapshot_id']:
            raise QualificationError('Reloaded snapshot has another id')
        work['stage'] = 'predict'
        _seed()
        frame = trainer.predict(
            snapshot,
            **prediction_options(args.case, durable,
                                 Path(args.scratch) / 'out',
                                 producer['classes'])).frame
    frame.to_parquet(Path(args.output) / 'reloaded.parquet')
    receipt.update(status='PASS', snapshot_id=snapshot.id, rows=len(frame))
    if args.device == 'mps':
        receipt['metal'] = metal_evidence(info)
    return receipt


def worker_main(args):
    _block_imports()
    receipt_path = Path(args.receipt)
    phase = 'producer' if args.command == 'produce' else 'consumer'
    receipt = dict(schema=RECEIPT_SCHEMA, phase=phase, case=args.case,
                   device=args.device, status='FAIL')
    code = 1
    try:
        receipt = (produce if args.command == 'produce' else consume)(args)
        code = 0
    except BaseException as error:  # noqa: B902 -- record, then fail.
        receipt.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    return code


# ---------------------------------------------------------------------------
# Controller.
# ---------------------------------------------------------------------------


def worker_env(device, roots):
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'DRQ_BLOCK_IMPORTS',
                 'DISKGRAPH_DATA_ROOT', 'DISKGRAPH_INGEST_BIN',
                 'KUMO_BP_OUTPUT_SPOOL_ROOT', 'KUMO_REQUIRE_GPU'):
        env.pop(name, None)
    # A task-owned HOME keeps ~/.cache/kumo-engine and other home caches
    # out of both workers, so nothing survives cleanup through them.
    env.update(HOME=str(roots['home']), USERPROFILE=str(roots['home']))
    env.update(TMPDIR=str(roots['tmp']), TEMP=str(roots['tmp']),
               TMP=str(roots['tmp']), XDG_CACHE_HOME=str(roots['cache']),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               SYNC_DATA_LOADER='1')
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    if device == 'mps':
        env['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    return env


def make_roots(base):
    roots = {name: base / name for name in ('scratch', 'cache', 'cwd', 'home')}
    for path in roots.values():
        path.mkdir(parents=True)
    # AF_UNIX socket paths are limited to about 104 bytes, and Torch data
    # loaders put sockets in TMPDIR. Use a short, task-owned directory.
    roots['tmp'] = Path(
        tempfile.mkdtemp(
            prefix='drq-', dir=os.environ.get('DRQ_TMP_BASE')
            or ('/tmp' if os.path.isdir('/tmp') else None)))
    return roots


def remove_roots(base, roots):
    """Delete every ephemeral root and prove that none survived."""
    targets = [Path(base), *roots.values()]
    for path in targets:
        shutil.rmtree(path, ignore_errors=True)
    survivors = [str(path) for path in targets if path.exists()]
    if survivors:
        raise QualificationError(f'Cleanup left directories: {survivors}')
    return [str(path) for path in targets]


def worker_command(python, script, command, case, device, version, **paths):
    result = [
        str(python), '-I',
        str(script), command, '--case', case, '--device', device,
        '--expected-version', version
    ]
    for key, value in paths.items():
        if value is True:
            result.append(f'--{key.replace("_", "-")}')
        elif value is not None:
            result += [f'--{key.replace("_", "-")}', str(value)]
    return result


def run_consumer(ctx, durable, label, *, device=None, env_extra=None,
                 skip_preflight=False):
    base = ctx['work'] / label
    if base.exists():
        shutil.rmtree(base)
    roots = make_roots(base)
    output = ctx['evidence'] / label
    output.mkdir(parents=True, exist_ok=True)
    device = device or ctx['device']
    env = worker_env(device, roots)
    env.update(env_extra or {})
    receipt = output / 'consumer-receipt.json'
    command = worker_command(ctx['python'], ctx['script'], 'consume',
                             ctx['case'], device, ctx['version'],
                             durable=durable, scratch=roots['scratch'],
                             producer_receipt=ctx['producer_receipt'],
                             receipt=receipt, output=output,
                             skip_preflight=skip_preflight or None)
    try:
        process = run_worker(command, cwd=roots['cwd'], env=env,
                             timeout=ctx['timeout'],
                             log_path=output / 'consumer.log')
    finally:
        remove_roots(base, roots)
    return process, receipt, output


DURABLE_FILES = ('inputs.parquet', 'baseline.parquet')


def require_durable_layout(durable):
    found = sorted(path.name for path in Path(durable).iterdir())
    expected = sorted([*DURABLE_FILES, 'model', 'snapshot'])
    if found != expected:
        raise QualificationError(
            f'Durable directory holds {found}, expected {expected}')


def run_case(ctx):
    """Run one positive case; return its result row."""
    case, device = ctx['case'], ctx['device']
    work, evidence = ctx['work'], ctx['evidence']
    durable = work / 'durable'
    producer_base = work / 'producer'
    durable.mkdir(parents=True)
    roots = make_roots(producer_base)
    raw = producer_base / 'raw'
    require_disjoint(durable, [producer_base, *roots.values()])
    receipt_path = evidence / 'producer-receipt.json'
    command = worker_command(ctx['python'], ctx['script'], 'produce', case,
                             device, ctx['version'], durable=durable,
                             scratch=roots['scratch'], raw=raw,
                             data_dir=ctx['data_dir'],
                             graph_metadata=ctx['graph_metadata'],
                             receipt=receipt_path)
    try:
        process = run_worker(command, cwd=roots['cwd'],
                             env=worker_env(device,
                                            roots), timeout=ctx['timeout'],
                             log_path=evidence / 'producer.log')
    finally:
        # Delete everything the producer owned except the durable exports.
        deleted = remove_roots(producer_base, roots)
    row = dict(producer_process=process, deleted_after_producer=deleted)
    if process['exit_code'] != 0:
        raise QualificationError(
            f'Producer exited with {process["exit_code"]}; see producer.log')
    producer = read_receipt(receipt_path, phase='producer', case=case,
                            device=device)
    require_device_work(producer, phase='producer', device=device)
    require_retrieval(producer, phase='producer', case=case, device=device)
    exports = {
        'model': tree_hashes(durable / 'model'),
        'snapshot': tree_hashes(durable / 'snapshot'),
    }
    inputs = {name: sha256_file(durable / name) for name in DURABLE_FILES}
    require_durable_layout(durable)
    shutil.copy2(durable / 'baseline.parquet', evidence / 'baseline.parquet')
    ctx['producer_receipt'] = receipt_path
    process, consumer_path, _ = run_consumer(ctx, durable, 'consumer')
    row['consumer_process'] = process
    if process['exit_code'] != 0:
        raise QualificationError(
            f'Consumer exited with {process["exit_code"]}; see consumer.log')
    consumer = read_receipt(consumer_path, phase='consumer', case=case,
                            device=device)
    if consumer.get('producer_receipt_sha256') != sha256_file(receipt_path):
        raise QualificationError('Consumer receipt names another producer')
    if consumer.get('kumo_engine') != producer.get('kumo_engine'):
        raise QualificationError('Producer and consumer used other wheels')
    require_device_work(consumer, phase='consumer', device=device)
    require_retrieval(consumer, phase='consumer', case=case, device=device)
    for name, hashes in exports.items():
        require_same_tree(hashes, durable / name, name)
    require_durable_layout(durable)
    for name, digest in inputs.items():
        if sha256_file(durable / name) != digest:
            raise QualificationError(f'{name} changed during the consumer run')
    import pandas as pd

    baseline = pd.read_parquet(evidence / 'baseline.parquet')
    reloaded = pd.read_parquet(evidence / 'consumer' / 'reloaded.parquet')
    row['comparison'] = compare_frames(case, baseline, reloaded,
                                       producer=producer, atol=ctx['atol'])
    row.update(
        model_files=exports['model'], snapshot_files=exports['snapshot'],
        inputs=inputs, snapshot_id=producer['snapshot_id'],
        snapshot_location=producer['snapshot_location'],
        model=producer['model'], metrics=producer['metrics'],
        producer_work=producer['work'], consumer_work=consumer['work'],
        environment={
            key: producer.get(key)
            for key in ('kumo_engine', 'import_path', 'python', 'platform',
                        'machine', 'system', 'torch', 'torch_cuda', 'faiss',
                        'gpu_name')
        })
    if device == 'mps':
        row['metal'] = dict(producer=producer['metal'],
                            consumer=consumer['metal'])
    ctx['producer'] = producer
    return row, durable, baseline


def _largest_file(root):
    return max((p for p in Path(root).rglob('*') if p.is_file()),
               key=lambda p: p.stat().st_size)


def _corrupt_largest(root):
    target = _largest_file(root)
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    return target


def _remove_largest(root):
    target = _largest_file(root)
    target.unlink()
    return target


def _remove_tree(root):
    shutil.rmtree(root)
    return root


def macos_major():
    """The macOS major version, or None off macOS or when it is unknown."""
    if platform.system() != 'Darwin':
        return None
    try:
        return int(platform.mac_ver()[0].split('.')[0])
    except ValueError:
        return None


def unavailable_device():
    """Pick a device this host cannot use, or None if every one is usable.

    Lightning maps 'gpu' to MPS on Apple Silicon, so a Mac with MPS on
    macOS 15 or later has no device that Kumo must reject. Before macOS 15,
    Kumo must reject MPS with a clear error.
    """
    import torch

    major = macos_major()
    if major is not None and major < MPS_MIN_MACOS:
        return 'mps_old_macos'
    if platform.system() != 'Darwin' or not torch.backends.mps.is_available():
        return 'mps'
    return None


# Each product rejection must name its cause in the consumer error.
EXPECTED_ERRORS = {
    'model_file_corrupt': r'model bundle .* does not match its digest',
    'model_file_missing': r'model bundle .* is missing',
    'snapshot_file_corrupt': r'snapshot export .* is corrupt',
    'snapshot_missing': r'No snapshot export at',
    'unavailable_device_mps': r'MPSAccelerator`? can not run on your system',
    'unavailable_device_mps_old_macos': (
        r"UnsupportedAcceleratorError: accelerator='mps' uses Apple MPS, "
        r'which requires macOS 15 or later'),
    'faiss_missing': r'FaissUnavailableError: FAISS is required',
}


def negative_checks(ctx, durable, baseline):
    """Prove that each failure is caught.

    Artifact, device and dependency checks pass only when the product fails
    with an error that names the cause. Prediction tampering checks pass
    only when the harness comparison rejects the changed output.
    """
    case = ctx['case']
    producer = ctx['producer']
    checks = []
    not_applicable = ctx.setdefault('not_applicable', [])

    def expect_rejected(name, action):
        try:
            detail = action()
        except QualificationError as error:
            checks.append(dict(name=name, detected=True, detail=str(error)))
            return
        except Exception as error:  # noqa: B902 -- unexpected, still record.
            checks.append(
                dict(name=name, detected=False,
                     detail=f'{type(error).__name__}: {error}'))
            return
        checks.append(dict(name=name, detected=False, detail=detail))

    def consumer_must_fail(name, mutate=None, **kwargs):
        """Detected only if the product fails with an error naming the cause.

        A zero exit, a timeout, a signal, a leftover process or an unrelated
        error means the product did not reject the input clearly.
        """
        label = 'neg-' + name.replace('_', '-')
        copy = ctx['work'] / f'{label}-durable'
        shutil.copytree(durable, copy)
        try:
            target = mutate(copy) if mutate else None
            try:
                process, receipt_path, _ = run_consumer(ctx, copy, label,
                                                        **kwargs)
            except QualificationError as error:
                return f'Consumer did not exit cleanly: {error}'
        finally:
            shutil.rmtree(copy, ignore_errors=True)
        receipt = json.loads(
            receipt_path.read_text()) if (receipt_path.is_file()) else {}
        detail = dict(exit_code=process['exit_code'],
                      error=receipt.get('error'),
                      mutated=str(target) if target else None)
        if process['exit_code'] != 1 or receipt.get('status') != 'FAIL':
            return f'Consumer did not reject it: {detail}'
        if not re.search(EXPECTED_ERRORS[name], receipt.get('error') or ''):
            return f'Consumer error does not name the cause: {detail}'
        raise QualificationError(json.dumps(detail, default=str))

    if case == ARTIFACT_CASE:
        mutations = {
            'model_file_corrupt': lambda d: _corrupt_largest(d / 'model'),
            'model_file_missing': lambda d: _remove_largest(d / 'model'),
            'snapshot_file_corrupt': lambda d: _corrupt_largest(d / 'snapshot'),
            'snapshot_missing': lambda d: _remove_tree(d / 'snapshot'),
        }
        for name, mutate in mutations.items():
            expect_rejected(name,
                            lambda n=name, m=mutate: consumer_must_fail(n, m))
        wrong = unavailable_device()
        if wrong is None:
            not_applicable.append(
                dict(name='unavailable_device',
                     reason='Every Kumo accelerator is usable on this host.'))
        else:
            name = f'unavailable_device_{wrong}'
            options = dict(device='mps', skip_preflight=True,
                           env_extra={'PYTORCH_ENABLE_MPS_FALLBACK': '0'})
            expect_rejected(name, lambda: consumer_must_fail(name, **options))
    if case == FAISS_CASE:
        expect_rejected(
            'faiss_missing', lambda: consumer_must_fail(
                'faiss_missing', env_extra={'DRQ_BLOCK_IMPORTS': 'faiss'}))

    def tampered(name, change):
        try:
            compare_frames(case, baseline, change(baseline.copy()),
                           producer=producer, atol=ctx['atol'])
        except QualificationError as error:
            if re.search(TAMPER_ERRORS.get(name, '.'), str(error)):
                raise
            return f'Rejected for another reason: {error}'
        return 'Tampered prediction passed'

    for name, change in tampers(case, baseline, ctx['atol']).items():
        expect_rejected(name, lambda n=name, c=change: tampered(n, c))
    return checks


# A tamper that keeps the output valid must fail the reload comparison
# itself, not an earlier output check.
TAMPER_ERRORS = {
    'prediction_score_shift': r'Prediction difference exceeds atol',
    'rank_swapped': r'changed outside a near tie',
}


def tampers(case, baseline, atol):
    """Changes to a correct output that the comparison must reject."""
    import numpy as np
    import pandas as pd

    column = next(c for c in ('SCORE', 'True_PROB', 'TARGET_PRED')
                  if c in baseline)
    first = baseline['ENTITY'].iloc[0]
    delta = 10 * atol + 1e-3

    def shift(f):
        """Move one score by more than atol and keep the output valid."""
        if case == 'binary':
            # Move away from the 0.5 threshold; keep the pair summing to one.
            p = f['True_PROB']
            row = ((p.sub(.5).abs() > 2 * delta) & p.between(delta, 1 - delta)
                   ).idxmax()
            step = delta if p[row] > .5 else -delta
            f.loc[row, 'True_PROB'] += step
            f.loc[row, 'False_PROB'] -= step
        elif case in ('multiclass', 'multilabel_ranking'):
            # Move mass from the second score to the lowest one, so each
            # entity still sums to one and keeps its top class and order.
            for _, group in f.groupby('ENTITY', sort=False):
                ordered = group['SCORE'].sort_values(ascending=False)
                if (len(ordered) > 2 and ordered.iloc[0] - ordered.iloc[1] >
                        2 * delta and ordered.iloc[1] - ordered.iloc[-1] >
                        2 * delta):
                    f.loc[ordered.index[1], 'SCORE'] -= delta
                    f.loc[ordered.index[-1], 'SCORE'] += delta
                    break
        else:
            f.loc[f.index[0], column] += delta
        return f

    result = {
        'prediction_score_shift': shift,
        'prediction_entity_missing': lambda f: f[f['ENTITY'] != first],
        'prediction_entity_duplicated': lambda f: pd.concat([f, f.iloc[:1]]),
    }
    if 'ANCHOR_TIMESTAMP' in baseline:
        result['timestamp_shifted'] = lambda f: f.assign(
            ANCHOR_TIMESTAMP=pd.to_datetime(f['ANCHOR_TIMESTAMP']) +
            pd.to_timedelta(1, unit='D'))
    if 'CLASS' in baseline:

        def replace_class(f):
            foreign = (f['CLASS'].max() + 1 if pd.api.types.is_numeric_dtype(
                f['CLASS']) else 'drq-foreign-class')
            f.loc[f.index[0], 'CLASS'] = foreign
            return f

        result.update(
            class_replaced=replace_class, class_dropped=lambda f: f.drop(
                f.index[f['ENTITY'] == first][-1:]))
    if case in RANKING_CASES:

        def swap_ranks(f):
            # Reorder two classes of the entity whose scores spread most.
            # Each class keeps its score; only the published rank changes.
            spread = f.groupby('ENTITY')['SCORE'].agg(np.ptp)
            rows = f.index[f['ENTITY'] == spread.idxmax()]
            pair, columns = [rows[0], rows[-1]], ['CLASS', 'SCORE']
            f.loc[pair, columns] = f.loc[pair[::-1], columns].to_numpy()
            return f

        result['rank_swapped'] = swap_ranks
    decision = 'PREDICTED' if 'PREDICTED' in baseline else (
        'TARGET_PRED' if case == 'binary' else None)
    if decision:

        def flip(f):
            f.loc[f.index[0], decision] = not f.loc[f.index[0], decision]
            return f

        result['decision_flipped'] = flip
    return result


def fixture_identity(data_dir, graph_metadata):
    """Name the fixture files the producers read, by content."""
    files = {
        f'data/{name}': digest
        for name, digest in tree_hashes(data_dir).items()
    }
    files['graphs.json'] = sha256_file(graph_metadata)
    return dict(data_sha256=hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest(),
                files=len(files))


def run_controller(args):
    script = Path(__file__).resolve()
    # Workers use this interpreter, whose environment holds the candidate.
    python = Path(sys.executable)
    wheel = Path(args.wheel).resolve()
    version = wheel_version(wheel)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A summary from an earlier, interrupted run must not stand for this one.
    (output / 'summary.json').unlink(missing_ok=True)
    work_root = Path(args.work_dir).resolve()
    require_disjoint(output, [work_root])
    summary = dict(
        status='RUNNING', label=args.label, device=args.device,
        attempt=args.attempt, wheel=dict(name=wheel.name,
                                         sha256=sha256_file(wheel),
                                         version=version),
        harness_sha256=sha256_file(script),
        tasks_sha256=sha256_file(tasks.__file__),
        fixture=fixture_identity(Path(args.data_dir), Path(args.graph_metadata)),
        python=str(python),
        python_version=platform.python_version(), system=platform.system(),
        machine=platform.machine(),
        os_release=platform.mac_ver()[0] or platform.release(),
        atol=args.atol, timeout=args.timeout,
        commit=args.commit, required_cases=list(args.case), cases={})
    try:
        summary['wheel']['installed_files_verified'] = verify_installed_wheel(
            wheel)
    except QualificationError as error:
        summary.update(status='FAIL', error=str(error))
        (output / 'summary.json'
         ).write_text(json.dumps(summary, indent=2, default=str) + '\n')
        print(f'Candidate wheel check failed: {error}', file=sys.stderr)
        return 1
    for case in args.case:
        evidence = output / case
        if evidence.exists():
            shutil.rmtree(evidence)
        evidence.mkdir(parents=True)
        work = work_root / case
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        ctx = dict(case=case, device=args.device, python=python, script=script,
                   version=version, timeout=args.timeout, atol=args.atol,
                   work=work, evidence=evidence,
                   data_dir=Path(args.data_dir).resolve(),
                   graph_metadata=Path(args.graph_metadata).resolve())
        row = dict(status='RUNNING')
        started = time.monotonic()
        try:
            details, durable, baseline = run_case(ctx)
            row.update(details)
            if args.negative_checks:
                row['negative_checks'] = negative_checks(ctx, durable, baseline)
                row['negative_checks_not_applicable'] = ctx.get(
                    'not_applicable', [])
                missed = [
                    c['name'] for c in row['negative_checks']
                    if not c['detected']
                ]
                if missed:
                    raise QualificationError(
                        f'Negative checks were not detected: {missed}')
            row['status'] = 'PASS'
        except Exception as error:  # noqa: B902 -- record per case.
            row.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            traceback.print_exc()
        row['seconds'] = round(time.monotonic() - started, 3)
        summary['cases'][case] = row
        (evidence / 'result.json'
         ).write_text(json.dumps(row, indent=2, default=str) + '\n')
        print(
            json.dumps(
                dict(case=case, status=row['status'], error=row.get('error'),
                     comparison=row.get('comparison'))), flush=True)
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    passed = all(summary['cases'].get(case, {}).get('status') == 'PASS'
                 for case in args.case)
    summary['status'] = 'PASS' if passed else 'FAIL'
    (output / 'summary.json'
     ).write_text(json.dumps(summary, indent=2, default=str) + '\n')
    return 0 if passed else 1


def required_labels(targets=REQUIRED_TARGETS):
    return [
        f'{name}-{device}-py{python}' for name, device in targets
        for python in PYTHONS
    ]


def host_mismatch(name, summary):
    """Explain why the evidence host is not the labeled platform."""
    system, machine = summary.get('system'), summary.get('machine')
    release = str(summary.get('os_release') or '')
    if name.startswith('macos-'):
        major = release.split('.')[0]
        if system != 'Darwin' or f'macos-{major}' != name:
            return f'Evidence ran on {system} {release}, not {name}.'
        return None
    wsl = 'microsoft' in release.lower()
    expected = {
        'linux-x86_64': ('Linux', 'x86_64', False),
        'linux-arm64': ('Linux', 'aarch64', False),
        'windows-wsl': ('Linux', 'x86_64', True),
    }.get(name)
    if expected != (system, machine, wsl):
        return f'Evidence ran on {system} {machine} {release}, not {name}.'
    return None


def summary_mismatch(label, summary):
    """Explain why a summary cannot stand for its label, else None."""
    name, device, python = label.rsplit('-', 2)
    if summary.get('device') != device:
        return f'Evidence ran device {summary.get("device")}, not {device}.'
    if not str(summary.get('python_version',
                           '')).startswith(python.removeprefix('py') + '.'):
        return f'Evidence ran Python {summary.get("python_version")}.'
    if summary.get('atol') != REQUIRED_ATOL:
        return f'Evidence used atol={summary.get("atol")}, not {REQUIRED_ATOL}.'
    if not (summary.get('wheel') or {}).get('installed_files_verified'):
        return 'The installed files were not checked against the wheel.'
    return host_mismatch(name, summary)


def build_report(evidence_dirs, *, commit=None, targets=REQUIRED_TARGETS):
    """Merge job summaries; a configuration without evidence is UNVERIFIED."""
    found = {}
    if isinstance(evidence_dirs, (str, Path)):
        evidence_dirs = [evidence_dirs]
    paths = [p for d in evidence_dirs for p in Path(d).rglob('summary.json')]
    for path in sorted(paths):
        summary = json.loads(path.read_text())
        label = summary.get('label')
        attempt = summary.get('attempt') or 0
        if label in found and found[label][1].get('attempt', 0) == attempt:
            raise QualificationError(f'Two summaries for {label} #{attempt}')
        # A re-run of failed jobs keeps the newest result per configuration.
        if label not in found or found[label][1].get('attempt', 0) < attempt:
            found[label] = (path, summary)
    # Every configuration must run one harness, one task module, one fixture
    # and one wheel version.
    identities = {
        key: {json.dumps(s.get(key), sort_keys=True)
              for _, s in found.values()}
        for key in ('harness_sha256', 'tasks_sha256', 'fixture')
    }
    identities['version'] = {(s.get('wheel') or {}).get('version')
                             for _, s in found.values()}
    mixed = sorted(key for key, values in identities.items()
                   if len(values) > 1)
    rows = []
    for label in required_labels(targets):
        path, summary = found.get(label, (None, None))
        for case in CASES:
            row = dict(label=label, case=case, status='UNVERIFIED',
                       reason='No evidence was produced for this case.')
            if summary is not None:
                result = summary.get('cases', {}).get(case)
                row.update(evidence=str(path), wheel=summary.get('wheel'),
                           harness_sha256=summary.get('harness_sha256'))
                if result is None and summary.get('status') == 'FAIL':
                    row.update(status='FAIL', reason=summary.get('error')
                               or 'The job failed before this case.')
                elif result is None:
                    row['reason'] = 'The job did not run this case.'
                elif summary.get('commit') != commit:
                    row.update(status='FAIL',
                               reason='Evidence names another commit.')
                elif mixed:
                    row.update(status='FAIL',
                               reason=f'Configurations ran different {mixed}.')
                elif not summary.get('tasks_sha256') or not (
                        summary.get('fixture') or {}).get('data_sha256'):
                    row.update(status='FAIL',
                               reason='Evidence lacks the task or fixture id.')
                elif summary_mismatch(label, summary):
                    row.update(status='FAIL',
                               reason=summary_mismatch(label, summary))
                else:
                    row.update(status=result.get('status', 'FAIL'),
                               reason=result.get('error'),
                               comparison=result.get('comparison'))
                    checks = result.get('negative_checks')
                    if checks is not None:
                        row['negative_checks'] = {
                            c['name']: c['detected']
                            for c in checks
                        }
                    if row['status'] == 'PASS' and (not checks or
                                                    not all(c['detected']
                                                            for c in checks)):
                        row.update(status='FAIL',
                                   reason='Negative checks are missing.')
                    name, device, _ = label.rsplit('-', 2)
                    needed = BASE_CHECKS.get(case, ()) + REQUIRED_CHECKS.get(
                        (name, device), {}).get(case, ()) + (
                            'prediction_score_shift', )
                    absent = [
                        n for n in needed
                        if n not in (row.get('negative_checks') or {})
                    ]
                    if row['status'] == 'PASS' and absent:
                        row.update(
                            status='FAIL',
                            reason=f'Required negative checks are '
                            f'missing: {absent}')
            rows.append(row)
    unexpected = sorted(set(found) - set(required_labels(targets)))
    status = 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'OPEN'
    return dict(status=status, commit=commit, rows=rows,
                unexpected_labels=unexpected)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'matrix.json'
     ).write_text(json.dumps(report, indent=2, default=str) + '\n')
    lines = [
        f'Desktop reload qualification: {report["status"]}', '',
        '| Configuration | Case | Status | Detail |',
        '| --- | --- | --- | --- |'
    ]
    for row in report['rows']:
        detail = row.get('reason') or json.dumps(row.get('comparison'))
        detail = str(detail).replace('|', '/').replace('\n', ' ')[:160]
        lines.append(
            f'| {row["label"]} | {row["case"]} | {row["status"]} | {detail} |')
    (output / 'matrix.md').write_text('\n'.join(lines) + '\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--wheel', required=True)
    run.add_argument('--device', choices=DEVICES, required=True)
    run.add_argument('--case', choices=CASES, action='append',
                     help='Default: all eight tasks.')
    run.add_argument('--data-dir', required=True)
    run.add_argument('--graph-metadata', required=True)
    run.add_argument('--work-dir', required=True)
    run.add_argument('--output-dir', required=True)
    run.add_argument('--attempt', type=int,
                     default=int(os.environ.get('GITHUB_RUN_ATTEMPT', '1')))
    run.add_argument('--label')
    run.add_argument('--commit')
    run.add_argument('--atol', type=float, default=REQUIRED_ATOL)
    run.add_argument('--timeout', type=float, default=1800)
    run.add_argument('--negative-checks', action='store_true')
    run.add_argument('--keep-work', action='store_true')
    report = commands.add_parser('report')
    report.add_argument('--evidence-dir', action='append', required=True)
    report.add_argument('--output-dir', required=True)
    report.add_argument('--commit', required=True)
    # CI produces the hosted rows only; the release report adds the Macs.
    report.add_argument('--targets', choices=('all', 'hosted'), default='all')
    for name in ('produce', 'consume'):
        worker = commands.add_parser(name)
        worker.add_argument('--case', choices=CASES, required=True)
        worker.add_argument('--device', choices=DEVICES, required=True)
        worker.add_argument('--expected-version', required=True)
        worker.add_argument('--durable', required=True)
        worker.add_argument('--scratch', required=True)
        worker.add_argument('--receipt', required=True)
        if name == 'produce':
            worker.add_argument('--raw', required=True)
            worker.add_argument('--data-dir', required=True)
            worker.add_argument('--graph-metadata', required=True)
        else:
            worker.add_argument('--producer-receipt', required=True)
            worker.add_argument('--output', required=True)
            worker.add_argument('--skip-preflight', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'run':
        args.case = args.case or list(CASES)
        return run_controller(args)
    if args.command == 'report':
        report = build_report(
            args.evidence_dir, commit=args.commit,
            targets=HOSTED_TARGETS if args.targets == 'hosted' else
            REQUIRED_TARGETS)
        write_report(report, args.output_dir)
        print(Path(args.output_dir, 'matrix.md').read_text())
        return 0 if report['status'] == 'PASS' else 1
    return worker_main(args)


if __name__ == '__main__':
    sys.exit(main())
