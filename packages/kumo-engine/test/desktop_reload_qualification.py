#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save, restart, reload and predict two GPU-smoke tasks on one wheel.

For the binary and static_link tasks of wheel_gpu_smoke.py (next to this
file), a producer process trains on the PR172 fixture, exports the model and
snapshot, and predicts. The controller ends it and deletes its HOME, TMPDIR,
caches, working directory, scratch and fixture copy. A new consumer process
loads the exports and predicts, and the two predictions must match. Run it
outside the checkout with the Python of a clean environment that holds the
candidate wheel. Use a new output folder for each attempt:

    python -I desktop_reload_qualification.py run --wheel <wheel> \
        --device cpu --data-dir <PR172 inputs> --output-dir <new dir> \
        --commit <full commit SHA>
    python -I desktop_reload_qualification.py report <evidence dir>...
"""
import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager, suppress
from pathlib import Path
from unittest.mock import patch

GPU_SMOKE = Path(__file__).with_name('wheel_gpu_smoke.py')
SPEC = importlib.util.spec_from_file_location(GPU_SMOKE.stem, GPU_SMOKE)
tasks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tasks)

TASKS = ('binary', 'static_link')
PYTHONS = ('3.11', '3.12', '3.13', '3.14')
TORCH = {'cpu': 'cpu', 'gpu': 'cuda', 'mps': 'mps'}
# CUDA kernels are not bit-reproducible, so CUDA uses the GPU smoke's bound.
ATOL = {'cpu': 1e-4, 'mps': 1e-4, 'gpu': 5e-3}
# CI runs the HOSTED rows on PYTHONS. A person runs the MAC rows with Python
# 3.12: macOS 14 on CPU (a VM is enough) and MPS on a bare-metal macOS 15.
MAC_PYTHON = '3.12'
HOSTED = ('linux-x86_64-cpu', 'linux-x86_64-gpu', 'linux-arm64-cpu',
          'windows-wsl-cpu', 'windows-wsl-gpu', 'macos-15-cpu', 'macos-26-cpu')
MAC = ('macos-14-cpu', 'macos-15-mps')
NOT_RUN = {
    'macos-26-mps': ('hosted macOS runners are VMs without MPS; needs a '
                     'bare-metal Mac on macOS 26'),
    'linux-arm64-gpu': ('no ARM64 GPU runner in this repository; the ARM64 '
                        'wheel has no GPU Faiss, and the DGX Spark guide '
                        'requires a source build'),
}
# Each failure case must fail a fresh consumer with an error that names it.
ERRORS = {
    'model_corrupt': r'model bundle .* does not match its digest',
    'model_missing': r'model bundle .* is missing',
    'snapshot_corrupt': r'snapshot export .* is corrupt',
    'snapshot_missing': r'No snapshot export at',
    'mps_missing': r'MPSAccelerator`? can not run on your system',
    'mps_too_old': r'UnsupportedAcceleratorError: .*macOS 15',
    'faiss_missing': r'FaissUnavailableError: FAISS is required',
}


def label(device):
    """Name the configuration from the OS, CPU, WSL kernel, macOS version,
    device and Python version that run this script."""
    if sys.platform == 'darwin':
        host = 'macos-' + platform.mac_ver()[0].split('.')[0]
    elif 'microsoft' in platform.release().lower():
        host = 'windows-wsl'
    else:
        host = 'linux-' + platform.machine().replace('aarch64', 'arm64')
    return f'{host}-{device}-py{sys.version_info[0]}.{sys.version_info[1]}'


def check_wheel(wheel):
    """Every file of the candidate wheel is installed unchanged, and
    kumo.engine is imported from one of those files."""
    import kumo.engine as kumo

    roots = [Path(sysconfig.get_path(name)) for name in ('purelib', 'platlib')]
    with zipfile.ZipFile(wheel) as archive:
        [record] = [n for n in archive.namelist() if n.endswith('-info/RECORD')]
        rows = list(csv.reader(archive.read(record).decode().splitlines()))
    verified = set()
    for name, digest, *_ in rows:
        if not digest or '.dist-info/' in name or '.data/' in name:
            continue
        path = next((r / name for r in roots if (r / name).is_file()), None)
        found = path and 'sha256=' + base64.urlsafe_b64encode(
            hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
        if found != digest:
            raise RuntimeError(f'{name} differs from the candidate wheel')
        verified.add(path.resolve())
    if Path(kumo.__file__).resolve() not in verified:
        raise RuntimeError(f'kumo.engine is not the wheel: {kumo.__file__}')
    return hashlib.sha256(Path(wheel).read_bytes()).hexdigest()


def mps_case():
    """The MPS failure case of this host, or None where MPS works."""
    import torch

    release = platform.mac_ver()[0]
    if release and int(release.split('.')[0]) < 15:
        return 'mps_too_old'
    return None if torch.backends.mps.is_available() else 'mps_missing'


@contextmanager
def observe(work):
    """Record the device of each batch, optimizer step and Faiss search."""
    import pytorch_lightning as pl
    import torch
    from kumo_core.modeling.retrieval import mips
    from torch.optim.optimizer import register_optimizer_step_post_hook

    def seen(key, module, outputs):
        values = outputs.values() if isinstance(outputs, dict) else [outputs]
        work[key].update([module.device.type] + [
            value.device.type for value in values
            if isinstance(value, torch.Tensor) and value.numel()
        ])

    class Proof(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, *args):
            seen('train', module, outputs)

        def on_predict_batch_end(self, trainer, module, outputs, *args):
            seen('predict', module, outputs)

    class Observed(pl.Trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), Proof()]
            super().__init__(*args, **kwargs)

    search = mips.MIPSFaiss._search

    def traced(index, *args):
        work['search'].add('torch' if index._cpu_index is not None else
                           'metal' if index._metal_index is not None else
                           'cuda' if index._res is not None else 'faiss-cpu')
        return search(index, *args)

    hook = register_optimizer_step_post_hook(
        lambda optimizer, *_: work['steps'].add(type(optimizer).__name__))
    try:
        with patch.object(pl, 'Trainer', Observed):
            with patch.object(mips.MIPSFaiss, '_search', traced):
                yield
    finally:
        hook.remove()


def worker(phase, task, device, durable, out, base, data):
    """Produce or consume one task in this process. Save the prediction and
    record what the process did."""
    if os.environ.get('DRQ_HIDE_FAISS'):
        sys.modules['faiss'] = None  # Each import of Faiss now fails.
    import random

    import numpy as np
    import torch
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    durable, out, base = Path(durable), Path(out), Path(base)
    files = kumo.FileConnector(root_dir=str(durable))
    seed = lambda: (random.seed(0), np.random.seed(0), torch.manual_seed(0))
    kumo.configure(scratch_dir=base / 'scratch')
    work = dict(train=set(), predict=set(), search=set(), steps=set())
    if phase == 'produce':
        # Only producer storage, which the controller deletes, holds the
        # source tables.
        graph = tasks.TASKS[task]['graph']
        shutil.copytree(Path(data) / graph, base / graph)
        graphs = json.loads((Path(data) / 'graphs.json').read_text())
        seed()
        prepared = tasks.prepare_task(task, base, graphs, base / 'task', {})
        shutil.copy2(prepared['input_file'], durable / 'inputs.parquet')
        snapshot, anchor = prepared['snapshot'], prepared['anchor']
        meta = dict(ids=prepared['ids'], classes=prepared['classes'],
                    anchor=anchor and str(anchor))
    else:
        meta = json.loads(out.with_name('produce.json').read_text())
    with observe(work):
        if phase == 'produce':
            seed()
            trainer = kumo.Trainer(tasks.make_plan(task), accelerator=device)
            trainer.fit(prepared['training_table'], snapshot=snapshot).export(
                kumo.ModelOutputConfig(files, path='model'))
            meta['model'] = trainer.result.id
            meta['snapshot'] = snapshot.export(
                kumo.SnapshotOutputConfig(files, path='snapshot'))
        else:
            trainer = kumo.Trainer.load_from_connector(files, path='model',
                                                       accelerator=device)
            snapshot = kumo.GraphSnapshot.load_from_connector(
                files, path=meta['snapshot'])
        # The prediction options of the GPU smoke.
        options = dict(entities=files['inputs.parquet'])
        if task == 'binary':
            options.update(
                binary_classification_threshold=.5,
                output_config=kumo.PredictionOutputConfig(
                    connector=kumo.FileConnector(root_dir=str(base / 'output')),
                    output_paths={'predictions': 'predictions'},
                    extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
        else:
            options['num_classes_to_return'] = len(meta['classes'])
        seed()
        work['predict'].clear()  # Only this prediction's batches count.
        frame = trainer.predict(snapshot, **options).frame
    frame.to_parquet(out.with_suffix('.parquet'))
    meta.update(work)
    out.with_suffix('.json').write_text(json.dumps(meta, default=sorted))


def spawn(phase, task, device, durable, out, data=None, **extra):
    """Run one worker in a new process. Then end its process group and delete
    each directory it could cache in: scratch, HOME, TMPDIR, XDG cache,
    working directory and fixture copy."""
    base = Path(tempfile.mkdtemp(prefix='drq-', dir='/tmp'))
    for name in ('home', 'cache', 'cwd', 'tmp', 'scratch'):
        (base / name).mkdir()
    # Kumo and DiskGraph variables could point at other binaries or storage.
    drop = ('KUMO_', 'DISKGRAPH_')
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env.update(extra, HOME=str(base / 'home'), TMPDIR=str(base / 'tmp'),
               XDG_CACHE_HOME=str(base / 'cache'),
               PYTORCH_ENABLE_MPS_FALLBACK='0')  # No silent CPU fallback.
    if device == 'cpu':  # A CPU row must not reach a GPU.
        env['CUDA_VISIBLE_DEVICES'] = ''
    command = [sys.executable, '-Iu', __file__, 'worker', phase, task, device]
    command += map(str, (durable, out, base, data))
    with open(out.with_suffix('.log'), 'wb') as log:
        process = subprocess.Popen(command, cwd=base / 'cwd', env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=3600)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            shutil.rmtree(base)


def qualify(task, device, data, out, work, result):
    """Produce, end and delete, then consume one task and compare the two
    predictions. Then make sure that each failure case fails a fresh
    consumer with an error that names it. Record each step in result."""
    import pandas as pd

    durable = work / task
    durable.mkdir()
    for phase in ('produce', 'consume'):
        if spawn(phase, task, device, durable, out / phase,
                 data if phase == 'produce' else None):
            raise RuntimeError(f'The {phase} process failed: {phase}.log')
        # Batches run only on the requested device, the search backend is the
        # expected one, and the consumer does not train.
        want = dict(train=[TORCH[device]], predict=[TORCH[device]])
        if phase == 'consume':
            want.update(train=[], steps=[])
        if task == 'static_link':
            cpu = 'torch' if sys.platform == 'darwin' else 'faiss-cpu'
            want['search'] = [dict(gpu='cuda', mps='metal').get(device, cpu)]
        record = json.loads((out / f'{phase}.json').read_text())
        done = result[phase] = {key: record[key] for key in want}
        if done != want:
            raise RuntimeError(f'The {phase} work is {done}, not {want}')
    meta = json.loads((out / 'produce.json').read_text())
    result.update(model=meta['model'], snapshot=meta['snapshot'])

    def align(phase):  # The GPU smoke's checks of one prediction.
        frame = pd.read_parquet(out / f'{phase}.parquet')
        options = dict(classes=meta['classes'])
        return (tasks.prediction_values(task, frame, meta['ids'],
                                        anchor=meta['anchor'], **options),
                tasks.prediction_decisions(task, frame, meta['ids'], **options))

    (before, chosen), (after, rechosen) = align('produce'), align('consume')
    result.update(
        tasks.compare_predictions(before, after, task=task, atol=ATOL[device],
                                  before_decisions=chosen,
                                  after_decisions=rechosen))
    cases = ('model_corrupt', 'model_missing', 'snapshot_corrupt',
             'snapshot_missing', mps_case())
    result['failure_cases'] = rejected = {}
    for name in filter(None, cases if task == 'binary' else ['faiss_missing']):
        copy, kind = work / name, name.split('_')[0]
        shutil.copytree(durable, copy)
        if kind in ('model', 'snapshot'):
            target = max((p for p in (copy / kind).rglob('*') if p.is_file()),
                         key=lambda p: p.stat().st_size)
            if name == 'snapshot_missing':
                shutil.rmtree(copy / kind)
            elif name == 'model_missing':
                target.unlink()
            else:
                content = bytearray(target.read_bytes())
                content[len(content) // 2] ^= 0xFF
                target.write_bytes(content)
        code = spawn('consume', task, 'mps' if kind == 'mps' else device, copy,
                     out / name, DRQ_HIDE_FAISS='1' if kind == 'faiss' else '')
        shutil.rmtree(copy)
        error = out / f'{name}.error'
        rejected[name] = error.read_text() if code and error.exists() else ''
        if not re.search(ERRORS[name], rejected[name]):
            raise RuntimeError(f'The consumer did not reject {name} clearly: '
                               f'exit {code}, {rejected[name]!r}')
    shutil.rmtree(durable)


def run(args):
    output = args.output_dir.resolve()
    if any((output / name).exists() for name in ('summary.json', *TASKS)):
        sys.exit(f'{output} holds an earlier attempt. Use a new --output-dir.')
    output.mkdir(parents=True, exist_ok=True)
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    summary = dict(label=label(args.device), commit=args.commit, started=now,
                   wheel=args.wheel.name, wheel_sha256=None,
                   cpus=os.cpu_count(), tasks={})
    work = Path(tempfile.mkdtemp(prefix='drq-work-', dir='/tmp'))
    try:
        for task in TASKS:
            started = time.monotonic()
            result = summary['tasks'][task] = {}
            try:  # A wrong wheel fails each task with the reason.
                summary['wheel_sha256'] = check_wheel(args.wheel)
                (output / task).mkdir()
                qualify(task, args.device, args.data_dir.resolve(),
                        output / task, work, result)
                result['status'] = 'PASS'
            except Exception as error:  # Record it, then run the next task.
                traceback.print_exc()
                result.update(status='FAIL',
                              error=f'{type(error).__name__}: {error}')
            result['seconds'] = round(time.monotonic() - started, 1)
            # The largest worker's peak memory. ru_maxrss is in KiB on Linux
            # and in bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            summary['peak_rss_mib'] = round(
                rss / (1 << 20 if sys.platform == 'darwin' else 1 << 10))
            print(task, json.dumps(result), flush=True)
            (output / 'summary.json').write_text(json.dumps(summary, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all(r['status'] == 'PASS'
                    for r in summary['tasks'].values()) else 1


def report(directories, hosted):
    """Print PASS, FAIL or UNVERIFIED for each configuration and task, and the
    rows that are not run. Return 1 unless every required row passes and all
    rows name one commit. `hosted` requires only the CI rows."""
    summaries = [
        json.loads(path.read_text()) for directory in directories
        for path in directory.rglob('summary.json')
    ]
    # Skip the summaries of other tools. The newest run of a row counts.
    summaries = sorted((s for s in summaries
                        if {'label', 'commit', 'started', 'tasks'} <= s.keys()),
                       key=lambda s: s['started'])
    found = {summary['label']: summary['tasks'] for summary in summaries}
    commits = sorted({summary['commit'] for summary in summaries})
    passed = len(commits) < 2
    lines = [
        '| Configuration | ' + ' | '.join(TASKS) + ' |',
        '| --- ' * (len(TASKS) + 1) + '|'
    ]
    rows = [(c, p) for c in HOSTED for p in PYTHONS]
    for config, python in rows + [(c, MAC_PYTHON) for c in MAC]:
        name = f'{config}-py{python}'
        row = [
            found.get(name, {}).get(task, {}).get('status', 'UNVERIFIED')
            for task in TASKS
        ]
        required = config in HOSTED or not hosted
        passed &= not required or row == ['PASS'] * len(TASKS)
        name += '' if required else ' (not required)'
        lines.append(f'| {name} | ' + ' | '.join(row) + ' |')
    lines.append('\nNot run and not required:')
    lines += [f'- {c}: {why}' for c, why in NOT_RUN.items()]
    print(f'Commits (one is required): {", ".join(commits) or "none"}', '',
          *lines, sep='\n')
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    qualification = commands.add_parser('run')
    qualification.add_argument('--device', choices=TORCH, required=True)
    qualification.add_argument('--commit', required=True)
    for flag in ('--wheel', '--data-dir', '--output-dir'):
        qualification.add_argument(flag, type=Path, required=True)
    matrix = commands.add_parser('report')
    matrix.add_argument('evidence', type=Path, nargs='+')
    matrix.add_argument('--hosted', action='store_true',
                        help='Require only the CI rows.')
    args = parser.parse_args()
    if args.command == 'run':
        return run(args)
    return report(args.evidence, args.hosted)


if __name__ == '__main__':
    if sys.argv[1:2] == ['worker']:
        try:
            worker(*sys.argv[2:])
        except BaseException as error:
            Path(sys.argv[6]).with_suffix('.error').write_text(
                f'{type(error).__name__}: {error}')
            raise
    else:
        sys.exit(main())
