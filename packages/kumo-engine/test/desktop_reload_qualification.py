#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save, restart, reload and predict two GPU-smoke tasks on one wheel.

Run it outside the checkout, next to wheel_gpu_smoke.py, with the Python of
the venv that holds the wheel. See docs/development/releasing-kumo-engine.md.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, suppress
from importlib import metadata, util
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

GPU_SMOKE = Path(__file__).with_name('wheel_gpu_smoke.py')
SPEC = util.spec_from_file_location(GPU_SMOKE.stem, GPU_SMOKE)
tasks = util.module_from_spec(SPEC)
SPEC.loader.exec_module(tasks)

TASKS = ('binary', 'static_link')
TORCH = {'cpu': 'cpu', 'gpu': 'cuda', 'mps': 'mps'}
# CUDA kernels are not bit-reproducible, so CUDA uses the GPU smoke's bound.
ATOL = {'cpu': 1e-4, 'mps': 1e-4, 'gpu': 5e-3}
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
    """Name the OS or macOS version, CPU, device and Python of this run."""
    if sys.platform == 'darwin':
        host = 'macos-' + platform.mac_ver()[0].split('.')[0]
    elif 'microsoft' in platform.release().lower():
        host = 'windows-wsl'
    else:
        host = 'linux-' + platform.machine().replace('aarch64', 'arm64')
    return f'{host}-{device}-py{sys.version_info[0]}.{sys.version_info[1]}'


def check_wheel(wheel):
    """Return the SHA-256 of the wheel that this venv installed and imports."""
    import kumo.engine as kumo

    source = metadata.distribution('kumo-engine').read_text('direct_url.json')
    url = urlsplit(json.loads(source or '{}').get('url', ''))
    if url.scheme != 'file' or url.netloc or url.query or url.fragment:
        raise RuntimeError('kumo-engine is not installed from a local wheel')
    path = Path(url2pathname(url.path)).resolve()
    if os.path.normcase(path) != os.path.normcase(wheel.resolve()):
        raise RuntimeError(f'kumo-engine is installed from {path}, not {wheel}')
    module = Path(kumo.__file__).resolve()
    if (sys.prefix == sys.base_prefix
            or not module.is_relative_to(Path(sys.prefix).resolve())):
        raise RuntimeError(f'kumo.engine is not imported from a venv: {module}')
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


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
        tensors = [v for v in values if torch.is_tensor(v) and v.numel()]
        work[key].update(x.device.type for x in [module, *tensors])

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

    step = lambda optimizer, *_: work['steps'].add(type(optimizer).__name__)
    with register_optimizer_step_post_hook(step):
        with patch.object(pl, 'Trainer', Observed):
            with patch.object(mips.MIPSFaiss, '_search', traced):
                yield


def worker(phase, task, device, durable, out, base, data):
    """Produce or consume one task here. Save the prediction and the work."""
    if os.environ.get('DRQ_HIDE_FAISS'):
        sys.modules['faiss'] = None  # Each import of Faiss now fails.
    import numpy as np
    import torch
    from kumoapi.jobs import MetadataField

    import kumo.engine as kumo

    durable, out, base = Path(durable), Path(out), Path(base)
    files = kumo.FileConnector(root_dir=str(durable))
    seed = lambda: (random.seed(0), np.random.seed(0), torch.manual_seed(0))
    kumo.configure(scratch_dir=base / 'scratch')
    work = dict(train=set(), predict=set(), search=set(), steps=set())
    if phase == 'produce':  # Only deleted producer storage holds the tables.
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
        options = dict(entities=files['inputs.parquet'])  # As the GPU smoke.
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
    """Run one worker in a new process. Then kill its process group and
    delete its scratch, HOME, TMPDIR, XDG cache, cwd and fixture copy."""
    base = Path(tempfile.mkdtemp(prefix='drq-', dir='/tmp'))
    for name in ('home', 'cache', 'cwd', 'tmp', 'scratch'):
        (base / name).mkdir()
    drop = ('KUMO_', 'DISKGRAPH_')  # Could point at other binaries or data.
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
    """Produce, delete, consume and compare one task. Then make sure that each
    failure case fails a fresh consumer clearly. Record each step in result."""
    import pandas as pd

    durable = work / task
    durable.mkdir()
    for phase in ('produce', 'consume'):
        if spawn(phase, task, device, durable, out / phase,
                 data if phase == 'produce' else None):
            raise RuntimeError(f'The {phase} process failed: {phase}.log')
        # Batches and searches run on the device. The consumer does not train.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    qualification = commands.add_parser('run')
    qualification.add_argument('--device', choices=TORCH, required=True)
    qualification.add_argument('--commit', required=True)
    for flag in ('--wheel', '--data-dir', '--output-dir'):
        qualification.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
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
            # The largest worker's ru_maxrss: KiB on Linux, bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            summary['peak_rss_mib'] = round(
                rss / (1 << 20 if sys.platform == 'darwin' else 1 << 10))
            print(task, json.dumps(result), flush=True)
            (output / 'summary.json').write_text(json.dumps(summary, indent=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all(r['status'] == 'PASS'
                    for r in summary['tasks'].values()) else 1


if __name__ == '__main__':
    if sys.argv[1:2] != ['worker']:
        sys.exit(main())
    try:
        worker(*sys.argv[2:])
    except BaseException as error:
        Path(sys.argv[6]).with_suffix('.error').write_text(
            f'{type(error).__name__}: {error}')
        raise
