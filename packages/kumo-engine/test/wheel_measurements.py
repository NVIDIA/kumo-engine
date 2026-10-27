# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record operation times and reuse the stress runner's resource monitor."""

import hashlib
import json
import logging
import os
import runpy
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def verify_loader_profile(observed, *, batch_size, workers):
    if (observed.get('train_batch_size') != batch_size
            or observed.get('eval_batch_size') != batch_size
            or observed.get('workers') != workers):
        raise ValueError('The actual batch size or workers differ from the fixed profile')


def verify_training_work(work, window, *, expected_batches):
    if (expected_batches < 220 or len(work) != 1
            or work[0].get('completed_batches') != expected_batches
            or work[0].get('early_stopped') is not False
            or window.get('completed_batches') != expected_batches
            or window.get('warmup_batches') != 20
            or window.get('timed_batches') != 200
            or not window.get('seconds', 0)):
        raise ValueError('The declared training work or timing window did not complete')


def measure_recommendations(before, after):
    import numpy as np

    keys = ['ENTITY', 'CLASS']
    for frame in (before, after):
        if frame.duplicated(keys).any() or not np.isfinite(frame['SCORE']).all():
            raise ValueError('Duplicate or nonfinite recommendation predictions')
    if not before.groupby('ENTITY').size().equals(after.groupby('ENTITY').size()):
        raise ValueError('Recommendation entity IDs or counts changed')
    joined = before.merge(after, on=keys, how='outer',
                          suffixes=('_before', '_after'), indicator=True,
                          validate='one_to_one')
    common = joined['_merge'] == 'both'
    difference = (joined.loc[common, 'SCORE_before']
                  - joined.loc[common, 'SCORE_after']).abs()
    return dict(status='MEASURED', qualified=False, gpu_tolerance=None,
                common_recommendations=int(common.sum()),
                removed_recommendations=int((joined['_merge'] == 'left_only').sum()),
                added_recommendations=int((joined['_merge'] == 'right_only').sum()),
                changed_entities=int(joined.loc[~common, 'ENTITY'].nunique()),
                max_abs_difference=float(difference.max()) if common.any() else None,
                difference_scope='Scores for shared entity/class coordinates only; membership changes remain separate')


@contextmanager
def fixed_loaders(measurements, report, *, batch_size, workers):
    import pytorch_lightning as pl
    import torch
    from kumo_core.train import diskgraph_nested_trainer as native
    from kumo_core.train.trainer import Trainer

    original_config = native._ttng_diskgraph_runtime_config
    original_create = Trainer.create_pl_trainer
    observations = report['loader_observations'] = []
    prediction_windows = report['model_prediction_windows'] = []

    class PredictionWindow(pl.Callback):
        def on_predict_start(self, trainer, module):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_predict_end(self, trainer, module):
            torch.cuda.synchronize()
            prediction_windows.append(dict(
                stage=(measurements.output / '.stage').read_text(),
                seconds=time.perf_counter() - self.started,
                scope='Lightning prediction callbacks after checkpoint and model setup'))

    def configure(*args, **kwargs):
        auto = original_config(*args, **kwargs)
        if auto is None:
            raise ValueError('Missing native loader configuration')
        return dict(auto, num_workers=workers)

    def create(trainer):
        data = trainer.datamodule
        observed = dict(train_batch_size=data.train_batch_size,
                        eval_batch_size=data.eval_loader_kwargs['batch_size'],
                        workers=data.num_workers,
                        stage=(measurements.output / '.stage').read_text())
        verify_loader_profile(observed, batch_size=batch_size, workers=workers)
        observations.append(observed)
        result = original_create(trainer)
        trainer.pl_trainer.callbacks.append(PredictionWindow())
        return result

    class RejectRetry(logging.Handler):
        def emit(self, record):
            if 'Temporarily decreasing the batch size' in record.getMessage():
                raise ValueError('Automatic memory retry invalidates the fixed profile')

    handler = RejectRetry()
    logger = logging.getLogger('kumo_core.train.trainer')
    logger.addHandler(handler)
    try:
        with (patch.object(native, '_ttng_diskgraph_runtime_config', configure),
              patch.object(Trainer, 'create_pl_trainer', create)):
            yield
    finally:
        logger.removeHandler(handler)


def binary_split_counts(values):
    if not values or any(value not in (False, True, 0, 1)
                         for value in values):
        raise ValueError('Binary evaluation needs finite binary answers')
    positive = sum(bool(value) for value in values)
    if not 0 < positive < len(values):
        raise ValueError('Binary evaluation needs both positive and negative answers')
    return dict(rows=len(values), positive=positive,
                negative=len(values) - positive)


class TrainingWindow:
    """Time 200 complete batches after 20 warmup batches."""

    def __init__(self, synchronize, *, timing=True):
        self.synchronize = synchronize
        self.timing = timing
        self.batches = 0
        self.optimizer_steps = 0
        self.started = None
        self.seconds = None

    def batch_start(self):
        if self.timing and self.batches == 20:
            self.synchronize()
            self.started = time.perf_counter()

    def batch_end(self, *, optimizer_steps):
        self.batches += 1
        self.optimizer_steps = optimizer_steps
        if self.timing and self.batches == 220:
            self.synchronize()
            self.seconds = time.perf_counter() - self.started

    def summary(self):
        return dict(completed_batches=self.batches,
                    optimizer_steps=self.optimizer_steps,
                    warmup_batches=20,
                    timed_batches=200 if self.seconds is not None else 0,
                    seconds=self.seconds)


class Measurements:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.monitor = None

    @contextmanager
    def stage(self, name, *, gpu=True):
        torch = sys.modules.get('torch') if gpu else None
        cuda = torch is not None and torch.cuda.is_available()
        (self.output / '.stage').write_text(name)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        row = dict(stage=name, status='FAIL')
        started = time.perf_counter()
        try:
            yield row
            row['status'] = 'COMPLETE'
        finally:
            if cuda:
                torch.cuda.synchronize()
            row['seconds'] = time.perf_counter() - started
            if cuda:
                row['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                row['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            self.rows.append(row)
            (self.output / 'measurements.json').write_text(
                json.dumps(self.rows, indent=2, allow_nan=False) + '\n')
            (self.output / '.stage').write_text('outside_operation_timer')

    def start_monitor(self, helper, scratch):
        self.helper = Path(helper).resolve()
        self.helper_sha256 = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        (self.output / '.monitor_stop').unlink(missing_ok=True)
        self.monitor_log = (self.output / 'monitor.log').open('w')
        self.monitor = subprocess.Popen(
            [sys.executable, str(self.helper), '--graph', 'commerce',
             '--scratch', '-', '--out', '-', '--_monitor', str(os.getpid()),
             str(self.output), str(scratch), '0.25', '5'],
            stdout=self.monitor_log, stderr=subprocess.STDOUT)
        for _ in range(40):
            if self.monitor.poll() is not None:
                raise RuntimeError('Resource monitor exited before measurement')
            if (self.output / 'monitor.jsonl').exists():
                return
            time.sleep(0.05)
        raise RuntimeError('Resource monitor produced no samples')

    def finish_monitor(self):
        if self.monitor is None:
            return
        (self.output / '.monitor_stop').touch()
        try:
            self.monitor.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.monitor.terminate()
            self.monitor.wait(timeout=15)
            raise RuntimeError('Resource monitor did not stop cleanly')
        finally:
            self.monitor_log.close()
        if self.monitor.returncode:
            raise RuntimeError('Resource monitor failed')
        helpers = runpy.run_path(str(self.helper))
        summary = helpers['summarize_monitor'](
            self.output / 'monitor.jsonl', helpers['_meminfo']()['MemTotal'])
        if not summary.get('overall', {}).get('samples'):
            raise RuntimeError('Resource monitor produced no usable samples')
        summary.update(helper_sha256=self.helper_sha256,
                       sample_interval_seconds=0.25,
                       heavy_sample_interval_seconds=5,
                       scope='worker process tree and exposed GPU',
                       peaks_are_sampled=True)
        peak = Path('/sys/fs/cgroup/memory.peak')
        if peak.exists():
            summary['container_peak_memory_bytes'] = int(peak.read_text())
            summary['container_peak_scope'] = 'Kernel high-water mark, including charged file cache and shared memory'
        (self.output / 'memory.json').write_text(
            json.dumps(summary, indent=2, allow_nan=False) + '\n')
