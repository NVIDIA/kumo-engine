# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record operation times and reuse the stress runner's resource monitor."""

import hashlib
import json
import logging
import math
import os
import runpy
import subprocess
import sys
import time
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from unittest.mock import patch


def verify_loader_profile(observed, *, batch_size, workers):
    if (observed.get('train_batch_size') != batch_size
            or observed.get('eval_batch_size') != batch_size
            or observed.get('workers') != workers):
        raise ValueError('The actual batch size or workers differ from the fixed profile')


def training_batch_examples(batch, module):
    seed = (getattr(module, 'lhs_name', None)
            or getattr(module, 'entity_table_name', None)
            or getattr(batch, 'input_type', None))
    try:
        size = batch[seed].batch_size
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Missing actual seed-row count in training batch') from error
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
        raise ValueError('Invalid actual seed-row count in training batch')
    return int(size)


def verify_training_counts(row, *, budget=None):
    work, window = row.get('training_work'), row.get('training_window', {})
    if not isinstance(work, list) or len(work) != 1 or not isinstance(window, dict):
        raise ValueError('Exactly one measured training workflow is required')
    record = work[0]
    fields = ('completed_batches', 'completed_examples', 'optimizer_steps')
    if (record.get('status') != 'COMPLETE'
            or type(record.get('early_stopped')) is not bool
            or any(type(record.get(key)) is not int or record[key] <= 0
                   or window.get(key) != record[key] for key in fields)):
        raise ValueError('Training work is missing, failed, or inconsistent')
    sizes = window.get('batch_examples', [])
    if (not isinstance(sizes, list) or len(sizes) != record['completed_batches']
            or any(type(size) is not int or size <= 0 for size in sizes)
            or sum(sizes) != record['completed_examples']):
        raise ValueError('Completed training examples differ from actual batches')
    if budget is not None and (
            not isinstance(budget, list) or len(budget) != 2
            or any(type(value) is not int or value <= 0 for value in budget)
            or record['completed_batches'] > budget[0] * budget[1]
            or record['early_stopped']):
        raise ValueError('Actual training work differs from the declared fault budget')


def allocate_memory_control(*, cpu_bytes=0, gpu_bytes=0):
    """Retain real allocations for a separate declared memory-growth trial."""
    for size in (cpu_bytes, gpu_bytes):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError('Memory-control byte counts must be nonnegative integers')
    buffers = []
    if cpu_bytes:
        host = bytearray(cpu_bytes)
        for offset in range(0, cpu_bytes, 4096):
            host[offset] = 1
        buffers.append(host)
    if gpu_bytes:
        import torch

        device = torch.ones(gpu_bytes, dtype=torch.uint8, device='cuda')
        torch.cuda.synchronize()
        buffers.append(device)
    return buffers, dict(cpu_bytes=int(cpu_bytes), gpu_bytes=int(gpu_bytes),
                         scope='Touched allocations retained through the declared workflow; resource monitors measure their effect')


def verify_training_work(work, window, *, expected_batches, expected_batch_size=None):
    if (expected_batches < 220 or len(work) != 1
            or work[0].get('completed_batches') != expected_batches
            or work[0].get('early_stopped') is not False
            or window.get('completed_batches') != expected_batches
            or window.get('warmup_batches') != 20
            or window.get('timed_batches') != 200
            or not window.get('seconds', 0)):
        raise ValueError('The declared training work or timing window did not complete')
    if expected_batch_size is not None:
        sizes = window.get('batch_examples', [])
        if (expected_batch_size < 1 or len(sizes) != expected_batches
                or any(size != expected_batch_size for size in sizes)
                or window.get('completed_examples') != expected_batches * expected_batch_size
                or window.get('timed_examples') != 200 * expected_batch_size):
            raise ValueError('Actual training examples differ from the fixed full-batch profile')


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
def fixed_loaders(measurements, report, *, batch_size, workers, full_batches=False):
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
        if full_batches:
            if not hasattr(data, 'drop_last_train'):
                raise ValueError('The native loader cannot request complete training batches')
            data.drop_last_train = True
        observed = dict(train_batch_size=data.train_batch_size,
                        eval_batch_size=data.eval_loader_kwargs['batch_size'],
                        workers=data.num_workers,
                        drop_last_train=getattr(data, 'drop_last_train', None),
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

    def __init__(self, synchronize, *, timing=True, control_delay_seconds=0):
        if (not math.isfinite(control_delay_seconds) or control_delay_seconds < 0
                or (control_delay_seconds and not timing)):
            raise ValueError('A timing control needs a finite nonnegative delay and an active timer')
        self.synchronize = synchronize
        self.timing = timing
        self.control_delay_seconds = control_delay_seconds
        self.observed_control_delay_seconds = 0
        self.batches = 0
        self.optimizer_steps = 0
        self.started = None
        self.seconds = None
        self.batch_examples = []

    def batch_start(self):
        if self.timing and self.batches == 20:
            self.synchronize()
            self.started = time.perf_counter()

    def batch_end(self, *, optimizer_steps, examples=None):
        if examples is not None and (isinstance(examples, bool)
                                     or not isinstance(examples, Integral) or examples < 1):
            raise ValueError('Invalid actual example count')
        self.batch_examples.append(int(examples) if examples is not None else None)
        self.batches += 1
        self.optimizer_steps = optimizer_steps
        if self.timing and self.batches == 220:
            self.synchronize()
            if self.control_delay_seconds:
                delay_started = time.perf_counter()
                time.sleep(self.control_delay_seconds)
                self.observed_control_delay_seconds = time.perf_counter() - delay_started
            self.seconds = time.perf_counter() - self.started

    def summary(self):
        def total(values):
            return sum(values) if all(value is not None for value in values) else None

        return dict(completed_batches=self.batches,
                    optimizer_steps=self.optimizer_steps,
                    warmup_batches=min(20, self.batches) if self.timing else 0,
                    timed_batches=200 if self.seconds is not None else 0,
                    batch_examples=list(self.batch_examples),
                    completed_examples=total(self.batch_examples),
                    warmup_examples=total(self.batch_examples[:20]) if self.timing else 0,
                    timed_examples=total(self.batch_examples[20:220]) if self.timing else 0,
                    control_delay_seconds=self.control_delay_seconds,
                    observed_control_delay_seconds=self.observed_control_delay_seconds,
                    seconds=self.seconds)


@contextmanager
def observe_training_work(report, *, timing=False, control_delay_seconds=0):
    """Count actual batches and seed rows, including interrupted fits."""
    import pytorch_lightning as pl
    import torch
    from kumo_core.train.trainer import Trainer

    work = report.setdefault('training_work', [])
    original_create = Trainer.create_pl_trainer

    class WorkCounter(pl.Callback):
        def __init__(self):
            self.window = TrainingWindow(torch.cuda.synchronize, timing=timing,
                                         control_delay_seconds=control_delay_seconds)
            self.record = None

        def on_train_start(self, trainer, module):
            self.record = dict(status='RUNNING')
            work.append(self.record)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.window.batch_start()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.window.batch_end(optimizer_steps=int(trainer.global_step),
                                  examples=training_batch_examples(batch, module))
            report['training_window'] = self.window.summary()
            self.record.update(report['training_window'])

        def on_train_end(self, trainer, module):
            stop = trainer.early_stopping_callback
            self.record.update(status='COMPLETE', current_epoch=int(trainer.current_epoch),
                               early_stopped=bool(stop and stop.stopped_epoch))

        def on_exception(self, trainer, module, error):
            if self.record is not None:
                self.record.update(status='FAIL', error=f'{type(error).__name__}: {error}')

    def create(trainer):
        result = original_create(trainer)
        trainer.pl_trainer.callbacks.append(WorkCounter())
        return result

    with patch.object(Trainer, 'create_pl_trainer', create):
        yield


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
