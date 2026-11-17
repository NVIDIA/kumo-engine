# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming numerical sensitivity from the original scoring forward pass.

The coordinate is the imputed, scaled, optionally logged and clamped value
entering the learnable numerical encoder. It is deliberately not a derivative
with respect to raw quantities, a causal effect, or a contribution allocation.
"""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
from kumo_core.encoders.target.numerical import NumericalTargetEncoder
from kumo_core.train.loss import MSELoss
from kumoapi.task import TaskType


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _encoded(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def bundle_fingerprints(model_dir: str) -> dict[str, Any]:
    """Hash the exact staged checkpoint, encoders and configuration once."""
    files = {}
    for path in sorted(Path(model_dir).rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            files[str(path.relative_to(model_dir))] = digest.hexdigest()
    return {'bundle_sha256': _digest(files), 'bundle_files_sha256': files}


logger = logging.getLogger(__name__)


def bind_walk_result(result, sample):
    """Bind local activation occurrences to retained source keys, never new edges."""
    result['status'] = 'truncated' if result['truncated'] else 'available'
    for walk in result['walks']:
        walk['source_pkey'] = sample['nodes'][walk['source_table']]['pkeys'][walk['source_occurrence']]
        for step in walk['steps']:
            if step['kind'] == 'relation':
                src, _, dst = step['type']
                step['source_pkey'] = sample['nodes'][src]['pkeys'][step['source_occurrence']]
                step['destination_pkey'] = sample['nodes'][dst]['pkeys'][step['destination_occurrence']]
            elif step['kind'] == 'self':
                step['pkey'] = sample['nodes'][step['table']]['pkeys'][step['occurrence']]
    return result


class LocalSensitivity:
    """One seed, one model forward, one autograd call, incremental artifacts."""

    requires_grad = True
    streaming = True

    def __init__(self, root: str, provenance: dict[str, Any],
                 numerical_sensitivity: bool = True,
                 walk_attribution: dict | None = None,
                 walk_on_unsupported: str = "raise"):
        self.walk_on_unsupported = walk_on_unsupported
        self.walk_config = walk_attribution
        self.walk_capture = None
        self.walk_status_counts = {}
        self.walk_failure = None
        self.walk_failure_unsupported = False
        self.numerical_sensitivity = numerical_sensitivity
        self.requires_grad = numerical_sensitivity
        self.attempted_rows = 0
        self.graph_failures = 0
        self.temporal_status_counts: dict[str, int] = {}
        self.generation_closed = False
        self.owns_sink = False
        self.root = Path(root)
        self.provenance = dict(provenance)
        self.rows = 0
        self.status_counts: dict[str, int] = {}
        self.failure: str | None = None

    def explain_start(self, module, datamodule, stage):
        from kumo_core.explain.walk_attribution import (
            SCHEMA_V1, UnsupportedWalkAttribution, WalkAttributionConfig,
            WalkCapture, select_schema)
        from kumo_core.explain.walk_graph import capture_numerics
        self.binary = module.task_type == TaskType.BINARY_CLASSIFICATION
        if self.binary:
            if self.numerical_sensitivity:
                # It differentiates a decoded scalar (contract section 7).
                raise ValueError('Numerical sensitivity requires scalar MSE '
                                 'regression; binary evidence requires it off')
        elif (module.task_type != TaskType.REGRESSION
                or not isinstance(module.loss, MSELoss) or
                not isinstance(module.target_encoder, NumericalTargetEncoder)):
            raise ValueError('local_sensitivity requires scalar MSE regression '
                             'with a numerical target encoder')
        if module.device.type != 'cpu':
            raise ValueError('local_sensitivity requires CPU execution')
        self.config = (None if self.walk_config is None else
                       WalkAttributionConfig(**self.walk_config))
        self.schema = (None if self.config is None else
                       select_schema(self.config, module.task_type))
        self.legacy = self.schema in (None, SCHEMA_V1) and not self.binary
        # The record coordinate of a binary task uses the default selection
        # (encoded class 1); a walk config may select the other class.
        self.record_selection = self.record_failure = None
        self.walk_selection = self.walk_selection_failure = None
        if not self.legacy:
            from kumo_core.explain.walk_binary import resolve_selection
            try:
                self.record_selection = resolve_selection(
                    module, WalkAttributionConfig(
                        output_index=0 if self.config is None else self.config.output_index))
            except UnsupportedWalkAttribution as error:
                if self.walk_on_unsupported == 'raise':
                    raise
                self.record_failure = str(error)
        if self.config is not None and not self.legacy:
            from kumo_core.explain.walk_attribution import method_module
            from kumo_core.explain.walk_binary import resolve_selection
            try:
                self.walk_selection = resolve_selection(module, self.config)
                # Qualification raises before any hook is registered.
                method_module(self.config.method).qualify(module, self.config)
            except UnsupportedWalkAttribution as error:
                if self.walk_on_unsupported == 'raise':
                    raise
                self.walk_selection_failure = str(error)
        if self.walk_config is not None and self.legacy and self.walk_on_unsupported == 'raise':
            with WalkCapture(module, self.config):
                pass
        if self.binary:
            self._binary_provenance(module, datamodule)
        else:
            self._regression_provenance(module, datamodule)
        # Precision of the original forward, applied again by offline replay.
        self.provenance['numerics'] = capture_numerics()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if any(self.root.iterdir()):
                raise FileExistsError('Evidence destination must be empty')
            with (self.root / '.capture-owner').open('x') as owner:
                owner.write(str(self.provenance['run_id']))
            self.owns_sink = True
            (self.root / 'rows').mkdir(exist_ok=True)
            (self.root / 'samples').mkdir(exist_ok=True)
            self._write_manifest(complete=False)
            (self.root / 'index.jsonl').write_text('')
        except OSError:
            self.failure = 'local_artifact_write_failed'
            logger.warning(
                'Local sensitivity sink unavailable; predictions continue')

    def _temporal_boundary(self):
        return {
            'sampler': 'at_or_before_seed_time',
            'strict_validation':
            'source_time_strictly_before_prediction_cutoff',
            'coordinate': 'unix_nanoseconds_from_native_sampler',
            'source_time_columns': {
                table: node.get('time_column')
                for table, node in self.provenance.get(
                    'model_reconstruction', {}).get(
                        'feature_schema', {}).get('node_types', {}).items()
            },
        }

    def _sampling_plan(self):
        # Sampling plans may be dataclasses/pydantic objects in native runtimes.
        plan = self.provenance['sampling_plan']
        if hasattr(plan, 'model_dump'):
            self.provenance['sampling_plan'] = plan.model_dump(mode='json')
        elif hasattr(plan, '__dataclass_fields__'):
            from dataclasses import asdict
            self.provenance['sampling_plan'] = asdict(plan)

    def _binary_provenance(self, module, datamodule):
        from kumo_core.explain.walk_binary import calibration_record
        selection = self.record_selection
        self.provenance.update({
            'schema_version': 1,
            'method': None,
            'origin': 'original_forward',
            'raw_value_coordinate': 'numerical_tensor_mapper_output',
            'signed_method': None,
            'input_coordinate': None,
            'output_coordinate': 'encoded_positive_class_probability',
            'numerical_sensitivity': False,
            'temporal_boundary': self._temporal_boundary(),
            'device': 'cpu',
            'loss': type(module.loss).__name__,
            'batch_size': 1,
            'sampler_seed': getattr(datamodule, 'base_seed', None),
            'sampling_plan': getattr(datamodule, 'sampling_plan', None),
            'target_transform': None,
            'task_type': 'binary_classification',
            'binary_output': None if selection is None else {
                'label_mapping': [dict(e) for e in selection.binary.label_mapping],
                'target_encoder_reverse': selection.binary.target_encoder_reverse,
                'encoded_positive_label': selection.binary.encoded_positive_label,
                'calibration': calibration_record(module),
            },
            'binary_output_failure': self.record_failure,
        })
        self._sampling_plan()

    def _regression_provenance(self, module, datamodule):
        self.provenance.update({
            'schema_version':
            1,
            'method':
            'absolute_gradient_decoded_output_wrt_transformed_numerical_input',
            'origin':
            'original_forward',
            'raw_value_coordinate':
            'numerical_tensor_mapper_output',
            'signed_method':
            'gradient_decoded_output_wrt_transformed_numerical_input',
            'input_coordinate':
            'transformed_numerical_input',
            'output_coordinate':
            'decoded_model_output',
            'numerical_sensitivity':
            self.numerical_sensitivity,
            'temporal_boundary': self._temporal_boundary(),
            'device':
            'cpu',
            'loss':
            'MSELoss',
            'batch_size':
            1,
            'sampler_seed':
            getattr(datamodule, 'base_seed', None),
            'sampling_plan':
            getattr(datamodule, 'sampling_plan', None),
            'target_transform': {
                'offset': module.target_encoder.offset,
                'scale': module.target_encoder.scale,
                'log_transform': module.target_encoder.log_transform,
            },
        })
        self._sampling_plan()

    def _write_manifest(self, *, complete: bool):
        if not self.owns_sink:
            return
        temporary = self.root / 'manifest.json.tmp'
        temporary.write_text(
            _encoded({
                **self.provenance,
                'complete': complete,
                'rows': self.rows,
                'failure': self.failure,
                'status_counts': self.status_counts,
                'walk_status_counts': self.walk_status_counts,
                'attempted_rows': self.attempted_rows,
                'graph_failures': self.graph_failures,
                'temporal_status_counts': self.temporal_status_counts,
                'generation_closed': self.generation_closed,
            }))
        os.replace(temporary, self.root / 'manifest.json')

    def explain_pre_step(self, batch, module, datamodule, stage, batch_idx):
        if batch[batch.input_type].batch_size != 1:
            raise ValueError('local_sensitivity requires one seed per batch')
        if self.walk_config is not None:
            from kumo_core.explain.walk_attribution import (
                WalkAttributionConfig, WalkCapture, UnsupportedWalkAttribution)
            try:
                self.walk_failure = None
                if not self.legacy:
                    from kumo_core.explain.walk_graph import GraphCapture
                    if self.walk_selection_failure is not None:
                        raise UnsupportedWalkAttribution(self.walk_selection_failure)
                    self.walk_capture = GraphCapture(module, self.config)
                else:
                    self.walk_capture = WalkCapture(module, WalkAttributionConfig(**self.walk_config))
                self.walk_capture.__enter__()
            except Exception as error:
                self.walk_failure_unsupported = isinstance(error, UnsupportedWalkAttribution)
                self.walk_failure = str(error) if self.walk_failure_unsupported else "capture_setup_failed"
                if self.walk_capture is not None:
                    self.walk_capture.__exit__(None, None, None)
                self.walk_capture = None
        # Clear tensors left by a prior batch, including unused encoders.
        for encoder in module.encoder.modules():
            if hasattr(encoder, '_saved_tensors'):
                del encoder._saved_tensors

    def _sample(self, batch, module) -> dict[str, Any]:
        nodes = {}
        for table in batch.node_types:
            block = batch[table]
            pkeys = block.get('pkey')
            count = int(block.num_nodes or 0)
            if pkeys is None and count:
                # The synthetic task table has no source primary key.
                if table == module.train_table_name:
                    pkeys = batch[batch.input_type].get('input_id')
                if pkeys is None:
                    raise ValueError(f'Missing source primary keys for {table}')
            nodes[table] = {
                'pkeys':
                _plain(pkeys) if pkeys is not None else [],
                'times_ns':
                _plain(block.get('time')),
                'node_ids':
                _plain(block.get('n_id')),
                'num_nodes':
                count,
                'identity_kind': ('task_row' if table == module.train_table_name
                                  else 'source_primary_key'),
                'num_sampled_nodes':
                _plain(block.get('num_sampled_nodes')),
                'batch':
                _plain(block.get('batch')),
                'batch_size':
                block.get('batch_size'),
                'input_id':
                _plain(block.get('input_id')),
                'seed_time':
                _plain(block.get('seed_time')),
            }
            frame = block.get('tf')
            if frame is not None:
                if any(not isinstance(value, torch.Tensor)
                       for value in frame.feat_dict.values()):
                    raise ValueError('Unsupported non-tensor feature capture')
                nodes[table]['feature_tensors'] = {
                    str(key): {
                        'values':
                        _plain(value),
                        'dtype':
                        str(value.dtype),
                        'shape':
                        list(value.shape),
                        'nonfinite': (_plain(~torch.isfinite(value))
                                      if value.is_floating_point() else None),
                        'nonfinite_kind': (_plain(
                            torch.isnan(value).to(torch.int8) +
                            2 * torch.isposinf(value).to(torch.int8) +
                            3 * torch.isneginf(value).to(torch.int8)) if
                                           value.is_floating_point() else None),
                        'columns':
                        frame.col_names_dict[key]
                    }
                    for key, value in frame.feat_dict.items()
                    if isinstance(value, torch.Tensor)
                }
                nodes[table]['numerical_features'] = {
                    column: _plain(frame.feat_dict[key][:, j])
                    for key, columns in frame.col_names_dict.items()
                    if str(key) in ('numerical', 'stype.numerical')
                    for j, column in enumerate(columns)
                }
        edges = []
        for kind in batch.edge_types:
            block = batch[kind]
            edges.append({
                'type':
                list(kind),
                'edge_index':
                _plain(block.edge_index),
                'num_sampled_edges':
                _plain(block.get('num_sampled_edges')),
                'original_edge_index':
                _plain(getattr(block, '_orig_edge_index', None))
            })
        return {
            'origin':
            'original_forward',
            'seed_table':
            batch.input_type,
            'train_table_name':
            module.train_table_name,
            'entity_table_name':
            getattr(module, 'entity_table_name', batch.input_type),
            'nodes':
            nodes,
            'edges':
            edges
        }

    def _cells(self, module, batch, output):
        entries = []
        for table, tf_encoder in module.encoder.enc_dict.items():
            if table == module.train_table_name:
                continue
            for encoder in tf_encoder.stype_encoder_dict.values():
                if not isinstance(encoder, FusedNumericalEncoder):
                    continue
                saved = getattr(encoder, '_saved_tensors', ())
                if not saved:
                    continue
                transformed = saved[0]
                tf = batch[table].tf
                matching = [
                    key for key, names in tf.col_names_dict.items()
                    if names == encoder.col_names
                ]
                if len(matching) != 1:
                    raise ValueError(f'Ambiguous numerical columns for {table}')
                raw = tf.feat_dict[matching[0]]
                keys = batch[table].get('pkey')
                if keys is None or len(keys) != transformed.size(0):
                    raise ValueError(
                        f'Missing numerical source mapping: {table}')
                entries.append((table, encoder, transformed, raw, keys))
        if not entries:
            return [], 'not_sampled', 'no_sampled_numerical_inputs'
        decoded = module.target_encoder.decode_tensor(
            module.loss.to_prediction(output['logits']))
        if decoded.numel() != 1:
            raise ValueError('local_sensitivity requires a scalar output')
        # Exactly one differentiation call; no replay or parameter backward.
        gradients = torch.autograd.grad(decoded.reshape(
            ()), [entry[2] for entry in entries], allow_unused=True,
                                        retain_graph=False)
        cells = []
        for (table, encoder, transformed, raw,
             keys), grad in zip(entries, gradients):
            for i, key in enumerate(keys):
                for j, column in enumerate(encoder.col_names):
                    signed = None if grad is None else float(grad[i, j])
                    value = None if signed is None else abs(signed)
                    finite = value is not None and math.isfinite(value)
                    cells.append({
                        'table':
                        table,
                        'pkey':
                        _plain(key),
                        'local_id':
                        i,
                        'column':
                        column,
                        'raw_value':
                        _plain(raw[i, j]),
                        'transformed_value':
                        _plain(transformed[i, j]),
                        'abs_gradient':
                        value if finite else None,
                        'signed_gradient':
                        signed if finite else None,
                        'mapper_value':
                        _plain(raw[i, j]),
                        'source_value':
                        None,
                        'source_value_status':
                        'not_provided',
                        'missing':
                        not bool(torch.isfinite(raw[i, j])),
                        'missing_kind':
                        ('nan' if bool(torch.isnan(raw[i, j]))
                         else 'positive_infinity' if bool(
                             torch.isposinf(raw[i, j])) else 'negative_infinity'
                         if bool(torch.isneginf(raw[i, j])) else None),
                        'input_dtype':
                        str(raw.dtype),
                        'status': ('available' if finite else 'unsupported'),
                        'reason': (None if finite else 'disconnected_input'
                                   if grad is None else 'nonfinite_gradient'),
                        'transform': {
                            'offset': _plain(encoder.offsets[j]),
                            'scale': _plain(encoder.scales[j]),
                            'log_transform': bool(encoder.log_mask[j]),
                            'impute_value': _plain(encoder.impute_values[j]),
                            'clamp': [-1000, 1000],
                            'excluded': bool(encoder.exclude_cols_mask[j]),
                        },
                    })
        values = [
            cell['abs_gradient'] for cell in cells
            if cell['abs_gradient'] is not None
        ]
        status = ('unsupported' if len(values) != len(cells) else
                  'zero_signal' if not any(values) else 'available')
        return cells, status, None

    def explain_post_step(self, batch, module, datamodule, step_output, stage,
                          batch_idx):
        self.attempted_rows += 1
        try:
            identity = self._identity(batch, step_output)
            step_output['prediction_id'] = [
                _digest([self.provenance['run_id'], identity])
            ]
            step_output['evidence_task_row_id'] = identity['task_row_id']
            step_output['evidence_cutoff_ns'] = identity['time_ns'] or [None]
            if self.failure is None:
                self._record_step(batch, module, step_output, batch_idx)
        except Exception:
            self.failure = 'local_artifact_capture_failed'
            logger.warning(
                'Local sensitivity capture unavailable; predictions continue')
        finally:
            if self.walk_capture is not None:
                self.walk_capture.__exit__(None, None, None)
                self.walk_capture = None
            for encoder in module.encoder.modules():
                if hasattr(encoder, '_saved_tensors'):
                    del encoder._saved_tensors

    def _identity(self, batch, step_output):
        seed = batch[batch.input_type]
        task_row_id = _plain(seed.get('input_id'))
        if not isinstance(task_row_id, list) or len(task_row_id) != 1:
            raise ValueError('Missing unique materialized task-row identity')
        return {
            'entity': _plain(step_output['pkey'][0]),
            'time_ns': _plain(step_output.get('time')),
            'task_row_id': task_row_id,
            # A binary record is identified by the raw head logit it explains.
            'output_coordinate': ('encoded_logit' if getattr(self, 'binary', False)
                                  else 'scalar'),
        }

    def _temporal_validation(self, sample, identity):
        spec = self.provenance.get('model_reconstruction', {})
        if not spec.get('metadata', {}).get('is_temporal', False):
            return {'status': 'not_applicable', 'reason': 'static_task'}
        cutoff = identity['time_ns']
        if not isinstance(cutoff, list) or len(cutoff) != 1:
            return {
                'status': 'unavailable',
                'reason': 'prediction_cutoff_unavailable'
            }
        columns = self.provenance['temporal_boundary']['source_time_columns']
        checked, violations, missing = 0, [], []
        for table, node in sample['nodes'].items():
            if not columns.get(table) or not node['num_nodes']:
                continue
            times = node.get('times_ns')
            if times is None or len(times) != node['num_nodes']:
                missing.append(table)
                continue
            for local_id, timestamp in enumerate(times):
                if timestamp is None:
                    missing.append(table)
                else:
                    checked += 1
                    if timestamp >= cutoff[0]:
                        violations.append({
                            'table': table,
                            'local_id': local_id,
                            'pkey': node['pkeys'][local_id],
                            'source_time_column': columns[table],
                            'time_ns': timestamp
                        })
        status = 'violated' if violations else (
            'unavailable' if missing else 'valid')
        return {
            'status':
            status,
            'checked_records':
            checked,
            'reason': ('strict_temporal_boundary_violation' if violations else
                       'source_timestamps_unavailable' if missing else None),
            'unavailable_tables':
            sorted(set(missing)),
            'violations':
            violations,
            'cutoff_ns':
            cutoff[0],
            'coordinate':
            'unix_nanoseconds'
        }

    def _v2_row(self, module, logits, record, sample, prediction_id):
        """The walk_attribution/v2 row of one original scoring forward."""
        from kumo_core.explain.walk_binary import output_block
        from kumo_core.explain.walk_graph import capture_numerics
        from kumo_core.explain.walk_schema import (
            explain_capture, minimal_row, row_numerics)
        identity = dict(config=self.config, origin='original_forward_attribution',
                        prediction_id=prediction_id,
                        sample_sha256=record['sample_sha256'])
        numerics = row_numerics(capture_numerics(),
                                str(logits.dtype).removeprefix('torch.'))
        if self.walk_capture is None:
            if self.walk_failure_unsupported:
                return minimal_row('unsupported', self.walk_failure, **identity)
            return minimal_row('failed', self.walk_failure or 'walk_capture_failed',
                               **identity)
        if record['graph_status'] != 'available':
            return minimal_row('failed', 'original_graph_missing', **identity)
        try:
            output = output_block(self.walk_selection, module, logits[0])
        except Exception as error:
            return minimal_row('failed', f'{type(error).__name__}: {error}', **identity)
        return explain_capture(self.walk_capture, selection=self.walk_selection,
                               output=output, sample=sample, numerics=numerics,
                               **identity)

    def _record_step(self, batch, module, step_output, batch_idx):
        identity = self._identity(batch, step_output)
        prediction_id = _digest([self.provenance['run_id'], identity])
        if (self.root / 'rows' / f'{prediction_id}.json').exists():
            raise ValueError('Duplicate prediction identity')
        logits = step_output['logits']
        if self.binary:
            from kumo_core.explain.walk_binary import emitted_probabilities
            # The emitted True_PROB, from the module's own calibrator and loss.
            prediction = float(emitted_probabilities(module, logits)[0, 1])
        else:
            decoded = module.target_encoder.decode_tensor(
                module.loss.to_prediction(logits))
            prediction = _plain(decoded.reshape(-1)[0])
        record: dict[str, Any] = {
            **identity, 'prediction_id': prediction_id,
            'prediction': prediction,
            'origin': 'original_forward',
            'sample_sha256': None,
            'cells': [],
            'status': 'failed',
            'reason': None,
            'graph_status': 'failed',
            'graph_reason': None
        }
        if not self.legacy:
            # Non-legacy records only (legacy v1 records keep their bytes).
            from kumo_core.explain.walk_binary import output_block
            if self.binary:
                record['prediction_coordinate'] = 'encoded_positive_class_probability'
            record['output'] = (None if self.record_selection is None else
                                output_block(self.record_selection, module, logits[0]))
        sample = None
        try:
            sample = self._sample(batch, module)
            sample_sha256 = _digest(sample)
            (self.root / 'samples' / f'{prediction_id}.json').write_text(
                _encoded(sample))
            record['sample_sha256'] = sample_sha256
            record['graph_status'] = 'available'
            record['temporal_validation'] = self._temporal_validation(
                sample, identity)
        except Exception:
            self.graph_failures += 1
            record['graph_reason'] = 'original_graph_capture_failed'
            record['temporal_validation'] = {
                'status': 'unavailable',
                'reason': 'original_graph_capture_failed'
            }
        if self.walk_config is not None and not self.legacy:
            record['walk_attribution'] = self._v2_row(module, logits, record,
                                                      sample, prediction_id)
        elif self.walk_config is not None:
            from kumo_core.explain.walk_attribution import UnsupportedWalkAttribution
            try:
                if self.walk_capture is None:
                    if self.walk_failure_unsupported:
                        raise UnsupportedWalkAttribution(self.walk_failure)
                    raise RuntimeError(self.walk_failure or 'capture unavailable')
                if record['graph_status'] != 'available':
                    raise ValueError('Original graph missing')
                record['walk_attribution'] = bind_walk_result(self.walk_capture.result(seed_index=0), sample)
                record['walk_attribution'].update(
                    origin='original_forward_attribution', prediction_id=prediction_id,
                    sample_sha256=record['sample_sha256'])
            except UnsupportedWalkAttribution as error:
                record['walk_attribution'] = {'status': 'unsupported', 'reason': str(error)}
            except Exception:
                record['walk_attribution'] = {'status': 'failed', 'reason': 'walk_capture_failed'}
            record['walk_attribution'].update(
                origin='original_forward_attribution', prediction_id=prediction_id,
                sample_sha256=record['sample_sha256'], config=self.config.to_dict(),
                method='hetero_lrp_gamma_reference_v1')
        try:
            if self.numerical_sensitivity:
                cells, status, reason = self._cells(module, batch, step_output)
            else:
                cells, status, reason = [], 'disabled', 'numerical_sensitivity_disabled'
            record.update(
                cells=cells, status=status, reason=reason,
                expected_cells=len(cells),
                available_cells=sum(c['status'] == 'available' for c in cells))
        except Exception as error:
            record[
                'reason'] = f'{type(error).__name__}: local sensitivity capture failed'
        finally:
            for encoder in module.encoder.modules():
                if hasattr(encoder, '_saved_tensors'):
                    del encoder._saved_tensors
        if self.walk_config is not None:
            state = record['walk_attribution']['status']
            self.walk_status_counts[state] = self.walk_status_counts.get(state, 0) + 1
        (self.root / 'rows' / f'{prediction_id}.json').write_text(
            _encoded(record))
        # Flush each row so memory remains bounded and interrupted runs expose
        # an honest partial index with complete=False in their manifest.
        with (self.root / 'index.jsonl').open('a') as stream:
            stream.write(
                _encoded({
                    k: v
                    for k, v in record.items() if k not in ('cells', 'walk_attribution')
                }) + '\n')
        temporal_state = record['temporal_validation']['status']
        self.temporal_status_counts[temporal_state] = (
            self.temporal_status_counts.get(temporal_state, 0) + 1)
        self.rows += 1
        self.status_counts[record['status']] = (
            self.status_counts.get(record['status'], 0) + 1)

    def close(self):
        """Release transient hooks even when prediction aborts before post-step."""
        if self.walk_capture is not None:
            self.walk_capture.__exit__(None, None, None)
            self.walk_capture = None

    def explain_end(self, module, datamodule, outputs, stage):
        if self.walk_capture is not None:
            self.walk_capture.__exit__(None, None, None)
            self.walk_capture = None
        try:
            self.generation_closed = True
            incomplete = sum(
                count for state, count in self.status_counts.items()
                if state not in {'available', 'zero_signal', 'disabled'})
            self._write_manifest(complete=(
                self.failure is None and not incomplete
                and not any(count for state, count in self.walk_status_counts.items() if state != "available")
                and not self.graph_failures
                and not self.temporal_status_counts.get('violated', 0)
                and not self.temporal_status_counts.get('unavailable', 0)
                and self.rows == self.attempted_rows))
        except OSError:
            logger.warning(
                'Local sensitivity manifest unavailable; predictions continue')


def load_scoring_module(model_dir: str, provenance: dict[str, Any]):
    """Reconstruct the frozen scorer without reopening its source data.

    Supports MSE regression and BCE binary classification; a binary module
    is returned with its restored calibrator. The bundle hash is verified
    before loading. The caller supplies a trusted model bundle. Evidence is
    declarative JSON; only the existing trusted checkpoint loading path loads
    model weights.
    """
    from types import SimpleNamespace

    from kumo_core.resolved_task import ResolvedNumericalTask
    from kumo_core.train.diskgraph_nested_trainer import (
        DiskgraphNestedTrainer,
        _has_train_table_time_col,
        _time_col_dict_from_schema,
    )
    from kumo_core.train.trainer import load_encoders
    from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime

    actual = bundle_fingerprints(model_dir)
    if actual['bundle_sha256'] != provenance.get('bundle_sha256'):
        raise ValueError('Scoring model does not match retained evidence')
    spec = provenance['model_reconstruction']
    task = ResolvedNumericalTask.from_dict(spec['task'])
    if task.task_type not in (TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION):
        raise ValueError('Retained scorer requires scalar regression or '
                         'binary classification')
    trainer = DiskgraphNestedTrainer.load(
        task, model_dir, diskgraph_task_dataset_bundle_dir='',
        infer_config_path=False, encoders=load_encoders(model_dir),
        diskgraph_runtime=LocalDiskgraphTrainingRuntime(), accelerator='cpu')
    trainer._prediction_mode = True
    manifest = SimpleNamespace(
        **{
            key: spec[key]
            for key in ('metadata', 'feature_schema', 'graph_runtime_metadata')
        })
    columns = spec['metadata'].get('train_table_columns') or {}
    trainer._configure_model_triple_from_manifest(
        manifest, time_col_dict_fn=_time_col_dict_from_schema,
        has_train_table_time_col=_has_train_table_time_col(columns))
    trainer.majority_sampling_weights = None
    trainer.create_pl_model(
        entity_table_names=[spec['metadata']['entity_table_name']],
        train_table_name=spec['metadata']['train_table_name'],
        ckpt_path=trainer.best_ckpt)
    module = trainer.pl_model
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        from kumo_core.train.loss import BinaryCrossEntropyLoss
        if not isinstance(module.loss, BinaryCrossEntropyLoss):
            raise ValueError('Retained binary scorer requires '
                             'BinaryCrossEntropyLoss')
    elif not isinstance(module.loss, MSELoss):
        raise ValueError('Retained scorer requires scalar MSE regression')
    return module.cpu().eval()
