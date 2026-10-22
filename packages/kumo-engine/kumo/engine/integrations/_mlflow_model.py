# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd

from kumo.engine.connector._extras import import_provider
from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot

_pyfunc = import_provider('mlflow.pyfunc', 'mlflow')
_schema = import_provider('mlflow.types.schema', 'mlflow')
_signature = import_provider('mlflow.models.signature', 'mlflow')

_DTYPES = {
    'PREDICTED': 'boolean',
    'SCORE': 'double',
    'TIMESTAMP': 'datetime',
    'False_PROB': 'double',
    'True_PROB': 'double',
}


class KumoModel(_pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from kumo.engine.trainer.trainer import Trainer

        self._trainer = Trainer.load(context.artifacts['bundle'])
        self._columns = list(context.model_config['columns'])
        self._snapshots: Dict[str, GraphSnapshot] = {}
        self._default = context.model_config.get('snapshot')
        if self._default:
            self._snapshot(self._default)

    def predict(self, context: Any, model_input: pd.DataFrame,
                params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        params = params or {}
        if len(model_input) == 0 and not params.get('population'):
            return pd.DataFrame(columns=self._columns)
        uri = params.get('snapshot') or self._default
        if not uri:
            raise ValueError("Pass params={'snapshot': <exported snapshot "
                             "URI>}.")
        return predict(self._trainer, self._snapshot(uri), model_input,
                       params)

    def _snapshot(self, uri: str) -> GraphSnapshot:
        if uri not in self._snapshots:
            connector, path = snapshot_location(uri)
            self._snapshots[uri] = GraphSnapshot.load_from_connector(
                connector, path=path)
        return self._snapshots[uri]


def predict(trainer: Any, snapshot: GraphSnapshot, rows: pd.DataFrame,
            params: Mapping[str, Any]) -> pd.DataFrame:
    from kumo.engine._scratch import scoped_scratch_dir
    from kumo.engine.connector.file import FileConnector

    time = params.get('prediction_time')
    time = None if time is None else pd.Timestamp(time)
    options = dict(
        binary_classification_threshold=params.get(
            'binary_classification_threshold'),
        num_classes_to_return=params.get('num_classes_to_return'),
    )
    trainer.scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=trainer.scratch_dir) as tmp, \
            scoped_scratch_dir(Path(tmp) / 'scratch'):
        if len(rows) == 0:
            return trainer.predict_snapshot(
                snapshot, prediction_time=None
                if time is None else time.to_pydatetime(), **options).frame
        rows = rows.rename(columns={'ENTITY': 'ENTITY1'})
        if time is not None and 'TIMESTAMP' not in rows:
            rows = rows.assign(TIMESTAMP=time)
        (Path(tmp) / 'rows').mkdir()
        rows.to_parquet(Path(tmp) / 'rows' / 'rows.parquet', index=False)
        return trainer.predict_snapshot(
            snapshot,
            entities=FileConnector(root_dir=str(Path(tmp) / 'rows'))[
                'rows.parquet'], **options).frame


def snapshot_location(uri: str) -> Tuple[Connector, str]:
    from kumo.engine.connector import connector_for_root

    parent, slash, name = uri.rstrip('/').rpartition('/')
    return connector_for_root(parent or slash or '.'), name


def signature(snapshot: GraphSnapshot, query: str) -> Tuple[Any, pd.DataFrame]:
    from kumo_core.train.formatting import prediction_output_columns
    from kumo_core.types import TaskType

    from kumo.engine.pquery import PredictiveQuery

    bound = PredictiveQuery.from_snapshot(snapshot, query)
    parsed = bound._parsed
    entity = snapshot.native_graph.table(parsed.entity_table)
    entity_type = _entity_type(str(entity.column(entity.primary_key).dtype))
    task_type = TaskType(parsed.task_type)
    dtypes = {
        **_DTYPES, 'TARGET_PRED':
        'boolean' if task_type == TaskType.BINARY_CLASSIFICATION else 'double'
    }
    inputs = _schema.Schema([
        _schema.ColSpec(entity_type, 'ENTITY'),
        _schema.ColSpec('datetime', 'TIMESTAMP', required=False),
    ])
    outputs = _schema.Schema([
        _schema.ColSpec(entity_type, 'ENTITY'),
        *(_schema.ColSpec(dtypes.get(name) or _schema.AnyType(), name)
          for name in prediction_output_columns(task_type)),
    ])
    params = _schema.ParamSchema([
        _schema.ParamSpec('snapshot', 'string', None),
        _schema.ParamSpec('prediction_time', 'string', None),
        _schema.ParamSpec('binary_classification_threshold', 'double', None),
        _schema.ParamSpec('num_classes_to_return', 'long', None),
        _schema.ParamSpec('population', 'boolean', False),
    ])
    example = pd.DataFrame({
        'ENTITY': {
            'long': [0, 1],
            'double': [0.0, 1.0],
        }.get(entity_type, ['entity-0', 'entity-1'])
    })
    return _signature.ModelSignature(inputs, outputs, params), example


def _entity_type(dtype: str) -> str:
    if dtype.startswith('int') or dtype == 'byte':
        return 'long'
    if dtype.startswith('float'):
        return 'double'
    return 'string'
