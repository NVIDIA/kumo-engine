# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from kumo.engine.connector._extras import import_provider
from kumo.engine.graph import GraphSnapshot
from kumo.engine.integrations._serving import snapshot_location

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
        from kumo.engine._deps import assets_through
        from kumo.engine._predictor import load_predictor

        self._reader = context.model_config.get('assets_from')
        bundle = context.artifacts['bundle']
        with assets_through(
                self._reader, bundle, lambda reader, _: (
                    f"pass model_config={{'assets_from': {reader}(...)}} to "
                    f"mlflow.pyfunc.load_model(model_uri, ...), with an "
                    f"identity that reads it")):
            self._predictor = load_predictor(bundle)
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
        return self._predictor.predict(model_input, self._snapshot(uri),
                                       params)

    def _snapshot(self, uri: str) -> GraphSnapshot:
        from kumo.engine._deps import reads

        if uri not in self._snapshots:
            connector, path = ((self._reader, uri)
                               if self._reader is not None
                               and reads(self._reader, uri) else
                               snapshot_location(uri))
            try:
                self._snapshots[uri] = GraphSnapshot.load_from_connector(
                    connector, path=path)
            except FileNotFoundError as error:
                reader = _remote_reader(uri)
                if reader is None or connector is self._reader:
                    raise
                raise FileNotFoundError(
                    f"{error} To read {uri} from here, pass "
                    f"model_config={{'assets_from': {reader}(...)}} to "
                    f"mlflow.pyfunc.load_model(model_uri, ...).") from error
        return self._snapshots[uri]


def _remote_reader(uri: str) -> Optional[str]:
    if uri.startswith('@'):
        return 'SnowflakeConnector'
    if uri.startswith('/Volumes/') and not os.path.isdir('/Volumes'):
        return 'DatabricksConnector'
    return None


def signature(snapshot: GraphSnapshot, query: str) -> Tuple[Any, pd.DataFrame]:
    from kumo_core.train.formatting import prediction_output_columns
    from kumo_core.types import TaskType

    from kumo.engine.pquery import PredictiveQuery

    bound = PredictiveQuery._from_snapshot(snapshot, query)
    parsed = bound._parsed
    entity = snapshot._native_graph.table(parsed.entity_table)
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
