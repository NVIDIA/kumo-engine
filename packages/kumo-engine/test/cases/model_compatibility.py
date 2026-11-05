# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay an approved older adtech model with the installed wheel."""

import os
from importlib.metadata import version

import pyarrow.parquet as parquet
from adtech_binary import LIMITS, QUERY, SPLIT, build_graph
from common import check_predictions
from kumoapi.jobs import MetadataField
from kumoapi.model_plan import TrainingTableGenerationPlan

import kumo.engine as kumo


def test_model_compatibility(tmp_path, record_testsuite_property):
    fixture = os.environ['ENGINE_COMPAT_EXPECTED']
    model = os.environ['ENGINE_COMPAT_MODEL']
    record_testsuite_property('model', model)
    record_testsuite_property('wheel', version('kumo-engine'))
    if not 1 <= parquet.ParquetFile(fixture).metadata.num_rows <= 20:
        raise ValueError('The approved fixture must contain 1 to 20 rows')
    expected = parquet.read_table(fixture).to_pandas()
    check_predictions(expected, expected, expected, 0)
    population = expected[['ENTITY', 'ANCHOR_TIMESTAMP']].copy()
    if population.isna().any().any() or population.duplicated().any():
        raise ValueError('The approved fixture needs unique, non-null keys')
    population.columns = ['ENTITY1', 'TIMESTAMP']
    record_testsuite_property('rows', len(expected))
    kumo.configure(scratch_dir=tmp_path)
    graph = build_graph(
        kumo.S3Connector(root_dir=os.environ['ENGINE_COMPAT_SOURCE']))
    query = kumo.PredictiveQuery(graph, QUERY)
    snapshot = graph.snapshot()
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    population.to_parquet(inputs / 'population.parquet', index=False)
    table = kumo.PredictionTable.from_source_table(
        kumo.FileConnector(root_dir=inputs)['population.parquet'], pq=query,
        snapshot=snapshot,
        plan=TrainingTableGenerationPlan(split=SPLIT, timeframe_step=7))
    if model.startswith('s3://'):
        trainer = kumo.Trainer.load_from_connector(kumo.S3Connector(),
                                                   path=model,
                                                   accelerator='gpu')
    else:
        trainer = kumo.Trainer.load(model, accelerator='gpu')
    observed = trainer.predict(
        table, snapshot=snapshot, binary_classification_threshold=0.5,
        output_config=kumo.PredictionOutputConfig(
            kumo.FileConnector(root_dir=tmp_path / 'predictions'),
            output_paths={'predictions': 'compatibility'},
            extra_fields=[MetadataField.ANCHOR_TIMESTAMP]))
    for name, value in check_predictions(
            expected, observed, observed,
            LIMITS['prediction_tolerance']).items():
        record_testsuite_property(name, value)
