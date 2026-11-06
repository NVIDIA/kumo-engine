# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay one approved older case model with the installed wheel."""

import os
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as parquet
from common import check_predictions
from kumoapi.jobs import MetadataField

import kumo.engine as kumo

CASES = ('adtech_binary', 'delivery_multiclass', 'delivery_multilabel',
         'delivery_multilabel_ranking', 'delivery_regression',
         'delivery_forecasting', 'commerce_static_link',
         'commerce_temporal_link', 'streaming_temporal_link',
         'community_regression', 'payments_binary', 'fraud_binary')
LINK_COUNTS = {
    'commerce_static_link': 5,
    'commerce_temporal_link': 10,
    'streaming_temporal_link': 50
}


def test_model_compatibility(tmp_path, record_testsuite_property):
    name = os.environ['ENGINE_COMPAT_CASE']
    if name not in CASES:
        raise ValueError(f'Choose ENGINE_COMPAT_CASE from {CASES}')
    case = import_module(name)
    fixture = Path(os.environ['ENGINE_COMPAT_FIXTURE'])
    model = os.environ['ENGINE_COMPAT_MODEL']
    source = os.environ['ENGINE_COMPAT_SOURCE']
    record_testsuite_property('case', name)
    record_testsuite_property('model', model)
    record_testsuite_property('wheel', version('kumo-engine'))
    # The largest current output is streaming's 50 links per input row.
    for part, limit in (('population', case.LIMITS['prediction_rows']),
                        ('expected', 1000)):
        if not 1 <= parquet.ParquetFile(
                fixture / f'{part}.parquet').metadata.num_rows <= limit:
            raise ValueError(f'{part}.parquet must contain 1 to {limit} rows')
    population = parquet.read_table(fixture / 'population.parquet').to_pandas()
    expected = parquet.read_table(fixture / 'expected.parquet').to_pandas()
    check_predictions(expected, expected, expected, 0)
    anchored = name != 'commerce_static_link'
    keys = ['ENTITY1'] + (['TIMESTAMP'] if anchored else [])
    if population[keys].isna().any().any() or population.duplicated(keys).any():
        raise ValueError('Population coordinates must be unique and non-null')
    output_keys = ['ENTITY'] + (['ANCHOR_TIMESTAMP'] if anchored else [])
    if set(population[keys].itertuples(index=False, name=None)) != set(
            expected[output_keys].itertuples(index=False, name=None)):
        raise ValueError('Expected coordinates must cover the input population')
    record_testsuite_property('input_rows', len(population))
    record_testsuite_property('output_rows', len(expected))
    kumo.configure(scratch_dir=tmp_path)
    if model.startswith('s3://'):
        trainer = kumo.Trainer.load_from_connector(kumo.S3Connector(),
                                                   path=model,
                                                   accelerator='gpu')
    else:
        trainer = kumo.Trainer.load(model, accelerator='gpu')
    graph = case.build_graph(kumo.S3Connector(root_dir=source))
    query = kumo.PredictiveQuery(graph, case.QUERY)
    snapshot = graph.snapshot()
    table = kumo.PredictionTable.from_source_table(
        kumo.FileConnector(root_dir=fixture)['population.parquet'], pq=query,
        snapshot=snapshot, plan=trainer.training_table_generation_plan)
    options = {
        'binary_classification_threshold': .5
    } if name.endswith('_binary') else {}
    if name in LINK_COUNTS:
        options['num_classes_to_return'] = LINK_COUNTS[name]
    observed = trainer.predict(
        table, snapshot=snapshot, **options,
        output_config=kumo.PredictionOutputConfig(
            kumo.FileConnector(root_dir=tmp_path / 'predictions'),
            output_paths={'predictions': 'compatibility'},
            extra_fields=[MetadataField.ANCHOR_TIMESTAMP] if anchored else []))
    if observed.num_rows != len(expected):
        raise ValueError(
            'Prediction row count differs from the approved fixture')
    for key, value in check_predictions(
            expected, observed, observed,
            case.LIMITS['prediction_tolerance']).items():
        record_testsuite_property(key, value)
