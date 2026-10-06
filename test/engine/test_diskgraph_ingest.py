# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock

import pytest
from kumo_core.types.schema import TS_FORMAT_METADATA, BaseColumnConfig
from kumoapi.encoder import MultiCategorical, NumericalList
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import TableConfig, TableMetadataConfig
from kumo.diskgraph.ingest import (
    prepare_diskgraph_converter_config,
    validate_diskgraph_ingest_config,
)
from kumo.graph import Column
from kumo.graph.column import EncoderGenerator
from kumo.graph.table import Table


def test_table_metadata_preserves_stringlist_dtype(monkeypatch):
    from kumo.pquery.AST import ast_node

    monkeypatch.setattr(ast_node, 'ALLOW_ARRAY_TARGETS', False)

    table = Table('customers', data_source='file')
    table._set_dtype_from_metadata(
        TableMetadataConfig(
            table_name='customers',
            source_name='customers',
            data_source='file',
            cols=[
                BaseColumnConfig('preferred_channels', dtype=Dtype.stringlist)
            ],
        ))

    config = table.config()
    preferred_channels = config.cols[0]

    assert isinstance(config, TableConfig)
    assert preferred_channels.dtype == Dtype.stringlist


def test_stringlist_infers_multicategorical_without_array_targets(monkeypatch):
    from kumo.pquery.AST import ast_node

    monkeypatch.setattr(ast_node, 'ALLOW_ARRAY_TARGETS', False)

    table = Table(
        'customers',
        data_source='file',
        cols=[Column('preferred_channels', dtype=Dtype.stringlist)],
    )

    table.infer_stypes(require_stats=False)
    preferred_channels = table.config().cols[0]

    assert Stype.multicategorical.supports_dtype(Dtype.stringlist)
    assert preferred_channels.dtype == Dtype.stringlist
    assert preferred_channels.stype == Stype.multicategorical


def test_multicategorical_internal_dtype_contract_matches_public_api():
    assert Stype.multicategorical.supports_dtype(Dtype.string)
    assert Stype.multicategorical.supports_dtype(Dtype.stringlist)
    assert Stype.multicategorical.supports_dtype(Dtype.intlist)
    assert not Stype.multicategorical.supports_dtype(Dtype.binary)
    assert not Stype.multicategorical.supports_dtype(Dtype.floatlist)


def test_list_feature_encoders_do_not_depend_on_array_targets(monkeypatch):
    from kumo.pquery.AST import ast_node

    monkeypatch.setattr(ast_node, 'ALLOW_ARRAY_TARGETS', False)

    assert EncoderGenerator.supports_encoder(
        Column('preferred_channels', dtype=Dtype.stringlist,
               stype=Stype.multicategorical),
        MultiCategorical(),
    )
    assert EncoderGenerator.supports_encoder(
        Column('token_ids', dtype=Dtype.intlist, stype=Stype.multicategorical),
        MultiCategorical(),
    )
    assert EncoderGenerator.supports_encoder(
        Column('embedding', dtype=Dtype.floatlist, stype=Stype.sequence),
        NumericalList(),
    )


def test_real_converter_preserves_exact_object_store_source_manifest():
    from diskgraph import DiskgraphConfigConverter

    config = {
        'name':
        'events_graph',
        'tables': [{
            'table_name': 'events',
            'data_source': 's3',
            'source_name': 'events',
            'data_version': 'v1',
            'pkey': {
                'name': 'id',
                'dtype': 'int64',
                'stype': 'ID',
            },
            'cols': [],
        }],
    }
    source = {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': 'gs://bucket/events/part.parquet',
            'size': 123,
            'e_tag': 'etag-value',
            'version': '789',
        }],
        'auth_profile':
        'ambient',
    }

    converted = DiskgraphConfigConverter(
        config,
        '/path/that/does/not/exist',
        {
            'events': 'v1'
        },
        source_specs={
            'events': {
                'source': source,
                'source_size': 123,
            },
        },
    ).convert()

    assert converted['tables'][0]['source'] == {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': 'gs://bucket/events/part.parquet',
            'size': 123,
            'e_tag': 'etag-value',
            'version': '789',
        }],
        'auth_profile':
        'ambient',
    }
    assert converted['tables'][0]['source_size'] == 123


def test_prepare_diskgraph_converter_config_preserves_list_dtype():
    graph_config = {
        'tables': [{
            'cols': [{
                'name': 'preferred_channels',
                'dtype': 'stringlist',
                'stype': 'multicategorical',
            }],
        }],
    }

    prepared = prepare_diskgraph_converter_config(graph_config)

    assert prepared['tables'][0]['cols'][0]['dtype'] == 'stringlist'
    assert graph_config['tables'][0]['cols'][0]['dtype'] == 'stringlist'


def test_prepare_diskgraph_converter_config_preserves_durable_delimiter():
    graph_config = {
        'tables': [{
            'cols': [{
                'name': 'tags',
                'dtype': 'string',
                'stype': 'multicategorical',
                'multicategorical_delimiter': '|',
            }],
        }],
    }

    prepared = prepare_diskgraph_converter_config(graph_config)

    assert prepared['tables'][0]['cols'][0]['multicategorical_delimiter'] == '|'
    assert graph_config['tables'][0]['cols'][0][
        'multicategorical_delimiter'] == '|'


def test_diskgraph_boundary_rejects_scalar_multicategorical_dtype():
    config = {
        'tables': [{
            'name':
            'TRANSACTIONS',
            'columns': [{
                'name': 'Time',
                'semantic_type': 'multicategorical',
                'ingested_dtype': 'string',
            }],
        }],
    }

    with pytest.raises(ValueError, match='list-shaped ingested dtype'):
        validate_diskgraph_ingest_config(config)


@pytest.mark.parametrize(
    'ingested_dtype',
    ['string_array', 'int64_array'],
)
def test_diskgraph_boundary_accepts_multicategorical_list_dtype(ingested_dtype):
    validate_diskgraph_ingest_config({
        'tables': [{
            'name':
            'items',
            'columns': [{
                'name': 'tags',
                'semantic_type': 'multicategorical',
                'ingested_dtype': ingested_dtype,
            }],
        }],
    })


@pytest.mark.parametrize(
    'ingested_dtype',
    ['float16_array', 'float32_array', 'float64_array'],
)
def test_diskgraph_boundary_rejects_unsupported_multicategorical_list_dtype(
        ingested_dtype):
    config = {
        'tables': [{
            'name':
            'items',
            'columns': [{
                'name': 'embedding',
                'semantic_type': 'multicategorical',
                'ingested_dtype': ingested_dtype,
            }],
        }],
    }

    with pytest.raises(ValueError, match='list-shaped ingested dtype'):
        validate_diskgraph_ingest_config(config)


def test_run_diskgraph_ingest_rejects_scalar_multicategorical_before_runner(
        monkeypatch, tmp_path):
    from kumo.diskgraph import ingest as ingest_module

    invalid_config = {
        'tables': [{
            'name':
            'TRANSACTIONS',
            'columns': [{
                'name': 'Time',
                'semantic_type': 'multicategorical',
                'ingested_dtype': 'string',
            }],
        }],
    }

    monkeypatch.setattr(ingest_module, 'to_json_dict', lambda _: {})
    monkeypatch.setattr(
        ingest_module,
        'normalize_diskgraph_source_fields',
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(ingest_module, 'convert_diskgraph_config',
                        lambda *args, **kwargs: invalid_config)
    run = getattr(ingest_module.run_diskgraph_ingest, '__wrapped__',
                  ingest_module.run_diskgraph_ingest)

    with pytest.raises(ValueError) as excinfo:
        run(
            MagicMock(),
            str(tmp_path / 'persist'),
            {'TRANSACTIONS': 'v1'},
            str(tmp_path / 'output'),
            str(tmp_path / 'dataplane'),
            'success',
        )

    message = str(excinfo.value)
    assert 'TRANSACTIONS' in message
    assert 'Time' in message
    assert 'Use categorical' in message


def test_run_diskgraph_ingest_hands_the_engine_no_databricks_config(
        monkeypatch, tmp_path):
    """Nothing reaches DiskGraph that would make it open a warehouse
    connection of its own: no ``--databricks-config-file`` and no
    ``DISKGRAPH_EPHEMERAL_DATABRICKS_CONFIG``."""
    from kumo_core.diskgraph.runner import IngestRunResult

    from kumo.diskgraph import ingest as ingest_module

    captured = {}

    def fake_run_ingest(_config, output_path, **kwargs):
        captured.update(kwargs)
        for filename in (
                'col_stats.json',
                'ingest_stats.json',
                'index_metadata.json',
                '_MANIFEST.json',
                '_SUCCESS',
        ):
            (output_path / filename).touch()
        (output_path / 'kumo_col_stats').mkdir()
        return IngestRunResult({}, 1, 1)

    monkeypatch.setattr(ingest_module, 'run_ingest', fake_run_ingest)
    heartbeat = MagicMock()

    ingest_module.run_diskgraph_ingest(
        None,
        str(tmp_path / 'persist'),
        {},
        str(tmp_path / 'output'),
        str(tmp_path / 'dataplane'),
        'success',
        subprocess_environment={'AWS_REGION': 'us-west-2'},
        heartbeat=heartbeat,
    )

    assert 'extra_args' not in captured
    assert 'cleanup_paths_after_phase' not in captured
    # ``None`` is the runner's "remove this name from the child", so the var
    # is unset in the subprocess rather than merely not set by us: a value
    # this process inherited must not reach the binary either.
    assert captured['subprocess_environment'] == {
        'AWS_REGION': 'us-west-2',
        'DISKGRAPH_EPHEMERAL_DATABRICKS_CONFIG': None,
    }
    assert captured['heartbeat'] is heartbeat


@pytest.mark.parametrize('stats,expected', [
    ({'stats': {'node_tables': 2, 'edge_relations': 1,
                'total_nodes': 7, 'total_edges': 8}},
     '2 node tables, 1 edge relations, 7 nodes, 8 edges'),
    ({'stats': None}, '0 node tables, 0 edge relations, 0 nodes, 0 edges'),
])
def test_run_diskgraph_ingest_logs_real_counts_from_nested_stats(
        stats, expected, monkeypatch, tmp_path, caplog):
    from kumo_core.diskgraph.runner import IngestRunResult

    from kumo.diskgraph import ingest as ingest_module

    def fake_run_ingest(_config, output_path, **_kwargs):
        for filename in (
                'col_stats.json',
                'ingest_stats.json',
                'index_metadata.json',
                '_MANIFEST.json',
                '_SUCCESS',
        ):
            (output_path / filename).touch()
        (output_path / 'kumo_col_stats').mkdir()
        return IngestRunResult(stats, 1, 1)

    monkeypatch.setattr(ingest_module, 'run_ingest', fake_run_ingest)
    caplog.set_level(logging.INFO)

    ingest_module.run_diskgraph_ingest(
        None,
        str(tmp_path / 'persist'),
        {},
        str(tmp_path / 'output'),
        str(tmp_path / 'dataplane'),
        'success',
    )

    assert f'diskgraph ingest completed: {expected}' in caplog.text


def test_run_diskgraph_ingest_preserves_no_space_error(monkeypatch, tmp_path):
    from kumo_core.diskgraph.runner import \
        DiskgraphNoSpaceError as CoreDiskgraphNoSpaceError

    from kumo.diskgraph import ingest as ingest_module
    from kumo.diskgraph.errors import DiskgraphNoSpaceError

    message = 'local SSD has 2.5% free; retry on a larger worker'

    def fake_run_ingest(*_args, **_kwargs):
        raise CoreDiskgraphNoSpaceError(message)

    monkeypatch.setattr(ingest_module, 'run_ingest', fake_run_ingest)

    with pytest.raises(DiskgraphNoSpaceError) as exc_info:
        ingest_module.run_diskgraph_ingest(
            None,
            str(tmp_path / 'persist'),
            {},
            str(tmp_path / 'output'),
            str(tmp_path / 'dataplane'),
            'success',
        )

    assert exc_info.value.get_user_message() == message
    assert exc_info.value.non_retryable is False


def test_run_diskgraph_ingest_logs_diagnostic_and_private_tail_separately(
        monkeypatch, tmp_path, caplog):
    from kumo_core.diskgraph.runner import (
        Diagnostic,
        DiagnosticCause,
        DiagnosticFields,
    )
    from kumo_core.diskgraph.runner import \
        NonRetryableDiskgraphError as CoreNonRetryableDiskgraphError

    from kumo.diskgraph import ingest as ingest_module
    from kumo.diskgraph.errors import NonRetryableDiskgraphError

    public_message = 'DiskGraph failed while encoding a source table'
    diagnostic_summary = 'partition offset exceeded an internal bound'
    canary = 'warehouse-row-value-must-require-debug'
    diagnostic = Diagnostic(
        diagnostic_id=f"dg-{'a' * 24}",
        visibility='ordinary',
        operation='ingest',
        category='internal',
        code='ingest.unexpected_failure',
        retryable=False,
        summary=diagnostic_summary,
        public=None,
        fields=DiagnosticFields(
            attempt=1,
            phase='encode',
            exit_code=None,
            signal=None,
        ),
        chain=(DiagnosticCause(
            kind='error',
            summary=diagnostic_summary,
        ), ),
        frames=(),
        fingerprint=f"dg-v1-{'b' * 64}",
        chain_truncated=False,
        frames_truncated=False,
        summary_truncated=False,
    )

    def fake_run_ingest(*_args, **_kwargs):
        raise CoreNonRetryableDiskgraphError(
            public_message,
            private_output_tail=canary,
            diagnostic=diagnostic,
        )

    monkeypatch.setattr(ingest_module, 'run_ingest', fake_run_ingest)

    with caplog.at_level(logging.DEBUG, logger=ingest_module.logger.name):
        with pytest.raises(NonRetryableDiskgraphError, match=public_message):
            ingest_module.run_diskgraph_ingest(
                None,
                str(tmp_path / 'persist'),
                {},
                str(tmp_path / 'output'),
                str(tmp_path / 'dataplane'),
                'success',
            )

    error_messages = '\n'.join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR)
    debug_messages = '\n'.join(
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG)
    assert public_message in error_messages
    assert diagnostic_summary in error_messages
    assert 'diagnostic_id=dg-' in error_messages
    assert canary not in error_messages
    assert canary in debug_messages


@pytest.mark.parametrize('missing_publication_file', [
    '_MANIFEST.json',
    '_SUCCESS',
])
def test_run_diskgraph_ingest_requires_local_publication_files(
        missing_publication_file, monkeypatch, tmp_path):
    from kumo_core.diskgraph.runner import IngestRunResult

    from kumo.diskgraph import ingest as ingest_module

    def fake_run_ingest(_config, output_path, **_kwargs):
        output_path = tmp_path / 'output'
        for filename in (
                'col_stats.json',
                'ingest_stats.json',
                'index_metadata.json',
                '_MANIFEST.json',
                '_SUCCESS',
        ):
            if filename != missing_publication_file:
                (output_path / filename).touch()
        (output_path / 'kumo_col_stats').mkdir()
        return IngestRunResult(
            stats={},
            final_thread_count=1,
            attempts=1,
        )

    monkeypatch.setattr(ingest_module, 'run_ingest', fake_run_ingest)

    with pytest.raises(RuntimeError, match=missing_publication_file):
        ingest_module.run_diskgraph_ingest(
            None,
            str(tmp_path / 'persist'),
            {},
            str(tmp_path / 'output'),
            str(tmp_path / 'dataplane'),
            'success',
        )


def test_prepare_diskgraph_converter_config_preserves_timestamp_format():
    graph_config = {
        'tables': [{
            'pkey': {
                'name': 'id',
                'dtype': 'string',
            },
            'cols': [{
                'name': 'event_ts',
                'dtype': 'string',
                'stype': 'timestamp',
                'metadata': {
                    TS_FORMAT_METADATA: '%Y/%m/%d',
                },
            }],
        }],
    }

    prepared = prepare_diskgraph_converter_config(graph_config)

    assert prepared['tables'][0]['cols'][0]['timestamp_format'] == '%Y/%m/%d'


def test_prepare_diskgraph_converter_config_skips_unit_echo_ts_format():
    # kumo overloads TS_FORMAT: when it equals the column's timestamp_unit it is
    # a unit sentinel (numeric-epoch column), NOT a strftime format (see
    # kumo/data/utils.py:248-252). DG's converter rejects a bare unit token as a
    # chrono format, so a unit-echo TS_FORMAT must not be injected as
    # timestamp_format; the unit already rides timestamp_unit.
    graph_config = {
        'tables': [{
            'pkey': {
                'name': 'id',
                'dtype': 'string',
            },
            'cols': [{
                'name': 'epoch_ts',
                'dtype': 'int',
                'stype': 'timestamp',
                'timestamp_unit': 's',
                'metadata': {
                    TS_FORMAT_METADATA: 's',
                },
            }],
        }],
    }

    prepared = prepare_diskgraph_converter_config(graph_config)

    col = prepared['tables'][0]['cols'][0]
    assert 'timestamp_format' not in col
    assert col['timestamp_unit'] == 's'


def test_real_ingest_encoded_date_requires_ts_format(tmp_path):
    # The real DiskGraph wheel reads an int YYYYMMDD column as a date only when
    # `timestamp_format` is set; without it the raw int is taken as epoch
    # seconds and collapses to 1970. Guards the kumo->DG format contract.
    import json
    from datetime import datetime, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq
    from diskgraph import ingest as rust_ingest

    def epoch(y, m, d):
        return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())

    data = tmp_path / 'events.parquet'
    index = tmp_path / 'index'
    index.mkdir()
    pq.write_table(
        pa.table({
            'id': pa.array([1, 2], type=pa.int64()),
            'ed': pa.array([20240131, 20230115], type=pa.int64()),
            'ed_noformat': pa.array([20240131, 20230115], type=pa.int64()),
        }), str(data))

    config = {
        'dataset_name':
        'ts',
        'num_shards':
        1,
        'tables': [{
            'name':
            'events',
            'primary_key':
            'id',
            'data_version':
            'v1',
            'source': {
                'type': 'parquet',
                'paths': [str(data)]
            },
            'columns': [
                {
                    'name': 'id',
                    'ingested_dtype': 'int64'
                },
                {
                    'name': 'ed',
                    'ingested_dtype': 'timestamp',
                    'semantic_type': 'timestamp',
                    'timestamp_format': '%Y%m%d'
                },
                {
                    'name': 'ed_noformat',
                    'ingested_dtype': 'timestamp',
                    'semantic_type': 'timestamp'
                },
            ],
        }],
    }
    rust_ingest(json.dumps(config), str(index), col_stats=True,
                cat_count_strategy='hybrid')
    stats = json.loads((index / 'col_stats.json').read_text())['events']

    # With the format, the encoded ints resolve to real 2023-2024 dates.
    assert stats['ed']['min_secs'] == epoch(2023, 1, 15)
    assert stats['ed']['max_secs'] == epoch(2024, 1, 31)
    # Without it, the raw int is mis-read as epoch seconds (1970-era).
    assert stats['ed_noformat']['max_secs'] == 20240131
    assert stats['ed_noformat']['max_secs'] < epoch(2000, 1, 1)
