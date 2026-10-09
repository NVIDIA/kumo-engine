# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import pathlib
import uuid
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from kumo_core.types import PredictionArtifactType
from pandas.testing import assert_frame_equal

from io_utils import fsutil, parquet_util
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.connector import ConnectorError
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.testing import withSnowflakeCredentials
from kumo.testing.rest import (
    writable_bigquery_connector,
    writable_snowflake_config,
)
from rest.backends.ml_backend import artifact_export
from rest.backends.ml_backend.artifact_export import (
    write_artifact_to_bigquery,
    write_artifact_to_databricks,
    write_artifact_to_s3,
    write_artifact_to_snowflake,
)
from rest.config.credentials import KeyPair
from rest.config.file_type import FileType
from rest.config.jobs import WriteMode
from rest.config.prediction_output import (
    PredictionStorageType,
    SnowflakePredictionOutput,
)
from statedb.config.common import ConnectorID
from statedb.utils.errors import ArtifactExportError


def make_test_df() -> pd.DataFrame:
    df = pd.DataFrame()
    df['ENTITY_ID'] = np.arange(0, 1 << 10)
    df['TIMESTAMP'] = 0
    return df


def prepare_input_data(path: str, chunking: bool = True) -> pd.DataFrame:
    df = make_test_df()
    if chunking:
        # This will write the dataframe into 2 part files.
        parquet_util.write_in_chunks(df, osp.join(path, 'part'), 4 * (1 << 12))
    else:
        df.to_parquet(osp.join(path, 'part0.parquet'))
    return df


def test_artifact_size_supports_file_and_dataset(tmp_path):
    single_file = tmp_path / 'processed_predictions.parquet'
    single_file.write_bytes(b'parquet-bytes')
    assert artifact_export._artifact_size(str(single_file)) == 13

    dataset = tmp_path / 'processed_embeddings.parquet'
    dataset.mkdir()
    (dataset / 'part-00000.parquet').write_bytes(b'1234')
    (dataset / 'part-00001.parquet').write_bytes(b'56789')
    assert artifact_export._artifact_size(str(dataset)) == 9


_TEST_CONNECTOR_ID = ConnectorID(f'connector_{"1" * 32}')


def test_databricks_export_rejects_other_schema_before_open():
    connector = DatabricksConnector(
        'databricks',
        host='https://example',
        cluster_id='cl',
        warehouse_id='wh',
        catalog='catalog',
        schema_name='authorized_schema',
    )

    with patch.object(connector, 'open') as mock_open, pytest.raises(
            ConnectorError, match='connector schema'):
        write_artifact_to_databricks(
            's3://internal-artifacts/predictions/',
            connector,
            'other_schema__kumo__predictions',
            WriteMode.OVERWRITE,
        )

    mock_open.assert_not_called()


@pytest.mark.parametrize(
    ('write_mode', 'replace'),
    [
        (WriteMode.OVERWRITE, True),
        (WriteMode.APPEND, False),
    ],
)
def test_databricks_export_honors_write_mode(
    tmp_path,
    write_mode,
    replace,
):
    artifact_path = tmp_path / 'predictions.parquet'
    artifact_path.write_bytes(b'parquet')
    connector = DatabricksConnector(
        'databricks',
        host='https://example',
        cluster_id='cl',
        warehouse_id='wh',
        catalog='catalog',
        schema_name='schema',
    )

    with (
            patch.object(connector, 'open'),
            patch.object(connector, 'write_artifacts_to_table') as write,
            patch.object(
                artifact_export,
                '_artifact_size',
                side_effect=AssertionError(
                    'known artifact size must not be rescanned'),
            ) as artifact_size,
    ):
        stats = write_artifact_to_databricks(
            str(artifact_path),
            connector,
            'schema__kumo__predictions',
            write_mode,
            known_size_bytes=240,
        )

    artifact_size.assert_not_called()
    assert stats.size_bytes == 240
    assert stats.display_path == 'catalog.schema.predictions'
    write.assert_called_once_with(
        'schema__kumo__predictions',
        str(artifact_path),
        replace=replace,
    )


def test_write_to_bigquery_rejects_s3_before_credentials_or_io(caplog):
    secret_path = 's3://secret-bucket/private/token/output/'
    connector = writable_bigquery_connector(_TEST_CONNECTOR_ID).as_connector()

    with (
            patch.object(connector, 'open', side_effect=AssertionError) as
            open_,
            patch.object(
                connector,
                'write_artifacts_to_table',
                side_effect=AssertionError,
            ) as write,
            patch.object(
                artifact_export.fsutil,
                'dirsize',
                side_effect=AssertionError,
            ) as dirsize,
            patch('google.auth.default', side_effect=AssertionError) as adc,
            patch.object(
                connector,
                '_use_credentials',
                side_effect=AssertionError,
            ) as secret_lookup,
            patch.object(
                connector,
                '_initialize_gcs_client',
                side_effect=AssertionError,
            ) as gcs_client,
            pytest.raises(
                ArtifactExportError,
                match='requires a gs:// artifact path',
            ) as exc_info,
    ):
        write_artifact_to_bigquery(
            secret_path,
            connector,
            write_mode=WriteMode.APPEND,
            table_name='predictions',
        )

    open_.assert_not_called()
    write.assert_not_called()
    dirsize.assert_not_called()
    adc.assert_not_called()
    secret_lookup.assert_not_called()
    gcs_client.assert_not_called()
    assert exc_info.value.non_retryable
    assert secret_path not in str(exc_info.value)
    assert secret_path not in caplog.text


@pytest.mark.parametrize(
    ('writer', 'connector_type', 'provider'),
    [
        (write_artifact_to_snowflake, SnowflakeConnector, 'Snowflake'),
        (write_artifact_to_bigquery, BigQueryConnector, 'BigQuery'),
    ],
)
def test_warehouse_append_rejects_staging_table_before_provider_io(
    writer,
    connector_type,
    provider,
):
    connector = Mock(spec=connector_type)

    with pytest.raises(ArtifactExportError,
                       match=f'{provider} staging table.*OVERWRITE'):
        writer(
            's3://internal-artifacts/predictions/',
            connector,
            WriteMode.APPEND,
            'predictions',
            'predictions_staging',
        )

    connector.open.assert_not_called()


def test_write_to_s3_parquet(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    input_path = osp.join(temp_path, 'input.parquet')
    fsutil.makedirs(input_path)
    df = prepare_input_data(input_path)
    output_path = osp.join(temp_path, "output.parquet")
    with patch.object(
            artifact_export.fsutil, 'copy_files_threaded',
            side_effect=AssertionError(
                'legacy directory exports must retain fsutil.copy')):
        stats = write_artifact_to_s3(input_path, output_path, FileType.PARQUET)
    assert stats.display_path == output_path + '/'
    assert stats.size_bytes > 0
    assert fsutil.exists(output_path)
    out_df = parquet_util.read_parquet(output_path)
    assert len(out_df) == len(df)


def test_write_to_parquet_copies_only_selected_files(tmp_path: pathlib.Path):
    input_path = tmp_path / 'input.parquet'
    input_path.mkdir()
    prepare_input_data(input_path.as_posix())
    parts = sorted(input_path.glob('*.parquet'))
    assert len(parts) >= 2
    stale = input_path / 'stale.parquet'
    stale.write_bytes(parts[0].read_bytes())

    output_path = tmp_path / 'output.parquet'
    write_artifact_to_s3(
        input_path.as_posix(),
        output_path.as_posix(),
        FileType.PARQUET,
        artifact_files=[part.as_posix() for part in parts],
    )

    output_parts = {
        path.name
        for path in output_path.iterdir() if path.suffix == '.parquet'
    }
    assert output_parts == {part.name for part in parts}
    assert stale.name not in output_parts


def test_selected_parquet_copy_retries_and_publishes_success_last(
        tmp_path: pathlib.Path):
    output_path = (tmp_path / 'output.parquet').as_posix()
    artifact_files = [
        f's3://internal/attempt/part-{index:03}.parquet' for index in range(3)
    ]
    attempts = []

    def copy_files(transfers, **kwargs):
        attempts.append((transfers, kwargs))
        assert not fsutil.exists(osp.join(output_path, '_SUCCESS'))
        if len(attempts) == 1:
            raise OSError('transient copy failure')

    with (
            patch.object(
                artifact_export.fsutil,
                'copy_files_threaded',
                side_effect=copy_files,
            ),
            patch('io_utils.retry.time.sleep') as sleep,
    ):
        write_artifact_to_s3(
            's3://internal/attempt/',
            output_path,
            FileType.PARQUET,
            artifact_files=artifact_files,
        )

    expected_transfers = [(path, osp.join(output_path, osp.basename(path)))
                          for path in artifact_files]
    assert attempts == [
        (expected_transfers, {
            'overwrite': True,
            'max_workers': artifact_export._PARQUET_COPY_MAX_WORKERS,
        }),
        (expected_transfers, {
            'overwrite': True,
            'max_workers': artifact_export._PARQUET_COPY_MAX_WORKERS,
        }),
    ]
    sleep.assert_called_once_with(60)
    assert fsutil.exists(osp.join(output_path, '_SUCCESS'))


def test_selected_parquet_copy_failure_propagates_without_success(
        tmp_path: pathlib.Path):
    output_path = (tmp_path / 'output.parquet').as_posix()

    with (
            patch.object(
                artifact_export.fsutil,
                'copy_files_threaded',
                side_effect=OSError('persistent copy failure'),
            ) as copy_files,
            patch('io_utils.retry.time.sleep') as sleep,
            pytest.raises(OSError, match='persistent copy failure'),
    ):
        write_artifact_to_s3(
            's3://internal/attempt/',
            output_path,
            FileType.PARQUET,
            artifact_files=[
                's3://internal/attempt/part-000.parquet',
                's3://internal/attempt/part-001.parquet',
            ],
        )

    assert copy_files.call_count == 5
    assert sleep.call_count == 4
    assert not fsutil.exists(osp.join(output_path, '_SUCCESS'))


# NOTE: CSV export with multiple parquet files fails.
@pytest.mark.parametrize("multi_sharded", [True, False])
def test_write_to_s3_csv(tmp_path: pathlib.Path, multi_sharded: bool):
    temp_path = tmp_path.resolve().as_posix()
    input_path = osp.join(temp_path, 'input.parquet')
    fsutil.makedirs(input_path)
    df = prepare_input_data(input_path, chunking=multi_sharded)
    output_path = osp.join(temp_path, "output.csv")
    stats = write_artifact_to_s3(input_path, output_path, FileType.CSV)
    if multi_sharded:
        # if artifact is multi sharded, write to output path as directory
        # with _SUCCESS file marker
        assert stats.display_path == output_path + '/'
        assert stats.size_bytes > 0
        assert fsutil.exists(output_path)
        output_files = fsutil.listdir(output_path)
        assert osp.join(output_path, '_KUMO_EXPORT') in output_files
        assert osp.join(output_path, '_SUCCESS') in output_files
        data_files = [
            path for path in output_files
            if osp.basename(path).startswith('part-')
        ]
        out_df = pd.concat([pd.read_csv(path) for path in data_files])
        assert len(out_df) == len(df)
    else:
        assert stats.display_path == output_path
        assert stats.size_bytes > 0
        assert fsutil.exists(output_path)
        out_df = pd.read_csv(output_path)
        assert len(out_df) == len(df)


def test_write_to_csv_preserves_uppercase_explicit_extension(
        tmp_path: pathlib.Path):
    input_path = (tmp_path / 'input.parquet').as_posix()
    fsutil.makedirs(input_path)
    expected = prepare_input_data(input_path, chunking=False)
    output_path = (tmp_path / 'output.CSV').as_posix()

    stats = write_artifact_to_s3(input_path, output_path, None)

    assert stats.display_path == output_path
    assert fsutil.exists(output_path)
    assert not fsutil.exists(output_path + '.csv')
    assert len(pd.read_csv(output_path)) == len(expected)


def test_write_to_csv_cleans_private_staging_on_converter_failure(
        tmp_path: pathlib.Path):
    input_path = tmp_path / 'input.parquet'
    input_path.mkdir()
    prepare_input_data(input_path.as_posix(), chunking=False)
    staged_dirs: list[pathlib.Path] = []

    def fail_conversion(_input: str, local_output: str) -> None:
        staging_dir = pathlib.Path(local_output).parent
        staged_dirs.append(staging_dir)
        assert staging_dir.stat().st_mode & 0o777 == 0o700
        raise RuntimeError('conversion failed')

    with pytest.raises(RuntimeError, match='conversion failed'):
        write_artifact_to_s3(
            input_path.as_posix(),
            (tmp_path / 'output.csv').as_posix(),
            FileType.CSV,
            csv_shard_converter=fail_conversion,
        )

    assert len(staged_dirs) == 1
    assert not staged_dirs[0].exists()


@pytest.mark.snowflake
@withSnowflakeCredentials
@pytest.mark.parametrize("with_staging", [True, False])
def test_write_to_snowflake(snowflake_key_pair_credentials: KeyPair,
                            with_staging: bool):
    # Existing S3 test data under the snowflake-integ-test/ prefix.
    # Must be under this prefix (not databricks-integ-test/) so the
    # unittest-iam-role STS credentials passed to Snowflake's COPY INTO
    # can access it — Snowflake reads the parquet directly from S3.
    input_path = ('s3://kumo-preprod-dataplane-bucket/'
                  'snowflake-integ-test/bp_output_test')
    id = str(uuid.uuid1()).upper().replace("-", "")

    connector_id = _TEST_CONNECTOR_ID
    connector_config = writable_snowflake_config(connector_id)
    connector = connector_config.as_connector('xva19026')
    connector.open(snowflake_key_pair_credentials.as_dict())

    if with_staging:
        write_mode = WriteMode.OVERWRITE
        staging_table_name = f'TEST_COPY_ARTIFACT_PRED+{id}_STAGING'
    else:
        write_mode = WriteMode.APPEND
        staging_table_name = None
    output_config = SnowflakePredictionOutput(
        storage_type=PredictionStorageType.SNOWFLAKE,
        artifact_type=PredictionArtifactType.PREDICTIONS,
        connector_id=ConnectorID(connector_id),
        table_name=f'TEST_COPY_ARTIFACT_PRED+{id}',
        staging_table_name=staging_table_name,
        write_mode=write_mode,
        extra_fields=[],
    )
    try:
        stats = write_artifact_to_snowflake(input_path, connector,
                                            output_config.write_mode,
                                            output_config.table_name,
                                            output_config.staging_table_name)
        assert stats.size_bytes > 0
        assert stats.display_path.endswith(output_config.table_name)

        # Read the output table
        out_df = connector.read_table(output_config.table_name)
        # The S3 test data contains 99 rows
        assert len(out_df) == 99
    finally:
        connector.drop_table_if_exists(output_config.table_name)
        if with_staging:
            assert output_config.staging_table_name is not None
            connector.drop_table_if_exists(output_config.staging_table_name)


def test_failed_s3_parquet_copy_retry(tmp_path: pathlib.Path):
    # Test that write_to_customer_s3 doesn't throw any errors after
    # retrying a failed operation. Simulate this case by removing some part
    # files from a successful job.
    temp_path = tmp_path.resolve().as_posix()
    input_path = osp.join(temp_path, 'input.parquet')
    fsutil.makedirs(input_path)
    df = prepare_input_data(input_path)

    output_path = osp.join(temp_path, "embed_output.parquet")
    stats = write_artifact_to_s3(input_path, output_path, None)

    # Make sure multiple files are created
    assert stats.size_bytes > 0
    output_files_list = fsutil.listdir(output_path)
    assert len(output_files_list) > 1
    # Remove the last file to simulate failed operation
    fsutil.rm(output_files_list[-1])

    # Retry the operation
    new_stats = write_artifact_to_s3(input_path, output_path, None)
    assert new_stats.size_bytes == stats.size_bytes
    # Make sure the missing file is re-created
    assert len(fsutil.listdir(output_path)) == len(output_files_list)
    out_df = parquet_util.read_parquet(output_path)
    assert_frame_equal(out_df, df)


def test_partitioned_export_allows_empty_destination_but_rejects_stale_data(
        tmp_path: pathlib.Path):
    input_path = (tmp_path / 'input.parquet').as_posix()
    fsutil.makedirs(input_path)
    prepare_input_data(input_path)
    output_path = (tmp_path / 'output.parquet').as_posix()
    fsutil.makedirs(output_path)

    write_artifact_to_s3(
        input_path,
        output_path,
        FileType.PARQUET,
        export_id='job-1',
    )

    fsutil.rm(osp.join(output_path, '_KUMO_EXPORT'))
    with pytest.raises(ArtifactExportError,
                       match='data from another export') as exc_info:
        write_artifact_to_s3(
            input_path,
            output_path,
            FileType.PARQUET,
            export_id='job-2',
        )
    assert exc_info.value.non_retryable


def test_csv_ready_renders_embedding_columns(tmp_path):
    df = pd.DataFrame()
    arr = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    df['embedding'] = arr.tolist()

    # Use parquet file to simulate production
    fname = os.path.join(tmp_path, "file.parquet")
    df.to_parquet(fname, index=False)
    df2 = parquet_util.read_parquet(fname)

    ready = artifact_export._csv_ready(df2)

    col = ready['embedding']
    assert col is not None
    assert col[0] == "1 2 3"
    assert col[1] == "4 5 6"
    assert col[2] == "7 8 9"


def test_the_hosted_refusal_carries_the_shared_wording():
    """The refusal is one rule, shared with the local sink; this activity only
    restates it in its own error type. The message comparison pins that the
    wording cannot diverge between the two backends.

    ``non_retryable`` is deliberately not asserted here: ``PublicException``
    already defaults it to ``True``, so an assertion on it holds whether or not
    ``_csv_ready`` sets it and would pin nothing.
    """
    from kumo_core.artifact_export import (
        MultiDimensionalCsvError,
        flatten_arrays_for_csv,
    )

    df = pd.DataFrame({'embedding': [np.array(((1, 2), (3, 4)))]})

    with pytest.raises(MultiDimensionalCsvError) as shared:
        flatten_arrays_for_csv(df)
    with pytest.raises(ArtifactExportError,
                       match='multi-dimensional array') as hosted:
        artifact_export._csv_ready(df)

    assert str(shared.value) == hosted.value.message


def test_an_unrelated_failure_is_not_reported_as_the_csv_refusal():
    """A frame can fail to render for reasons that have nothing to do with
    array dimensionality, and such a failure must not reach a customer worded
    as the one refusal this activity knows how to word -- nor marked
    permanently non-retryable on that basis."""
    df = pd.DataFrame({'embedding': [np.array([0.1, 0.2])]})

    def unrelated(_frame):
        raise ValueError('Wrong number of dimensions. values.ndim > ndim')

    with patch.object(artifact_export, 'flatten_arrays_for_csv', unrelated):
        with pytest.raises(Exception) as raised:
            artifact_export._csv_ready(df)

    assert type(raised.value) is ValueError
    assert 'Wrong number of dimensions' in str(raised.value)


@pytest.mark.parametrize('value', [np.array([0.1, 0.2]), [0.1, 0.2]])
def test_an_array_cell_renders_as_the_shared_rule_renders_it(value):
    """A CSV destination holds one string per cell either way, so the same
    frame exported by either backend reads back identically."""
    from kumo_core.artifact_export import flatten_arrays_for_csv

    df = pd.DataFrame({'EMBEDDING': [value]})

    assert (artifact_export._csv_ready(df)['EMBEDDING'].tolist() ==
            flatten_arrays_for_csv(df)['EMBEDDING'].tolist() == ['0.1 0.2'])


def test_the_export_marker_is_the_digest_both_backends_write(tmp_path):
    """The hosted export writes whatever ``export_marker`` returns, rather than
    a digest of its own.

    That is all this asserts -- the function against itself -- because that is
    all this backend decides. The digest itself stays compatible with markers
    already sitting in customer buckets, and is pinned literally in
    ``packages/kumo-core/tests/test_artifact_export.py::
    test_the_marker_digests_the_identity_it_is_given``.
    """
    from kumo_core.artifact_export import EXPORT_MARKER_NAME, export_marker

    temp_path = tmp_path.resolve().as_posix()
    input_path = osp.join(temp_path, 'input.parquet')
    fsutil.makedirs(input_path)
    prepare_input_data(input_path, chunking=True)
    output_path = osp.join(temp_path, 'output')

    write_artifact_to_s3(input_path, output_path, FileType.PARQUET,
                         export_id='job-1')

    marker = fsutil.read_text(osp.join(output_path, EXPORT_MARKER_NAME))
    assert marker == export_marker('job-1')


@pytest.mark.parametrize('fail_conversion', [False, True])
def test_selected_csv_uses_hosted_scratch_and_cleans_it(tmp_path, monkeypatch,
                                                        fail_conversion):
    from io_utils.fsutil import common
    scratch = tmp_path / 'hosted-scratch'
    scratch.mkdir()
    monkeypatch.setattr(common, '_temp_prefix', lambda: str(scratch) + '/')
    source = tmp_path / 'source.parquet'
    source.write_bytes(b'converter-fixture')
    staged = []

    def convert(input_path, output_path):
        assert input_path == str(source)
        path = pathlib.Path(output_path)
        assert path.parent.parent == scratch
        staged.append(path)
        path.write_text('ENTITY\n1\n')
        if fail_conversion:
            raise RuntimeError('injected conversion failure')

    def export():
        return write_artifact_to_s3(str(source), str(tmp_path / 'output.csv'),
                                    FileType.CSV, artifact_files=[str(source)],
                                    csv_shard_converter=convert)

    if fail_conversion:
        with pytest.raises(RuntimeError, match='conversion failure'):
            export()
    else:
        export()
        assert (tmp_path / 'output.csv').read_text() == 'ENTITY\n1\n'
    assert staged and all(not path.parent.exists() for path in staged)
    assert not list(scratch.iterdir())


@pytest.mark.parametrize('selected', [False, True])
def test_hosted_selected_and_legacy_share_destination_rules(
        tmp_path, monkeypatch, selected):
    from kumo_core import artifact_export as shared
    source = tmp_path / 'source.parquet'
    source.write_bytes(b'parquet-fixture')
    destination = str(tmp_path / 'output') + '/'
    calls = []
    prepare = shared.prepare_export_destination
    publish = shared.publish_export_destination

    def preparing(*args, **kwargs):
        calls.append('prepare')
        return prepare(*args, **kwargs)

    def publishing(*args, **kwargs):
        assert (tmp_path / 'output' / source.name).exists()
        assert not (tmp_path / 'output' / '_SUCCESS').exists()
        calls.append('publish')
        return publish(*args, **kwargs)

    monkeypatch.setattr(shared, 'prepare_export_destination', preparing)
    monkeypatch.setattr(shared, 'publish_export_destination', publishing)
    kwargs = {'artifact_files': [str(source)]} if selected else {}
    write_artifact_to_s3(str(source), destination, FileType.PARQUET,
                         export_id='job-1', **kwargs)
    assert calls == ['prepare', 'publish']
    assert (tmp_path / 'output' / '_SUCCESS').exists()
    with pytest.raises(ArtifactExportError, match='another export'):
        write_artifact_to_s3(str(source), destination, FileType.PARQUET,
                             export_id='job-2', **kwargs)
    assert calls == ['prepare', 'publish', 'prepare']


@pytest.mark.parametrize('selected', [False, True])
def test_hosted_empty_destination_and_failed_recopy_can_retry(
        tmp_path, monkeypatch, selected):
    from kumo_core.artifact_export import EXPORT_MARKER_NAME

    from rest.backends.ml_backend import artifact_export

    source = tmp_path / 'source.parquet'
    pd.DataFrame({'ENTITY': [1]}).to_parquet(source)
    target = tmp_path / 'output'
    target.mkdir()
    kwargs = {'artifact_files': [str(source)]} if selected else {}

    def export():
        return write_artifact_to_s3(str(source),
                                    str(target) + '/', FileType.PARQUET,
                                    export_id='job-1', **kwargs)

    copy = artifact_export._write_artifact_to_s3_parquet

    def interrupted(*args, **kwargs):
        assert not (target / '_SUCCESS').exists()
        copy(*args, **kwargs)
        raise RuntimeError('transfer interrupted')

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr(artifact_export, '_write_artifact_to_s3_parquet',
                          interrupted)
            with pytest.raises(RuntimeError, match='transfer interrupted'):
                export()
        assert (target / EXPORT_MARKER_NAME).exists()
        assert not (target / '_SUCCESS').exists()
        export()
        assert (target / '_SUCCESS').exists()
        assert pd.read_parquet(target / source.name).ENTITY.tolist() == [1]


@pytest.mark.parametrize('selected', [False, True])
def test_hosted_csv_retry_replaces_partially_uploaded_part(
        tmp_path, monkeypatch, selected):
    source = tmp_path / 'source.parquet'
    expected = pd.DataFrame({'ENTITY': [10, 20]})
    expected.to_parquet(source)
    target = tmp_path / 'output'
    kwargs = {'artifact_files': [str(source)]} if selected else {}

    def export():
        write_artifact_to_s3(str(source),
                             str(target) + '/', FileType.CSV, export_id='job-1',
                             **kwargs)

    def interrupted(source, destination, **kwargs):
        pathlib.Path(destination).write_bytes(
            pathlib.Path(source).read_bytes()[:10])
        raise RuntimeError('partial upload')

    with monkeypatch.context() as patch:
        patch.setattr(fsutil, 'copy_file', interrupted)
        with pytest.raises(RuntimeError, match='partial upload'):
            export()
    assert (target / 'part-00000.csv').stat().st_size == 10
    assert not (target / '_SUCCESS').exists()
    export()
    assert_frame_equal(pd.read_csv(target / 'part-00000.csv'), expected)
    assert (target / '_SUCCESS').exists()
