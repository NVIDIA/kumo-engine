# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import random
import tempfile
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.types.conversions import dtype_to_pandas
from kumo_core.types.schema import IngestedTableFingerprint
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from kumo.connector.connector import (
    Connector,
    DuplicateColumnsError,
    SourceTableNotFoundError,
)
from kumo.connector.csv_connector import CSVConnector, read_csv_safe
from kumo.connector.utils import ColumnDtypeMismatchError, List, Optional
from kumo.data import get_persist_dir
from kumo.graph.graph import Graph
from kumo.graph.table import Table
from kumo.testing.data import financial_csv_connector
from kumo.utils import detect_csv_sep
from kumo.utils.csv import NoValidSeparatorFoundError


def test_csv_connector_config():
    connector = CSVConnector('csv', root_dir='my_test_dir')

    assert asdict(connector.config()) == {
        '_target_': 'kumo.connector.csv_connector.CSVConnector',
        'name': 'csv',
        'root_dir': 'my_test_dir',
        'suffix': 'csv',
        'sep': None,
        'na_values': None,
        'compression': 'infer',
        'on_bad_lines': 'skip',
        'escapechar': None,
    }
    assert connector == Connector.from_config(connector.config())


def test_export_table_projects_selected_columns_and_preserves_na_values(
        tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # The connector writes under the deploy_env-derived data persist dir
    # (kumo.data.get_persist_dir); read exported artifacts back from there.
    persist_dir = Path(get_persist_dir())

    table_name = "providers"
    csv_path = source_dir / f"{table_name}.csv"
    with csv_path.open("w") as f:
        f.write("id,amount,nullable_code,zip\n")
        for i in range(60_000):
            nullable_code = "?" if i == 7 else f"code_{i}"
            f.write(f"{i},{i * 10},{nullable_code},{100_000_000 + i}\n")
        f.write("60000,600000,code_60000,3062223ND\n")

    connector = CSVConnector("csv", root_dir=str(source_dir), na_values="?")
    table_fp = IngestedTableFingerprint(
        connector_name=connector.name,
        source_name=table_name,
        cols={
            "id": Dtype.int64,
            "amount": Dtype.int64,
            "nullable_code": Dtype.string,
        },
        pkey="id",
        time_col=None,
        end_time_col=None,
    )

    metrics = connector.export_table(table_name, table_fp, "v1")

    export_path = persist_dir / "ingest" / table_fp.ingest_path_fpid / "v1"
    table = pq.read_table(export_path)
    assert metrics.num_rows == 60_001
    assert table.column_names == ["id", "amount", "nullable_code"]
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("amount").type == pa.int64()
    assert table.schema.field("nullable_code").type == pa.string()
    assert table.column("nullable_code").null_count == 1


def test_export_table_allows_decimal_formatted_int_values(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # The connector writes under the deploy_env-derived data persist dir
    # (kumo.data.get_persist_dir); read exported artifacts back from there.
    persist_dir = Path(get_persist_dir())

    table_name = "transactions"
    csv_path = source_dir / f"{table_name}.csv"
    with csv_path.open("w") as f:
        f.write("id,amount,zip\n")
        f.write("1,10.0,100000000\n")
        f.write("2,11.0,3062223ND\n")

    connector = CSVConnector("csv", root_dir=str(source_dir))
    table_fp = IngestedTableFingerprint(
        connector_name=connector.name,
        source_name=table_name,
        cols={
            "id": Dtype.int64,
            "amount": Dtype.int64,
        },
        pkey="id",
        time_col=None,
        end_time_col=None,
    )

    metrics = connector.export_table(table_name, table_fp, "v1")

    export_path = persist_dir / "ingest" / table_fp.ingest_path_fpid / "v1"
    table = pq.read_table(export_path)
    assert metrics.num_rows == 2
    assert table.column_names == ["id", "amount"]
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("amount").type == pa.int64()


def test_export_table_handles_multi_block_and_multi_file_inputs(tmp_path):
    """Regression test: across multiple CSV blocks (and multiple shards),
    integer columns with decimal-formatted values in later blocks must not
    crash schema inference, alphanumeric values in unselected columns must
    not affect the output, and a projected string column must round-trip."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # The connector writes under the deploy_env-derived data persist dir
    # (kumo.data.get_persist_dir); read exported artifacts back from there.
    persist_dir = Path(get_persist_dir())

    table_name = "events"
    # Sharded CSVs live in a directory matching ``table_name``. The first
    # shard's "amount" column looks like int64; the second shard contains
    # decimal-formatted ints, which would force a schema mismatch at the
    # ParquetWriter without explicit pinning. The unselected "extra" column
    # also carries alphanumeric values that must not affect the output.
    table_dir = source_dir / table_name
    table_dir.mkdir()
    shards = [
        [f"{i},10,9876ABC{i},code_{i}\n" for i in range(2000)],
        [f"{2000 + i},10.5,9876ABC{i},code_{i}\n" for i in range(2000)],
    ]
    for shard_idx, rows in enumerate(shards):
        csv_path = table_dir / f"shard_{shard_idx}.csv"
        with csv_path.open("w") as f:
            f.write("id,amount,extra,label\n")
            f.writelines(rows)

    connector = CSVConnector("csv", root_dir=str(source_dir))
    table_fp = IngestedTableFingerprint(
        connector_name=connector.name,
        source_name=table_name,
        cols={
            "id": Dtype.int64,
            "amount": Dtype.int64,
            "label": Dtype.string,
        },
        pkey="id",
        time_col=None,
        end_time_col=None,
    )

    metrics = connector.export_table(table_name, table_fp, "v1")

    export_path = persist_dir / "ingest" / table_fp.ingest_path_fpid / "v1"
    table = pq.read_table(export_path)
    assert metrics.num_rows == 4000
    # Unselected "extra" column must not appear in the output.
    assert table.column_names == ["id", "amount", "label"]
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("amount").type == pa.int64()
    assert table.schema.field("label").type == pa.string()


def test_load_csv():
    connector = financial_csv_connector()

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]

    df = connector.read_table('TRANS', pkey, cols)

    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]

    chunk = connector.read_table('TRANS', pkey, cols, num_rows=5)
    assert df[:5].equals(chunk)

    chunk = connector.read_table('TRANS', pkey, cols, num_rows=3, offset=2)
    assert df[2:5].equals(chunk)


def test_load_shards():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'shard_test')
    connector = CSVConnector('csv', root_dir=root_dir)

    df1 = connector.read_table('full')
    df2 = connector.read_table('sharded_csv')

    assert len(df1) == len(df2)


def test_load_csv_sample():
    connector = financial_csv_connector()

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]

    df = connector.read_table_sample('TRANS', pkey, cols, num_rows=12)

    assert df.shape[0] == 12
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]

    df = connector.read_table('TRANS', pkey, cols, num_rows=1234)
    assert df.shape[0] == 1234

    random.seed(10)
    frac = 0.01
    df = connector.read_table_sample('TRANS', pkey, cols, frac=frac)
    assert df.shape[0] == 50


def test_detect_csv_sep(tmp_path: Path):
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', 'csv_data', 'FINANCIAL', 'TRANS.csv')
    assert detect_csv_sep(path) == ';'

    path = (tmp_path / "text.csv").as_posix()
    with open(path, "w") as f:
        f.write('Adventure|Animation|Children|Comedy|Fantasy')
    assert detect_csv_sep(path) == '|'


def test_detect_csv_sep_for_empty_file(tempdir):
    fsutil.makedirs(tempdir.name)
    file_path = os.path.join(tempdir.name, 'no_data.csv')
    with open(file_path, "w") as f:
        f.write('\n')
    with pytest.raises(NoValidSeparatorFoundError,
                       match="Could not auto-detect a delimiter"):
        assert detect_csv_sep(file_path)
    fsutil.rm(tempdir.name)


def test_detect_csv_sep_for_windows(tempdir):
    fsutil.makedirs(tempdir.name)
    file_path = os.path.join(tempdir.name, 'windows.csv')
    with open(file_path, "w") as f:
        f.write('1,2,3\r\n4,5,6\r\n7,8,9\r\n')
    assert detect_csv_sep(file_path) == ','


def test_get_sep(tempdir):
    df1 = pd.DataFrame(data={
        'col1': [1, 2, 3, 4],
        'col2': [1, 2, 3, 4],
    })
    df2 = pd.DataFrame(data={
        'col1': [1, 2, 3, 4],
        'col2': [1, 2, 3, 4],
    })
    table_path = os.path.join(tempdir.name, 'data')
    fsutil.makedirs(table_path, exist_ok=True)
    df1.to_csv(os.path.join(table_path, 'file1.csv'), index=False, sep='|')
    df2.to_csv(os.path.join(table_path, 'file2.csv'), index=False, sep='|')

    connector = CSVConnector('csv', root_dir=table_path)
    assert connector._get_sep(table_path) == '|'


def test_get_sep_single_col(tempdir):
    df = pd.DataFrame(data={
        'col1': [1, 2, 3, 4],
    })
    table_path = os.path.join(tempdir.name, 'data')
    fsutil.makedirs(table_path)
    df.to_csv(os.path.join(table_path, 'file1.csv'), index=False, sep='|')

    connector = CSVConnector('csv', root_dir=table_path)
    assert connector._get_sep(table_path) == ','


@pytest.mark.parametrize("data,dtype,expect_result", [
    ('1\n2', Dtype.int, [1, 2]),
    ('1\n2', Dtype.date, [
        pd.Timestamp('1970-01-01 00:00:01'),
        pd.Timestamp('1970-01-01 00:00:02'),
    ]),
    ('1\n2', Dtype.string, ["1", "2"]),
    ('1.0\n2.1', Dtype.int, None),
    ('a\n2.1', Dtype.int, None),
    ('a\n2.1', Dtype.date, [pd.NaT, pd.NaT]),
    ('true\nFalse\n0\n1', Dtype.bool, [True, False, False, True]),
])
def test_read_csv_safe(data: str, dtype: Dtype,
                       expect_result: Optional[List[Any]]):
    io = StringIO(f'A\n{data}')
    dtypes = {'A': dtype}
    if expect_result is None:
        with pytest.raises(ColumnDtypeMismatchError):
            read_csv_safe(io, dtypes=dtypes)
    else:
        df = read_csv_safe(io, dtypes=dtypes)
        assert list(df['A']) == expect_result


def test_read_csv_safe_infers_int_from_float_like():
    io = StringIO('A\n1.0\n2.0\n3.0\n')
    df = read_csv_safe(io, dtypes={})
    assert df['A'].dtype == pd.Int64Dtype()
    assert list(df['A']) == [1, 2, 3]


def test_read_csv_safe_parsing_types(csv_edge_case_data,
                                     write_csv_edge_case_files):
    """Test read_csv_safe on combinations of test data and separators (checks
    that the read operation does not fail and that columns are separated
    correctly when expected to be)
    - Write each of 4 test data cases with the separators: , : ; \t |
    - Test reading all 4 test cases with ";" as the separator, plus the
        inferred separator if different
    - Two of the four tests have structural issues that result in sep inference
        defaulting to "," regardless of how the file is written
    """

    data = csv_edge_case_data()  # non-timestamp data
    # two of the four tests have structural issues preventing sep inference
    structure_issues = [
        "missing_and_empty_fields",
        "delimiter_mixes",
    ]
    for separator in [';', ',', '\t', '|', ':']:
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_csv_edge_case_files(data, tmp_dir, separator=separator)

            # Get all csv files in the test directory
            files = fsutil.listdir(path=tmp_dir, suffix='csv')
            # defaults used by kumo
            default_kwargs = {
                'sep': ';',
                'compression': 'infer',
                'on_bad_lines': 'skip',
                'escapechar': None,
                'nrows': 10
            }

            # some test data written out with some delimeters won't parse
            # correctly, expectations of that captured here
            expected_shapes = {
                ('01_missing_and_empty_fields.csv', ';'): (6, 1),
                ('01_missing_and_empty_fields.csv', ','): (5, 5),
                ('02_formatting_and_special_characters.csv', ','): (8, 5),
                ('03_complex_and_invalid_structure.csv', ','): (6, 5),
                ('04_column_duplication.csv', ','): (2, 6),
                ('05_delimiter_mixes.csv', ','): (3, 5),
                ('01_missing_and_empty_fields.csv', '\t'): (5, 1),
                ('02_formatting_and_special_characters.csv', '\t'): (8, 5),
                ('03_complex_and_invalid_structure.csv', '\t'): (6, 5),
                ('04_column_duplication.csv', '\t'): (2, 6),
                ('05_delimiter_mixes.csv', '\t'): (3, 1),
                ('01_missing_and_empty_fields.csv', '|'): (6, 1),
                ('02_formatting_and_special_characters.csv', '|'): (8, 5),
                ('03_complex_and_invalid_structure.csv', '|'): (6, 5),
                ('04_column_duplication.csv', '|'): (2, 6),
                ('05_delimiter_mixes.csv', '|'): (3, 1),
                ('01_missing_and_empty_fields.csv', ':'): (6, 1),
                ('02_formatting_and_special_characters.csv', ':'): (8, 5),
                ('03_complex_and_invalid_structure.csv', ':'): (6, 5),
                ('04_column_duplication.csv', ':'): (2, 6),
                ('05_delimiter_mixes.csv', ':'): (3, 1),
            }

            # Test each file
            for file_path in files:
                # Try reading with read_csv_safe (no dtypes passed in,
                # how its used in context)
                try:
                    # basic check that file is read without failure
                    try:
                        result_df = read_csv_safe(file_path, {},
                                                  **default_kwargs)
                    except DuplicateColumnsError:
                        assert "04_column_duplication.csv" in file_path
                        continue

                    # infer separator correctly and read with that
                    inferred_sep = detect_csv_sep(file_path)

                    if not any(x in file_path for x in structure_issues):
                        assert inferred_sep == separator

                    # in case where tests writes with semicolon, table with
                    # some rows forced to be ; becomes normal
                    elif separator == ';' and "delimiter_mixes" in file_path:
                        assert inferred_sep == ';'
                    else:
                        # remaining cases default to "," inference
                        assert inferred_sep == ','

                    # next, try reading with the inferred separator
                    #  (if it isn't the one already tried)
                    if inferred_sep != default_kwargs['sep']:
                        inferred_kwargs = default_kwargs.copy()
                        inferred_kwargs.update({'sep': inferred_sep})

                        try:
                            result_df = read_csv_safe(file_path, {},
                                                      **inferred_kwargs)
                        except DuplicateColumnsError:
                            assert "04_column_duplication.csv" in file_path
                            continue

                        # basic check that columns were separated correctly
                        # and rows read

                        assert result_df.shape[0] == expected_shapes[(
                            file_path.split('/')[-1], separator)][0]

                except Exception as e:
                    pytest.fail(f"Failed to read {file_path}: {str(e)}")


def test_read_csv_safe_index():
    io = StringIO('A\n1\n5\n')
    dtypes = {'A': Dtype.int}
    df = read_csv_safe(io, dtypes=dtypes, index_col='A')
    assert list(df.index.values) == [1, 5]


def test_csv_connector_list_tables():
    connector = financial_csv_connector()
    # TODO(Dulloor): source_name is not populated by get_table_metadata
    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    assert len(table_metadata_configs) == 15

    with pytest.raises(SourceTableNotFoundError):
        table_metadata_configs = connector.get_table_metadata()

    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'shard_test')
    connector = CSVConnector('csv', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    assert len(table_metadata_configs) == 2
    table_names = {t.source_name for t in table_metadata_configs}
    assert table_names == {'full', 'sharded_csv'}


def test_csv_col_dtypes_infer_temporal():
    """
    Test to check that if DATE/TIME columns are type STRING or INT,
    they are inferred as temporal types. This is only done for CSVConnector
    and skipped for Connector types.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        data = {
            'DATE_1': [
                '2024-01-01 00:00:00', '2024-02-15 00:00:00',
                '2024-03-30 00:00:00', '2024-04-12 00:00:00',
                '2024-05-25 00:00:00'
            ],
            'TIME_1':
            [1704060000, 1707984000, 1711800000, 1712908800, 1716609600]
        }
        df = pd.DataFrame(data)

        table_name = 'infer_int_string_temporal_test'
        table_path = os.path.join(tmp_dir, f'{table_name}.csv')
        df.to_csv(table_path, index=False)

        connector = CSVConnector('csv', root_dir=tmp_dir)
        table_metadata_configs = connector.get_table_metadata([table_name])
        expected_dtypes = {
            table_name: {
                'DATE_1': Dtype.date,
                'TIME_1': Dtype.time,
            }
        }

        for col in table_metadata_configs[0].cols:
            assert col.dtype == expected_dtypes[table_name][col.name]


def test_csv_col_types_infer_and_cast():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'movielens')
    connector = CSVConnector('csv', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)
    graph.add('tag_small')
    graph.add('tags_epoch_small')
    graph.add('synthetic_table')

    expected_dtypes = {
        'tag_small': {
            'userId': Dtype.int64,
            'movieId': Dtype.int64,
            'tag': Dtype.string,
            'timestamp': Dtype.time
        },
        'tags_epoch_small': {
            'userId': Dtype.int64,
            'movieId': Dtype.int64,
            'tag': Dtype.string,
            'timestamp': Dtype.time
        },
        'synthetic_table': {
            'index': Dtype.int64,
            'textcol': Dtype.string,
            'integer': Dtype.int64,
            'multicat1': Dtype.string,
            'lists': Dtype.string,
            'date': Dtype.date,
        }
    }
    for table in table_metadata_configs:
        # test the two versions of the tag table
        if 'tag' in table.source_name or 'synthetic' in table.source_name:
            for col in table.cols:
                graph[table.name].add_col(col.name, dtype=col.dtype)
                assert col.dtype == expected_dtypes[table.source_name][col.name]

            graph[table.name].infer_stypes()
            # infer timestamp unit:
            graph[table.name].infer_dtypes_from_stypes()


def test_csv_connector_all_types(write_stype_dtype_test_csv):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = write_stype_dtype_test_csv(tmp_dir)

        # Read the CSV file to get the column names
        with open(output_path, 'r') as f:
            header = f.readline().strip().split(',')

        # Create a single dtypes dictionary for all columns
        expected_dtypes = {}
        expected_stypes = {}

        for column in header:
            # Extract the dtype from the column name (format: stype__dtype)
            parts = column.split('__')
            if len(parts) < 2:
                continue

            stype_str = parts[0]
            dtype_str = parts[1]
            # Map the dtype string to Dtype
            try:
                dtype = getattr(Dtype, dtype_str)
                # Add to the dtypes dictionary
                expected_dtypes[column] = dtype

                stype = getattr(Stype, stype_str)
                expected_stypes[column] = stype
            except AttributeError:
                # Skip if the dtype is not a valid kumo dtype
                continue

        connector = CSVConnector('csv', root_dir=tmp_dir)
        table_metadata_configs = connector.get_table_metadata()
        types_table = [
            table for table in table_metadata_configs
            if table.source_name == 'stype_dtype_combinations'
        ][0]
        table = Table.from_source_config(types_table,
                                         name='stype_dtype_combinations',
                                         data_source=types_table.data_source,
                                         connector=connector)

        table.infer_stypes()
        table.infer_dtypes_from_stypes()

        for col in table.cols:
            # Implementation of lists via csv strings is a todo
            if expected_dtypes[col.name] in [
                    Dtype.floatlist, Dtype.intlist, Dtype.stringlist
            ]:
                continue

            if expected_dtypes[col.name] == Dtype.time:
                # DiskGraph does not auto-detect the fixture's time-only
                # HH:MM:SS values, so name inference leaves them as strings.
                assert col.dtype == Dtype.string
                assert col.stype != Stype.timestamp
                continue
            elif expected_dtypes[col.name] == Dtype.binary:
                assert col.dtype == Dtype.string
            elif expected_dtypes[col.name] == Dtype.timedelta:
                assert col.dtype == Dtype.int64
            # integers and floats always inferred as *64 for csv (
            # but can be explicitly set otherwise)
            elif col.dtype == Dtype.int64:
                assert expected_dtypes[col.name].is_int()
            elif col.dtype == Dtype.float64:
                assert expected_dtypes[col.name].is_float()
            elif col.dtype == Dtype.time:
                assert expected_dtypes[col.name].is_temporal()
            # not expected to be inferred, but can be explicitly set:

            else:
                assert col.dtype == expected_dtypes[col.name]

            if expected_stypes[col.name] == Stype.ID:
                assert col.stype in [Stype.ID, Stype.categorical]
            # test data too few rows for string to be inferred as text
            elif expected_stypes[col.name] == Stype.text:
                assert col.stype in [Stype.text, Stype.categorical]
            elif expected_stypes[col.name] == Stype.sequence:
                assert col.stype == Stype.categorical
            else:
                assert col.stype == expected_stypes[col.name]

        # force a new read + cast

        recast = connector.read_table_sample('stype_dtype_combinations',
                                             cols=table.cols, skip_cache=True,
                                             cast_results=True)
        # 31 valid (stype, dtype) pairs plus the fixture's two custom columns.
        # Was 35: `text__binary` and `sequence__binary` are no longer valid
        # pairs now that `Stype` is kumoapi's, which accepts `binary` only for
        # `categorical` and `ID`.
        assert recast.shape == (1000, 33)

        # test some manual type edits
        type_mod_dict = {
            'numerical__int': (Dtype.time, Stype.timestamp),
            'numerical__int32': (Dtype.int32, Stype.numerical),
            'numerical__int64': (Dtype.int64, Stype.numerical),
            'numerical__float32': (Dtype.float32, Stype.numerical),
            'numerical__float64': (Dtype.float64, Stype.numerical),
            'numerical__float': (Dtype.float, Stype.categorical),
            'categorical__binary': (Dtype.binary, Stype.categorical),
            'numerical__timedelta': (Dtype.timedelta, Stype.numerical),
            'ID__int': (Dtype.int, Stype.ID),
            'ID__int32': (Dtype.int32, Stype.ID),
            'ID__int64': (Dtype.int64, Stype.ID),
            'ID__float': (Dtype.float, Stype.ID),
            'ID__float32': (Dtype.float32, Stype.ID),
            'ID__float64': (Dtype.float64, Stype.ID),
            'ID__string': (Dtype.string, Stype.ID),
            'ID__binary': (Dtype.binary, Stype.ID),
        }
        for col, (dtype, stype) in type_mod_dict.items():
            table[col].dtype = dtype
            table[col].stype = stype

        recast = connector.read_table_sample('stype_dtype_combinations',
                                             cols=table.cols, skip_cache=True,
                                             cast_results=True)
        # 31 valid (stype, dtype) pairs plus the fixture's two custom columns.
        # Was 35: `text__binary` and `sequence__binary` are no longer valid
        # pairs now that `Stype` is kumoapi's, which accepts `binary` only for
        # `categorical` and `ID`.
        assert recast.shape == (1000, 33)


def test_csv_connector_timestamp_handling(csv_edge_case_data,
                                          write_csv_edge_case_files):
    # Covers only the DG-native read path: timestamp stype/dtype inference +
    # pandas `read_table_sample` casting over the edge-case timestamp CSVs. The
    # Spark-cast row-nullification assertions have no DG-native equivalent on
    # the connector boundary.
    data = csv_edge_case_data(timestamps=True)
    expected_dtype_by_table = {
        # ISO 8601 values share a DiskGraph-supported format.
        '01_standard_iso8601': Dtype.time,
        # Date-only ISO values share a DiskGraph-supported format.
        '02_date_only': Dtype.time,
        # RFC3339 Z and numeric offsets share a supported offset format.
        '03_datetime_with_timezone': Dtype.time,
        # RFC3339 Z and UTC numeric offsets share a supported offset format.
        '03_datetime_with_timezone_utc': Dtype.time,
        # DiskGraph auto-detection does not support 12-hour AM/PM formats.
        '04_twelve_hour_am_pm_format': Dtype.string,
        # Only one mixed TZ value matches the selected supported format.
        '05_mixed_formats_tz': Dtype.string,
        # Only one mixed non-TZ value uses a supported format.
        '05_mixed_formats_ntz': Dtype.string,
        # Only one malformed value resolves, leaving 75% apply failures.
        '06_malformed_timestamps': Dtype.string,
        # Each integer epoch fixture resolves to its magnitude-derived unit.
        '07_unix_timestamps_s': Dtype.time,
        '07_unix_timestamps_ms': Dtype.time,
        '07_unix_timestamps_us': Dtype.time,
        '07_unix_timestamps_ns': Dtype.time,
        # Non-null dashed dates and malformed values are unsupported.
        '08_missing_and_null_placeholders': Dtype.string,
        # Zoned and unzoned fractional values do not share a detected format.
        '09_milliseconds_TZ': Dtype.string,
        # Unzoned fractional ISO timestamps share a supported format.
        '09_milliseconds_NTZ': Dtype.time,
        # Slash, dotted, and dashed values do not share a supported format.
        '10_locale_formats_mixed': Dtype.string,
        # Bare year (%Y) and year-month (%Y%m) integers are day-less, so the
        # name guard keeps them numeric instead of inventing a day.
        '11_integer_encoded_dates_1': Dtype.int64,
        '11_integer_encoded_dates_2': Dtype.int64,
        # Full encoded dates/datetimes still resolve to a temporal dtype.
        '11_integer_encoded_dates_3': Dtype.time,
        '12_integer_encoded_dates_with_time': Dtype.time,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_csv_edge_case_files(data, tmp_dir, separator=',')

        connector = CSVConnector('csv', root_dir=tmp_dir)
        table_metadata_configs = connector.get_table_metadata()
        assert {table.source_name
                for table in table_metadata_configs
                } == set(expected_dtype_by_table)
        for table_config in table_metadata_configs:
            expected_dtype = expected_dtype_by_table[table_config.source_name]
            timestamp_config = next(col for col in table_config.cols
                                    if col.name == 'timestamp')
            assert timestamp_config.dtype == expected_dtype

            # 1. create table and infer types
            table = Table.from_source_config(
                table_config, name=table_config.source_name[3:],
                data_source=table_config.data_source, connector=connector)
            table.infer_stypes()
            table.infer_dtypes_from_stypes()
            assert table['timestamp'].dtype == expected_dtype
            if expected_dtype.is_temporal():
                assert table['timestamp'].stype == Stype.timestamp
            else:
                # Compatibility change: unsupported name-matched formats now
                # stay strings instead of becoming all-null timestamps.
                assert table['timestamp'].stype != Stype.timestamp

            # 2. read table sample with pandas (including casting)
            table_sample = connector.read_table_sample(table_config.source_name,
                                                       cols=table.cols,
                                                       skip_cache=True,
                                                       cast_results=True)

            assert table_sample.shape[0] > 0
            assert table_sample.shape[1] == 4


def test_csv_connector_type_inference(tempdir):
    tables_path = os.path.join(tempdir.name, 'tables')
    os.makedirs(tables_path, exist_ok=True)
    dummy_table_path = os.path.join(tables_path, 'DUMMY.csv')

    pd.DataFrame(
        {
            'int': ['1', '2', '3'],
            'float_int_valued': ['1.0', '2.0', '3.0'],
            'float_int_valued_missing': ['1.0', '2.0', pd.NA],
            'float': ['1.1', '2.2', '3.3'],
            'int_leading_zero': ['01', '02', '03'],
            'float_leading_zero': ['01.0', '02.0', '03.0'],
            'int_missing': ['1', '2', pd.NA]
        }, dtype='string').to_csv(dummy_table_path, index=False)

    connector = CSVConnector('csv', root_dir=tables_path)

    actual = [col.dtype for col in connector.get_table_metadata()[0].cols]
    expected = [
        Dtype.int64, Dtype.int64, Dtype.int64, Dtype.float64, Dtype.string,
        Dtype.string, Dtype.int64
    ]

    assert actual == expected

    df = connector.read_table('DUMMY')

    actual_pandas = df.dtypes.to_list()
    expected_pandas = [dtype_to_pandas(e) for e in expected]

    assert actual_pandas == expected_pandas


def test_csv_num_rows():
    connector = financial_csv_connector()
    # single file
    assert connector._get_table_num_rows('LOAN') == 682

    # directory with multiple files
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', '..', 'test', 'csv_data', 'shard_test')

    shard_connector = CSVConnector('csv', root_dir, na_values='?')
    assert shard_connector._get_table_num_rows('sharded_csv') == 77


def test_write_csv_table(tempdir):
    data = [{
        "name": "ted",
        "age": 34,
        "value": 0.10
    }, {
        "name": "barney",
        "age": 35,
        "value": 1.2
    }]
    df = pd.DataFrame.from_dict(data, )

    fsutil.makedirs(tempdir.name, exist_ok=True)
    connector = CSVConnector('csv', root_dir=tempdir.name)
    connector.write_pandas('test', df)
    assert fsutil.exists(osp.join(tempdir.name, 'test.csv'))


def test_csv_connector_load_full_s3_dataset(tempdir,
                                            require_kumo_public_datasets_s3):
    connector = CSVConnector(
        'csv', root_dir="s3://kumo-public-datasets/financial/csv/")

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('TRANS', pkey, cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [
        dtype_to_pandas(col[2]) for col in cols
    ]
    assert df.shape[0] == 1056320


def test_csv_connector_load_string_ids(tempdir,
                                       require_kumo_public_datasets_s3):
    connector = CSVConnector('csv',
                             root_dir="s3://kumo-public-datasets/test_csv/")

    df = connector.read_table('test_csv')
    # Note that if it becomes 3, the test should fail since id `01` and `1`
    # are different ids.
    assert len(df['id'].unique()) == 6

    pkey = 'id'
    cols = [
        ('id', Stype.ID, Dtype.string),
        ('value', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('test_csv', pkey, cols)
    assert len(df.index.unique()) == 6


def test_csv_connector_check_encoding():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'encoding_test')
    connector = CSVConnector('csv', root_dir=root_dir)
    # Verify that get_table_metadata doesn't fail on the unsupported files
    metadata = connector.get_table_metadata()
    # Expect only the table in utf-8 encoding to be included:
    assert len(metadata) == 1
    assert metadata[0].table_name == 'inutf8'


@pytest.mark.parametrize('contents', [
    'id,empty,code\n9007199254740993,,001\n9007199254740994,,002\n',
    'id,value\n1,10.0\n2,20\n',
    'id,value\n1,true\n2,false\n',
])
def test_shared_csv_metadata_and_preview(tmp_path, contents):
    from kumo_core.sources.object_store import ObjectStoreSourceReader
    from pyarrow.fs import LocalFileSystem

    (tmp_path / 'events.csv').write_text(contents)
    hosted = CSVConnector('csv', root_dir=str(tmp_path))
    local = ObjectStoreSourceReader(LocalFileSystem(), root=str(tmp_path))
    metadata = hosted._get_table_cols_metadata('events')
    assert {
        column.name: column.dtype
        for column in metadata.cols
    } == local.dtypes('events')
    expected = hosted._load(str(tmp_path / 'events.csv'), num_rows=2)
    empty = local.head('events', 0)
    assert empty.num_rows == 0
    assert empty.schema.names == list(local.dtypes('events'))
    actual = local.head('events', 2).to_pandas()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize('compression', [None, 'gzip', 'zip'])
def test_shared_csv_preview_preserves_pandas_dialects(tmp_path, compression):
    path = tmp_path / 'events.csv'
    pd.DataFrame({
        'id': [1, 2],
        'value': [10, 20]
    }).to_csv(path, index=False, sep=' ', compression=compression)
    result = read_csv_safe(str(path), {}, sep=r'\s+', compression=compression)
    assert result.to_dict('list') == {'id': [1, 2], 'value': [10, 20]}
