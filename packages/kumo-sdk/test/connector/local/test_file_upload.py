# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# test/connector/test_file_upload.py

import asyncio
import csv
import io
import os
import tempfile
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumoai_studio.connector.file_upload_connector import FileUploadConnector
from kumoai_studio.connector.utils import (
    CHUNK_SIZE,
    MAX_LOCAL_FILE_SIZE,
    MAX_REMOTE_FILE_SIZE,
    MAX_UPLOAD_PARTS,
    _complete_table_upload,
    _local_upload_part_size,
    _resolve_upload_url,
    _same_origin_upload_auth_headers,
    _sanitize_columns,
    _start_table_upload,
    _upload_single_file,
    multi_put_bounded,
    sanitize_file,
)

# --------------------------------------------------------------------
# Global patch: stub client so tests never require real authentication
# --------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_global_client():
    with patch("kumoai_studio.connector.file_upload_connector.global_state") as gs1, \
            patch("kumoai_studio._require_project"), \
            patch("kumoai_studio.connector.utils.global_state") as gs2, \
            patch("kumoai_studio.connector.utils.multi_put_bounded",
                  new_callable=MagicMock), \
            patch("kumoai_studio.connector.utils.asyncio.run_coroutine_threadsafe") \
            as run_coroutine:
        gs1.client = MagicMock()
        gs1._project_id = 'project-id'
        gs2.client = MagicMock()
        run_coroutine.return_value.result.return_value = []
        yield


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =========================
# Local File Tests
# =========================


@pytest.mark.parametrize(
    ('api_url', 'raw_url', 'expected'),
    [
        ('https://api.example.com', '/v1/upload?token=a%2Fb&part=1',
         'https://api.example.com/v1/upload?token=a%2Fb&part=1'),
        ('https://api.example.com/public_api', '/v1/upload?token=a%2Fb&part=1',
         'https://api.example.com/public_api/v1/upload?token=a%2Fb&part=1'),
        ('https://api.example.com/public_api/',
         '/public_api/v1/upload?token=already-prefixed',
         ('https://api.example.com/public_api/v1/upload?'
          'token=already-prefixed')),
        ('https://api.example.com/api', 'v1/upload?token=plain',
         'https://api.example.com/api/v1/upload?token=plain'),
        ('https://api.example.com/public_api',
         'https://uploads.example.net/part?token=external',
         'https://uploads.example.net/part?token=external'),
    ],
)
def test_resolve_upload_url_preserves_path_query_and_absolute_external(
        api_url, raw_url, expected):
    from kumoai_studio.connector import utils
    utils.global_state.client._url = api_url

    assert _resolve_upload_url(raw_url) == expected


@pytest.mark.parametrize('raw_url', [
    '//evil.example/upload',
    '///evil.example/upload',
    'https:evil.example/upload',
    'ftp://uploads.example.net/part',
])
def test_resolve_upload_url_rejects_malformed_or_non_http_targets(raw_url):
    from kumoai_studio.connector import utils
    utils.global_state.client._url = 'https://api.example.com'

    with pytest.raises(ValueError):
        _resolve_upload_url(raw_url)


def test_upload_auth_headers_are_same_origin_only():
    from kumoai_studio.connector import utils
    utils.global_state.client._url = 'https://api.example.com/base'
    utils.global_state.client._api_key = 'owner-bound-secret'

    relative = _resolve_upload_url('/v1/upload?token=relative')
    same_origin = 'https://api.example.com:443/v1/upload?token=absolute'
    external = 'https://uploads.example.net/part?token=external'

    expected = {'X-API-Key': 'owner-bound-secret'}
    assert _same_origin_upload_auth_headers(relative) == expected
    assert _same_origin_upload_auth_headers(same_origin) == expected
    assert _same_origin_upload_auth_headers(external) == {}


def test_complete_upload_keeps_server_validation_enabled():
    from kumoai_studio.connector import utils

    _complete_table_upload(
        table_name='test_table',
        file_type='parquet',
        upload_path='s3://bucket/temp/test_table.parquet',
        upload_id='upload-id',
        parts_metadata=[],
        project_id='project-id',
    )

    complete_upload = cast(
        MagicMock,
        utils.global_state.client.connector_api.complete_file_upload)
    request = complete_upload.call_args.args[0]
    assert request.validate_data is True


@pytest.mark.parametrize('file_type', ['parquet', 'csv'])
def test_connector_upload_differing_file_types(file_type, temp_dir):
    path = os.path.join(temp_dir, "test." + file_type)
    if file_type == 'parquet':
        pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    else:
        pd.DataFrame({'a': [1, 2, 3]}).to_csv(path, index=False)

    other_type = "csv" if file_type == 'parquet' else "parquet"
    connector = FileUploadConnector(file_type=other_type)
    # message text may vary; just assert type and a key phrase
    with pytest.raises(ValueError) as ei:
        connector.upload(name="test_table", path=path)
    assert f"must end with .{other_type}" in str(ei.value)


def test_connector_upload_small_file(temp_dir):
    # Create a small dummy parquet file
    path = os.path.join(temp_dir, "small.parquet")
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)

    with patch('kumoai_studio.connector.utils._upload_single_file'
               ) as mock_upload:

        # Uploading table via new FileUploadConnector method
        parquet_connector = FileUploadConnector(file_type="parquet")
        parquet_connector.upload("test_table", path)

        # Check that the mock was called exactly once
        mock_upload.assert_called_once()

        # Check the arguments
        args, kwargs = mock_upload.call_args
        assert args == ("test_table", path)
        assert kwargs.get("tqdm_bar_position", 0) == 0


def test_connector_upload_exact_local_limit_as_one_logical_table(temp_dir):
    path = os.path.join(temp_dir, "large.parquet")
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)

    with patch('os.path.getsize', return_value=MAX_LOCAL_FILE_SIZE), \
            patch('kumoai_studio.connector.utils._upload_single_file') as upload:
        FileUploadConnector(file_type="parquet").upload("test_table", path)

    upload.assert_called_once_with("test_table", path, project_id="project-id")


def test_connector_rejects_file_above_local_limit(temp_dir):
    path = os.path.join(temp_dir, "large.parquet")
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)

    with patch('os.path.getsize', return_value=MAX_LOCAL_FILE_SIZE + 1), \
            patch('kumoai_studio.connector.utils._upload_single_file') as upload, \
            pytest.raises(ValueError,
                          match='exceeds the 100.0 GiB upload limit'):
        FileUploadConnector(file_type="parquet").upload("test_table", path)

    upload.assert_not_called()


def test_exact_local_limit_fits_the_multipart_contract():
    from kumoai_studio.connector import utils
    part_size = _local_upload_part_size(MAX_LOCAL_FILE_SIZE)

    assert part_size > CHUNK_SIZE
    _start_table_upload('large', 'parquet', MAX_LOCAL_FILE_SIZE,
                        part_size_bytes=part_size)
    start_upload = cast(
        MagicMock, utils.global_state.client.connector_api.start_file_upload)
    request = start_upload.call_args.args[0]
    assert request.num_parts <= MAX_UPLOAD_PARTS


def test_multi_put_keeps_only_the_concurrency_window_in_memory():
    state = {'generated': 0, 'completed': 0, 'peak': 0}

    def parts():
        for value in range(10):
            state['generated'] += 1
            state['peak'] = max(state['peak'],
                                state['generated'] - state['completed'])
            yield bytes([value])

    async def put(**kwargs):
        await asyncio.sleep(0)
        state['completed'] += 1
        return kwargs['part_no'] + 1, 'etag'

    with patch('kumoai_studio.connector.utils._put_with_retry_streamed', put):
        result = asyncio.run(
            multi_put_bounded(
                urls=[f'https://upload/{i}' for i in range(10)],
                data_iter=parts(),
                concurrency=2,
            ))

    assert len(result) == 10
    assert state['peak'] <= 2


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_sanitize_file_with_messy_columns(extension, temp_dir):
    """Test that file sanitization correctly cleans column names."""
    # Create DataFrame with messy column names
    messy_df = pd.DataFrame({
        'First Name': ['Alice', 'Bob', 'Charlie'],
        'Age(yrs)': [25, 30, 35],
        'Email@Address':
        ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
        'Income$$$': [50000, 60000, 70000]
    })

    original_path = os.path.join(temp_dir, f"messy_columns.{extension}")
    if extension == 'parquet':
        messy_df.to_parquet(original_path, index=False)
    else:
        messy_df.to_csv(original_path, index=False)

    # Test sanitization
    sanitized_path, was_changed = sanitize_file(original_path)

    # Should have been changed
    assert was_changed is True
    assert sanitized_path != original_path

    # Check that sanitized file has clean column names
    expected_columns = ['First_Name', 'Age_yrs', 'Email_Address', 'Income']
    if extension == 'parquet':
        sanitized_table = pq.read_table(sanitized_path)
        assert sanitized_table.column_names == expected_columns
        sanitized_df = sanitized_table.to_pandas()
    else:
        sanitized_df = pd.read_csv(sanitized_path)
        assert sanitized_df.columns.tolist() == expected_columns

    # Verify data integrity
    assert len(sanitized_df) == len(messy_df)
    assert (
        sanitized_df['First_Name'].tolist() == messy_df['First Name'].tolist())
    assert (sanitized_df['Age_yrs'].tolist() == messy_df['Age(yrs)'].tolist())

    # Clean up temp file
    os.unlink(sanitized_path)


def test_sanitize_file_with_duplicate_columns(temp_dir):
    """Test that file sanitization correctly cleans column names."""
    # Column blocks
    columns_block_0 = ['First Name', 'Age(yrs)', 'Email@Address', 'Income$$$']
    columns_block_1 = ['First Name', 'Age(yrs)', 'First Name_1']
    columns_block_2 = ['First Name', 'First Name_2']

    # Data blocks
    rows_block_0 = [['Alice', 25, 'alice@test.com', 50000],
                    ['Bob', 30, 'bob@test.com', 60000],
                    ['Charlie', 35, 'charlie@test.com', 70000]]
    rows_block_1 = [['Eve', 42, 'Ecila'], ['Odenkirk', 24, 'Bob'],
                    ['Sheen', 66, 'EilrahC']]
    rows_block_2 = [['Cooper', 'Repooc'], ['De Niro', 'Orin Ed'],
                    ['Puth', 'Htup']]

    # Combine columns and rows
    combined_columns = columns_block_0 + columns_block_1 + columns_block_2
    combined_rows = [
        r0 + r1 + r2
        for r0, r1, r2 in zip(rows_block_0, rows_block_1, rows_block_2)
    ]
    print(combined_rows)

    # Write CSV manually to preserve duplicate column names exactly
    original_path = os.path.join(temp_dir, "messy_columns.csv")
    with open(original_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(combined_columns)
        writer.writerows(combined_rows)

    # Run sanitization
    sanitized_path, was_changed = sanitize_file(original_path)
    assert was_changed is True

    # Load sanitized CSV
    sanitized_df = pd.read_csv(sanitized_path)
    expected_columns = [
        'First_Name', 'Age_yrs', 'Email_Address', 'Income', 'First_Name_1',
        'Age_yrs__1', 'First_Name_1_1', 'First_Name_2', 'First_Name_2_1'
    ]
    assert sanitized_df.columns.tolist() == expected_columns

    # Clean up temp file
    os.unlink(sanitized_path)


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_sanitize_file_no_changes_needed(extension, temp_dir):
    """Test that file with clean columns returns original path."""
    # Create DataFrame with clean column names
    clean_df = pd.DataFrame({
        'first_name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com']
    })

    original_path = os.path.join(temp_dir, f"clean_columns.{extension}")
    if extension == 'parquet':
        clean_df.to_parquet(original_path, index=False)
    else:
        clean_df.to_csv(original_path, index=False)

    # Test sanitization
    sanitized_path, was_changed = sanitize_file(original_path)

    # Should not have been changed
    assert was_changed is False
    assert sanitized_path == original_path


def test_sanitize_columns_helper():
    """Test the _sanitize_columns helper function directly."""
    # Test messy column names
    messy_columns = [
        'First Name', 'Age(yrs)', 'Email@Address', 'Income$$$', '___leading',
        'trailing___'
    ]
    clean_columns, was_changed = _sanitize_columns(messy_columns)

    # The regex [^0-9A-Za-z]+ replaces non-alphanumeric with _,
    # then strip("_") removes leading/trailing _
    expected_clean = [
        'First_Name', 'Age_yrs', 'Email_Address', 'Income', 'leading',
        'trailing'
    ]
    assert clean_columns == expected_clean
    assert was_changed is True

    # Test already clean columns
    clean_columns_input = ['first_name', 'age', 'email']
    clean_result, was_changed = _sanitize_columns(clean_columns_input)

    assert clean_result == clean_columns_input
    assert was_changed is False


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
@pytest.mark.parametrize('complete_fails', [False, True])
@patch('kumoai_studio.connector.utils._start_table_upload')
@patch('kumoai_studio.connector.utils._complete_table_upload')
@patch('kumoai_studio.connector.utils.multi_put_bounded',
       new_callable=MagicMock)
@patch('kumoai_studio.connector.utils.asyncio.run_coroutine_threadsafe')
@patch('os.unlink')
def test_temp_file_cleanup_with_sanitization(mock_unlink, mock_run_coroutine,
                                             mock_multi_put, mock_complete,
                                             mock_start, complete_fails,
                                             extension, temp_dir):
    """Test that temporary files created during sanitization are properly
    cleaned up.
    """
    # Create DataFrame with messy column names that will require sanitization
    messy_df = pd.DataFrame({
        'First Name': ['Alice', 'Bob', 'Charlie'],
        'Age(yrs)': [25, 30, 35]
    })

    original_path = os.path.join(temp_dir, f"messy_upload.{extension}")
    if extension == 'parquet':
        messy_df.to_parquet(original_path, index=False)
    else:
        messy_df.to_csv(original_path, index=False)

    # Creates a connector with the same type as the file extension
    connector = FileUploadConnector(file_type=extension)

    # Mock the upload components
    mock_response = MagicMock()
    mock_response.presigned_part_urls = {0: "http://fake-url"}
    mock_response.temp_upload_path = "fake-path"
    mock_response.upload_id = "fake-id"
    mock_start.return_value = mock_response
    mock_multi_put.return_value = []
    mock_run_coroutine.return_value.result.return_value = []
    if complete_fails:
        mock_complete.side_effect = RuntimeError("completion failed")

    # Test upload table which should trigger sanitization and cleanup
    if complete_fails:
        with pytest.raises(RuntimeError, match="completion failed"):
            connector.upload("test_table", original_path)
    else:
        connector.upload("test_table", original_path)

    # Verify that os.unlink was called (temp file cleanup)
    # Should be called once for the sanitized temp file
    assert mock_unlink.call_count >= 1

    # Verify upload components were called
    mock_start.assert_called_once()
    mock_multi_put.assert_called_once()
    mock_complete.assert_called_once()


def test_oversized_sanitized_file_is_rejected_and_removed(temp_dir):
    original_path = os.path.join(temp_dir, "original.csv")
    sanitized_path = os.path.join(temp_dir, "sanitized.csv")
    with open(original_path, "w") as f:
        f.write("bad column\nvalue\n")
    with open(sanitized_path, "w") as f:
        f.write("bad_column\nvalue\n")

    with patch('kumoai_studio.connector.utils.sanitize_file',
               return_value=(sanitized_path, True)), \
            patch('os.path.getsize', return_value=MAX_LOCAL_FILE_SIZE + 1), \
            patch('kumoai_studio.connector.utils._start_table_upload') as start, \
            pytest.raises(ValueError, match='exceeds the 100.0 GiB upload limit'):
        _upload_single_file("test_table", original_path)

    start.assert_not_called()
    assert os.path.exists(original_path)
    assert not os.path.exists(sanitized_path)


# =========================
# Remote FS helpers & cases
# =========================


class _FakeFS:
    """Minimal fsspec-like FS for tests."""
    def __init__(self, info_map=None, ls_list=None, open_map=None):
        self._info_map = info_map or {}
        self._ls_list = ls_list
        self._open_map = open_map or {}

    def info(self, path):
        if path in self._info_map:
            return self._info_map[path]
        raise FileNotFoundError(path)

    def ls(self, path, detail=True):
        if self._ls_list is not None:
            return self._ls_list
        return []

    def open(self, path, mode="rb"):
        data = self._open_map.get(path)
        if data is None:
            raise FileNotFoundError(path)
        bio = io.BytesIO(data)

        class Ctx:
            def __enter__(self, *a, **k):  # noqa
                return bio

            def __exit__(self, *a):  # noqa
                bio.close()

        return Ctx()


def _parquet_bytes_with_schema(cols_and_types: list[tuple[str, pa.DataType]],
                               rows=1) -> bytes:
    arrays, names = [], []
    for name, typ in cols_and_types:
        names.append(name)
        if pa.types.is_integer(typ):
            arrays.append(pa.array([1] * rows, type=typ))
        elif pa.types.is_string(typ):
            arrays.append(pa.array(["x"] * rows, type=typ))
        elif pa.types.is_floating(typ):
            arrays.append(pa.array([1.0] * rows, type=typ))
        else:
            arrays.append(pa.array([None] * rows, type=pa.null()))
    tbl = pa.table(arrays, names=names)
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    return buf.getvalue()


# -------------------------
# Remote single-file (CSV)
# -------------------------

CSV_EDGE_CASES = [
    ("utf8_bom_and_crlf_semicolon",
     b"\xef\xbb\xbfFirst Name;Age(yrs)\r\nAlice;25\r\n", True, None),
    ("quoted_commas", b'col1,col2\n"hello, world",42\n', True, None),
    ("quoted_newline", b'c1,c2\n"hello\nworld",1\n', True, None),
    ("only_header_no_rows", b"c1,c2\n", False, r"zero data rows|zero rows"),
    ("empty_file", b"", False, r"empty|auto-detect a delimiter"),
    ("deny_list_delimiter_detected", b"a b c d e\n1 2 3 4 5\n", True, None),
]


@patch('kumoai_studio.connector.utils._make_filesystem')
@patch('kumoai_studio.connector.utils._start_table_upload_with_parts')
@patch('kumoai_studio.connector.utils._complete_table_upload')
@patch('kumoai_studio.connector.utils.multi_put_bounded',
       new_callable=MagicMock)
@patch('kumoai_studio.connector.utils.asyncio.run_coroutine_threadsafe')
@pytest.mark.parametrize('case', CSV_EDGE_CASES,
                         ids=[c[0] for c in CSV_EDGE_CASES])
def test_remote_single_file_csv_edge_cases(mock_run_coroutine, mock_mput,
                                           mock_complete, mock_start,
                                           mock_fs_factory, case):
    name, data, should_pass, fail_re = case
    url = f"s3://bucket/path/{name}.csv"
    fs = _FakeFS(info_map={url: {
        "type": "file",
        "size": len(data)
    }}, open_map={url: data})
    mock_fs_factory.return_value = fs

    mock_resp = MagicMock()
    mock_resp.presigned_part_urls = {"0": "http://u0"}
    mock_resp.temp_upload_path = "tmp"
    mock_resp.upload_id = "id"
    mock_start.return_value = mock_resp
    mock_mput.return_value = []
    mock_run_coroutine.return_value.result.return_value = []

    csv_conn = FileUploadConnector(file_type="csv")

    if should_pass:
        csv_conn.upload("csv_edge_tbl", url)
        mock_start.assert_called_once()
        mock_mput.assert_called_once()
        mock_complete.assert_called_once()
    else:
        with pytest.raises(Exception, match=fail_re):
            csv_conn.upload("csv_edge_tbl", url)


# -------------------------
# Remote single-file (Parquet)
# -------------------------


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_single_file_parquet_zero_rows_raises(mock_fs_factory):
    empty_tbl = pa.table([pa.array([], type=pa.int32())], names=["a"])
    buf = io.BytesIO()
    pq.write_table(empty_tbl, buf)
    b = buf.getvalue()
    url = "s3://bucket/empty.parquet"

    fs = _FakeFS(info_map={url: {
        "type": "file",
        "size": len(b)
    }}, open_map={url: b})
    mock_fs_factory.return_value = fs

    pq_conn = FileUploadConnector(file_type="parquet")

    with pytest.raises(ValueError, match="zero rows"):
        pq_conn.upload("empty_tbl", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_single_file_parquet_invalid_col_names(mock_fs_factory):
    df = pd.DataFrame({"First Name": [1], "Income$$$": [2]})
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    b = buf.getvalue()
    url = "s3://bucket/bad_names.parquet"

    fs = _FakeFS(info_map={url: {
        "type": "file",
        "size": len(b)
    }}, open_map={url: b})
    mock_fs_factory.return_value = fs

    pq_conn = FileUploadConnector(file_type="parquet")

    with pytest.raises(ValueError) as ei:
        pq_conn.upload("bad_names_tbl", url)
    msg = str(ei.value)
    # Be flexible on wording; ensure both bad names are referenced
    assert "First Name" in msg and "Income$$$" in msg


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_single_file_parquet_invalid_dtypes(mock_fs_factory):
    struct_type = pa.struct([("x", pa.int32())])
    tbl = pa.table([pa.array([{"x": 1}], type=struct_type)], names=["s"])
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    b = buf.getvalue()
    url = "s3://bucket/invalid_dtype.parquet"

    fs = _FakeFS(info_map={url: {
        "type": "file",
        "size": len(b)
    }}, open_map={url: b})
    mock_fs_factory.return_value = fs

    pq_conn = FileUploadConnector(file_type="parquet")

    with pytest.raises(ValueError) as ei:
        pq_conn.upload("invalid_dtype_tbl", url)
    msg = str(ei.value)
    assert "Unsupported Parquet Data Types" in msg or ("unsupported"
                                                       in msg.lower())
    assert "s" in msg


@patch('kumoai_studio.connector.utils._make_filesystem')
@pytest.mark.parametrize(
    "desc, arrow_type, builder",
    [
        # struct -> unsupported
        ("struct", pa.struct([("x", pa.int32())]), lambda t: pa.array([{
            "x": 1
        }], type=t)),
        # list<struct> -> unsupported
        ("list_of_struct", pa.list_(pa.struct([("x", pa.int32())
                                               ])), lambda t: pa.array([[{
                                                   "x": 1
                                               }, {
                                                   "x": 2
                                               }]], type=t)),
        # map<string,int> -> unsupported
        ("map_string_int", pa.map_(pa.string(),
                                   pa.int32()), lambda t: pa.array([{
                                       "a": 1,
                                       "b": 2
                                   }], type=t)),
        # list<binary> -> unsupported (list elements must be int/float/string)
        ("list_of_binary", pa.list_(
            pa.binary()), lambda t: pa.array([[b"a", b"b"]], type=t)),
    ],
    ids=lambda c: c if isinstance(c, str) else None,
)
def test_remote_single_file_parquet_tricky_types_rejected(
        mock_fs_factory, desc, arrow_type, builder):
    # Build a single-column table "bad" with the given tricky/unsupported.
    arr = builder(arrow_type)
    tbl = pa.table([arr], names=["bad"])
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    b = buf.getvalue()

    url = f"s3://bucket/{desc}.parquet"
    fs = _FakeFS(
        info_map={url: {
            "type": "file",
            "size": len(b)
        }},
        open_map={url: b},
    )
    mock_fs_factory.return_value = fs

    conn = FileUploadConnector(file_type="parquet")
    with pytest.raises(ValueError) as ei:
        conn.upload(f"{desc}_tbl", url)

    msg = str(ei.value).lower()
    # Should be rejected by validate_parquet_schema, not reach upload
    assert ("unsupported parquet data types" in msg) or ("unsupported" in msg)
    assert "bad" in msg


# -------------------------
# Remote directory: mixed / mismatches / ignores
# -------------------------


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_mixed_types_error(mock_fs_factory):
    url = "s3://bucket/dir/"
    fs = _FakeFS(
        info_map={url: {
            "type": "directory"
        }},
        ls_list=[
            {
                "name": url + "a.parquet",
                "size": 10,
                "type": "file"
            },
            {
                "name": url + "b.csv",
                "size": 10,
                "type": "file"
            },
        ],
    )
    mock_fs_factory.return_value = fs
    pq_conn = FileUploadConnector(file_type="parquet")
    with pytest.raises(
            ValueError,
            match=r"(Mixed CSV and Parquet|also contains CSV files)"):
        pq_conn.upload("mixed", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_parquet_schema_mismatch_error(mock_fs_factory):
    url = "s3://bucket/dirp/"
    p1 = url + "p1.parquet"
    p2 = url + "p2.parquet"

    b1 = _parquet_bytes_with_schema([("a", pa.int32()), ("b", pa.string())],
                                    rows=2)
    b2 = _parquet_bytes_with_schema([("a", pa.int32()), ("c", pa.string())],
                                    rows=3)

    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            p1: {
                "type": "file",
                "size": len(b1)
            },
            p2: {
                "type": "file",
                "size": len(b2)
            }
        },
        ls_list=[{
            "name": p1,
            "size": len(b1),
            "type": "file"
        }, {
            "name": p2,
            "size": len(b2),
            "type": "file"
        }],
        open_map={
            p1: b1,
            p2: b2
        },
    )
    mock_fs_factory.return_value = fs

    pq_conn = FileUploadConnector(file_type="parquet")
    with pytest.raises(ValueError, match="mismatch"):
        pq_conn.upload("parq_dir", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_csv_header_mismatch_error(mock_fs_factory):
    url = "s3://bucket/dcsv/"
    c1 = url + "a.csv"
    c2 = url + "b.csv"
    b1 = b"col1,col2\n1,2\n"
    b2 = b"col1,colX\n3,4\n"

    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            c1: {
                "type": "file",
                "size": len(b1)
            },
            c2: {
                "type": "file",
                "size": len(b2)
            }
        },
        ls_list=[{
            "name": c1,
            "size": len(b1),
            "type": "file"
        }, {
            "name": c2,
            "size": len(b2),
            "type": "file"
        }],
        open_map={
            c1: b1,
            c2: b2
        },
    )
    mock_fs_factory.return_value = fs

    csv_conn = FileUploadConnector(file_type="csv")
    with pytest.raises(ValueError, match="header mismatch|mismatch"):
        csv_conn.upload("csv_dir", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_parquet_dtype_mismatch_same_names(mock_fs_factory):
    # Directory with two parquet shards: same column names, different
    # dtypes for 'val'
    url = "s3://bucket/parq_dtype_mismatch/"
    p1 = url + "a.parquet"
    p2 = url + "b.parquet"

    b1 = _parquet_bytes_with_schema([("id", pa.int32()), ("val", pa.string())])
    b2 = _parquet_bytes_with_schema([("id", pa.int32()), ("val", pa.int32())])

    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            p1: {
                "type": "file",
                "size": len(b1)
            },
            p2: {
                "type": "file",
                "size": len(b2)
            },
        },
        ls_list=[
            {
                "name": p1,
                "size": len(b1),
                "type": "file"
            },
            {
                "name": p2,
                "size": len(b2),
                "type": "file"
            },
        ],
        open_map={
            p1: b1,
            p2: b2
        },
    )
    mock_fs_factory.return_value = fs

    conn = FileUploadConnector(file_type="parquet")
    with pytest.raises(ValueError, match=r"(schema mismatch|mismatch)"):
        conn.upload("parq_dtype_mismatch_tbl", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_oversize_file_error(mock_fs_factory):
    url = "s3://bucket/ovr/"
    c1 = url + "big.csv"
    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            c1: {
                "type": "file",
                "size": MAX_REMOTE_FILE_SIZE + 1
            }
        },
        ls_list=[{
            "name": c1,
            "size": MAX_REMOTE_FILE_SIZE + 1,
            "type": "file"
        }],
        open_map={},
    )
    mock_fs_factory.return_value = fs

    csv_conn = FileUploadConnector(file_type="csv")
    with pytest.raises(ValueError, match="exceed 1GB|1GB"):
        csv_conn.upload("big_dir", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
@patch('kumoai_studio.connector.utils._start_table_upload_with_parts')
@patch('kumoai_studio.connector.utils._complete_table_upload')
@patch('kumoai_studio.connector.utils.multi_put_bounded',
       new_callable=MagicMock)
@patch('kumoai_studio.connector.utils.asyncio.run_coroutine_threadsafe')
def test_remote_directory_ignores_irrelevant_files(mock_run_coroutine,
                                                   mock_mput, mock_complete,
                                                   mock_start,
                                                   mock_fs_factory):
    url = "s3://bucket/mixed_dir/"
    c1 = url + "a.csv"
    junk1 = url + "notes.txt"
    junk2 = url + ".DS_Store"
    subdir = url + "nested/"
    b1 = b"col1,col2\n1,2\n"

    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            c1: {
                "type": "file",
                "size": len(b1)
            },
            junk1: {
                "type": "file",
                "size": 5
            },
            junk2: {
                "type": "file",
                "size": 10
            },
            subdir: {
                "type": "directory"
            }
        },
        ls_list=[{
            "name": c1,
            "size": len(b1),
            "type": "file"
        }, {
            "name": junk1,
            "size": 5,
            "type": "file"
        }, {
            "name": junk2,
            "size": 10,
            "type": "file"
        }, {
            "name": subdir,
            "type": "directory"
        }],
        open_map={c1: b1},
    )
    mock_fs_factory.return_value = fs

    mock_resp = MagicMock()
    mock_resp.presigned_part_urls = {"0": "http://u0"}
    mock_resp.temp_upload_path = "tmp"
    mock_resp.upload_id = "id"
    mock_start.return_value = mock_resp
    mock_mput.return_value = []
    mock_run_coroutine.return_value.result.return_value = []

    csv_conn = FileUploadConnector(file_type="csv")
    csv_conn.upload(
        "mixed_tbl",
        url,
    )

    assert mock_start.call_count == 1
    assert mock_mput.call_count == 1
    assert mock_complete.call_count == 1


@patch('kumoai_studio.connector.utils._make_filesystem')
@pytest.mark.parametrize(
    "connector_type, present_ext, missing_regex",
    [
        ("csv", "parquet",
         r"(no \.csv files|no .*csv .*found|no matching .*csv)"),
        ("parquet", "csv",
         r"(no \.parquet files|no .*parquet .*found|no matching .*parquet)"),
    ],
)
def test_remote_directory_file_type_enforced(mock_fs_factory, connector_type,
                                             present_ext, missing_regex):
    url = "s3://bucket/dir_type_enforced/"

    # Build a directory manifest containing ONLY files of `present_ext`
    if present_ext == "parquet":
        f1 = url + "part-0001.parquet"
        data = io.BytesIO()
        pq.write_table(pa.table({"a": pa.array([1], type=pa.int32())}), data)
        b = data.getvalue()
        fs = _FakeFS(
            info_map={
                url: {
                    "type": "directory"
                },
                f1: {
                    "type": "file",
                    "size": len(b)
                }
            },
            ls_list=[{
                "name": f1,
                "size": len(b),
                "type": "file"
            }],
            open_map={f1: b},
        )
    else:  # present_ext == "csv"
        f1 = url + "part-0001.csv"
        b = b"a,b\n1,2\n"
        fs = _FakeFS(
            info_map={
                url: {
                    "type": "directory"
                },
                f1: {
                    "type": "file",
                    "size": len(b)
                }
            },
            ls_list=[{
                "name": f1,
                "size": len(b),
                "type": "file"
            }],
            open_map={f1: b},
        )

    mock_fs_factory.return_value = fs

    conn = FileUploadConnector(file_type=connector_type)
    with pytest.raises(ValueError, match=missing_regex):
        conn.upload("dir_enforced_tbl", url)


@patch('kumoai_studio.connector.utils._make_filesystem')
def test_remote_directory_parquet_bad_shard_includes_filename(mock_fs_factory):
    url = "s3://bucket/dir_parquet_bad_shard/"
    good = url + "a.parquet"
    bad = url + "b.parquet"

    # helper to build quick parquet bytes
    def _pbytes(cols):
        tbl = pa.table(cols)
        buf = io.BytesIO()
        pq.write_table(tbl, buf)
        return buf.getvalue()

    # Good shard: simple, supported schema
    good_bytes = _pbytes({
        "id": pa.array([1, 2], type=pa.int32()),
        "val": pa.array([10, 20], type=pa.int32())
    })

    # Bad shard: introduce unsupported type (e.g., struct)
    bad_arr = pa.array([{"x": 1}], type=pa.struct([("x", pa.int32())]))
    bad_bytes = _pbytes({"id": pa.array([1], type=pa.int32()), "val": bad_arr})

    fs = _FakeFS(
        info_map={
            url: {
                "type": "directory"
            },
            good: {
                "type": "file",
                "size": len(good_bytes)
            },
            bad: {
                "type": "file",
                "size": len(bad_bytes)
            },
        },
        ls_list=[
            {
                "name": good,
                "size": len(good_bytes),
                "type": "file"
            },
            {
                "name": bad,
                "size": len(bad_bytes),
                "type": "file"
            },
        ],
        open_map={
            good: good_bytes,
            bad: bad_bytes
        },
    )
    mock_fs_factory.return_value = fs

    conn = FileUploadConnector(file_type="parquet")

    # Expect validation to fail before upload, and include the bad
    # filename in the message
    with pytest.raises(ValueError) as ei:
        conn.upload("tbl_bad_shard", url)

    msg = str(ei.value)
    # Must mention the offending shard's filename
    assert "b.parquet" in msg or bad in msg
    # And ideally mention unsupported types (keep regex flexible
    # across wording)
    assert ("Unsupported Parquet Data Types" in msg) or ("unsupported"
                                                         in msg.lower())


# -------------------------
# Optional helper present?
# -------------------------


def test_validate_url_ext_helper():
    try:
        from kumoai_studio.connector.utils import _validate_url_ext
    except Exception:
        pytest.skip("validate_url_ext not exported in this build")
    assert _validate_url_ext("s3://b/x.csv", None) == "csv"
    assert _validate_url_ext("gs://b/y.parquet", "parquet") == "parquet"
    with pytest.raises(ValueError, match="must end with .parquet"):
        _validate_url_ext("abfs://x/file.csv", "parquet")
    with pytest.raises(ValueError, match="must end with .csv or .parquet"):
        _validate_url_ext("s3://b/noext", None)
