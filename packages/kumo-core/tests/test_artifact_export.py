# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The export rules both backends obey."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from kumo_core.artifact_export import (
    EXPORT_MARKER_NAME,
    MULTI_DIMENSIONAL_CSV,
    MultiDimensionalCsvError,
    export_marker,
    flatten_arrays_for_csv,
)


def test_the_marker_is_named_what_both_backends_look_for() -> None:
    assert EXPORT_MARKER_NAME == "_KUMO_EXPORT"


def test_the_marker_digests_the_identity_it_is_given() -> None:
    """A cloud-written directory and a local retry recognise each other only
    while both digest the identity the same way."""
    assert export_marker("job-1") == hashlib.sha256(b"job-1").hexdigest()


def test_two_identities_are_told_apart() -> None:
    assert export_marker("job-1") != export_marker("job-2")


def test_an_embedding_column_becomes_a_space_separated_string() -> None:
    frame = pd.DataFrame({
        "ENTITY": [1, 2],
        "EMBEDDING": [np.array([0.1, 0.2]),
                      np.array([0.3, 0.4])],
    })

    assert flatten_arrays_for_csv(frame)["EMBEDDING"].tolist() == [
        "0.1 0.2", "0.3 0.4"
    ]


def test_a_list_column_is_rendered_like_an_array_one() -> None:
    """A frame handed straight to an export carries lists where one read back
    from Parquet carries arrays; both land the same way."""
    frame = pd.DataFrame({"E": [1], "M": [[1, 2, 3]]})

    assert flatten_arrays_for_csv(frame)["M"].tolist() == ["1 2 3"]


def test_an_arrow_backed_list_column_is_rendered_too() -> None:
    """A list column is not object dtype, and would otherwise walk past this
    into PyArrow's ``Unsupported Type: list<item: double>``."""
    frame = pa.table({
        "E": pa.array([1]),
        "M": pa.array([[0.1, 0.2]]),
    }).to_pandas(types_mapper=pd.ArrowDtype)

    assert flatten_arrays_for_csv(frame)["M"].tolist() == ["0.1 0.2"]


@pytest.mark.parametrize(
    "value",
    [
        np.array([[1, 2], [3, 4]]),
        [[1, 2], [3]],
        [np.array([1]), np.array([2])],
    ],
)
def test_a_nested_array_is_refused_rather_than_flattened(value: object) -> None:
    frame = pd.DataFrame({"E": [1], "M": [value]})

    with pytest.raises(MultiDimensionalCsvError, match="multi-dimensional"):
        flatten_arrays_for_csv(frame)


def test_the_refusal_is_still_catchable_as_a_value_error() -> None:
    """Giving the refusal a type of its own is for callers that translate it;
    a caller that only wants "the frame was rejected" keeps working with no
    import, which is how the local sink and its suite read it."""
    assert issubclass(MultiDimensionalCsvError, ValueError)


def test_the_refusal_is_worded_once() -> None:
    assert "multi-dimensional array" in MULTI_DIMENSIONAL_CSV
    assert "Parquet" in MULTI_DIMENSIONAL_CSV


def test_the_callers_frame_is_left_alone() -> None:
    """The frame belongs to the caller: an export must not rewrite the job's
    own rows on its way out."""
    frame = pd.DataFrame({"M": [np.array([1, 2])]})

    flatten_arrays_for_csv(frame)

    assert isinstance(frame["M"][0], np.ndarray)


def test_a_frame_with_nothing_to_render_is_returned_as_it_stands() -> None:
    """No copy at all where no column needs one."""
    frame = pd.DataFrame({"A": [1, 2], "B": [1.5, 2.5]})

    assert flatten_arrays_for_csv(frame) is frame


def test_file_metadata_uses_hosted_batch_bound_and_preserves_source(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.artifact_export import (
        artifact_rows_per_batch,
        local_parquet_batches,
        prepared_artifact_file,
    )
    source = tmp_path / 'scores.parquet'
    table = pa.table({'ENTITY': np.arange(400000, dtype=np.int64)})
    pq.write_table(table, source)
    before = source.read_bytes()
    batches = list(local_parquet_batches(str(source)))
    assert len(batches) > 1
    assert max(batch.num_rows
               for batch in batches) <= artifact_rows_per_batch(8)
    scratch_root = tmp_path / 'private-export'
    scratch_root.mkdir()
    with prepared_artifact_file(str(source), {'JOB_TIMESTAMP': 'fixed'},
                                scratch_root=str(scratch_root)) as staged:
        assert Path(staged).resolve().is_relative_to(scratch_root.resolve())
        assert pq.ParquetFile(staged).metadata.num_rows == 400000
        assert pq.ParquetFile(staged).schema_arrow.names == [
            'ENTITY', 'JOB_TIMESTAMP'
        ]
    assert source.read_bytes() == before


@pytest.mark.parametrize('existing_empty', [False, True])
def test_multipart_failure_and_failed_recopy_remain_retryable(
        tmp_path, monkeypatch, existing_empty):
    import pyarrow.parquet as pq
    from kumo_core.artifact_export import export_selected_files
    from kumo_core.object_store_sink import _ArrowExportFiles
    from pyarrow.fs import LocalFileSystem

    sources = []
    for index in range(2):
        path = tmp_path / f'part-{index}.parquet'
        pq.write_table(pa.table({'id': [index]}), path)
        sources.append(str(path))
    target = tmp_path / 'output'
    if existing_empty:
        target.mkdir()
    fs = _ArrowExportFiles(LocalFileSystem(), None)
    success = target / '_SUCCESS'

    def interrupted_copy(pairs, **kwargs):
        assert not success.exists()
        fs.copy_file(*pairs[0])
        raise RuntimeError('interrupted transfer')

    def export():
        return export_selected_files(sources,
                                     str(target) + '/', fs=fs, identity='job-1')

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr(fs, 'copy_files_threaded', interrupted_copy)
            with pytest.raises(RuntimeError, match='interrupted transfer'):
                export()
        assert not success.exists()
        assert (target /
                EXPORT_MARKER_NAME).read_text() == export_marker('job-1')
        assert pq.read_table(target / 'part-0.parquet')['id'].to_pylist() == [0]
        export()
        assert success.exists()
        assert pq.read_table(target / 'part-1.parquet')['id'].to_pylist() == [1]

    def denied(path):
        raise PermissionError('delete denied')

    with monkeypatch.context() as patch:
        patch.setattr(fs, 'rm', denied)
        patch.setattr(
            fs, 'copy_files_threaded',
            lambda *a, **kw: pytest.fail('copy before marker removal'))
        with pytest.raises(PermissionError, match='delete denied'):
            export()
    assert success.exists()


def test_csv_retry_replaces_partially_uploaded_part(tmp_path, monkeypatch):
    from kumo_core.artifact_export import export_selected_files
    from kumo_core.object_store_sink import _ArrowExportFiles
    from pyarrow.fs import LocalFileSystem

    source = tmp_path / 'source.parquet'
    expected = pd.DataFrame({'id': [10, 20]})
    expected.to_parquet(source)
    target = tmp_path / 'output'
    fs = _ArrowExportFiles(LocalFileSystem(), None)

    def export():
        export_selected_files([str(source)],
                              str(target) + '/', fs=fs, identity='job-1',
                              file_type='csv')

    def interrupted(source, destination, **kwargs):
        Path(destination).write_bytes(Path(source).read_bytes()[:6])
        raise RuntimeError('partial upload')

    with monkeypatch.context() as patch:
        patch.setattr(fs, 'copy_file', interrupted)
        with pytest.raises(RuntimeError, match='partial upload'):
            export()
    assert (target / 'part-00000.csv').stat().st_size == 6
    assert not (target / '_SUCCESS').exists()
    export()
    pd.testing.assert_frame_equal(pd.read_csv(target / 'part-00000.csv'),
                                  expected)
    assert (target / '_SUCCESS').exists()


@pytest.mark.parametrize('directory', [True, False])
def test_sidecars_are_written_before_success(tmp_path, monkeypatch,
                                             directory):
    import pyarrow.parquet as pq
    from kumo_core.artifact_export import export_selected_files
    from kumo_core.object_store_sink import _ArrowExportFiles
    from pyarrow.fs import LocalFileSystem

    source = tmp_path / 'part-0.parquet'
    pq.write_table(pa.table({'id': [1]}), source)
    target = tmp_path / 'output'
    fs = _ArrowExportFiles(LocalFileSystem(), None)
    touched = fs.touch

    def touch(path):
        assert (target / 'meta.json').read_bytes() == b'{}'
        touched(path)

    monkeypatch.setattr(fs, 'touch', touch)
    export_selected_files([str(source)],
                          str(target) + ('/' if directory else ''), fs=fs,
                          identity='job-1', sidecars={'meta.json': b'{}'})
    if directory:
        assert (target / '_SUCCESS').exists()
    else:
        assert not (target / 'meta.json').exists()
        assert pq.read_table(tmp_path / 'output.parquet').num_rows == 1
