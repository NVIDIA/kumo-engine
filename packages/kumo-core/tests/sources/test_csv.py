# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading CSV the way the hosted connector reads it.

Every expectation here was measured against
``kumo.connector.csv_connector.CSVConnector`` on the same bytes, so a
divergence is a real one rather than a guess about pandas.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.object_store import ObjectStoreSourceReader
from kumo_core.sources.types import ReadMode
from kumoapi.typing import Dtype
from pyarrow.fs import LocalFileSystem


@pytest.mark.parametrize('compressed', [False, True])
def test_local_csv_stream_decodes_compressed_parts(tmp_path, compressed):
    from kumo_core.sources.csv_format import open_reader, parse_options

    path = tmp_path / ('part.csv.gz' if compressed else 'part.csv')
    contents = b'id;value\n1;hello\n2;world\n'
    path.write_bytes(gzip.compress(contents) if compressed else contents)
    with open_reader(LocalFileSystem(), str(path),
                     parse=parse_options(';')) as reader:
        assert reader.read_all().to_pydict() == {
            'id': [1, 2],
            'value': ['hello', 'world']
        }


def write(root: Path, name: str, **files: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    for stem, body in files.items():
        (directory / f"{stem}.csv").write_text(body, encoding="utf-8")
    return directory


def reader_over(root: Path) -> ObjectStoreSourceReader:
    return ObjectStoreSourceReader(LocalFileSystem(), root=str(root))


def staged(root: Path, name: str, tmp_path: Path, columns=None):
    reader = reader_over(root)
    return reader.prepare(name, str(tmp_path / "out"), columns=columns
                          or reader.dtypes(name), mode=ReadMode.STREAMED)


def test_a_semicolon_file_keeps_its_columns_apart(tmp_path):
    """The delimiter decides the column *names*, so getting it wrong does not
    merely mistype the graph -- it makes the cloud's graph unbuildable."""
    write(tmp_path, "t", t="a;b\n1;2\n3;4\n")
    assert reader_over(tmp_path).dtypes("t") == {
        "a": Dtype.int64,
        "b": Dtype.int64,
    }


def test_a_column_of_whole_floats_is_an_integer_column(tmp_path):
    """Every warehouse NUMBER/DOUBLE export lands here, and an FK written as
    ``1.0`` would otherwise become a float key."""
    write(tmp_path, "t", t="a\n1.0\n2.0\n3.0\n")
    assert reader_over(tmp_path).dtypes("t") == {"a": Dtype.int64}
    table = staged(tmp_path, "t", tmp_path)
    assert pq.read_table(sorted(
        table.paths)).column("a").to_pylist() == [1, 2, 3]


@pytest.mark.parametrize("body", [
    "a,b\n1,\n2,\n",
    "a,b\n1,NA\n2,NA\n",
    "a,b\n1,NULL\n2,NULL\n",
])
def test_a_column_that_is_entirely_missing_is_text(body, tmp_path):
    """``unsupported`` maps to no Arrow type at all, so such a column fails
    downstream rather than merely disagreeing with the cloud."""
    write(tmp_path, "t", t=body)
    assert reader_over(tmp_path).dtypes("t")["b"] is Dtype.string


def test_one_na_token_does_not_defeat_the_leading_zero_guard(tmp_path):
    """The hosted sample is read through pandas, which has already turned
    ``N/A`` into a missing value before the zip code is looked at."""
    write(tmp_path, "t", t="z\n01234\nN/A\n05678\n")
    assert reader_over(tmp_path).dtypes("t") == {"z": Dtype.string}


@pytest.mark.parametrize("body,expected", [
    ("a\n0x10\n0x20\n", Dtype.string),
    ("a\n2026-01-01\n2026-01-02\n", Dtype.string),
    ("a\n007\n008\n", Dtype.string),
])
def test_arrow_is_never_left_to_infer_a_csv_type(body, expected, tmp_path):
    """Left to itself Arrow reads these as 16, a date and 7; the hosted
    connector infers through pandas over text and keeps all of them."""
    write(tmp_path, "t", t=body)
    assert reader_over(tmp_path).dtypes("t") == {"a": expected}


@pytest.mark.parametrize("body,values", [
    ("a\n+5\n6\n", [5, 6]),
    ("a\n1_000\n2000\n", [1000, 2000]),
])
def test_an_integer_arrow_cannot_parse_is_still_read(body, values, tmp_path):
    """Both spellings are integers to the platform and unreadable to Arrow's
    CSV parser. The text is rewritten in the scan so Arrow reaches the same
    value, rather than the column being called something else here."""
    write(tmp_path, "t", t=body)
    assert reader_over(tmp_path).dtypes("t") == {"a": Dtype.int64}
    table = staged(tmp_path, "t", tmp_path)
    assert pq.read_table(sorted(table.paths)).column("a").to_pylist() == values


def test_an_integer_wider_than_a_double_is_not_rounded(tmp_path):
    """Only a float-demoted column goes through a double to become an
    integer. Sending every integer that way would round this one."""
    write(tmp_path, "t", t="a\n9007199254740993\n1\n")
    table = staged(tmp_path, "t", tmp_path)
    assert pq.read_table(sorted(
        table.paths)).column("a").to_pylist() == [9007199254740993, 1]


def test_a_number_too_wide_for_an_integer_column_is_refused(tmp_path):
    """The platform raises ``OverflowError`` out of an ``except ValueError``
    here, so the table is unreadable there too. Refusing keeps the two in
    step; reading it as a double would build a different graph in silence."""
    write(tmp_path, "t", t="a\n99999999999999999999\n1\n")
    with pytest.raises(PermanentSourceError, match="too wide"):
        reader_over(tmp_path).dtypes("t")


def test_a_value_below_the_sample_is_typed_the_way_the_platform_types_it(
        tmp_path):
    """Exercised through ``dtypes`` rather than a hand-passed type, or the
    sample size this pins would never be read."""
    write(tmp_path, "t", t="a\n" + "1\n" * 1000 + "ABC\n2\n")
    assert reader_over(tmp_path).dtypes("t") == {"a": Dtype.string}


def test_a_value_below_the_sample_refuses_the_read_rather_than_widening(
        tmp_path):
    """Past the sample nothing can see the conflict, so both sides type the
    column an integer and then meet the value at read time. The platform's
    export refuses it -- measured, in
    ``test/connector/test_csv_hosted_core_parity.py`` -- and nulling it here
    would stage a graph over data the cloud will not ingest, with the key on
    that row silently gone.
    """
    write(tmp_path, "t", t="a\n" + "1\n" * 100100 + "ABC\n2\n")
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"a": Dtype.int64}
    with pytest.raises(PermanentSourceError, match="ABC"):
        staged(tmp_path, "t", tmp_path)


def test_a_whole_float_below_the_sample_is_still_read_as_its_integer(tmp_path):
    """The spelling every warehouse export writes, and the one the row above
    must not be confused with: the platform pins an integer column to a double
    so that ``10.0`` survives, and a reader that refused it -- or nulled it --
    would drop a foreign key the cloud keeps.
    """
    write(tmp_path, "t", t="a\n" + "1\n" * 100100 + "10.0\n2\n")
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"a": Dtype.int64}
    written = pq.read_table(sorted(staged(tmp_path, "t", tmp_path).paths))
    assert written.schema.field("a").type == pa.int64()
    assert written.column("a").to_pylist()[-3:] == [1, 10, 2]


def test_an_infinity_below_the_sample_survives_a_float_column(tmp_path):
    """Arrow's own float parser reads it, so the platform stages it; a
    stricter grammar here would turn it into a gap."""
    write(tmp_path, "t", t="a\n" + "1.5\n" * 100100 + "inf\n2.5\n")
    written = pq.read_table(sorted(staged(tmp_path, "t", tmp_path).paths))
    assert written.column("a").to_pylist()[-2:] == [float("inf"), 2.5]


@pytest.mark.parametrize("first,second", [
    ("c\n100\n200\n", "c\nABC\nDEF\n"),
    ("c\nABC\nDEF\n", "c\n100\n200\n"),
])
def test_the_type_does_not_depend_on_which_part_sorts_first(
        first, second, tmp_path):
    """Sampling only ``paths[0]`` made a numeric part beside a textual one read
    as an integer and fail at ingest, or as text, purely by file name."""
    write(tmp_path, "t", a=first, b=second)
    assert reader_over(tmp_path).dtypes("t") == {"c": Dtype.string}


def test_a_ragged_row_in_the_sniffed_window_is_refused(tmp_path):
    """The platform parses the window it sniffed the separator from and
    refuses the file when that fails, before any bad-line handling applies."""
    write(tmp_path, "t", t="a,b\n1,2\n3,4,5\n6,7\n")
    with pytest.raises(PermanentSourceError, match="delimiter"):
        reader_over(tmp_path).dtypes("t")


def test_a_ragged_row_past_the_sniffed_window_is_still_refused(tmp_path):
    """Beyond the window the separator was checked in, the row is met by the
    reader instead.

    The platform's two halves answer this differently and only one of them
    stages anything: its metadata reader is pandas with ``on_bad_lines='skip'``
    and types the table, while its export is Arrow and refuses it. Refusing at
    the catalog is therefore the same outcome one step earlier -- a table
    neither backend can ingest -- and staging every other row would instead
    hand ingest a table shorter than its source with nothing said."""
    rows = ["a,b"] + [f"{index},{index}" for index in range(20_000)]
    rows.insert(15_000, "1,2,3")
    write(tmp_path, "t", t="\n".join(rows) + "\n")
    with pytest.raises(PermanentSourceError):
        staged(tmp_path, "t", tmp_path)


def test_an_empty_part_is_skipped_rather_than_failing_the_table(tmp_path):
    """A zero-byte object is what an interrupted export leaves behind."""
    directory = write(tmp_path, "t", a="c\n1\n2\n")
    (directory / "b.csv").write_text("", encoding="utf-8")
    assert staged(tmp_path, "t", tmp_path).num_rows == 2


def test_a_csv_column_named_like_a_pandas_index_is_kept(tmp_path):
    """The reserved name is a Parquet artefact; in a CSV it is just a column,
    and dropping it loses data the hosted connector reports."""
    write(tmp_path, "t", t="__index_level_0__,a\n1,2\n3,4\n")
    assert reader_over(tmp_path).dtypes("t") == {
        "__index_level_0__": Dtype.int64,
        "a": Dtype.int64,
    }


def test_a_parquet_column_named_like_a_pandas_index_is_still_dropped(tmp_path):
    directory = tmp_path / "p"
    directory.mkdir()
    pq.write_table(
        pa.table({
            "__index_level_0__": pa.array([1], pa.int64()),
            "a": pa.array([2], pa.int64()),
        }), directory / "part-0.parquet")
    assert reader_over(tmp_path).dtypes("p") == {"a": Dtype.int64}


def test_a_missing_value_does_not_widen_an_integer_column(tmp_path):
    """Measured against the hosted connector, which keeps this ``int64``."""
    write(tmp_path, "t", t="a,b\n1,x\n2,y\n,z\n")
    assert reader_over(tmp_path).dtypes("t")["a"] is Dtype.int64


@pytest.mark.parametrize("token", ["NA", "null", ""])
def test_a_missing_token_is_staged_as_text_in_a_text_column(token, tmp_path):
    """Two null sets are in play and they are not the same list. Pandas'
    decides the *type*, so a column carrying ``NA`` is still an integer; Arrow's
    decides the *value*, and the hosted export leaves it at Arrow's default,
    where a string column is never nulled at all. Nulling one here would stage
    a gap where the cloud stages the two letters.
    """
    write(tmp_path, "t", t=f"a,b\n1,x\n{token},{token}\n2,z\n")
    reader = reader_over(tmp_path)
    assert reader.dtypes("t") == {"a": Dtype.int64, "b": Dtype.string}
    table = staged(tmp_path, "t", tmp_path)
    written = pq.read_table(sorted(table.paths))
    assert written.column("a").to_pylist() == [1, None, 2]
    assert written.column("b").to_pylist() == ["x", token, "z"]


@pytest.mark.parametrize("token", ["None", "<NA>"])
def test_the_two_na_tokens_arrow_does_not_know_refuse_a_number(token, tmp_path):
    """Pandas reads these as missing and Arrow does not, so the column is an
    integer on both sides and neither reader's parser will take the value. The
    platform's export raises; so does this one, rather than inventing a gap the
    cloud would never have."""
    write(tmp_path, "t", t=f"a\n1\n{token}\n2\n")
    assert reader_over(tmp_path).dtypes("t") == {"a": Dtype.int64}
    with pytest.raises(PermanentSourceError, match=token):
        staged(tmp_path, "t", tmp_path)


def test_a_boolean_column_with_a_gap_is_text_the_way_pandas_reads_it(tmp_path):
    """Pandas can only give a column the ``bool`` dtype when every value is
    one; a single missing value leaves it an object column, which is a string
    to Kumo. Stripping the gap before asking would call it a boolean here and
    disagree with the cloud on a dtype that reaches ``graph_id``.
    """
    write(tmp_path, "t", t="a\ntrue\nfalse\nNA\n")
    assert reader_over(tmp_path).dtypes("t") == {"a": Dtype.string}
    written = pq.read_table(sorted(staged(tmp_path, "t", tmp_path).paths))
    assert written.column("a").to_pylist() == ["true", "false", "NA"]


@pytest.mark.parametrize("declared", [Dtype.date, Dtype.time])
def test_a_declared_temporal_column_is_not_staged_as_text(declared, tmp_path):
    """A graph may declare a type CSV inference never produces. The hosted
    export has no Arrow spelling for the temporal family either, and leaves
    such a column to Arrow rather than to text -- so pinning it to a string
    here would stage a string where the cloud stages a timestamp.
    """
    write(tmp_path, "t", t="a\n2026-01-01 10:00:00\n2026-01-02 11:00:00\n")
    table = staged(tmp_path, "t", tmp_path, columns={"a": declared})
    written = pq.read_table(sorted(table.paths))
    assert pa.types.is_temporal(written.schema.field("a").type)
    assert str(written.column("a")[0]) == "2026-01-01 10:00:00"


def test_the_two_spellings_of_normalization_agree(tmp_path):
    """The rewrite exists twice, per value and as a scan expression, and a
    column's type is decided by one while its data is produced by the other.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as pds
    from kumo_core.csv_rules import normalize_numeric
    from kumo_core.sources import csv_format

    values = ["1_000", "+5", "7", "+1_2", "-3", "1_0.5", "+0.25"]
    table = pa.table({"a": pa.array(values, pa.string())})
    scanned = pds.dataset(table).scanner(columns={
        "a": csv_format.numeric(pc.field("a"))
    }).to_table()
    assert scanned.column("a").to_pylist() == [
        normalize_numeric(value) for value in values
    ]


def test_a_multi_part_csv_is_streamed_a_part_at_a_time(tmp_path):
    """One shard per part, which is what shows the read went through the
    shared streaming path rather than materialising the table first."""
    write(tmp_path, "t", a="c\n1\n2\n", b="c\n3\n4\n")
    table = staged(tmp_path, "t", tmp_path)
    assert table.num_rows == 4
    assert len(table.paths) > 1


def test_a_delimiter_the_platform_denies_is_not_used(tmp_path):
    """The sniffer answers a space-separated file with a space, which the
    platform refuses to accept as a delimiter. Taking it would split this file
    into two columns the cloud does not have.
    """
    write(tmp_path, "t", t="a b\n1 2\n3 4\n")
    assert list(reader_over(tmp_path).dtypes("t")) == ["a b"]
