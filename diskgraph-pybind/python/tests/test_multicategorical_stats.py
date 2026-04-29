"""
Pybind-layer smoke test for multicategorical column stats.

This loads the diskgraph wheel runtime and drives the pybind `ingest` API,
then reads the emitted
`kumo_col_stats/col_stats_full/<table>.json`, and asserts the three
multicategorical keys tt_executor also emits are present with the expected
shape for all four multicat column shapes (string + delimiter, list<utf8>,
list<int64>, list<float64>).

Runs as part of the normal `pytest diskgraph-pybind/python/tests` suite; the
cost is one pybind ingest run per test session. The test is effectively a
contract check on the key names and Python boundary — the detailed runtime
behavior is exercised by the Rust integration tests in
`diskgraph-ingest/tests/ingest_multicategorical_e2e.rs`.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import diskgraph


def _build_multicat_fixture(tmp_path):
    """Write a parquet with one row per input kind and a matching config."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    index_dir = tmp_path / "index"

    item_ids = pa.array([0, 1, 2, 3], type=pa.int64())
    # String column with comma-delimited tokens.
    str_tags = pa.array(
        ["kids,outdoor,sale", "audio,pro", "kids", ""],
        type=pa.utf8(),
    )
    # list<utf8>
    tag_list = pa.array(
        [["sports", None, "sale"], ["kids"], ["pro", "audio"], ["kids", ""]],
        type=pa.list_(pa.utf8()),
    )
    # list<int64>
    id_list = pa.array(
        [[10, None, 20, 30], [40], [10], [1, 2]],
        type=pa.list_(pa.int64()),
    )
    # list<float64>
    val_list = pa.array(
        [[1.0, None, 2.5], [1.0], [3.5], []],
        type=pa.list_(pa.float64()),
    )
    table = pa.table({
        "item_id": item_ids,
        "str_tags": str_tags,
        "tag_list": tag_list,
        "id_list": id_list,
        "val_list": val_list,
    })
    parquet_path = data_dir / "items.parquet"
    pq.write_table(table, str(parquet_path))

    config = {
        "dataset_name": "mc_stats_test",
        "num_shards": 1,
        "tables": [{
            "name": "items",
            "columns": [
                {"name": "item_id", "ingested_dtype": "int64"},
                {
                    "name": "str_tags",
                    "ingested_dtype": "string",
                    "semantic_type": "multicategorical",
                    "multicategorical_delimiter": ","
                },
                {
                    "name": "tag_list",
                    "ingested_dtype": "string_array",
                    "semantic_type": "multicategorical"
                },
                {
                    "name": "id_list",
                    "ingested_dtype": "int64_array",
                    "semantic_type": "multicategorical"
                },
                {
                    "name": "val_list",
                    "ingested_dtype": "float64_array",
                    "semantic_type": "multicategorical"
                },
            ],
            "primary_key": "item_id",
            "data_version": "v1",
            "source": {"type": "parquet", "paths": [str(parquet_path)]}
        }]
    }
    diskgraph.ingest(
        json.dumps(config),
        str(index_dir),
        col_stats=True,
        cat_count_strategy="hybrid",
    )

    stats_path = index_dir / "kumo_col_stats" / "col_stats_full" / "items.json"
    with open(stats_path) as f:
        stats = json.load(f)
    return stats


@pytest.fixture(scope="module")
def multicat_stats(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mc_stats")
    return _build_multicat_fixture(tmp_path)


def test_multicat_string_scalar_keys_present(multicat_stats):
    # MULTI_CATEGORIES_SEPARATOR = "," for the string + delim column.
    sep = multicat_stats["str_tags__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False"]
    assert sep["value"] == ","
    # MULTI_CATEGORY_COUNTS reflects comma-splitting.
    counts = multicat_stats["str_tags__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"]
    arr = counts["value"]
    m = {k: v for k, v in arr}
    assert m.get("kids") == 2
    assert m.get("outdoor") == 1
    assert m.get("sale") == 1
    assert m.get("audio") == 1
    assert m.get("pro") == 1
    # Empty-string row contributes one `""` token (non-null-row semantics).
    assert m.get("") == 1
    # Approx num-unique key is present.
    assert "str_tags__kumo.multicategorical__NUM_UNIQUE_MULTI__True" in multicat_stats


def test_multicat_stringarray_null_separator(multicat_stats):
    # Native list → separator is null, not a string.
    sep = multicat_stats["tag_list__kumo.multicategorical__MULTI_CATEGORIES_SEPARATOR__False"]
    assert sep["value"] is None
    counts = multicat_stats["tag_list__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"]
    m = {k: v for k, v in counts["value"]}
    # sports×1, sale×1, kids×2, pro×1, audio×1, ""×1.
    # Native null list elements are dropped.
    assert m.get("kids") == 2
    assert m.get("sports") == 1
    assert m.get("sale") == 1
    assert m.get("pro") == 1
    assert m.get("audio") == 1
    assert m.get("") == 1


def test_multicat_intlist_numeric_keys(multicat_stats):
    # list<int64>: counts keys emitted as JSON numbers for tt_executor parity.
    counts = multicat_stats["id_list__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"]
    m = {k: v for k, v in counts["value"]}
    # int keys — tolerate either int or str in case downstream path changes.
    def get(key):
        return m.get(key) or m.get(str(key))
    assert get(10) == 2
    assert get(20) == 1
    assert get(30) == 1
    assert get(40) == 1
    assert get(1) == 1
    assert get(2) == 1
    assert get(0) is None


def test_multicat_floatlist_numeric_keys(multicat_stats):
    counts = multicat_stats["val_list__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False"]
    m = {k: v for k, v in counts["value"]}
    def get(key):
        # DG stringifies floats with Display, so 1.0 -> "1" (matches Spark/GE
        # behavior of dropping the trailing zero on whole-number floats); the
        # native float key and `str(float)` ("1.0") forms are also accepted in
        # case a future backend serializes them differently.
        if (
            isinstance(key, float)
            and key.is_integer()
            and m.get(str(int(key))) is not None
        ):
            return m[str(int(key))]
        return m.get(key) or m.get(str(key))
    assert get(1.0) == 2
    assert get(2.5) == 1
    assert get(3.5) == 1
    assert get(0.0) is None


def test_multicat_universal_stats_present(multicat_stats):
    # Every multicat column should also get COUNT / NUM_NA under the
    # kumo.multicategorical stype key.
    for col in ("str_tags", "tag_list", "id_list", "val_list"):
        for stat in ("COUNT", "NUM_NA", "NA_FRACTION", "INVALID_FRACTION"):
            key = f"{col}__kumo.multicategorical__{stat}__False"
            assert key in multicat_stats, f"missing {key}"
