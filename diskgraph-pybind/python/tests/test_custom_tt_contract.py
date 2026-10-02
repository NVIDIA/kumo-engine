"""The custom-TT contract, as a Python consumer decodes it.

The Rust tests prove each token corresponds to a reader arm. They cannot prove
the other half: that the token names a predicate PyArrow actually has, and that
the predicate means what the arm means. Every consumer of this contract turns a
token into `pa.types.is_<token>`, so that step is asserted here.
"""

from __future__ import annotations

import diskgraph
import pyarrow as pa
import pytest

CONTRACT = diskgraph.CUSTOM_TT_CONTRACT

TYPE_KEYS = [
    "entity_types",
    "scalar_types",
    "list_target_containers",
    "list_target_item_types",
    "timestamp_types",
    "weight_types",
    "weight_list_containers",
    "weight_list_item_types",
]

NAME_KEYS = [
    "entity_column_prefix",
    "target_column_prefix",
    "timestamp_column",
    "weight_output_column",
]


def _tokens() -> set[str]:
    return {token for key in TYPE_KEYS for token in CONTRACT[key]}


def test_the_contract_carries_every_key_a_consumer_reads():
    assert set(CONTRACT) == set(TYPE_KEYS) | set(NAME_KEYS)


@pytest.mark.parametrize("key", NAME_KEYS)
def test_each_name_is_a_non_empty_string(key):
    assert isinstance(CONTRACT[key], str) and CONTRACT[key]


@pytest.mark.parametrize("key", TYPE_KEYS)
def test_each_type_list_is_a_non_empty_list_of_strings(key):
    value = CONTRACT[key]
    assert isinstance(value, list) and value
    assert all(isinstance(token, str) and token for token in value)


def test_every_token_names_a_predicate_pyarrow_has():
    """The whole contract is decoded through `pa.types.is_<token>`. A token
    with no such predicate would raise at the consumer, or -- worse, if the
    consumer used `getattr(..., None)` -- read as "no type matches"."""
    missing = sorted(t for t in _tokens() if not hasattr(pa.types, f"is_{t}"))
    assert not missing, f"no pa.types.is_* for: {missing}"


@pytest.mark.parametrize(
    ("token", "matching", "not_matching"),
    [
        ("int8", pa.int8(), pa.int16()),
        ("int16", pa.int16(), pa.int32()),
        ("int32", pa.int32(), pa.int64()),
        ("int64", pa.int64(), pa.int32()),
        ("float32", pa.float32(), pa.float64()),
        ("float64", pa.float64(), pa.float32()),
        # utf8 only: the reader downcasts to `StringArray`, so a large string
        # must not match.
        ("string", pa.string(), pa.large_string()),
        ("boolean", pa.bool_(), pa.int64()),
        ("decimal128", pa.decimal128(10, 2), pa.float64()),
        ("timestamp", pa.timestamp("ns"), pa.date32()),
        ("list", pa.list_(pa.int64()), pa.large_list(pa.int64())),
        ("large_list", pa.large_list(pa.int64()), pa.list_(pa.int64())),
        ("fixed_size_list", pa.list_(pa.int64(), 2), pa.list_(pa.int64())),
    ],
)
def test_each_predicate_means_what_the_reader_means(token, matching,
                                                    not_matching):
    predicate = getattr(pa.types, f"is_{token}")
    assert predicate(matching)
    assert not predicate(not_matching)


def test_a_time_zone_aware_timestamp_is_still_a_timestamp():
    """The reader downcasts on the Rust array type and the zone lives in the
    `DataType`, so a tz-aware column reads as a timestamp there. The Python
    predicate has to agree, or a consumer would refuse a column DiskGraph
    reads."""
    assert pa.types.is_timestamp(pa.timestamp("ns", tz="Etc/UTC"))


def test_link_targets_and_weights_accept_both_variable_list_widths():
    assert CONTRACT["list_target_containers"] == ["list", "large_list"]
    assert set(CONTRACT["weight_list_containers"]) == set(
        CONTRACT["list_target_containers"])


def test_a_scalar_weight_takes_a_decimal_and_a_list_weight_does_not():
    assert "decimal128" in CONTRACT["weight_types"]
    assert "decimal128" not in CONTRACT["weight_list_item_types"]


def test_an_entity_key_is_narrower_than_a_scalar_target():
    assert set(CONTRACT["entity_types"]) < set(CONTRACT["scalar_types"])
    assert "boolean" not in CONTRACT["entity_types"]
