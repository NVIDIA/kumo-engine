"""Independent fixtures for the canonical source boundary, not dataset acceptance."""

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test.kumo_pytorch.aml_source import (
    prepare_aml_sources,
    load_account_id_map,
    account_identity_map,
)


def account(bank, number, variant="hi-small"):
    return json.dumps([variant, "account", str(bank), number], separators=(",", ":"))


def transaction(**overrides):
    row = {
        "timestamp": "2022/09/01 00:20",
        "from_bank": "001",
        "from_account": "0007",
        "to_bank": "2",
        "to_account": "0008",
        "amount_paid": "12.1234567890",
        "payment_currency": "USD",
        "amount_received": "10.0000000001",
        "receiving_currency": "EUR",
        "payment_format": "Wire",
        "is_laundering": "DO_NOT_READ",
    }
    return {**row, **overrides}


def source_files(tmp_path, rows=None, members=None):
    tx, binding = tmp_path / "source.parquet", tmp_path / "binding.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            rows if rows is not None else [transaction()],
            schema=pa.schema([pa.field(name, pa.string()) for name in transaction()]),
        ),
        tx,
        row_group_size=2,
    )
    members = (
        members if members is not None else [account(1, "0007"), account(2, "0008")]
    )
    pq.write_table(
        pa.table(
            {
                "entity_id": members,
                "first_observed_time": [datetime(2022, 9, 1)] * len(members),
            }
        ),
        binding,
    )
    checksums = {
        "transactions": hashlib.sha256(tx.read_bytes()).hexdigest(),
        "accounts_binding": hashlib.sha256(binding.read_bytes()).hexdigest(),
    }
    return tx, binding, checksums


def prepare(tmp_path, rows=None, members=None, **kwargs):
    tx, binding, checksums = source_files(tmp_path, rows, members)
    return prepare_aml_sources(
        tx,
        binding,
        tmp_path / "output",
        "hi-small",
        expected_checksums=checksums,
        **kwargs,
    )


def test_exact_fields_keys_precision_utc_and_label_exclusion(tmp_path):
    manifest = prepare(tmp_path)
    tx = pq.read_table(tmp_path / "output/transactions.parquet")
    row = tx.to_pylist()[0]
    assert set(tx.column_names) == {
        "transaction_id",
        "event_time",
        "event_time_seconds",
        "sender_id",
        "receiver_id",
        "amount_paid",
        "payment_currency",
        "amount_received",
        "receiving_currency",
        "payment_format",
    }
    assert row["transaction_id"] == 0
    assert row["sender_id"] == 0
    assert account_identity_map(tmp_path / "output")[account(1, "0007")] == 0
    assert row["amount_paid"] == Decimal("12.1234567890")
    assert row["amount_received"] == Decimal("10.0000000001")
    assert str(tx.schema.field("event_time").type) == "timestamp[ms, tz=UTC]"
    assert row["event_time"].isoformat() == "2022-09-01T00:20:00+00:00"
    assert manifest["tables"]["transactions"]["rows"] == 1
    assert manifest["tables"]["account_events"]["rows"] == 2
    for name, record in manifest["tables"].items():
        path = tmp_path / "output" / record["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert "is_laundering" not in pq.read_schema(path).names
    assert json.loads((tmp_path / "output/manifest.json").read_text()) == manifest


def test_self_transfer_is_one_touch(tmp_path):
    manifest = prepare(tmp_path, [transaction(to_bank="01", to_account="0007")])
    events = pq.read_table(tmp_path / "output/account_events.parquet").to_pylist()
    assert len(events) == 1
    assert events[0]["direction"] == "self"
    assert events[0]["event_id"] == 0
    assert manifest["self_transfers"] == 1


def test_physical_identity_survives_batch_and_rowgroup_boundaries(tmp_path):
    rows = [transaction(amount_paid=str(i)) for i in range(12)]
    tx, binding, sums = source_files(tmp_path, rows)
    for size in (1, 3, 20):
        prepare_aml_sources(
            tx,
            binding,
            tmp_path / f"batch-{size}",
            "hi-small",
            expected_checksums=sums,
            batch_size=size,
        )
    one = pq.read_table(tmp_path / "batch-1/account_events.parquet")
    for size in (3, 20):
        assert one.equals(
            pq.read_table(tmp_path / f"batch-{size}/account_events.parquet")
        )
    outgoing = [row for row in one.to_pylist() if row["direction"] == "outgoing"]
    assert [row["source_row"] for row in outgoing] == list(range(12))
    assert [row["event_id"] for row in outgoing] == sorted(
        row["event_id"] for row in outgoing
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"amount_paid": "1.00000000001"},
        {"amount_received": "1.00000000001"},
        {"amount_paid": "1e2"},
        {"amount_received": "NaN"},
        {"amount_paid": "10000000000000000000000000000"},
        {"timestamp": "2022/02/30 00:20"},
        {"timestamp": "2022/9/1 00:20"},
        {"timestamp": "2022/09/01 00:20:00"},
        {"timestamp": "0000/09/01 00:20"},
        {"from_bank": "-1"},
        {"from_bank": "1.0"},
        {"to_account": "8"},
        {"payment_format": ""},
        {"receiving_currency": ""},
    ],
)
def test_rejects_invalid_values_without_completion_marker(tmp_path, overrides):
    with pytest.raises(ValueError, match="physical source row 0"):
        prepare(tmp_path, [transaction(**overrides)])
    assert not (tmp_path / "output/manifest.json").exists()


@pytest.mark.parametrize(
    "members",
    [
        [account(1, "0007")],
        [account(1, "0007"), account(1, "0007"), account(2, "0008")],
        [account(1, "0007", "li-small"), account(2, "0008")],
        [account("001", "0007"), account(2, "0008")],
    ],
)
def test_unknown_duplicate_or_noncanonical_binding_rejected(tmp_path, members):
    with pytest.raises(ValueError):
        prepare(tmp_path, members=members)
    assert not (tmp_path / "output/manifest.json").exists()


def test_negative_and_maximum_exact_decimal(tmp_path):
    prepare(
        tmp_path,
        [
            transaction(
                amount_paid="-9999999999999999999999999999.9999999999",
                amount_received="+0.1",
            )
        ],
    )
    row = pq.read_table(tmp_path / "output/transactions.parquet").to_pylist()[0]
    assert row["amount_paid"] == Decimal("-9999999999999999999999999999.9999999999")
    assert row["amount_received"] == Decimal("0.1000000000")


def test_existing_output_and_bad_hash_are_never_overwritten(tmp_path):
    tx, binding, sums = source_files(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("existing")
    with pytest.raises(FileExistsError):
        prepare_aml_sources(tx, binding, out, "hi-small", expected_checksums=sums)
    assert sentinel.read_text() == "existing"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_aml_sources(
            tx,
            binding,
            tmp_path / "wrong",
            "hi-small",
            expected_checksums={**sums, "transactions": "0" * 64},
        )
    assert not (tmp_path / "wrong").exists()


def test_context_and_symlink_sources_rejected(tmp_path):
    tx, binding, sums = source_files(tmp_path)
    link = tmp_path / "link.parquet"
    link.symlink_to(tx)
    with pytest.raises(ValueError, match="Symlinks"):
        prepare_aml_sources(
            link, binding, tmp_path / "output", "hi-small", expected_checksums=sums
        )
    with pytest.raises(ValueError, match="context/"):
        prepare_aml_sources(
            tx,
            binding,
            tmp_path / "context/output",
            "hi-small",
            expected_checksums=sums,
        )


def test_source_order_is_preserved_and_cutoff_filtering_is_external(tmp_path):
    prepare(
        tmp_path,
        [
            transaction(timestamp="2022/09/02 00:00"),
            transaction(timestamp="2022/09/01 00:00"),
        ],
    )
    rows = pq.read_table(tmp_path / "output/transactions.parquet").to_pylist()
    assert [row["transaction_id"] for row in rows] == [0, 1]
    assert rows[0]["event_time"] > rows[1]["event_time"]


def test_required_pins_and_source_schema(tmp_path):
    tx, binding, sums = source_files(tmp_path)
    with pytest.raises(ValueError, match="Both exact"):
        prepare_aml_sources(
            tx, binding, tmp_path / "missing-pin", "hi-small", expected_checksums={}
        )
    data = pq.read_table(tx).drop(["amount_received"])
    pq.write_table(data, tx)
    sums["transactions"] = hashlib.sha256(tx.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="amount_received"):
        prepare_aml_sources(
            tx, binding, tmp_path / "missing-field", "hi-small", expected_checksums=sums
        )


def test_availability_epoch_transport_nulls_and_identity_metadata(tmp_path):
    prepare(
        tmp_path,
        [transaction(amount_paid=None, payment_currency=None, payment_format=None)],
    )
    accounts = pq.read_table(tmp_path / "output/accounts.parquet").to_pylist()
    tx = pq.read_table(tmp_path / "output/transactions.parquet").to_pylist()[0]
    assert accounts[0] == {
        "account_id": 0,
        "entity_id": account(1, "0007"),
        "first_observed_time": datetime(2022, 9, 1, tzinfo=timezone.utc),
    }
    assert tx["event_time_seconds"] == int(
        datetime(2022, 9, 1, 0, 20, tzinfo=timezone.utc).timestamp()
    )
    assert (
        tx["amount_paid"] is None
        and tx["payment_currency"] is None
        and tx["payment_format"] is None
    )
    assert "is_sender" not in tx and "hour" not in tx
    assert (
        pq.read_schema(tmp_path / "output/transactions.parquet")
        .field("amount_paid")
        .nullable
    )


def test_bounded_roots_keep_full_history_future_and_counterparties_without_expanding(
    tmp_path,
):
    rows = [
        transaction(),
        transaction(from_bank="2", from_account="0008", to_bank="3", to_account="0009"),
        transaction(timestamp="2022/09/10 00:00", to_bank="3", to_account="0009"),
        transaction(to_bank="1", to_account="0007"),
    ]
    members = [
        account(3, "0009"),
        account(1, "0007"),
        account(2, "0008"),
        account(9, "isolated"),
    ]
    tx, binding, sums = source_files(tmp_path, rows, members)
    id_map = load_account_id_map(
        binding, "hi-small", expected_sha256=sums["accounts_binding"]
    )
    manifest = prepare_aml_sources(
        tx,
        binding,
        tmp_path / "slice",
        "hi-small",
        expected_checksums=sums,
        root_entity_ids=[account(1, "0007")],
        batch_size=2,
    )
    output = pq.read_table(tmp_path / "slice/transactions.parquet").to_pylist()
    assert [row["transaction_id"] for row in output] == [0, 2, 3]
    assert [
        row["event_id"]
        for row in pq.read_table(tmp_path / "slice/account_events.parquet").to_pylist()
    ] == [0, 1, 4, 5, 6]
    mapped = account_identity_map(tmp_path / "slice")
    assert mapped == {
        name: id_map[name] for name in members if name != account(9, "isolated")
    }
    assert manifest["scope"] == {
        "kind": "selected_root_full_histories",
        "complete_root_histories": True,
        "complete_history_root_entity_ids": [account(1, "0007")],
        "counterparty_scope": "endpoint_accounts_and_selected_root_transactions_only",
        "source_transactions_scanned": 4,
        "accepted_accounts": 4,
        "time_filter": None,
        "cutoff_filter": None,
    }
    assert output[1]["event_time"] > datetime(2022, 9, 5, tzinfo=timezone.utc)


def test_root_filter_does_not_confuse_same_account_number_at_different_bank(tmp_path):
    members = [account(1, "0007"), account(2, "0007"), account(2, "0008")]
    manifest = prepare(
        tmp_path,
        [transaction(from_bank="2"), transaction()],
        members,
        root_entity_ids=[account(1, "0007")],
        batch_size=1,
    )
    assert manifest["tables"]["transactions"]["rows"] == 1
    assert pq.read_table(tmp_path / "output/transactions.parquet")[
        "transaction_id"
    ].to_pylist() == [1]


def test_numeric_identity_is_binding_order_and_batch_independent(tmp_path):
    tx, binding, sums = source_files(
        tmp_path, members=[account(2, "0008"), account(1, "0007")]
    )
    before = load_account_id_map(
        binding, "hi-small", expected_sha256=sums["accounts_binding"]
    )
    table = pq.read_table(binding).take(pa.array([1, 0]))
    pq.write_table(table, binding)
    after = load_account_id_map(
        binding,
        "hi-small",
        expected_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),
    )
    assert before == after == {account(1, "0007"): 0, account(2, "0008"): 1}
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_account_id_map(
            binding, "hi-small", expected_sha256=sums["accounts_binding"]
        )


@pytest.mark.parametrize(
    "roots", [[], [account(3, "unknown")], [account(1, "0007", "li-small")]]
)
def test_invalid_root_scope_fails_before_creating_output(tmp_path, roots):
    with pytest.raises(ValueError):
        prepare(tmp_path, root_entity_ids=roots)
    assert not (tmp_path / "output").exists()


def test_binding_requires_real_whole_second_availability(tmp_path):
    tx, binding, sums = source_files(tmp_path)
    for availability in (
        None,
        datetime(2022, 9, 1, microsecond=1),
        datetime(2022, 9, 2),
    ):
        pq.write_table(
            pa.table(
                {
                    "entity_id": [account(1, "0007"), account(2, "0008")],
                    "first_observed_time": pa.array(
                        [availability] * 2, type=pa.timestamp("us")
                    ),
                }
            ),
            binding,
        )
        sums["accounts_binding"] = hashlib.sha256(binding.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="availability"):
            prepare_aml_sources(
                tx,
                binding,
                tmp_path / str(availability),
                "hi-small",
                expected_checksums=sums,
            )


def test_prepared_helpers_reject_tampered_table(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "output/accounts.parquet"
    table = pq.read_table(path)
    pq.write_table(table.take(pa.array([1, 0])), path)
    with pytest.raises(ValueError, match="changed"):
        account_identity_map(tmp_path / "output")
