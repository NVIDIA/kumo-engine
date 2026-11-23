# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from tfm.contracts.schema import TransactionEvent
from tfm.data.adapters.pragmatiq import (
    events_for_flat_fields,
    generate_pragmatiq_fixture,
    load_pragmatiq_dataset,
)
from tfm.data.sequence_builders.entity_sequences import (
    build_entity_sequences,
    build_sequences_at_label_cutoffs,
)
from tfm.features.tokenizers.financial import TransactionFieldTokenizer


def test_pragmatiq_roundtrip_fixture(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=3, events_per_user=6, seed=1)
    assert (root / "events.parquet").exists()
    assert (root / "profiles.parquet").exists()
    assert (root / "transfers.parquet").exists()
    assert (root / "labels" / "fraud.parquet").exists()
    assert (root / "labels" / "default_12m.parquet").exists()

    dataset = load_pragmatiq_dataset(root)
    assert dataset.events
    assert dataset.profiles
    assert dataset.transfers
    assert "app" in dataset.dropped_sources
    assert "fraud" in dataset.labels
    assert "default_12m" in dataset.labels
    assert any(event.labels.get("fraud") is not None for event in dataset.events)
    assert all(event.event_id for event in dataset.events)
    sources = {event.categorical["source"] for event in dataset.events}
    assert sources == {"transaction", "app"}


def test_events_for_flat_fields_drops_app_and_tokenizes(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=2, events_per_user=6, seed=2)
    dataset = load_pragmatiq_dataset(root)
    with __import__("pytest").warns(UserWarning, match="Dropped non-transaction"):
        flat_events = events_for_flat_fields(dataset.events)
    assert flat_events
    assert all(ev.categorical.get("source") == "transaction" for ev in flat_events)
    assert all(
        "merchant_name" in ev.categorical or "merchant" in ev.categorical for ev in flat_events
    )
    assert all("amount" in ev.numeric for ev in flat_events)

    seqs = build_entity_sequences(flat_events, datetime(2024, 12, 31, tzinfo=UTC), max_events=8)
    tok = TransactionFieldTokenizer()
    ids, attn = tok.encode_sequence(seqs[0], max_length=64)
    assert ids[0] == tok.bos_id
    assert sum(attn) > 0


def test_committed_pragmatiq_fixture_loads() -> None:
    from pathlib import Path

    root = Path("tests/fixtures/pragmatiq")
    dataset = load_pragmatiq_dataset(root)
    assert len(dataset.events) > 0
    assert dataset.profiles
    assert "fraud" in dataset.labels


def test_eval_ts_cutoff_excludes_future_events(tmp_path) -> None:
    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=2, events_per_user=6, seed=0)
    dataset = load_pragmatiq_dataset(root)
    row = dataset.labels["default_12m"][0]
    entity_events = [e for e in dataset.events if e.entity_id == row.entity_id]
    seqs = build_entity_sequences(entity_events, row.timestamp, max_events=16)
    assert seqs
    assert all(ensure_before(event, row.timestamp) for event in seqs[0].events)


def test_sequences_at_label_cutoffs_drop_post_eval_events() -> None:
    from tfm.contracts.schema import TransactionEvent
    from tfm.data.adapters.pragmatiq import LabelRow

    early = TransactionEvent(
        entity_id="u0",
        event_id="e0",
        event_timestamp=datetime(2024, 1, 15, tzinfo=UTC),
    )
    late = TransactionEvent(
        entity_id="u0",
        event_id="e1",
        event_timestamp=datetime(2024, 3, 15, tzinfo=UTC),
    )
    row = LabelRow(
        entity_id="u0",
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
        label=1,
        task="default_12m",
    )
    pairs = build_sequences_at_label_cutoffs([early, late], [row], max_events=16)
    assert len(pairs) == 1
    seq, matched = pairs[0]
    assert matched is row
    assert [event.event_id for event in seq.events] == ["e0"]
    assert seq.cutoff_time == row.timestamp


def test_label_tables_without_label_column_are_skipped(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = generate_pragmatiq_fixture(tmp_path / "book", n_users=2, events_per_user=4, seed=0)
    extra = pa.table(
        {
            "user_id": pa.array(["u0000"], type=pa.string()),
            "ts": pa.array([datetime(2024, 1, 2, tzinfo=UTC)]),
            "treated": pa.array([0], type=pa.int8()),
        }
    )
    pq.write_table(extra, root / "labels" / "comm_uplift.parquet")
    dataset = load_pragmatiq_dataset(root)
    assert "fraud" in dataset.labels
    assert "comm_uplift" not in dataset.labels


def ensure_before(event: TransactionEvent, cutoff: datetime) -> bool:
    return event.event_timestamp <= cutoff


def test_label_free_loader_never_reads_or_stratifies_by_fraud(tmp_path):
    book = generate_pragmatiq_fixture(tmp_path / "book", n_users=20, events_per_user=6)
    labels = book / "labels/fraud.parquet"
    labels.write_bytes(b"invalid parquet must not be read")
    loaded = load_pragmatiq_dataset(book, max_entities=12, sample_seed=7, load_labels=False)
    assert len({event.entity_id for event in loaded.events}) == 12
    assert not loaded.labels and all(not event.labels for event in loaded.events)
