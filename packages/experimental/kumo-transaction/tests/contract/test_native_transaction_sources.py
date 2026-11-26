# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builtin source hooks expose original rows, static attributes and dated facts."""

from datetime import timedelta

import pyarrow.parquet as pq
import pytest

from tfm.config import load_recipe_config
from tfm.data.adapters.registry import table_source
from tfm.data.synthetic.generator import generate_synthetic_events


def test_streamed_synthetic_source_matches_reference_events(tmp_path, monkeypatch):
    from tfm.data.adapters import registry

    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")

    def forbidden(*args, **kwargs):
        raise AssertionError("Native source hooks must not invoke object corpus loading")

    monkeypatch.setattr(registry, "load_adapter_events", forbidden)
    source = table_source(config.data, seed=config.seed, fixture_dir=tmp_path)
    events = generate_synthetic_events(
        seed=config.seed,
        n_entities=config.data.n_entities,
        events_per_entity=config.data.events_per_entity,
        label_delay=timedelta(hours=12),
    )
    actual = pq.read_table(source.files["events"])
    assert actual.num_rows == len(events)
    declaration = next(table for table in source.tables if table.name == "events")
    assert actual[declaration.numeric["amount"]].to_pylist() == [
        event.numeric["amount"] for event in events
    ]
    assert actual["fraud_value"].to_pylist() == [event.labels["fraud"].value for event in events]
    assert actual["id"].to_pylist() == [event.event_id for event in events]
    assert source.graph_numeric["events"] == {
        "amount_log": "amount_log",
        "present": "amount_present",
    }
    assert source.future_numeric["events"] == {"amount": "amount"}


def test_pragmatiq_source_retains_static_profile_contract_and_dated_milestones(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("TFM_PRAGMATIQ_ROOT", raising=False)
    config = load_recipe_config("recipes/blueprint_v2/config.smoke.yaml")
    source = table_source(config.data, seed=config.seed, fixture_dir=tmp_path)
    users = next(table for table in source.tables if table.name == "users")
    assert users.immutable
    assert {"country", "plan"} <= users.categorical.keys()
    facts = next(table for table in source.tables if table.name == "facts")
    assert facts.event_time == facts.available_at == "at"
    assert source.profile_paths == {"milestones": ("facts",)}
    assert (
        pq.read_table(source.files["events"]).num_rows
        == config.data.n_entities * config.data.events_per_entity
    )
    assert source.event_labels == {"events": ("fraud_value", "fraud_available_us")}


def test_native_book_source_rejects_label_like_input_keys(tmp_path):
    from tfm.contracts.errors import SchemaError
    from tfm.data.adapters.native_transactions import _ScalarBook

    with _ScalarBook(tmp_path, "amount") as book:
        with pytest.raises(SchemaError, match="looks like a label"):
            book.event(
                "owner",
                "event",
                0,
                "transaction",
                {"amount": 1.0},
                {},
                {" Is_Fraud ": "yes"},
                None,
                None,
            )


def test_synthetic_stream_constructs_only_requested_records(monkeypatch):
    from tfm.data.synthetic import generator

    original = generator.TransactionEvent
    calls = []

    def observed(**kwargs):
        calls.append(kwargs["event_id"])
        return original(**kwargs)

    monkeypatch.setattr(generator, "TransactionEvent", observed)
    events = generator.iter_synthetic_events(n_entities=1000, events_per_entity=1000)
    assert calls == []
    first = next(events)
    assert calls == [first.event_id]
    events.close()


def test_tabformer_source_scans_rows_without_object_loading(tmp_path, monkeypatch):
    from tfm.data.adapters import tabformer

    root = tabformer.write_tabformer_fixture(tmp_path / "book", n_users=8, events_per_user=3)
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    data = config.data.model_copy(
        update={
            "adapter": "tabformer",
            "path": str(root),
            "split_protocol": "precomputed",
            "n_entities": 8,
        }
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Native source hooks must not construct an object corpus")

    monkeypatch.setattr(tabformer, "load_tabformer_adapter", forbidden)
    monkeypatch.setattr(tabformer, "_table_to_events", forbidden)
    source = table_source(data, seed=42, fixture_dir=tmp_path / "projected")
    events = pq.read_table(source.files["events"])
    assert events.num_rows == 8 * 3 * 3
    assert set(events["partition"].to_pylist()) == {"train", "validation", "test"}
    assert len(set(events["id"].to_pylist())) == events.num_rows
    assert (
        "fraud_value"
        not in next(table for table in source.tables if table.name == "events").numeric.values()
    )
    assert pq.read_table(source.files["transfers"]).num_rows == 0


def test_native_financial_projection_and_flat_delivery_need_no_python_tokenization(
    tmp_path, monkeypatch
):
    import torch

    from tfm.contracts.graph import utc_us
    from tfm.data.native_financial import financial_source
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_transaction_anchors import sequence_anchors
    from tfm.data.native_views import SourceRows
    from tfm.data.splits.temporal import parse_timestamp
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    source = table_source(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    reference = generate_synthetic_events(
        seed=config.seed,
        n_entities=config.data.n_entities,
        events_per_entity=config.data.events_per_entity,
    )
    fields = tuple(config.features.fields)
    expected = TransactionFieldTokenizer(fields=fields).event_tokens(reference[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Native flat delivery cannot tokenize/fit Python histories")

    monkeypatch.setattr(TransactionFieldTokenizer, "fit", forbidden)
    monkeypatch.setattr(TransactionFieldTokenizer, "collate", forbidden)
    monkeypatch.setattr(TransactionFieldTokenizer, "event_tokens", forbidden)
    source, fixed = financial_source(source, fields, tmp_path / "financial")
    row = pq.read_table(source.files["events"]).slice(0, 1).to_pylist()[0]
    assert [row[f"flat_{field}"] for field in fields] == expected
    with NativeInputs(
        source,
        root="users",
        history_paths={"events": ("events",)},
        anchor_factory=lambda source, path: sequence_anchors(source, config, path),
        work_dir=tmp_path / "work",
        max_events=config.data.sequence_length,
        batch_size=4,
        fixed_vocabularies=fixed,
        fit_view="train",
        views={
            "train": {
                "events": SourceRows(
                    before_us=utc_us(parse_timestamp(config.data.temporal_split.train_end))
                )
            }
        },
    ) as inputs:
        (batch, *_) = list(inputs.batches("train", mode="flat", max_tokens=26))
        assert batch.input_ids.shape == (4, 26)
        assert (batch.input_ids[:, 0] == 1).all()
        assert (
            batch.input_ids[
                batch.attention_mask.sum(1).sub(1).new_tensor(range(4)),
                batch.attention_mask.sum(1).sub(1),
            ]
            == 2
        ).all()
        assert torch.isfinite(batch.numeric_features).all()
        assert ((batch.labels == -100) == ~batch.attention_mask.bool()).all()


def test_native_flat_window_groups_store_each_event_once_and_share_token_ids(tmp_path):

    from tfm.config import PretrainWindowsConfig
    from tfm.contracts.graph import utc_us
    from tfm.data.native_financial import financial_source
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_transaction_anchors import sequence_anchors
    from tfm.data.native_views import SourceRows
    from tfm.data.native_windows import flat_window_source
    from tfm.data.sequence_builders.entity_sequences import build_entity_windows
    from tfm.data.splits.temporal import parse_timestamp
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"pretrain_windows": PretrainWindowsConfig(events=3, max_tokens=40)}
            )
        }
    )
    original = table_source(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    fields = tuple(config.features.fields)
    projected, fixed = financial_source(original, fields, tmp_path / "financial")
    projected = flat_window_source(projected, config, tmp_path / "windows")
    assert (
        pq.ParquetFile(projected.files["events"]).metadata.num_rows
        == pq.ParquetFile(original.files["events"]).metadata.num_rows
    )
    cutoff = parse_timestamp(config.data.temporal_split.train_end)
    source_events = generate_synthetic_events(
        seed=config.seed,
        n_entities=config.data.n_entities,
        events_per_entity=config.data.events_per_entity,
    )
    reference = build_entity_windows(
        [event for event in source_events if event.event_timestamp < cutoff],
        cutoff,
        window_events=3,
    )
    tokenizer = TransactionFieldTokenizer(fields=fields)
    expected = sorted(
        tuple(token for event in row.events for token in tokenizer.event_tokens(event))
        for row in reference
    )
    # Window roots and regular owners have mutually exclusive fixed paths to
    # one stored event table. Both streams must share each field's native IDs.
    with NativeInputs(
        projected,
        root="users",
        history_paths={"events": ("events",), "window_events": ("window_events",)},
        anchor_factory=lambda source, path: sequence_anchors(source, config, path),
        work_dir=tmp_path / "work",
        max_events=12,
        batch_size=4,
        fixed_vocabularies=fixed,
        exclusive_terminal_paths=True,
        fit_split="pretrain",
        fit_view="train",
        views={"train": {"events": SourceRows(before_us=utc_us(cutoff))}},
    ) as inputs:
        state = inputs.task("flat").fitted_state.to_dict()
        lookup = {}
        for identity, encoder in state["encoders"].items():
            if encoder["kind"] != "vocabulary":
                continue
            members = dict(state["spaces"]["model"]["members"][identity])
            for value, local in encoder["entries"]:
                lookup[members[local]] = value
        actual = []
        for batch in inputs.batches("pretrain", mode="flat", max_tokens=40, history_limit=3):
            for row, mask in zip(batch.input_ids, batch.attention_mask, strict=True):
                actual.append(
                    tuple(
                        lookup[int(value)]
                        for value in row[mask.bool()]
                        if int(value) not in (0, 1, 2, 3)
                    )
                )
        assert sorted(actual) == expected
        (probe, *_) = list(
            inputs.batches(
                "train",
                mode="flat",
                max_tokens=145,
                history_limit=config.data.sequence_length,
                view="train",
            )
        )
        assert all(
            int(value) in lookup or int(value) in (0, 1, 2, 3)
            for value in probe.input_ids.flatten()
        )


def test_tabformer_opt_out_samples_independently_of_fraud_outcomes(tmp_path):
    from tfm.data.adapters import tabformer

    root = tabformer.write_tabformer_fixture(tmp_path / "book", n_users=12, events_per_user=3)
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml").data.model_copy(
        update={
            "adapter": "tabformer",
            "path": str(root),
            "max_entities": 4,
            "stratify_by_fraud": False,
            "load_labels": True,
        }
    )
    selected = []
    for attempt in (0, 1):
        source = table_source(config, seed=42, fixture_dir=tmp_path / f"source{attempt}")
        selected.append(pq.read_table(source.files["users"])["id"].to_pylist())
        for path in root.rglob("*.parquet"):
            table = pq.read_table(path)
            if "Is Fraud?" in table.column_names:
                import pyarrow as pa

                table = table.set_column(
                    table.schema.get_field_index("Is Fraud?"),
                    "Is Fraud?",
                    pa.array(["Yes"] * len(table)),
                )
                pq.write_table(table, path)
    assert selected[0] == selected[1]
    assert set(pq.read_table(source.files["events"])["fraud_value"].to_pylist()) == {1.0}


def test_precomputed_sequence_anchor_ids_include_the_partition(tmp_path):
    from tfm.data.adapters.native_transactions import _ScalarBook
    from tfm.data.native_source import NativeSourceFiles
    from tfm.data.native_transaction_anchors import sequence_anchors

    config = load_recipe_config("recipes/blueprint_v2/config.smoke.yaml")
    config.data.split_protocol = "precomputed"
    config.data.load_labels = False
    instant = 1704067200000000
    with _ScalarBook(tmp_path / "book", "amount") as book:
        for partition in ("train", "validation", "test"):
            book.event(
                "one",
                partition,
                instant,
                "transaction",
                {"amount": 1},
                {},
                {},
                1.0,
                instant,
                partition=partition,
            )
        source = book.tables(config.data)
    with NativeSourceFiles(source, tmp_path / "normalized") as normalized:
        path = sequence_anchors(normalized, config, tmp_path / "anchors.parquet")
        rows = pq.read_table(path).to_pylist()
        assert len({r["sample_id"] for r in rows}) == len(rows) == 3
        assert {r["split"] for r in rows} == {"train", "validation", "test"}
        assert len({r["cutoff"] for r in rows}) == 1


def test_lazy_reference_ties_use_original_event_id_before_history_limit(tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    import pyarrow as pa

    from tfm.data.adapters.native_transactions import _ScalarBook
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_reference import ReferenceRows

    config = load_recipe_config("recipes/blueprint_v2/config.smoke.yaml")
    instant = 1704067200000000
    with _ScalarBook(tmp_path / "book", "amount") as book:
        book.event("one", "z", instant, "transaction", {"amount": 9}, {}, {}, None, None)
        book.event("one", "a", instant, "transaction", {"amount": 1}, {}, {}, None, None)
        source = book.tables(config.data)
    physical = pq.read_table(source.files["events"])
    ids = physical["id"].to_pylist()
    pq.write_table(
        physical.take(pa.array([ids.index("z"), ids.index("a")])), source.files["events"]
    )
    from datetime import UTC, datetime

    requested = tmp_path / "requested.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["one"],
                "sample_id": ["one"],
                "split": ["train"],
                "cutoff": [datetime(2024, 1, 2, tzinfo=UTC)],
            }
        ),
        requested,
    )
    with NativeInputs(
        source,
        root="users",
        history_paths={"events": ("events",)},
        anchors=requested,
        work_dir=tmp_path / "work",
        max_events=2,
        batch_size=1,
    ) as inputs:
        db = inputs.normalized.connection
        db.execute("CREATE TABLE event_partitions (code INTEGER PRIMARY KEY,partition TEXT)")
        db.executemany("INSERT INTO event_partitions VALUES (?,?)", [(0, None), (1, None)])
        references = ReferenceRows(inputs)
        assert [e.event_id for e in references.sequence("one").events] == ["a", "z"]
        inputs.max_events = 1
        assert [e.event_id for e in references.sequence("one").events] == ["z"]


def test_tabformer_csv_source_streams_public_csv_reader(tmp_path):
    import csv

    from tfm.data.adapters import tabformer

    root = tabformer.write_tabformer_fixture(tmp_path / "book", n_users=4, events_per_user=2)
    rows = []
    for name in ("train", "val", "test"):
        path = root / "temporal_split" / f"{name}.parquet"
        rows.extend(pq.read_table(path).to_pylist())
        path.unlink()
    with (root / "raw/card_transaction.v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    data = config.data.model_copy(
        update={
            "adapter": "tabformer",
            "path": str(root),
            "split_protocol": "temporal",
            "load_labels": False,
            "n_entities": 4,
        }
    )
    source = table_source(data, seed=42, fixture_dir=tmp_path / "projected")
    actual = pq.read_table(source.files["events"])
    assert actual.num_rows == len(rows)
    assert actual["fraud_value"].null_count == len(rows)
