# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchors read from a table, and the order they are consumed in."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kumo.pytorch.anchors import _Permutation
from kumo.pytorch.api import (
    ALL_ANCHORS,
    NON_TEMPORAL,
    Aggregate,
    AnchorTable,
    Batching,
    BatchSpec,
    CandidateColumn,
    CandidateRequest,
    CandidateSetSpec,
    CandidateSource,
    CandidateTarget,
    Compare,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    EventSource,
    EventSpec,
    ExternalTarget,
    Field,
    Fields,
    InMemoryEngine,
    InMemoryRows,
    Input,
    LabelTarget,
    Padded,
    Parquet,
    Raw,
    RelationSpec,
    Rows,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    TargetValue,
    TimeWindow,
    ValidationError,
    ZScore,
    check,
    padded_positions,
    prepare,
    shard_positions,
)
from kumo.pytorch.engines.in_memory.anchors import InMemoryAnchors
from kumo.pytorch.task import _mark_padding

USERS = [{"user_id": index, "score": float(index)} for index in range(1, 13)]
EVENTS = [
    {"event_id": 100 + index, "user_id": index, "occurred_at": 50, "amount": 1.0}
    for index in range(1, 13)
]
ANCHORS = [
    {"sid": f"s{index}", "user_id": index, "at": 100, "split": "train" if index % 3 else "valid"}
    for index in range(1, 13)
]
SCHEMA = DataSchema(
    tables=(
        TableSpec(
            "users",
            "user_id",
            {"user_id": DType("int64"), "score": DType("float64")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "events",
            "event_id",
            {
                "event_id": DType("int64"),
                "user_id": DType("int64"),
                "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                "amount": DType("float64"),
            },
            time_column="occurred_at",
        ),
    ),
    relations=(RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)
DATASET = Dataset(tables={"users": InMemoryRows(rows=USERS), "events": InMemoryRows(rows=EVENTS)})


def _contract(*, temporal: bool = True) -> DataLoaderSpec:
    values: dict[str, object] = {"score": Field(column="score")}
    if temporal:
        values["seen"] = Aggregate(
            Rows(path=("user_events",), time=TimeWindow("occurred_at", None, 0)),
            "count",
            dtype=DType("int64"),
        )
    return DataLoaderSpec(
        SCHEMA,
        SampleSpec(entity=EntitySpec("users", entity_type="user"), entity_values=values),
        BatchSpec(
            inputs={
                "entity": Input(
                    EntitySource(),
                    Fields(fields=tuple(Encode(field=name, encoding=Raw()) for name in values)),
                    Separate(),
                )
            },
            batching=Batching(max_samples=4),
        ),
    )


def _table(**overrides: object) -> AnchorTable:
    settings: dict[str, object] = {
        "source": InMemoryRows(rows=ANCHORS),
        "entity_id": "user_id",
        "as_of_time": "at",
        "sample_id": "sid",
        "split": "split",
    }
    settings.update(overrides)
    return AnchorTable(**settings)


class TestOrdering:
    """The default order an engine may adopt: `shard_positions`."""

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 5, 8, 17, 64, 100, 1000, 4097])
    def test_a_shuffle_is_a_permutation_not_an_approximation(self, size: int) -> None:
        order = _Permutation(size, seed=99)
        assert sorted(order(position) for position in range(size)) == list(range(size))

    def test_every_anchor_belongs_to_exactly_one_rank(self) -> None:
        seen = [
            position
            for rank in range(7)
            for position in shard_positions(
                1000, seed=3, epoch=1, shuffle=True, rank=rank, world_size=7
            )
        ]
        assert sorted(seen) == list(range(1000))

    def test_epochs_differ_and_repeat_exactly(self) -> None:
        first = list(shard_positions(500, seed=3, epoch=0, shuffle=True))
        second = list(shard_positions(500, seed=3, epoch=1, shuffle=True))
        assert first != second
        assert first == list(shard_positions(500, seed=3, epoch=0, shuffle=True))

    def test_unshuffled_order_is_the_table_order(self) -> None:
        assert list(shard_positions(10, rank=1, world_size=3)) == [1, 4, 7]

    def test_a_rank_must_exist_in_its_world(self) -> None:
        with pytest.raises(ValueError, match="rank must identify"):
            list(shard_positions(10, rank=3, world_size=3))


def _requests_for(split: str) -> list[SampleRequest]:
    return [
        SampleRequest(row["sid"], row["user_id"], row["at"], split=row["split"])
        for row in ANCHORS
        if row["split"] == split
    ]


def _check_anchors(contract: DataLoaderSpec, table: AnchorTable) -> list[str]:
    issues = check(contract, DATASET, InMemoryEngine(), anchors=table)
    return [item.path for item in issues if item.path.startswith("anchors")]


def _scores(batches) -> list[float]:
    return [
        value
        for batch in batches
        for value, padding in zip(
            batch.inputs["entity"]["score"]["values"].tolist(),
            batch.metadata["is_padding"].tolist(),
            strict=True,
        )
        if not padding
    ]


class TestMaterialize:
    """Anchors bind once, at materialize; each pass then chooses a split."""

    def test_fitting_reads_only_the_fit_split(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            assert task.count("train") == 8
            assert task.count("valid") == 4
            assert task.count() == 12

    def test_anchors_spanning_splits_need_a_fit_split(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            with pytest.raises(ValueError, match="pass fit_split"):
                plan.materialize(_table())

    def test_requests_carrying_splits_need_a_fit_selector_too(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            with pytest.raises(ValueError, match="pass fit_split"):
                plan.materialize(_requests_for("train"))
            assert plan.materialize(_requests_for("train"), fit_split="train").count("train") == 8

    def test_one_task_serves_every_split(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            valid = _scores(task.batches(split="valid", batch_size=4))
            train = _scores(task.batches(split="train", batch_size=4))
        assert sorted(valid) == [3.0, 6.0, 9.0, 12.0]
        assert len(train) == 8

    def test_an_unknown_split_is_refused_when_asked_for(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            with pytest.raises(ValueError, match="no anchors in split\\(s\\) \\['test'\\]"):
                task.batches(split="test", batch_size=4)

    def test_a_split_cannot_be_applied_to_requests(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            with pytest.raises(ValueError, match="requests carry their own"):
                task.batches(_requests_for("train"), split="train", batch_size=4)

    def test_table_and_requests_yield_the_same_samples(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            from_table = {
                sample_id: score
                for batch in task.batches(split="train", batch_size=3, shuffle=True, seed=5)
                for sample_id, score in zip(
                    batch.metadata["sample_ids"],
                    batch.inputs["entity"]["score"]["values"].tolist(),
                    strict=True,
                )
            }
            from_requests = {
                sample_id: score
                for batch in task.batches(_requests_for("train"), batch_size=3)
                for sample_id, score in zip(
                    batch.metadata["sample_ids"],
                    batch.inputs["entity"]["score"]["values"].tolist(),
                    strict=True,
                )
            }
        assert from_table == from_requests


class TestFittedStateReuse:
    """Fitted input state is reusable under new targets; target state is not."""

    def _with_target(self) -> DataLoaderSpec:
        contract = _contract()
        return replace(
            contract,
            sample=replace(
                contract.sample,
                targets={
                    "future": Aggregate(
                        Rows(path=("user_events",), time=TimeWindow("occurred_at", 0, 10)),
                        "count",
                        dtype=DType("int64"),
                    )
                },
            ),
            batch=replace(
                contract.batch, targets={"future": LabelTarget(source="future", encoding=Raw())}
            ),
        )

    def test_complete_state_is_reused_without_refitting(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            train = plan.materialize(_table(), fit_split="train")
            scoring = plan.materialize(_table(), fitted=train.fitted_state)
        assert scoring.fitted_state == train.fitted_state

    def test_input_state_carries_into_a_contract_with_new_targets(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            pretrain = plan.materialize(_table(), fit_split="train")
        with prepare(self._with_target(), DATASET, InMemoryEngine()) as plan:
            finetune = plan.materialize(
                _table(), fitted=pretrain.fitted_state.inputs, fit_split="train"
            )
        assert finetune.fitted_state["inputs"] == pretrain.fitted_state["inputs"]
        assert finetune.fingerprints["inputs"] == pretrain.fingerprints["inputs"]
        assert finetune.fingerprints["contract"] != pretrain.fingerprints["contract"]
        assert "future" in finetune.fitted_state["targets"]

    def test_complete_state_is_refused_under_different_targets(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            pretrain = plan.materialize(_table(), fit_split="train")
        with prepare(self._with_target(), DATASET, InMemoryEngine()) as plan:
            with pytest.raises(ValueError, match="reuse fitted.inputs"):
                plan.materialize(_table(), fitted=pretrain.fitted_state)

    def test_state_is_refused_under_different_inputs(self) -> None:
        with prepare(_contract(temporal=False), DATASET, InMemoryEngine()) as plan:
            other = plan.materialize(_table(), fit_split="train")
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            with pytest.raises(ValueError, match="different inputs"):
                plan.materialize(_table(), fitted=other.fitted_state.inputs, fit_split="train")


class TestOrderingGuarantees:
    """What any engine's order must satisfy, checked on the reference task."""

    def _epoch(self, anchors, *, world_size: int, epoch: int = 0, shuffle: bool = True):
        train = frozenset({"train"})
        count = anchors.count(train)
        return [
            list(
                padded_positions(
                    anchors.order(
                        train,
                        seed=11,
                        epoch=epoch,
                        shuffle=shuffle,
                        rank=rank,
                        world_size=world_size,
                    ),
                    count=count,
                    world_size=world_size,
                )
            )
            for rank in range(world_size)
        ]

    @pytest.mark.parametrize("world_size", [1, 2, 3, 5, 8, 11])
    def test_exactly_once_disjoint_and_equal(self, world_size: int) -> None:
        anchors = InMemoryAnchors(_table())
        ranks = self._epoch(anchors, world_size=world_size)
        real = [position for rank in ranks for position, padding in rank if not padding]
        assert sorted(real) == list(range(anchors.count(frozenset({"train"}))))
        assert len({len(rank) for rank in ranks}) == 1

    def test_the_order_is_a_pure_function_of_seed_and_epoch(self) -> None:
        anchors = InMemoryAnchors(_table())
        assert self._epoch(anchors, world_size=3) == self._epoch(anchors, world_size=3)
        assert self._epoch(anchors, world_size=3) != self._epoch(anchors, world_size=3, epoch=1)

    def test_padding_repeats_only_a_rank_s_own_anchor(self) -> None:
        anchors = InMemoryAnchors(_table())
        for rank in self._epoch(anchors, world_size=3):
            own = {position for position, padding in rank if not padding}
            assert all((position in own for position, padding in rank if padding))

    def test_an_order_that_overfills_a_rank_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ranks must partition"):
            list(padded_positions(iter(range(5)), count=6, world_size=3))

    def test_every_rank_gets_the_same_number_of_batches(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            per_rank = [
                list(
                    task.batches(
                        split="train", batch_size=2, shuffle=True, seed=3, rank=rank, world_size=3
                    )
                )
                for rank in range(3)
            ]
        assert len({len(batches) for batches in per_rank}) == 1
        flags = [flag for batches in per_rank for b in batches for flag in b.metadata["is_padding"]]
        assert sum(bool(flag) for flag in flags) == 1
        real_ids = [
            sample_id
            for batches in per_rank
            for b in batches
            for sample_id, flag in zip(
                b.metadata["sample_ids"], b.metadata["is_padding"].tolist(), strict=True
            )
            if not flag
        ]
        assert sorted(real_ids) == sorted(row["sid"] for row in ANCHORS if row["split"] == "train")

    def test_the_padding_marker_never_leaks_into_request_metadata(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            batches = list(task.batches(split="train", batch_size=4, rank=2, world_size=3))
        assert all(
            "_is_padding" not in entry for b in batches for entry in b.metadata["request_metadata"]
        )


class TestPaddingIdentity:
    """Caller IDs have no reserved padding suffix (data-and-samples.md)."""

    @staticmethod
    def _requests(ids):
        return [
            SampleRequest(sid, index + 1, 10 + index, metadata={"source": f"row-{index}"})
            for index, sid in enumerate(ids)
        ]

    @staticmethod
    def _materialize(plan, requests, source):
        if source == "table":
            return plan.materialize(
                AnchorTable(
                    source=InMemoryRows(
                        rows=[
                            {
                                "sid": request.sample_id,
                                "user_id": request.entity_id,
                                "at": request.as_of_time,
                                **request.metadata,
                            }
                            for request in requests
                        ]
                    ),
                    entity_id="user_id",
                    as_of_time="at",
                    sample_id="sid",
                    metadata=("source",),
                )
            )
        return plan.materialize(requests)

    @staticmethod
    def _assert_rows(batches, ids, positions, padding):
        assert [sid for batch in batches for sid in batch.metadata["sample_ids"]] == ids
        assert len(set(ids)) == len(ids)
        assert [p for batch in batches for p in batch.metadata["is_padding"].tolist()] == padding
        assert [i for batch in batches for i in batch.metadata["entity_ids"].tolist()] == [
            index + 1 for index in positions
        ]
        assert [t for batch in batches for t in batch.metadata["as_of_time"].tolist()] == [
            10 + index for index in positions
        ]
        assert [m for batch in batches for m in batch.metadata["request_metadata"]] == [
            {"split": None, "source": f"row-{index}"} for index in positions
        ]
        assert [
            value
            for batch in batches
            for value in batch.inputs["entity"]["score"]["values"].tolist()
        ] == [float(index + 1) for index in positions]

    @pytest.mark.parametrize("source", ["table", "materialized_requests", "explicit_requests"])
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize(
        ("ids", "world_size", "expected_ids", "positions", "padding"),
        [
            # Rank 1 owns positions 1 and 3, then repeats position 1. The caller
            # already owns b#padding2, so the repeated b must choose another ID.
            (
                ["a", "b", "c", "b#padding2", "e"],
                2,
                ["b", "b#padding2", "b#padding3"],
                [1, 3, 1],
                [False, False, True],
            ),
            (
                ["a", 1, "c", "1#padding2", "e"],
                2,
                [1, "1#padding2", "1#padding3"],
                [1, 3, 1],
                [False, False, True],
            ),
            # Several occupied suffixes must be skipped, without renaming them.
            (
                ["a", "b", "c", "b#padding4", "e", "b#padding5", "g", "b#padding6", "i"],
                2,
                ["b", "b#padding4", "b#padding5", "b#padding6", "b#padding7"],
                [1, 3, 5, 7, 1],
                [False, False, False, False, True],
            ),
            (
                ["a", "b", "c", "d", "e"],
                2,
                ["b", "d", "b#padding2"],
                [1, 3, 1],
                [False, False, True],
            ),
            (
                ["a", "b", "c", "b#padding2", "e"],
                1,
                ["a", "b", "c", "b#padding2", "e"],
                list(range(5)),
                [False] * 5,
            ),
        ],
    )
    def test_padding_preserves_real_identity_and_row_ownership(
        self, source, batch_size, ids, world_size, expected_ids, positions, padding
    ) -> None:
        requests = self._requests(ids)
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = self._materialize(plan, requests, source)
            settings = {
                "batch_size": batch_size,
                "rank": world_size - 1,
                "world_size": world_size,
            }
            explicit = requests if source == "explicit_requests" else None
            batches = list(task.batches(explicit, **settings))
            self._assert_rows(batches, expected_ids, positions, padding)
            assert [b.metadata["sample_ids"] for b in task.batches(explicit, **settings)] == [
                b.metadata["sample_ids"] for b in batches
            ]
        assert [request.sample_id for request in requests] == ids
        assert all("_is_padding" not in request.metadata for request in requests)

    @pytest.mark.parametrize("source", ["table", "materialized_requests"])
    def test_padding_in_a_later_chunk_is_unique_and_resumable(self, source) -> None:
        requests = self._requests(["a", "b", "c", "d", "e", "f", "g", "b#padding1", "i"])
        # The first chunk owns b/d/f. The final chunk owns b#padding1 and the
        # repeat of b; its locally generated b#padding1 must avoid the real row.
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = self._materialize(plan, requests, source)
            settings = {"batch_size": 3, "chunk_batches": 1, "rank": 1, "world_size": 2}
            batches = list(task.batches(**settings))
            self._assert_rows(
                batches,
                ["b", "d", "f", "b#padding1", "b#padding2"],
                [1, 3, 5, 7, 1],
                [False, False, False, False, True],
            )
            partial = task.batches(**settings)
            next(partial)
            resumed = task.batches(**settings)
            resumed.load_state_dict(partial.state_dict())
            assert [b.metadata["sample_ids"] for b in resumed] == [
                b.metadata["sample_ids"] for b in batches[1:]
            ]

    def test_padding_avoids_later_real_ids_and_other_padding_ids(self) -> None:
        requests = self._requests([1, "1", 1, "1", "1#padding2", "1#padding3"])
        flags = [False, False, True, True, False, False]
        marked = _mark_padding(requests, flags)
        # Integer 1 and string "1" are distinct real identities, but both use
        # the same string prefix for padding. Reserve all real rows first.
        assert [request.sample_id for request in marked] == [
            1,
            "1",
            "1#padding4",
            "1#padding5",
            "1#padding2",
            "1#padding3",
        ]
        for before, after, padding in zip(requests, marked, flags, strict=True):
            if padding:
                assert after == replace(
                    before,
                    sample_id=after.sample_id,
                    metadata={**before.metadata, "_is_padding": True},
                )
            else:
                assert after is before


class TestResume:
    def test_a_saved_position_continues_the_same_sequence(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            settings = {"split": "train", "batch_size": 2, "shuffle": True, "seed": 9, "epoch": 4}
            full = [b.metadata["sample_ids"] for b in task.batches(**settings)]
            first = task.batches(**settings)
            next(first)
            state = first.state_dict()
            resumed = task.batches(**settings)
            resumed.load_state_dict(state)
            rest = [b.metadata["sample_ids"] for b in resumed]
        assert rest == full[1:]

    def test_a_position_is_refused_under_another_configuration(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            first = task.batches(split="train", batch_size=2, seed=1)
            next(first)
            with pytest.raises(ValueError, match="different batch configuration"):
                task.batches(split="train", batch_size=2, seed=2).load_state_dict(
                    first.state_dict()
                )


class TestIdentity:
    def test_a_task_carries_its_fingerprints_and_output_schema(self) -> None:
        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            again = plan.materialize(_table(), fit_split="train")
            other = plan.materialize(_requests_for("train"), fit_split="train")
        assert set(task.fingerprints) == {
            "contract",
            "inputs",
            "dataset",
            "anchors",
            "input_state",
            "target_state",
            "output_schema",
            "task",
        }
        assert task.fingerprints == again.fingerprints
        assert task.fingerprints["anchors"] != other.fingerprints["anchors"]
        assert set(task.output_schema["inputs"]) == {"entity"}

    def test_the_artifact_records_the_task_identity(self, tmp_path) -> None:
        import json

        with prepare(_contract(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            task.to_artifact().save(tmp_path)
        written = json.loads((tmp_path / "task.json").read_text())
        assert written["fingerprints"] == dict(task.fingerprints)


class TestRefusals:
    def test_a_missing_prediction_time_is_refused(self) -> None:
        issues = check(_contract(), DATASET, InMemoryEngine(), anchors=_table(as_of_time=None))
        issue = next(item for item in issues if item.path == "anchors.as_of_time")
        assert "no column holding each anchor's prediction time" in issue.message

    def test_any_clocked_table_requires_a_prediction_time(self) -> None:
        assert _check_anchors(_contract(temporal=False), _table(as_of_time=None)) == [
            "anchors.as_of_time"
        ]

    def test_a_schema_without_clocks_refuses_one(self) -> None:
        static = DataLoaderSpec(
            DataSchema(
                tables=(next(table for table in SCHEMA.tables if table.name == "users"),),
                relations=(),
            ),
            _contract(temporal=False).sample,
            _contract(temporal=False).batch,
        )
        dataset = Dataset(tables={"users": InMemoryRows(rows=USERS)})
        issues = check(static, dataset, InMemoryEngine(), anchors=_table())
        assert [item.path for item in issues] == ["anchors.as_of_time"]
        assert check(static, dataset, InMemoryEngine(), anchors=_table(as_of_time=None)) == ()

    def test_materialize_refuses_before_the_table_is_read(self) -> None:
        engine = InMemoryEngine()
        read: list[AnchorTable] = []
        engine.materialize_anchors = read.append
        with prepare(_contract(), DATASET, engine) as plan:
            with pytest.raises(ValidationError) as error:
                plan.materialize(_table(as_of_time=None), fit_split="train")
        assert [item.path for item in error.value.issues] == ["anchors.as_of_time"]
        assert read == []

    def test_a_label_cannot_be_both_supplied_and_computed(self) -> None:
        contract = DataLoaderSpec(
            SCHEMA,
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                targets={
                    "y": Aggregate(
                        Rows(path=("user_events",), time=TimeWindow("occurred_at", 0, 10)),
                        "count",
                        dtype=DType("int64"),
                    )
                },
            ),
            BatchSpec(
                inputs={
                    "entity": Input(
                        EntitySource(),
                        Fields(fields=(Encode(field="score", encoding=Raw()),)),
                        Separate(),
                    )
                },
                targets={"y": LabelTarget(source="y", encoding=Raw())},
                batching=Batching(max_samples=4),
            ),
        )
        assert "anchors.targets" in _check_anchors(contract, _table(targets={"y": "label"}))

    def test_an_engine_refuses_a_format_it_does_not_read(self) -> None:
        assert "anchors.source" in _check_anchors(
            _contract(), _table(source=Parquet("/lake/anchors"))
        )

    def test_the_padding_marker_is_reserved(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _table(metadata=("_is_padding",))


class TestCandidates:
    """A candidate set is a nested column, not a second table to join."""

    def _contract_with_candidates(self) -> DataLoaderSpec:
        return DataLoaderSpec(
            SCHEMA,
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                entity_values={"score": Field(column="score")},
                candidate_sets={
                    "peers": CandidateSetSpec(
                        entity=EntitySpec("users", entity_type="user"),
                        values={"score": Field(column="score")},
                        targets={"clicked": ExternalTarget(dtype=DType("bool"))},
                    )
                },
            ),
            BatchSpec(
                inputs={
                    "entity": Input(
                        EntitySource(),
                        Fields(fields=(Encode(field="score", encoding=Raw()),)),
                        Separate(),
                    ),
                    "peers": Input(
                        CandidateSource("peers"),
                        Fields(fields=(Encode(field="score", encoding=Raw()),)),
                        Padded(),
                    ),
                },
                targets={
                    "clicked": CandidateTarget(input="peers", source="clicked", encoding=Raw())
                },
                batching=Batching(max_samples=4),
            ),
        )

    def _table(self) -> AnchorTable:
        rows = [
            {
                "sid": f"s{index}",
                "user_id": index,
                "at": 100,
                "split": "train",
                "peers": [
                    {"peer_id": 1 + index % 12, "label": True},
                    {"peer_id": 1 + (index + 3) % 12, "label": False},
                ],
            }
            for index in range(1, 5)
        ]
        return AnchorTable(
            source=InMemoryRows(rows=rows),
            entity_id="user_id",
            as_of_time="at",
            sample_id="sid",
            split="split",
            candidates={
                "peers": CandidateColumn(
                    column="peers", entity_id="peer_id", targets={"clicked": "label"}
                )
            },
        )

    def _requests(self) -> list[SampleRequest]:
        return [
            SampleRequest(
                row["sid"],
                row["user_id"],
                row["at"],
                split="train",
                candidates={
                    "peers": tuple(
                        CandidateRequest(
                            entry["peer_id"], targets=(TargetValue("clicked", entry["label"]),)
                        )
                        for entry in row["peers"]
                    )
                },
            )
            for row in self._table().source.rows
        ]

    def test_an_undeclared_candidate_set_is_refused(self) -> None:
        table = replace(
            self._table(),
            candidates={
                **self._table().candidates,
                "strangers": CandidateColumn(column="strangers", entity_id="peer_id"),
            },
        )
        assert _check_anchors(self._contract_with_candidates(), table) == [
            "anchors.candidates.strangers"
        ]

    def test_a_candidate_label_cannot_be_both_supplied_and_computed(self) -> None:
        contract = self._contract_with_candidates()
        peers = contract.sample.candidate_sets["peers"]
        computed = replace(
            peers,
            targets={
                "clicked": Aggregate(
                    Rows(path=("user_events",), time=TimeWindow("occurred_at", 0, 10)),
                    "count",
                    dtype=DType("int64"),
                )
            },
        )
        contract = replace(
            contract, sample=replace(contract.sample, candidate_sets={"peers": computed})
        )
        assert _check_anchors(contract, self._table()) == ["anchors.candidates.peers.targets"]

    def test_a_nested_column_matches_caller_supplied_candidates(self) -> None:
        contract = self._contract_with_candidates()
        requests = self._requests()
        with prepare(contract, DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(self._table(), fit_split="train")
            from_table = next(iter(task.batches(split="train", batch_size=4)))
            from_requests = next(iter(task.batches(requests, batch_size=4)))
        assert from_table.inputs["peers"]["score"]["values"].equal(
            from_requests.inputs["peers"]["score"]["values"]
        )
        assert from_table.targets["clicked"]["values"].equal(
            from_requests.targets["clicked"]["values"]
        )

    def test_candidate_order_is_the_column_order(self) -> None:
        contract = self._contract_with_candidates()
        with prepare(contract, DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(self._table(), fit_split="train")
            batch = next(iter(task.batches(batch_size=4)))
        assert batch.targets["clicked"]["values"][0].tolist() == [1.0, 0.0]

    def test_an_anchor_with_no_candidates_is_empty_not_an_error(self) -> None:
        table = self._table()
        rows = [dict(row) for row in table.source.rows]
        rows[0]["peers"] = None
        contract = self._contract_with_candidates()
        with prepare(contract, DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(
                replace(table, source=InMemoryRows(rows=rows)), fit_split="train"
            )
            batch = next(iter(task.batches(split="train", batch_size=4)))
        assert batch.inputs["peers"]["valid"][0].tolist() == [False, False]


STATIC_SCHEMA = DataSchema(
    tables=(
        TableSpec(
            "users",
            "user_id",
            {"user_id": DType("int64"), "score": DType("float64")},
            time_column=NON_TEMPORAL,
        ),
        TableSpec(
            "events",
            "event_id",
            {
                "event_id": DType("int64"),
                "user_id": DType("int64"),
                "occurred_at": DType("timestamp", unit="s", timezone="UTC"),
                "amount": DType("float64"),
            },
            time_column=NON_TEMPORAL,
        ),
    ),
    relations=(RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)
TIMED_COUNT = Aggregate(
    Rows(path=("user_events",), time=TimeWindow("occurred_at", None, 0)),
    "count",
    dtype=DType("int64"),
)


class TestTimeDeclaredAnywhereNeedsAnAnchorTime:
    """No table declares a clock, yet the contract still reads time somewhere."""

    def _static(self, sample: SampleSpec, inputs: dict[str, Input]) -> DataLoaderSpec:
        return DataLoaderSpec(
            STATIC_SCHEMA, sample, BatchSpec(inputs=inputs, batching=Batching(max_samples=4))
        )

    def test_a_timed_candidate_aggregate_requires_an_anchor_time(self) -> None:
        contract = self._static(
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                entity_values={"score": Field(column="score")},
                candidate_sets={
                    "peers": CandidateSetSpec(
                        entity=EntitySpec("users", entity_type="user"), values={"seen": TIMED_COUNT}
                    )
                },
            ),
            {
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field="score", encoding=Raw()),)),
                    Separate(),
                ),
                "peers": Input(
                    CandidateSource("peers"),
                    Fields(fields=(Encode(field="seen", encoding=Raw()),)),
                    Padded(),
                ),
            },
        )
        assert contract.requires_anchor_time
        engine = InMemoryEngine()
        read: list[AnchorTable] = []
        engine.materialize_anchors = read.append
        with prepare(contract, DATASET, engine) as plan:
            with pytest.raises(ValidationError) as error:
                plan.materialize(_table(as_of_time=None), fit_split="train")
        assert "anchors.as_of_time" in [item.path for item in error.value.issues]
        assert read == []

    def test_a_timed_comparison_at_the_root_requires_one(self) -> None:
        contract = self._static(
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                entity_values={
                    "active": Compare(TIMED_COUNT, "greater_than", 0, dtype=DType("bool"))
                },
            ),
            {
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field="active", encoding=Raw()),)),
                    Separate(),
                )
            },
        )
        assert contract.requires_anchor_time

    def test_an_explicit_time_column_on_a_root_field_requires_one(self) -> None:
        sample = SampleSpec(
            entity=EntitySpec("users", entity_type="user"),
            entity_values={"score": Field(column="score", time_column="score")},
        )
        contract = DataLoaderSpec(
            STATIC_SCHEMA,
            sample,
            BatchSpec(
                inputs={
                    "entity": Input(
                        EntitySource(),
                        Fields(fields=(Encode(field="score", encoding=Raw()),)),
                        Separate(),
                    )
                },
                batching=Batching(max_samples=4),
            ),
        )
        assert contract.requires_anchor_time

    def test_a_contract_with_no_time_anywhere_still_needs_none(self) -> None:
        contract = self._static(
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                entity_values={"score": Field(column="score")},
            ),
            {
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field="score", encoding=Raw()),)),
                    Separate(),
                )
            },
        )
        assert not contract.requires_anchor_time


class TestDistributedTokenBudgets:
    def _budgeted(self) -> DataLoaderSpec:
        return DataLoaderSpec(
            SCHEMA,
            SampleSpec(
                entity=EntitySpec("users", entity_type="user"),
                entity_values={"score": Field(column="score")},
                events={
                    "history": EventSpec(
                        rows=Rows(path=("user_events",)), fields={"amount": Field(column="amount")}
                    )
                },
            ),
            BatchSpec(
                inputs={
                    "entity": Input(
                        EntitySource(),
                        Fields(fields=(Encode(field="score", encoding=Raw()),)),
                        Separate(),
                    ),
                    "history": Input(
                        EventSource(("history",)),
                        Fields(fields=(Encode(field="amount", encoding=Raw()),)),
                        Padded(),
                    ),
                },
                batching=Batching(max_samples=2, max_tokens=5, length_input="history"),
            ),
        )

    def test_a_token_budget_is_refused_across_ranks(self) -> None:
        with prepare(self._budgeted(), DATASET, InMemoryEngine()) as plan:
            task = plan.materialize(_table(), fit_split="train")
            with pytest.raises(ValueError, match="max_tokens with world_size > 1"):
                task.batches(split="train", rank=0, world_size=2)
            with pytest.raises(ValueError, match="max_tokens with world_size > 1"):
                task.batches(_requests_for("train"), rank=1, world_size=2)
            assert list(task.batches(split="train"))


def _scored_contract() -> DataLoaderSpec:
    contract = _contract(temporal=False)
    return replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                "entity": Input(
                    EntitySource(),
                    Fields(fields=(Encode(field="score", encoding=ZScore()),)),
                    Separate(),
                )
            },
        ),
    )


def _fitted_mean(task) -> float:
    return task.output("entity").field(name="score").encoder.mean


def _mean(ids: list[int]) -> float:
    return sum(float(index) for index in ids) / len(ids)


class TestFitSelection:
    """Unassigned or held-out anchors never reach fitted state by default."""

    def _table_with_unassigned(self) -> AnchorTable:
        rows = [
            {**row, "split": None if row["user_id"] % 3 == 0 else row["split"]} for row in ANCHORS
        ]
        return _table(source=InMemoryRows(rows=rows))

    def _fit(self, table: AnchorTable, **selection: object):
        with prepare(_scored_contract(), DATASET, InMemoryEngine()) as plan:
            return plan.materialize(table, **selection)

    def test_a_table_with_a_split_column_requires_a_selector(self) -> None:
        with pytest.raises(ValueError, match="pass fit_split"):
            self._fit(self._table_with_unassigned())

    def test_a_named_split_excludes_unassigned_rows(self) -> None:
        task = self._fit(self._table_with_unassigned(), fit_split="train")
        assert _fitted_mean(task) == pytest.approx(_mean([i for i in range(1, 13) if i % 3]))

    def test_several_named_splits(self) -> None:
        table = _table()
        task = self._fit(table, fit_split={"train", "valid"})
        assert _fitted_mean(task) == pytest.approx(_mean(list(range(1, 13))))

    def test_unassigned_rows_only_when_asked_for(self) -> None:
        task = self._fit(self._table_with_unassigned(), fit_split={None})
        assert _fitted_mean(task) == pytest.approx(_mean([3, 6, 9, 12]))

    def test_every_anchor_only_when_asked_for(self) -> None:
        task = self._fit(self._table_with_unassigned(), fit_split=ALL_ANCHORS)
        assert _fitted_mean(task) == pytest.approx(_mean(list(range(1, 13))))

    def test_a_table_without_splits_fits_everything_and_refuses_a_name(self) -> None:
        table = _table(split=None)
        assert _fitted_mean(self._fit(table)) == pytest.approx(_mean(list(range(1, 13))))
        with pytest.raises(ValueError, match="declare no splits"):
            self._fit(table, fit_split="train")
        assert _fitted_mean(self._fit(table, fit_split=ALL_ANCHORS)) == pytest.approx(
            _mean(list(range(1, 13)))
        )

    def test_a_selector_must_name_something(self) -> None:
        with pytest.raises(ValueError, match="at least one split"):
            self._fit(_table(), fit_split=set())
