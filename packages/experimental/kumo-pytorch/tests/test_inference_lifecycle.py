# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train, write, load in a fresh process, infer and serve (G-ARTIFACT-001,
G-INFERENCE-001, G-SERVING-001).

The guarantees' golden cases fix raw input values; these tests cover what the
golden format can't express: fresh processes, encoded batches, targets and
masking, and refusals.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import torch

from conformance.assertions import write_parquet
from conformance.cases import GoldenCase, discover_cases
from kumo.pytorch import api
from kumo.pytorch.assembly import ARTIFACT_DIGESTS, artifact_digests
from tests.reference_models.batch_definitions import BATCH_SPECS
from tests.reference_models.data_schema import SYNTHETIC_SCHEMA
from tests.reference_models.fixtures import SAMPLES, requests_for
from tests.reference_models.fixtures import _tables as _recipe_tables
from tests.test_diskgraph_backend import (
    _fixture_rows,
    _pragma_contract,
    _requests,
    _schema,
    _temporal_contract,
    _write_parquet,
)
from tests.test_docs_code import _blocks

ROOT = Path(__file__).parents[1]
Rows = dict[str, list[dict[str, object]]]
CONTRACTS: dict[str, Callable[[api.DataSchema], api.DataLoaderSpec]] = {
    "temporal": _temporal_contract,
    "pragma": _pragma_contract,
}
ENGINES = ("in_memory", "diskgraph")
SEED, EPOCH = 7, 2


def assert_same(actual: object, expected: object, path: str = "batch") -> None:
    """Exact equality through mappings, sequences and tensors (dtype included)."""

    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_same(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


def assert_same_batches(actual: list[api.Batch], expected: list[api.Batch]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert_same(left.inputs, right.inputs, "inputs")
        assert_same(left.targets, right.targets, "targets")
        assert_same(left.metadata, right.metadata, "metadata")


def _dataset(engine_name: str, rows: Rows, directory: Path) -> tuple[api.Dataset, api.Engine]:
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
        directory.mkdir(parents=True, exist_ok=True)
        dataset = _write_parquet(rows, directory / "parquet")
        engine = api.DiskGraphEngine(index=directory / "index", num_threads=1, num_shards=1)
        return dataset, engine
    return api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()}), (
        api.InMemoryEngine()
    )


def _train(contract: api.DataLoaderSpec, dataset: api.Dataset, engine: api.Engine, out: Path):
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            task.to_artifact().save(out)
            batches = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
            return task.fingerprints, task.fitted_state.to_dict(), batches


def _infer(
    artifact: api.Artifact,
    dataset: api.Dataset,
    engine: api.Engine,
    *,
    inference: bool = True,
    output_format: str = "inspectable",
) -> list[api.Batch]:
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(
            _requests(), fitted=artifact.fitted_state, inference=inference
        ) as task:
            return list(
                task.batches(batch_size=1, seed=SEED, epoch=EPOCH, output_format=output_format)
            )


def _fresh_process(
    artifact_dir: Path, contract: str, engine_name: str, directory: Path, out: Path
) -> None:
    """Runs in a child interpreter: load, materialize without fitting, save batches."""

    artifact = api.Artifact.load(artifact_dir)
    if engine_name == "diskgraph":
        dataset = api.Dataset(
            {
                table: api.Parquet(str(directory / "parquet" / f"{table}.parquet"))
                for table in _fixture_rows()
            }
        )
        engine: api.Engine = api.DiskGraphEngine(
            index=directory / "index", num_threads=1, num_shards=1
        )
    else:
        dataset, engine = _dataset(engine_name, _fixture_rows(), directory)
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fitted=artifact.fitted_state) as task:
            training = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
            fingerprints = dict(task.fingerprints)
    inference = _infer(artifact, dataset, engine)
    torch.save(
        {
            "fingerprints": fingerprints,
            "fitted": artifact.fitted_state.to_dict(),
            "training": training,
            "inference": inference,
        },
        out,
    )


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("contract_name", sorted(CONTRACTS))
def test_a_fresh_process_rebuilds_the_same_batches(
    tmp_path: Path, contract_name: str, engine_name: str
) -> None:
    contract = CONTRACTS[contract_name](_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    fingerprints, fitted, training = _train(contract, dataset, engine, tmp_path / "artifact")
    inference = _infer(api.Artifact.load(tmp_path / "artifact"), dataset, engine)

    out = tmp_path / "child.pt"
    code = (
        "from pathlib import Path; from tests.test_inference_lifecycle import _fresh_process; "
        f"_fresh_process(Path({str(tmp_path / 'artifact')!r}), {contract_name!r}, "
        f"{engine_name!r}, Path({str(tmp_path)!r}), Path({str(out)!r}))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    child = torch.load(out, weights_only=False)

    assert child["fingerprints"] == dict(fingerprints)
    assert child["fitted"] == json.loads(json.dumps(fitted))
    assert_same_batches(child["training"], training)
    assert_same_batches(child["inference"], inference)


@pytest.mark.parametrize("name", sorted(BATCH_SPECS))
def test_every_fitted_state_kind_survives_the_artifact(tmp_path: Path, name: str) -> None:
    # Vocabulary, Buckets and Text (BPE) encodings, shared ID spaces with field-name and
    # value tokens, composite tokens, matching groups and fitted thresholds.
    contract = api.DataLoaderSpec(SYNTHETIC_SCHEMA, SAMPLES[name], BATCH_SPECS[name])
    dataset = api.Dataset({key: api.InMemoryRows(rows) for key, rows in _recipe_tables().items()})
    requests = requests_for(name)
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path)
            expected = list(task.batches(batch_size=2, seed=SEED))
            state = task.fitted_state.to_dict()
            bound = task.bound_contract()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "contract.json",
        "fitted-state.json",
        "manifest.json",
        "task.json",
    ]
    artifact = api.Artifact.load(tmp_path)
    assert json.loads(json.dumps(state)) == artifact.fitted_state.to_dict()
    assert artifact.bound_contract == bound
    with api.prepare(artifact.contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state) as task:
            assert_same_batches(list(task.batches(batch_size=2, seed=SEED)), expected)
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            assert all(not batch.targets for batch in task.batches(batch_size=2, seed=SEED))


def _labelled_contract() -> tuple[api.DataLoaderSpec, Rows]:
    columns = {"user_id": api.DType("int64"), "country": api.DType("string")}
    schema = api.DataSchema(
        (api.TableSpec("users", "user_id", columns, time_column=api.NON_TEMPORAL),), ()
    )
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("users"),
            entity_values={"country": api.Field(column="country")},
            targets={"tier": api.ExternalTarget(dtype=api.DType("string"))},
        ),
        api.BatchSpec(
            {
                "user": api.Input(
                    api.EntitySource(),
                    api.Fields((api.Encode("country", api.Vocabulary()),), row=api.Columns()),
                    api.Separate(),
                )
            },
            targets={"tier": api.LabelTarget("tier", api.Vocabulary())},
            batching=api.Batching(max_samples=4),
        ),
    )
    rows: Rows = {"users": [{"user_id": i, "country": c} for i, c in ((1, "US"), (2, "FR"))]}
    return contract, rows


def _labelled(*tiers: str) -> list[api.SampleRequest]:
    return [
        api.SampleRequest(f"s{index}", 1 + index % 2, 10, targets=(api.TargetValue("tier", tier),))
        for index, tier in enumerate(tiers)
    ]


def _tier_ids(task: api.Task) -> dict[object, int]:
    return dict(task.target("tier").field(name="value").encoder.entries)


def test_a_reloaded_label_vocabulary_is_the_trained_one(tmp_path: Path) -> None:
    contract, rows = _labelled_contract()
    dataset = api.Dataset({"users": api.InMemoryRows(rows["users"])})
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(_labelled("gold", "gold", "silver")) as task:
            trained = _tier_ids(task)
            # binding leaves targets declared, so a fine-tune of the bound contract fits its own
            assert task.bound_contract().batch.targets["tier"].encoding.values is None
            task.to_artifact().save(tmp_path)
    artifact = api.Artifact.load(tmp_path)

    # new data whose labels include one the training data never had
    new_rows = [{"user_id": 1, "country": "DE"}, {"user_id": 2, "country": "US"}]
    with api.prepare(
        artifact.contract, api.Dataset({"users": api.InMemoryRows(new_rows)}), api.InMemoryEngine()
    ) as plan:
        with plan.materialize(
            _labelled("bronze", "bronze", "silver"), fitted=artifact.fitted_state
        ) as task:
            assert _tier_ids(task) == trained
            batch = next(task.batches())
    unknown = api.Vocabulary().unknown_id
    assert batch.targets["tier"]["values"].tolist() == [unknown, unknown, trained["silver"]]


@pytest.mark.parametrize("engine_name", ENGINES)
def test_inference_emits_no_targets_and_keys_every_output(tmp_path: Path, engine_name: str) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    _, _, training = _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fitted=artifact.fitted_state, inference=True) as task:
            assert task.inference
            assert task.output_schema["targets"] == {}
            inference = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
    assert all(training_batch.targets for training_batch in training)
    for batch, training_batch in zip(inference, training, strict=True):
        assert batch.targets == {}
        assert_same(batch.inputs, training_batch.inputs)
        assert_same(batch.metadata, training_batch.metadata)


@pytest.mark.parametrize("engine_name", ENGINES)
def test_inference_turns_masking_off(tmp_path: Path, engine_name: str) -> None:
    contract = _pragma_contract(_schema())
    unmasked = replace(contract, batch=replace(contract.batch, targets={}))
    assert unmasked.input_fingerprint == contract.input_fingerprint
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    _, _, training = _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    inference = _infer(artifact, dataset, engine)
    with api.prepare(unmasked, dataset, engine) as plan:
        with plan.materialize(
            _requests(), fitted=artifact.fitted_state.inputs, fit_split="train"
        ) as task:
            plain = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))

    corrupted = [batch.targets["masked_values"]["corruption"].any() for batch in training]
    assert any(corrupted)
    for batch, reference in zip(inference, plain, strict=True):
        assert batch.targets == {}
        assert_same(batch.inputs, reference.inputs)


def _with_future_rows(rows: Rows) -> Rows:
    cutoff = _requests()[0].as_of_time
    future = []
    for user_id in (2, 3):
        template = next(row for row in rows["card_transactions"] if row["user_id"] == user_id)
        for index, at in enumerate((cutoff + timedelta(seconds=1), cutoff + timedelta(days=1))):
            future.append(
                {
                    **template,
                    "card_transaction_id": 900_000 + user_id * 10 + index,
                    "occurred_at": at,
                    "amount": 99_999.0,
                    "channel": "store",
                    "description": "future row",
                }
            )
    return {**rows, "card_transactions": [*rows["card_transactions"], *future]}


@pytest.mark.parametrize("engine_name", ENGINES)
def test_inference_reads_no_row_after_the_cutoff(tmp_path: Path, engine_name: str) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path / "now")
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    future_dataset, future_engine = _dataset(
        engine_name, _with_future_rows(_fixture_rows()), tmp_path / "future"
    )

    before = _infer(artifact, dataset, engine, inference=False)
    after = _infer(artifact, future_dataset, future_engine, inference=False)
    # the future rows are real: training-mode labels read them
    assert not torch.equal(
        after[0].targets["future_spend"]["values"], before[0].targets["future_spend"]["values"]
    )
    assert_same_batches(
        _infer(artifact, future_dataset, future_engine), _infer(artifact, dataset, engine)
    )


def test_inference_never_asks_the_engine_for_a_target(tmp_path: Path) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    built: list[api.SampleSpec] = []
    with api.prepare(artifact.contract, dataset, engine) as plan:
        source = plan._source
        original = source.build

        def spy(spec: api.SampleSpec, requests: Any) -> Any:
            built.append(spec)
            return original(spec, requests)

        source.build = spy
        with plan.materialize(_requests(), fitted=artifact.fitted_state, inference=True) as task:
            list(task.batches(batch_size=2))
    # caller-supplied labels read no rows, so only their declarations remain
    assert built and all(
        isinstance(target, api.ExternalTarget) for spec in built for target in spec.targets.values()
    )


def test_inference_never_fits(tmp_path: Path) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with pytest.raises(ValueError, match="inference never fits"):
            plan.materialize(_requests(), inference=True)
        with plan.materialize(_requests(), fit_split="train") as task:
            fitted = task.fitted_state
        with pytest.raises(ValueError, match="nothing to fit in inference"):
            plan.materialize(_requests(), fitted=fitted, fit_split="train", inference=True)
        with pytest.raises(ValueError, match="nothing to fit in inference"):
            plan.materialize(_requests(), fitted=fitted.inputs, fit_split="train", inference=True)


def test_inference_scores_every_anchor_even_when_training_drops_the_last_batch(
    tmp_path: Path,
) -> None:
    case = _golden("C-SERVING-001")
    contract = replace(
        case.contract,
        batch=replace(case.contract.batch, batching=api.Batching(max_samples=2, drop_last=True)),
    )
    requests = list(case.requests)
    with api.prepare(contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            assert sum(batch.batch_size for batch in task.batches()) == 2
            task.to_artifact().save(tmp_path)
    artifact = api.Artifact.load(tmp_path)
    with api.prepare(artifact.contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            scored = [i for batch in task.batches() for i in batch.metadata["sample_ids"]]
    assert scored == [request.sample_id for request in requests]
    served = api.ServingAssembler(artifact).assemble(case.tables, requests, batch_size=2)
    assert [i for batch in served for i in batch.metadata["sample_ids"]] == scored


def test_diskgraph_model_output_inference_matches_inspectable(tmp_path: Path) -> None:
    contract = _pragma_contract(_schema())
    dataset, engine = _dataset("diskgraph", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    model = _infer(artifact, dataset, engine, output_format="model")
    inspectable = _infer(artifact, dataset, engine)
    for left, right in zip(model, inspectable, strict=True):
        assert left.targets == {}
        assert_same(left.inputs, right.inputs)


@pytest.mark.parametrize("output_format", ["inspectable", "model"])
@pytest.mark.parametrize("contract_name", sorted(CONTRACTS))
def test_diskgraph_inference_equals_reference_inference(
    tmp_path: Path, contract_name: str, output_format: str
) -> None:
    contract = CONTRACTS[contract_name](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    expected = {b.metadata["sample_ids"][0]: b for b in _infer(artifact, dataset, engine)}
    disk_dataset, disk_engine = _dataset("diskgraph", _fixture_rows(), tmp_path / "diskgraph")
    actual = _infer(artifact, disk_dataset, disk_engine, output_format=output_format)
    assert sorted(expected) == sorted(b.metadata["sample_ids"][0] for b in actual)
    for batch in actual:
        assert_same_batches([batch], [expected[batch.metadata["sample_ids"][0]]])


def _rows_for(users: set[int]) -> Rows:
    rows = _fixture_rows()
    return {
        "users": [row for row in rows["users"] if row["user_id"] in users],
        "accounts": [row for row in rows["accounts"] if row["user_id"] in users],
        "card_transactions": [row for row in rows["card_transactions"] if row["user_id"] in users],
        "merchants": rows["merchants"],
    }


@pytest.mark.parametrize("contract_name", sorted(CONTRACTS))
def test_serving_equals_batch_inference(tmp_path: Path, contract_name: str) -> None:
    contract = CONTRACTS[contract_name](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    with api.prepare(artifact.contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fitted=artifact.fitted_state, inference=True) as task:
            expected = list(task.batches(batch_size=1, seed=SEED))

    # one request per call, each with only its own user's rows; batches compare by sample ID
    # because length bucketing orders the batch path
    server = api.ServingAssembler(artifact)
    served = {
        request.sample_id: server.assemble(_rows_for({request.entity_id}), [request], seed=SEED)
        for request in _requests()
    }
    for batch in expected:
        assert_same_batches(served[batch.metadata["sample_ids"][0]], [batch])


def _pragma_without_text(schema: api.DataSchema) -> api.DataLoaderSpec:
    contract = _pragma_contract(schema)
    history = contract.batch.inputs["history"]
    fields = tuple(f for f in history.representation.fields if f.field != "description")
    masked = contract.batch.targets["masked_values"]
    return replace(
        contract,
        batch=replace(
            contract.batch,
            inputs={
                **contract.batch.inputs,
                "history": replace(
                    history, representation=replace(history.representation, fields=fields)
                ),
            },
            targets={"masked_values": replace(masked, fields=("amount", "channel"))},
        ),
    )


@pytest.mark.parametrize(
    ("contract_factory", "text"),
    [(_pragma_contract, True), (_pragma_without_text, False)],
    ids=["text", "no_text"],
)
def test_a_customer_without_history_is_scored_on_every_path(
    tmp_path: Path, contract_factory: Callable[[api.DataSchema], api.DataLoaderSpec], text: bool
) -> None:
    """A reloaded artifact scores customers with no events before the cutoff (none at
    all, or only later ones), while other customers' events fill the table: the
    history is empty and fully masked, the entity inputs are kept, and every path
    gives the reference's batch."""

    contract = contract_factory(_schema())
    _train(contract, *_dataset("in_memory", _fixture_rows(), tmp_path), tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    encoders = artifact.fitted_state.to_dict()["encoders"].values()
    assert any(encoder["kind"] == "text" for encoder in encoders) == text

    cutoff = _requests()[0].as_of_time
    rows = _fixture_rows()
    rows["users"] += [{**rows["users"][0], "user_id": user_id} for user_id in (4, 5)]
    later = {**rows["card_transactions"][0], "card_transaction_id": 900_001, "user_id": 5}
    rows["card_transactions"].append({**later, "occurred_at": cutoff + timedelta(seconds=1)})
    assert all(row["user_id"] != 4 for row in rows["card_transactions"])
    requests = [api.SampleRequest(f"user-{user_id}", user_id, cutoff) for user_id in (4, 5)]

    def infer(engine_name: str, output_format: str = "inspectable") -> dict[str, api.Batch]:
        dataset, engine = _dataset(engine_name, rows, tmp_path / f"{engine_name}-{output_format}")
        with api.prepare(artifact.contract, dataset, engine) as plan:
            with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
                batches = task.batches(batch_size=1, seed=SEED, output_format=output_format)
                return {batch.metadata["sample_ids"][0]: batch for batch in batches}

    reference = infer("in_memory")
    assert sorted(reference) == ["user-4", "user-5"]
    for batch in reference.values():
        assert batch.targets == {}
        history = batch.inputs["history"]
        assert history["value_ids"].numel() == 0
        assert history["sequence"]["event_indices"].numel() == 0
        for name in ("history", "history_calendar", "history_timing"):
            inputs = batch.inputs[name]
            valid = [inputs[key] for key in ("valid", "field_valid") if key in inputs]
            valid += [field["valid"] for field in inputs.values() if isinstance(field, Mapping)]
            assert valid and not any(mask.any() for mask in valid), name
        entity = batch.inputs["entity"]
        assert entity["valid"].all() and entity["value_ids"].shape == (1, 3)

    server = api.ServingAssembler(artifact)
    for request in requests:
        own = {
            name: [row for row in values if row.get("user_id") == request.entity_id]
            for name, values in rows.items()
            if name != "merchants"
        }
        served = server.assemble({**own, "merchants": rows["merchants"]}, [request], seed=SEED)
        assert_same_batches(served, [reference[request.sample_id]])
    for output_format in ("inspectable", "model"):
        actual = infer("diskgraph", output_format)
        assert actual.keys() == reference.keys()
        for sample_id, batch in actual.items():
            assert_same_batches([batch], [reference[sample_id]])


def test_serving_reads_arrow_tables(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    contract = _pragma_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    rows = _rows_for({2, 3})
    tables = {name: pa.Table.from_pylist(values) for name, values in rows.items() if values}
    tables["merchants"] = pa.Table.from_pylist(rows["merchants"])
    server = api.ServingAssembler(artifact)
    assert_same_batches(server.assemble(tables, _requests()), server.assemble(rows, _requests()))


def test_serving_refuses_rows_it_would_have_to_guess(tmp_path: Path) -> None:
    contract = _pragma_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    server = api.ServingAssembler(api.Artifact.load(tmp_path / "artifact"))
    rows = _rows_for({2})
    request = _requests()[:1]
    spec = "ServingAssembler.rows_spec"
    with pytest.raises(ValueError, match=r"required tables \['card_transactions'\].*" + spec):
        server.assemble({k: v for k, v in rows.items() if k != "card_transactions"}, request)
    with pytest.raises(ValueError, match=r"\['devices'\], which .* does not declare.*" + spec):
        server.assemble({**rows, "devices": []}, request)
    broken = [{k: v for k, v in row.items() if k != "amount"} for row in rows["card_transactions"]]
    with pytest.raises(ValueError, match=r"rows\['card_transactions'\]\[0\] lacks .*'amount'"):
        server.assemble({**rows, "card_transactions": broken}, request)
    with pytest.raises(ValueError, match="at least one request"):
        server.assemble(rows, [])


def test_state_is_refused_under_a_different_schema_or_contract(tmp_path: Path) -> None:
    contract = _pragma_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")

    tables = [
        replace(table, columns={**table.columns, "nickname": api.DType("string")})
        if table.name == "users"
        else table
        for table in contract.schema.tables
    ]
    other_schema = replace(contract, schema=replace(contract.schema, tables=tuple(tables)))
    rows = _fixture_rows()
    rows["users"] = [{**row, "nickname": None} for row in rows["users"]]
    with api.prepare(other_schema, *_dataset("in_memory", rows, tmp_path)) as plan:
        with pytest.raises(ValueError, match="fitted for different inputs"):
            plan.materialize(_requests(), fitted=artifact.fitted_state)

    other_targets = replace(contract, batch=replace(contract.batch, targets={}))
    with api.prepare(other_targets, dataset, engine) as plan:
        with pytest.raises(ValueError, match="belongs to a different contract"):
            plan.materialize(_requests(), fitted=artifact.fitted_state)


def test_weights_are_refused_under_other_fitted_state(tmp_path: Path) -> None:
    contract = _pragma_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    weights = {"embedding.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2)}
    api.save_weights(tmp_path / "model.pt", weights, artifact.fitted_state)
    assert_same(api.load_weights(tmp_path / "model.pt", artifact.fitted_state), weights)

    refit_rows = _fixture_rows()
    refit_rows["users"] = [{**row, "country": "ZZ"} for row in refit_rows["users"]]
    _train(contract, *_dataset("in_memory", refit_rows, tmp_path), tmp_path / "refit")
    refit = api.Artifact.load(tmp_path / "refit")
    assert refit.fitted_state.contract_fingerprint == artifact.fitted_state.contract_fingerprint
    with pytest.raises(ValueError, match="fitted_state_fingerprint"):
        api.load_weights(tmp_path / "model.pt", refit.fitted_state)

    other = replace(contract, batch=replace(contract.batch, targets={}))
    _train(other, dataset, engine, tmp_path / "other")
    with pytest.raises(ValueError, match="contract_fingerprint"):
        api.load_weights(tmp_path / "model.pt", api.Artifact.load(tmp_path / "other").fitted_state)


def test_an_inconsistent_artifact_is_refused(tmp_path: Path) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")

    def tampered(name: str, change: Callable[[str], str], *, redigest: bool = False) -> Path:
        copy = tmp_path / f"tampered-{len(list(tmp_path.glob('tampered-*')))}"
        copy.mkdir()
        texts = {path.name: path.read_text() for path in (tmp_path / "artifact").iterdir()}
        texts[name] = change(texts[name])
        if redigest:
            manifest = json.loads(texts["manifest.json"])
            manifest[ARTIFACT_DIGESTS] = artifact_digests(texts)
            texts["manifest.json"] = json.dumps(manifest)
        for file, text in texts.items():
            (copy / file).write_text(text)
        return copy

    other = replace(contract, batch=replace(contract.batch, targets={}))
    for name, change in (
        ("contract.json", lambda _text: other.to_json()),
        ("fitted-state.json", lambda text: text.replace("C021", "C099")),
        ("task.json", _grow_output_vocabulary),
        ("manifest.json", _other_contract_fingerprint),
    ):
        with pytest.raises(ValueError, match=f"{name} does not match the digest manifest.json"):
            api.Artifact.load(tampered(name, change))
    # Digests that agree don't make files that disagree with the contract acceptable.
    for name, change, field in (
        ("task.json", _other_task_contract, "fingerprints.contract"),
        ("task.json", _grow_output_vocabulary, "output_schema.inputs"),
        ("manifest.json", _other_contract_fingerprint, "contract_fingerprint"),
    ):
        with pytest.raises(ValueError, match=rf"{name} field {re.escape(field)} is not what"):
            api.Artifact.load(tampered(name, change, redigest=True))
    with pytest.raises(ValueError, match="do not reproduce the written fitted state"):
        api.Artifact.load(tampered("fitted-state.json", _ghost_input, redigest=True))
    (tmp_path / "artifact" / "fitted-state.json").unlink()
    with pytest.raises(FileNotFoundError, match="fitted-state.json"):
        api.Artifact.load(tmp_path / "artifact")


def _grow_output_vocabulary(text: str) -> str:
    task = json.loads(text)
    task["output_schema"]["inputs"]["entity"]["country"]["vocabulary_size"] += 1
    return json.dumps(task)


def _other_task_contract(text: str) -> str:
    task = json.loads(text)
    task["fingerprints"]["contract"] = "0" * 64
    return json.dumps(task)


def _other_contract_fingerprint(text: str) -> str:
    manifest = json.loads(text)
    manifest["contract_fingerprint"] = "0" * 64
    return json.dumps(manifest)


def _ghost_input(text: str) -> str:
    state = json.loads(text)
    state["inputs"]["ghost"] = {"bindings": []}
    return json.dumps(state)


def _golden(case_id: str) -> GoldenCase:
    return discover_cases()[case_id]


def _golden_dataset(case: GoldenCase) -> api.Dataset:
    return api.Dataset({name: api.InMemoryRows(rows) for name, rows in case.tables.items()})


def _golden_values(
    batches: list[api.Batch], fitted: api.FittedState
) -> dict[str, dict[str, object]]:
    """Decode each sample's entity columns: raw values as they are, vocabulary IDs to keys."""

    state = fitted.to_dict()
    keys = {
        name: {index: value for value, index in encoder["entries"]}
        for name, identity, _ in state["inputs"]["entity"]["bindings"]
        for encoder in (state["encoders"][identity],)
        if encoder["kind"] == "vocabulary"
    }
    values: dict[str, dict[str, object]] = {}
    for batch in batches:
        for index, sample_id in enumerate(batch.metadata["sample_ids"]):
            values[sample_id] = {
                name: keys[name][value] if name in keys else value
                for name, field in batch.inputs["entity"].items()
                for value in (field["values"][index].item(),)
            }
    return values


def _expected(case: GoldenCase) -> dict[str, dict[str, object]]:
    return {sample: answer["entity_values"] for sample, answer in case.expected.items()}


def test_golden_artifact_case_reloads_in_a_fresh_process(tmp_path: Path) -> None:
    case = _golden("C-ARTIFACT-001")
    requests = list(case.requests)
    with api.prepare(case.contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path / "artifact")
    code = (
        "import json, sys; from pathlib import Path; from kumo.pytorch import api; "
        "from tests.test_inference_lifecycle import _golden, _golden_dataset, "
        "_golden_values; "
        "case = _golden('C-ARTIFACT-001'); "
        f"artifact = api.Artifact.load(Path({str(tmp_path / 'artifact')!r})); "
        "plan = api.prepare(artifact.contract, _golden_dataset(case), api.InMemoryEngine()); "
        "task = plan.materialize(list(case.requests), fitted=artifact.fitted_state); "
        "batches = list(task.batches(batch_size=8)); "
        "print(json.dumps(_golden_values(batches, artifact.fitted_state)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == _expected(case)


def test_golden_inference_case_answers_in_inference_mode(tmp_path: Path) -> None:
    case = _golden("C-INFERENCE-001")
    requests = list(case.requests)
    future = api.Aggregate(
        api.Rows(("user_events",), api.TimeWindow("at", 0, 100)), "count", dtype=api.DType("int64")
    )
    contract = replace(
        case.contract,
        sample=replace(case.contract.sample, targets={"future_events": future}),
        batch=replace(
            case.contract.batch,
            targets={"future_events": api.LabelTarget("future_events", api.Raw())},
        ),
    )
    with api.prepare(contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path)
    artifact = api.Artifact.load(tmp_path)
    with api.prepare(artifact.contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=artifact.fitted_state, inference=True) as task:
            batches = list(task.batches(batch_size=8))
    assert all(not batch.targets for batch in batches)
    assert _golden_values(batches, artifact.fitted_state) == _expected(case)


def test_golden_serving_case_answers_from_supplied_rows(tmp_path: Path) -> None:
    case = _golden("C-SERVING-001")
    requests = list(case.requests)
    with api.prepare(case.contract, _golden_dataset(case), api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path)
    artifact = api.Artifact.load(tmp_path)
    served = {}
    for request in requests:
        rows = {
            "users": [row for row in case.tables["users"] if row["user_id"] == request.entity_id],
            "events": [row for row in case.tables["events"] if row["user_id"] == request.entity_id],
        }
        batches = api.ServingAssembler(artifact).assemble(rows, [request])
        served.update(_golden_values(batches, artifact.fitted_state))
    assert served == _expected(case)


def test_the_lifecycle_example_runs_in_separate_processes(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.lifecycle_app", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "serving equals batch inference: True" in result.stdout
    assert (tmp_path / "predictions.json").is_file()


# A label richer than the inputs: DiskGraph runs the inputs, not the label.
STAMP = api.DType("timestamp", unit="s", timezone="UTC")
LABEL_SCHEMA = api.DataSchema(
    tables=(
        api.TableSpec(
            "users", "user_id", {"user_id": api.DType("int64")}, time_column=api.NON_TEMPORAL
        ),
        api.TableSpec(
            "sessions",
            "session_id",
            {"session_id": api.DType("int64"), "user_id": api.DType("int64"), "started_at": STAMP},
            time_column="started_at",
        ),
        api.TableSpec(
            "events",
            "event_id",
            {
                "event_id": api.DType("int64"),
                "user_id": api.DType("int64"),
                "session_id": api.DType("int64"),
                "at": STAMP,
                "amount": api.DType("decimal128", precision=10, scale=2),
            },
            time_column="at",
        ),
    ),
    relations=(
        api.RelationSpec("user_events", "users", "user_id", "events", "user_id"),
        api.RelationSpec("user_sessions", "users", "user_id", "sessions", "user_id"),
        api.RelationSpec("session_events", "sessions", "session_id", "events", "session_id"),
    ),
)
LABEL_ROWS: Rows = {
    "users": [{"user_id": 1}, {"user_id": 2}],
    "sessions": [
        {"session_id": 1, "user_id": 1, "started_at": 2},
        {"session_id": 2, "user_id": 1, "started_at": 14},
        {"session_id": 3, "user_id": 2, "started_at": 11},
    ],
    "events": [
        {"event_id": 1, "user_id": 1, "session_id": 1, "at": 3, "amount": Decimal("1.50")},
        {"event_id": 2, "user_id": 1, "session_id": 1, "at": 8, "amount": Decimal("2.25")},
        {"event_id": 3, "user_id": 1, "session_id": 1, "at": 10, "amount": Decimal("4.00")},
        {"event_id": 4, "user_id": 1, "session_id": 2, "at": 15, "amount": Decimal("8.10")},
        {"event_id": 5, "user_id": 2, "session_id": 3, "at": 12, "amount": Decimal("0.75")},
        {"event_id": 6, "user_id": 2, "session_id": 3, "at": 30, "amount": None},
    ],
}
LABEL_REQUESTS = (
    api.SampleRequest("user-1-at-10", 1, 10),
    api.SampleRequest("user-1-at-16", 1, 16),
    api.SampleRequest("user-2-at-10", 2, 10),
)
_AFTER = api.TimeWindow("at", 0, 100)
_COUNT = api.DType("int64")
#: label -> (target, what DiskGraph says when asked to train it)
LABELS: dict[str, tuple[api.Aggregate, str]] = {
    "row_filter": (
        api.Aggregate(
            api.Rows(("user_events",), _AFTER, filters=(api.RowFilter("amount", "not_null"),)),
            "count",
            dtype=_COUNT,
        ),
        "does not support row filters",
    ),
    "timed_intermediate_path": (
        api.Aggregate(api.Rows(("user_sessions", "session_events"), _AFTER), "count", dtype=_COUNT),
        "cannot time-filter intermediate table 'sessions'",
    ),
    "decimal_operand": (
        api.Aggregate(api.Rows(("user_events",), _AFTER), "max", column="amount"),
        "does not support decimal128 operands",
    ),
}


def _label_contract(
    target: api.Aggregate, entity_values: Mapping[str, Any] | None = None
) -> api.DataLoaderSpec:
    history = api.Rows(("user_events",), api.TimeWindow("at", stop=0))
    values = {
        "event_count": api.Aggregate(history, "count", dtype=_COUNT),
        **(entity_values or {}),
    }
    return api.DataLoaderSpec(
        schema=LABEL_SCHEMA,
        sample=api.SampleSpec(
            entity=api.EntitySpec("users", entity_type="user"),
            entity_values=values,
            targets={"label": target},
        ),
        batch=api.BatchSpec(
            inputs={
                "entity": api.Input(
                    api.EntitySource(),
                    api.Fields(
                        tuple(api.Encode(name, api.Raw()) for name in values), row=api.Columns()
                    ),
                    api.Separate(),
                )
            },
            targets={"label": api.LabelTarget("label", api.Raw())},
        ),
    )


def _label_parquet(directory: Path) -> tuple[api.Dataset, api.DiskGraphEngine]:
    pytest.importorskip("diskgraph")
    directory.mkdir(parents=True)
    tables = {table.name: table for table in LABEL_SCHEMA.tables}
    for name, rows in LABEL_ROWS.items():
        write_parquet(tables[name], rows, directory / f"{name}.parquet")
    dataset = api.Dataset(
        {name: api.Parquet(str(directory / f"{name}.parquet")) for name in LABEL_ROWS}
    )
    return dataset, api.DiskGraphEngine(index=directory / "index", num_threads=1, num_shards=1)


def _trained_label_artifact(contract: api.DataLoaderSpec, out: Path) -> api.Artifact:
    memory = api.Dataset({name: api.InMemoryRows(rows) for name, rows in LABEL_ROWS.items()})
    with api.prepare(contract, memory, api.InMemoryEngine()) as plan:
        with plan.materialize(list(LABEL_REQUESTS)) as task:
            assert next(task.batches(batch_size=8)).targets["label"]
            task.to_artifact().save(out)
    return api.Artifact.load(out)


@pytest.mark.parametrize("label", sorted(LABELS))
def test_diskgraph_scores_a_contract_whose_label_it_cannot_build(
    tmp_path: Path, label: str
) -> None:
    target, refusal = LABELS[label]
    artifact = _trained_label_artifact(_label_contract(target), tmp_path / "artifact")
    memory = api.Dataset({name: api.InMemoryRows(rows) for name, rows in LABEL_ROWS.items()})
    with api.prepare(artifact.contract, memory, api.InMemoryEngine(), inference=True) as plan:
        with plan.materialize(list(LABEL_REQUESTS), fitted=artifact.fitted_state) as task:
            expected = list(task.batches(batch_size=8))
    dataset, engine = _label_parquet(tmp_path / "diskgraph")

    # training still refuses at check(), before any data is opened
    assert any(refusal in issue.message for issue in api.check(artifact.contract, dataset, engine))
    with pytest.raises(api.ValidationError, match=refusal):
        api.prepare(artifact.contract, dataset, engine)
    assert api.check(artifact.contract, dataset, engine, inference=True) == ()
    with api.prepare(artifact.contract, dataset, engine, inference=True) as plan:
        assert plan.inference and plan.contract.fingerprint == artifact.contract.fingerprint
        with plan.materialize(list(LABEL_REQUESTS), fitted=artifact.fitted_state) as task:
            assert task.inference
            for output_format in ("inspectable", "model"):
                actual = list(task.batches(batch_size=8, output_format=output_format))
                assert_same_batches(actual, expected)


def test_an_inference_plan_materializes_only_inference(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    artifact = _trained_label_artifact(_label_contract(LABELS["row_filter"][0]), tmp_path / "a")
    dataset, engine = _label_parquet(tmp_path / "diskgraph")
    anchors_path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": [request.sample_id for request in LABEL_REQUESTS],
                "user_id": [request.entity_id for request in LABEL_REQUESTS],
                "at": pa.array(
                    [request.as_of_time for request in LABEL_REQUESTS], pa.timestamp("s", "UTC")
                ),
            }
        ),
        anchors_path,
    )
    anchors = api.AnchorTable(
        api.Parquet(str(anchors_path)), entity_id="user_id", as_of_time="at", sample_id="sample_id"
    )
    with api.prepare(artifact.contract, dataset, engine, inference=True) as plan:
        for training in (list(LABEL_REQUESTS), anchors):
            with pytest.raises(ValueError, match="prepared for inference"):
                plan.materialize(training, inference=False)
            with pytest.raises(ValueError, match="inference never fits"):
                plan.materialize(training)
        # the anchor table is checked under the plan's mode
        with plan.materialize(anchors, fitted=artifact.fitted_state) as task:
            from_table = list(task.batches(batch_size=8))
        with plan.materialize(list(LABEL_REQUESTS), fitted=artifact.fitted_state) as task:
            from_requests = list(task.batches(batch_size=8))
    assert [batch.metadata["sample_ids"] for batch in from_table] == [
        tuple(request.sample_id for request in LABEL_REQUESTS)
    ]
    for left, right in zip(from_table, from_requests, strict=True):
        assert left.targets == {}
        assert_same(left.inputs, right.inputs)


def test_an_inference_plan_still_refuses_an_input_the_engine_cannot_run(tmp_path: Path) -> None:
    filtered = api.Aggregate(
        api.Rows(
            ("user_events",),
            api.TimeWindow("at", stop=0),
            filters=(api.RowFilter("amount", "not_null"),),
        ),
        "count",
        dtype=_COUNT,
    )
    contract = _label_contract(LABELS["row_filter"][0], {"priced_events": filtered})
    dataset, engine = _label_parquet(tmp_path / "diskgraph")
    issues = api.check(contract, dataset, engine, inference=True)
    assert [(issue.path, issue.code) for issue in issues] == [
        ("sample.entity_values.priced_events.source.filters", "DG-ROW-FILTER")
    ]
    with pytest.raises(api.ValidationError, match="priced_events"):
        api.prepare(contract, dataset, engine, inference=True)


def test_an_inference_check_still_refuses_a_malformed_target() -> None:
    broken = api.Aggregate(api.Rows(("user_events",), _AFTER), "sum", column="no_such_column")
    contract = _label_contract(broken)
    memory = api.Dataset({name: api.InMemoryRows(rows) for name, rows in LABEL_ROWS.items()})
    assert api.check(contract, memory, api.InMemoryEngine(), inference=True)


def _infer_with(
    contract: api.DataLoaderSpec,
    fitted: api.FittedState,
    dataset: api.Dataset,
    engine: api.Engine,
) -> tuple[list[api.Batch], api.Task]:
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fitted=fitted, inference=True) as task:
            return list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH)), task


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("contract_name", sorted(CONTRACTS))
def test_inference_from_the_input_half_equals_inference_from_the_whole_state(
    tmp_path: Path, contract_name: str, engine_name: str
) -> None:
    contract = CONTRACTS[contract_name](_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    state = artifact.fitted_state
    complete, _ = _infer_with(artifact.contract, state, dataset, engine)
    input_only, task = _infer_with(artifact.contract, state.inputs, dataset, engine)

    fitted_targets = bool(task.manifest["targets"]) or bool(state.to_dict()["targets"])
    assert fitted_targets == (contract_name == "temporal")
    for left, right in zip(input_only, complete, strict=True):
        assert left.targets == {}
        assert_same(left.inputs, right.inputs)
        if fitted_targets:
            # the batch names the state that encoded it: the input half alone
            assert left.metadata["artifact_fingerprint"] == task.manifest["artifact_fingerprint"]
            assert left.metadata["artifact_fingerprint"] != right.metadata["artifact_fingerprint"]
            left = replace(
                left,
                metadata={**left.metadata, "artifact_fingerprint": None},
            )
            right = replace(right, metadata={**right.metadata, "artifact_fingerprint": None})
        assert_same(left.metadata, right.metadata)
    assert not task.fitted_state.has_targets
    assert task.fitted_state == state.inputs
    assert task.fitted_state.fingerprint == state.inputs.fingerprint
    assert task.output_schema == artifact.output_schema | {"targets": {}}
    with pytest.raises(ValueError, match="only the input half"):
        task.to_artifact()
    with pytest.raises(ValueError, match="complete fitted state"):
        api.ServingAssembler(replace(artifact, fitted_state=state.inputs))


def test_inference_from_the_input_half_reads_no_target_row(tmp_path: Path) -> None:
    contract = _temporal_contract(_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    built: list[api.SampleSpec] = []
    with api.prepare(artifact.contract, dataset, engine) as plan:
        source = plan._source
        original = source.build

        def spy(spec: api.SampleSpec, requests: Any) -> Any:
            built.append(spec)
            return original(spec, requests)

        source.build = spy
        with plan.materialize(
            _requests(), fitted=artifact.fitted_state.inputs, inference=True
        ) as task:
            list(task.batches(batch_size=2))
    assert built and all(
        isinstance(target, api.ExternalTarget) for spec in built for target in spec.targets.values()
    )


@pytest.mark.parametrize("engine_name", ENGINES)
def test_a_pretrained_input_half_embeds_under_a_contract_without_targets(
    tmp_path: Path, engine_name: str
) -> None:
    pretrain = _pragma_contract(_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    _train(pretrain, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    embed = replace(artifact.contract, batch=replace(artifact.contract.batch, targets={}))
    assert embed.input_fingerprint == artifact.contract.input_fingerprint

    expected, _ = _infer_with(artifact.contract, artifact.fitted_state, dataset, engine)
    actual, task = _infer_with(embed, artifact.fitted_state.inputs, dataset, engine)
    assert task.fitted_state == artifact.fitted_state.inputs
    for left, right in zip(actual, expected, strict=True):
        assert left.targets == {}
        assert_same(left.inputs, right.inputs)


def test_the_guide_shows_the_lifecycle_app_definitions() -> None:
    page = ROOT / "docs" / "06-inference-and-serving.md"
    (shop,) = [block for block in _blocks(page) if block.markers.get("file") == "shop.py"]
    definitions = shop.code[shop.code.index("DAY = 86_400") :]
    assert definitions in (ROOT / "tests" / "lifecycle_app.py").read_text()
