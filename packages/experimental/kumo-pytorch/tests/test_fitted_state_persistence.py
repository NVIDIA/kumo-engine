# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts, fitted state and artifacts cross process boundaries (G-ARTIFACT-001).

Pickling covers what a DataLoader worker or a multi-process job receives;
`Artifact.load` gives back the writing task's artifact, or refuses; the fit
record covers where the state came from, kept through reuse and outside every
fingerprint.
"""

from __future__ import annotations

import json
import pickle
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kumo.pytorch import api
from kumo.pytorch.assembly import ARTIFACT_DIGESTS, artifact_digests, pin_fitted_state
from kumo.pytorch.contract._frozen import FrozenMapping
from tests.reference_models.batch_definitions import BATCH_SPECS
from tests.reference_models.data_schema import SYNTHETIC_SCHEMA
from tests.reference_models.fixtures import SAMPLES, requests_for
from tests.reference_models.fixtures import _tables as _recipe_tables
from tests.test_diskgraph_backend import _fixture_rows, _requests, _schema
from tests.test_inference_lifecycle import (
    CONTRACTS,
    ENGINES,
    EPOCH,
    SEED,
    _dataset,
    _train,
    assert_same,
    assert_same_batches,
)

WORKERS = 2
_DELETE = object()
T = TypeVar("T")


def _recipe(name: str) -> tuple[api.DataLoaderSpec, api.Dataset, list[api.SampleRequest]]:
    contract = api.DataLoaderSpec(SYNTHETIC_SCHEMA, SAMPLES[name], BATCH_SPECS[name])
    dataset = api.Dataset({key: api.InMemoryRows(rows) for key, rows in _recipe_tables().items()})
    return contract, dataset, list(requests_for(name))


def _batches(
    contract: api.DataLoaderSpec,
    dataset: api.Dataset,
    engine: api.Engine,
    requests: list[api.SampleRequest],
    fitted: api.FittedState,
    *,
    inference: bool = False,
    output_format: str = "inspectable",
) -> list[api.Batch]:
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(requests, fitted=fitted, inference=inference) as task:
            return list(
                task.batches(batch_size=1, seed=SEED, epoch=EPOCH, output_format=output_format)
            )


def _redigest(artifact: Path) -> None:
    """Record the current files' digests, so only the consistency checks can refuse."""

    manifest = json.loads((artifact / "manifest.json").read_text())
    texts = {name: (artifact / name).read_text() for name in manifest[ARTIFACT_DIGESTS]}
    manifest[ARTIFACT_DIGESTS] = artifact_digests(texts)
    (artifact / "manifest.json").write_text(json.dumps(manifest))


def _round_trip(value: T) -> T:
    return pickle.loads(pickle.dumps(value))


def _assert_frozen(contract: api.DataLoaderSpec) -> None:
    tables = contract.schema.tables
    assert all(isinstance(table.columns, FrozenMapping) for table in tables)
    inputs = contract.batch.inputs
    assert isinstance(inputs, FrozenMapping)
    with pytest.raises(TypeError):
        inputs["extra"] = next(iter(inputs.values()))  # type: ignore[index]
    with pytest.raises(AttributeError):
        inputs._items = {}  # type: ignore[misc]


@pytest.mark.parametrize("name", sorted(BATCH_SPECS))
def test_every_fitted_state_kind_pickles_to_the_same_batches(tmp_path: Path, name: str) -> None:
    contract, dataset, requests = _recipe(name)
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as task:
            task.to_artifact().save(tmp_path)
            expected = list(task.batches(batch_size=2, seed=SEED))
            fitted = _round_trip(task.fitted_state)
            bound = task.bound_contract()
    assert fitted == task.fitted_state
    assert fitted.fingerprint == task.fitted_state.fingerprint
    assert fitted.fit == task.fitted_state.fit
    for value in (contract, bound):
        copy = _round_trip(value)
        assert copy == value and copy.fingerprint == value.fingerprint
        _assert_frozen(copy)

    artifact = api.Artifact.load(tmp_path)
    copy = _round_trip(artifact)
    assert copy.contract == artifact.contract
    assert copy.bound_contract == artifact.bound_contract
    assert copy.fitted_state == artifact.fitted_state
    assert copy.fitted_state.fit == artifact.fitted_state.fit
    assert (copy.fingerprints, copy.output_schema, copy.manifest) == (
        artifact.fingerprints,
        artifact.output_schema,
        artifact.manifest,
    )
    with api.prepare(copy.contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests, fitted=copy.fitted_state) as task:
            assert_same_batches(list(task.batches(batch_size=2, seed=SEED)), expected)
        with plan.materialize(requests, fitted=fitted) as task:
            assert_same_batches(list(task.batches(batch_size=2, seed=SEED)), expected)


def test_state_used_by_diskgraph_pickles_without_its_engine(tmp_path: Path) -> None:
    # DiskGraph attaches vocabulary lookups and packers to the state; they are not state.
    contract = CONTRACTS["temporal"](_schema())
    dataset, engine = _dataset("diskgraph", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            expected = list(task.batches(batch_size=1, seed=SEED, output_format="model"))
            fitted = _round_trip(task.fitted_state)
    assert fitted == task.fitted_state
    assert_same_batches(
        _batches(contract, dataset, engine, list(_requests()), fitted, output_format="model"),
        expected,
    )


class _ShardedInference(IterableDataset[api.Batch]):
    """One worker's share of batch inference, built from what the parent pickled."""

    def __init__(
        self,
        artifact: api.Artifact,
        dataset: api.Dataset,
        engine: api.Engine,
        requests: list[api.SampleRequest],
        output_format: str,
    ) -> None:
        self.artifact = artifact
        self.dataset = dataset
        self.engine = engine
        self.requests = requests
        self.output_format = output_format

    def __iter__(self) -> Iterator[api.Batch]:
        worker = get_worker_info()
        assert worker is not None, "built for DataLoader workers"
        yield from _batches(
            self.artifact.contract,
            self.dataset,
            self.engine,
            self.requests[worker.id :: worker.num_workers],
            self.artifact.fitted_state,
            inference=True,
            output_format=self.output_format,
        )


@pytest.mark.parametrize(
    ("engine_name", "output_format"),
    [("in_memory", "inspectable"), ("diskgraph", "model")],
)
def test_spawned_dataloader_workers_build_the_same_batches(
    tmp_path: Path, engine_name: str, output_format: str
) -> None:
    contract = CONTRACTS["pragma"](_schema())
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    artifact = api.Artifact.load(tmp_path / "artifact")
    requests = list(_requests())
    expected = [
        batch
        for worker in range(WORKERS)
        for batch in _batches(
            artifact.contract,
            dataset,
            engine,
            requests[worker::WORKERS],
            artifact.fitted_state,
            inference=True,
            output_format=output_format,
        )
    ]
    loader = DataLoader(
        _ShardedInference(artifact, dataset, engine, requests, output_format),
        batch_size=None,
        num_workers=WORKERS,
        multiprocessing_context="spawn",
    )
    received = {batch.metadata["sample_ids"][0]: batch for batch in loader}
    assert sorted(received) == sorted(request.sample_id for request in requests)
    for batch in expected:
        assert_same_batches([received[batch.metadata["sample_ids"][0]]], [batch])


@pytest.mark.parametrize("name", sorted(BATCH_SPECS))
@pytest.mark.parametrize("inference", [False, True])
def test_a_saved_artifact_loads_equal_to_the_task(
    tmp_path: Path, name: str, inference: bool
) -> None:
    contract, dataset, requests = _recipe(name)
    with api.prepare(contract, dataset, api.InMemoryEngine()) as plan:
        with plan.materialize(requests) as trained:
            fitted = trained.fitted_state
        with plan.materialize(requests, fitted=fitted, inference=inference) as task:
            task.to_artifact().save(tmp_path / "artifact")
            artifact = task.to_artifact()

    loaded = api.Artifact.load(tmp_path / "artifact")
    assert loaded == artifact
    assert loaded.fitted_state.fit == artifact.fitted_state.fit
    assert loaded != replace(artifact, fitted_state=artifact.fitted_state.inputs)
    assert loaded.fitted_state.inputs == artifact.fitted_state.inputs != loaded.fitted_state
    loaded.save(tmp_path / "again")
    for file in ("contract.json", "fitted-state.json", "task.json", "manifest.json"):
        assert (tmp_path / "again" / file).read_bytes() == (
            tmp_path / "artifact" / file
        ).read_bytes()


def _future_spend_encoders(state: api.FittedState) -> dict[str, Any]:
    return {key: value for key, value in state["encoders"].items() if "future_spend" in key}


@pytest.mark.parametrize("engine_name", ENGINES)
def test_an_artifact_input_half_is_reused_under_other_targets(
    tmp_path: Path, engine_name: str
) -> None:
    pretrain = CONTRACTS["temporal"](_schema())
    finetune = replace(
        pretrain,
        batch=replace(
            pretrain.batch,
            targets={"future_spend": api.LabelTarget("future_spend", api.ZScore())},
        ),
    )
    assert finetune.input_fingerprint == pretrain.input_fingerprint
    assert finetune.fingerprint != pretrain.fingerprint
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path / "pretrain")
    with api.prepare(pretrain, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            artifact = task.to_artifact()
            artifact.save(tmp_path / "artifact")
            pretrained = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
            pretrained_state = task.fitted_state

    loaded = api.Artifact.load(tmp_path / "artifact")
    assert loaded == artifact
    inputs = loaded.fitted_state.inputs
    dataset, engine = _dataset(engine_name, _fixture_rows(), tmp_path / "finetune")
    with api.prepare(finetune, dataset, engine) as plan:
        with plan.materialize(_requests(), fitted=inputs, fit_split="train") as task:
            tuned = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
            state = task.fitted_state
    assert state.inputs == inputs == pretrained_state.inputs
    assert sorted(state["targets"]) == ["future_spend"]
    assert sorted(pretrained_state["targets"]) == ["future_spend", "next_transaction"]
    assert _future_spend_encoders(state) != _future_spend_encoders(pretrained_state)
    assert state.fit["targets"]["contract"] == finetune.fingerprint
    assert state.fit["inputs"]["contract"] == pretrain.fingerprint
    for left, right in zip(tuned, pretrained, strict=True):
        assert_same(left.inputs, right.inputs, "inputs")
        assert left.targets.keys() == {"future_spend"}


def test_an_artifact_with_inconsistent_fitted_state_is_refused(tmp_path: Path) -> None:
    contract = CONTRACTS["pragma"](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            task.to_artifact().save(tmp_path / "artifact")

    def changed(name: str, text: str) -> Path:
        directory = tmp_path / "changed"
        shutil.rmtree(directory, ignore_errors=True)
        shutil.copytree(tmp_path / "artifact", directory)
        (directory / name).write_text(text)
        _redigest(directory)
        return directory

    other = replace(contract, batch=replace(contract.batch, targets={}))
    with pytest.raises(
        ValueError, match="fitted under a different contract: its contract fingerprint"
    ):
        api.Artifact.load(changed("contract.json", other.to_json() + "\n"))

    def refused(change: Callable[[Any], Any], match: str) -> None:
        written = change(json.loads((tmp_path / "artifact" / "fitted-state.json").read_text()))
        with pytest.raises(ValueError, match=match):
            api.Artifact.load(changed("fitted-state.json", json.dumps(written)))

    def edited(written: dict[str, Any], path: tuple[str, ...], value: object) -> dict[str, Any]:
        node = written
        for key in path[:-1]:
            node = node[key]
        if value is _DELETE:
            del node[path[-1]]
        else:
            node[path[-1]] = value
        return written

    refused(lambda w: edited(w, ("format_version",), 3), "format_version 3")
    refused(lambda w: edited(w, ("fitted_for",), _DELETE), "does not record the contract")
    refused(lambda w: edited(w, ("fit",), ["not", "an", "object"]), "malformed fit record")
    refused(lambda w: edited(w, ("lengths",), [1]), "malformed length summary")
    refused(lambda w: [w], "not a fitted-state object")
    refused(lambda w: edited(w, ("lengths",), None), "has no length summary")
    refused(
        lambda w: edited(w, ("lengths", "mean"), w["lengths"]["mean"] + 1.0),
        "length summary in .* is not the one manifest.json records",
    )
    refused(
        lambda w: edited(w, ("targets", "ghost"), {"bindings": [], "thresholds": {}, "groups": []}),
        r"holds target state for \['ghost'\], but this contract's targets need it for \[\]",
    )
    refused(
        lambda w: edited(w, ("targets",), _DELETE),
        "has no target state object",
    )
    refused(
        lambda w: edited(w, ("inputs", "history", "bindings"), None),
        "changed: .*'NoneType' object is not iterable",
    )

    def renumbered_category(written: dict[str, Any]) -> dict[str, Any]:
        entries = written["encoders"]['["shared","categories"]']["entries"]
        entries[0][1] += 1000
        return written

    refused(renumbered_category, "ID space 'pragma' does not number the IDs")
    refused(
        lambda w: edited(w, ("inputs", "ghost"), {"bindings": []}),
        "do not reproduce the written fitted state",
    )


def _with_test_anchor(requests: tuple[api.SampleRequest, ...]) -> list[api.SampleRequest]:
    held_out = api.SampleRequest("user-4-held-out", 4, datetime(2024, 12, 1), split="test")
    return [*requests, held_out]


def test_the_fit_record_says_where_state_was_fitted_and_survives_reuse(tmp_path: Path) -> None:
    contract = CONTRACTS["temporal"](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            task.to_artifact().save(tmp_path / "trained")
            record = {
                "contract": contract.fingerprint,
                "dataset": task.fingerprints["dataset"],
                "anchors": task.fingerprints["anchors"],
                "fit_split": ["train"],
            }
            inputs = {**record, "statistics": task.fitted_state.fit["inputs"]["statistics"]}
            assert task.fitted_state.fit == {"inputs": inputs, "targets": record}
            trained = task.fitted_state

        # more anchors, same training split: the same values from different anchors
        with plan.materialize(_with_test_anchor(_requests()), fit_split="train") as task:
            assert task.fitted_state == trained
            assert task.fitted_state.fingerprint == trained.fingerprint
            assert task.fitted_state.fit["inputs"]["anchors"] != record["anchors"]

    artifact = api.Artifact.load(tmp_path / "trained")
    written = json.loads((tmp_path / "trained" / "fitted-state.json").read_text())
    assert written["fit"] == artifact.fitted_state.fit == {"inputs": inputs, "targets": record}
    assert artifact.manifest["fitted_state_fingerprint"] == trained.fingerprint

    rows = _fixture_rows()
    rows["merchants"] = rows["merchants"][:-1]
    other, other_engine = _dataset("in_memory", rows, tmp_path)
    with api.prepare(artifact.contract, other, other_engine) as plan:
        with plan.materialize(
            _with_test_anchor(_requests()), fitted=artifact.fitted_state, inference=True
        ) as task:
            assert task.fingerprints["dataset"] != record["dataset"]
            assert task.fitted_state.fit == {"inputs": inputs, "targets": record}
            task.to_artifact().save(tmp_path / "scored")
        reloaded = api.Artifact.load(tmp_path / "scored").fitted_state
        assert reloaded.fit == {"inputs": inputs, "targets": record}
        assert reloaded.fingerprint == trained.fingerprint

        # reusing the input half only: the targets are fitted, and recorded, here
        with plan.materialize(
            _requests(), fitted=artifact.fitted_state.inputs, fit_split="train"
        ) as task:
            assert task.fitted_state.fit["inputs"] == inputs
            assert task.fitted_state.fit["targets"] == {
                **record,
                "dataset": task.fingerprints["dataset"],
            }


def test_weights_follow_the_fitted_values_not_the_fit_record(tmp_path: Path) -> None:
    contract = CONTRACTS["temporal"](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            weights: Mapping[str, torch.Tensor] = {"w": torch.ones(2)}
            api.save_weights(tmp_path / "model.pt", weights, task.fitted_state)
        with plan.materialize(_with_test_anchor(_requests()), fit_split="train") as task:
            assert task.fitted_state.fit != {}
            loaded = api.load_weights(tmp_path / "model.pt", task.fitted_state)
    assert torch.equal(loaded["w"], weights["w"])


@pytest.mark.parametrize("contract_name", sorted(CONTRACTS))
def test_a_loaded_state_gives_the_writing_task_batches_and_metadata(
    tmp_path: Path, contract_name: str
) -> None:
    # pragma sets Batching.length_input, so its length summary reaches batch metadata
    contract = CONTRACTS[contract_name](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            task.to_artifact().save(tmp_path / "artifact")
            expected = list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH))
            manifest = dict(task.manifest)
            bound = task.bound_contract()
        written = json.loads((tmp_path / "artifact" / "fitted-state.json").read_text())
        assert pin_fitted_state(contract, written) == bound
        loaded = api.Artifact.load(tmp_path / "artifact").fitted_state
        with plan.materialize(_requests(), fitted=loaded) as task:
            assert task.manifest == manifest
            assert_same_batches(list(task.batches(batch_size=1, seed=SEED, epoch=EPOCH)), expected)

    if written["targets"]:
        path = tmp_path / "artifact" / "fitted-state.json"
        path.write_text(json.dumps(dict(written, targets={})))
        _redigest(tmp_path / "artifact")
        with pytest.raises(ValueError, match="nothing is refitted to fill the gap"):
            api.Artifact.load(tmp_path / "artifact")


def test_an_artifact_with_a_changed_length_summary_is_refused(tmp_path: Path) -> None:
    contract = CONTRACTS["pragma"](_schema())
    dataset, engine = _dataset("in_memory", _fixture_rows(), tmp_path)
    _train(contract, dataset, engine, tmp_path / "artifact")
    path = tmp_path / "artifact" / "fitted-state.json"
    written = json.loads(path.read_text())
    written["lengths"]["mean"] += 1.0
    path.write_text(json.dumps(written))
    with pytest.raises(ValueError, match="fitted-state.json does not match the digest"):
        api.Artifact.load(tmp_path / "artifact")
    _redigest(tmp_path / "artifact")
    with pytest.raises(ValueError, match="length summary in .* is not the one manifest.json"):
        api.Artifact.load(tmp_path / "artifact")
