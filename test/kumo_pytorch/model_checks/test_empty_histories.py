"""Real empty/mixed AML histories through input checks, forward and backward."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import kumo.pytorch.api as r

from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.involvement import verify_inputs, verify_batch
from test.kumo_pytorch.masked_values import (
    original_tokens,
    verify_targets as verify_masked,
    verify_loss as masked_loss,
)
from test.kumo_pytorch.next_token import (
    verify_targets as verify_next,
    verify_loss as next_loss,
)
from test.kumo_pytorch.models.aml import build_model
from .fixtures import CUTOFF, VALUE_FIELDS, build_fixture
from .test_aml import CASES, assert_equal

T = int(CUTOFF.timestamp())


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=("diskgraph",))
def source(request, tmp_path_factory):
    fixture = build_fixture()
    # Existing, available roots with no touching transactions, not absent IDs.
    fixture.tables["accounts"].extend(
        dict(account_id=i, first_observed_time=CUTOFF - timedelta(days=50))
        for i in (5, 6)
    )
    reference = {
        "events": [
            dict(
                id=row["transaction_id"],
                time=row["event_time_seconds"],
                accounts={str(row["sender_id"]), str(row["receiver_id"])},
                sender=str(row["sender_id"]),
                receiver=str(row["receiver_id"]),
                values={name: row[name] for name in VALUE_FIELDS},
            )
            for row in fixture.tables["transactions"]
        ]
    }
    directory = tmp_path_factory.mktemp("aml-empty-histories")
    dataset = fixture.parquet_dataset(directory / "sources")
    engine = r.DiskGraphEngine(index=directory / "index", num_threads=1)
    return fixture, dataset, engine, reference, request.param


@pytest.mark.parametrize("mixed", (False, True), ids=("all-empty", "mixed"))
@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
def test_empty_history_inputs_and_training(source, selection, family, mixed):
    fixture, dataset, engine, reference, backend = source
    graph = family in {"graphsage", "combined"}
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    roots = (5, 1, 6) if mixed else (5, 6)
    requests = tuple(
        r.SampleRequest(
            f"empty-probe-{i}",
            root,
            CUTOFF,
            targets=(r.TargetValue("label", float(i % 2)),),
            split="test",
        )
        for i, root in enumerate(roots)
    )
    anchors = {
        row.sample_id: dict(
            sample_id=row.sample_id, entity_id=str(row.entity_id), as_of_time=T * 10**9
        )
        for row in (*fixture.training, *requests)
    }
    labels = {row.sample_id: row.targets[0].value for row in requests}

    def execution(rows):
        return tuple(
            replace(
                row,
                as_of_time=CUTOFF - timedelta(seconds=int(graph)),
                metadata={"prediction_time_ns": T * 10**9} if graph else {},
                targets=() if native else row.targets,
            )
            for row in rows
        )

    requests = execution(requests)
    contract = build_contract(fixture.schema, family, native=native, max_events=11)
    with r.prepare(contract, dataset, engine) as plan:
        with plan.materialize(
            execution(fixture.training), fit_split="train", encoder_cutoff=T - 1
        ) as task:
            fingerprint = task.fitted_state.fingerprint
            batches = list(
                task.batches(
                    requests,
                    batch_size=3,
                    seed=7,
                    output_format="model",
                )
            )
            assert len(batches) == 1
            batch = batches[0]
            assert list(batch.metadata["sample_ids"]) == [
                row.sample_id for row in requests
            ]
            assert not batch.metadata["is_padding"].any()
            identities = {str(i): i for i in range(1, 7)}
            state = task.fitted_state.to_dict()
            if family == "pragma":
                original = original_tokens(batch, anchors, reference, state, identities)
                verify_masked(batch, original)
                assert len(original) == (42 if mixed else 0)
            elif native:
                sequences = verify_next(batch, anchors, reference, state, identities)
                assert [len(row) for row in sequences] == (
                    [0, 76, 0] if mixed else [0, 0]
                )
            else:
                verify_inputs(
                    batch,
                    family,
                    anchors,
                    reference,
                    identities,
                    [anchors[row.sample_id] for row in fixture.training],
                )
            before = deepcopy(batch)
            model = build_model(task, selection, family, dim=8, seed=17)
            result = model.training_step(batch)
            assert torch.isfinite(result.logits).all() and torch.isfinite(result.loss)
            if family == "pragma":
                report = masked_loss(batch, result)
            elif native:
                report = next_loss(batch, result)
            else:
                report = verify_batch(batch, result, labels)
                assert report["eligible"] == len(roots), (
                    "Empty histories must retain their external labels"
                )
            if native and not mixed:
                assert result.weight == 0 and report["mean_loss"] is None
                assert result.loss.item() == 0
            else:
                assert result.weight > 0
            result.loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(
                torch.isfinite(gradient).all() for gradient in gradients
            )
            if native and not mixed:
                assert all(not gradient.any() for gradient in gradients)
                # The runner must skip optimizer.step for a zero-weight batch.
            else:
                assert any(gradient.any() for gradient in gradients)
            for section in ("inputs", "targets", "metadata"):
                assert_equal(getattr(batch, section), getattr(before, section))
            assert task.fitted_state.fingerprint == fingerprint
