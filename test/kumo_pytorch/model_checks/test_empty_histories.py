# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real empty/mixed AML histories through input checks, forward and backward."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.involvement import verify_batch, verify_inputs
from test.kumo_pytorch.masked_values import original_tokens
from test.kumo_pytorch.masked_values import verify_loss as masked_loss
from test.kumo_pytorch.masked_values import verify_targets as verify_masked
from test.kumo_pytorch.models import build_model
from test.kumo_pytorch.next_token import verify_loss as next_loss
from test.kumo_pytorch.next_token import verify_targets as verify_next

import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import (
    CUTOFF,
    VALUE_FIELDS,
    build_fixture,
    probe_anchor_table,
    sweep_cases,
    sweep_contract,
    sweep_native,
    sweep_requests,
    verify_sweep_targets,
)
from .test_aml import assert_equal

CASES = sweep_cases()

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
    native = sweep_native(selection, family)
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

    requests = sweep_requests(execution(requests), selection, family)
    fit_requests = sweep_requests(execution(fixture.training), selection, family)
    contract = sweep_contract(fixture.schema, selection, family, max_events=11)
    with r.prepare(contract, dataset, engine) as plan:
        with (
            plan.materialize(
                probe_anchor_table(plan, fit_requests),
                fit_split="train",
                encoder_cutoff=T - 1,
            ) as fitted,
            plan.materialize(
                probe_anchor_table(plan, requests), fitted=fitted.fitted_state
            ) as task,
        ):
            fingerprint = task.fitted_state.fingerprint
            batches = list(
                task.batches(
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
            from test.kumo_pytorch.lifecycle import finite_outputs

            assert finite_outputs(result.logits) and torch.isfinite(result.loss)
            if family == "pragma":
                report = masked_loss(batch, result)
            elif native:
                report = next_loss(batch, result)
            else:
                if isinstance(result.logits, dict):
                    from test.kumo_pytorch.targets import verify_stored_loss

                    counts = verify_sweep_targets(
                        batch,
                        requests,
                        selection,
                        family,
                        state,
                        fitting_requests=fit_requests,
                    )
                    verify_stored_loss(selection["binding"], batch, result)
                    report = {"eligible": len(roots)}
                    assert all(n == len(roots) for n in counts.values())
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
