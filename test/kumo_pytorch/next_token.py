"""Independent AML next-token source, alignment and objective checks."""

import math

from .involvement import history
from .reference_tokens import history_tokens


def verify_targets(batch, anchors, reference, state, identities):
    """Reconstruct full historical tokens; inputs/answers remove opposite ends."""
    assert not batch.metadata["is_padding"].any(), (
        "Source comparison needs unpadded requests"
    )
    assert set(batch.inputs) == {"history"} and set(batch.targets) == {"next_token"}
    roles = state["spaces"]["history"]["roles"]
    assert roles == {
        "pad": 0,
        "unknown": 1,
        "mask": 2,
        "entity": 3,
        "event": 4,
        "separator": 5,
    }
    stream, target = batch.inputs["history"], batch.targets["next_token"]
    width = stream["value_ids"].shape[1]
    assert (
        stream["valid"].shape
        == target["values"].shape
        == target["valid"].shape
        == stream["value_ids"].shape
    )
    sequences = []
    for owner, sid in enumerate(batch.metadata["sample_ids"]):
        anchor = anchors[sid]
        cutoff = anchor["as_of_time"] // 10**9
        assert int(batch.metadata["as_of_time"][owner]) == cutoff
        assert (
            int(batch.metadata["entity_ids"][owner]) == identities[anchor["entity_id"]]
        )
        sequence = history_tokens(anchor, history(anchor, reference), state)
        length = max(len(sequence) - 1, 0)
        assert length <= width, "History was truncated"
        assert stream["value_ids"][owner].tolist() == sequence[:-1] + [roles["pad"]] * (
            width - length
        ), "Historical inputs/order differ"
        assert target["values"][owner].tolist() == sequence[1:] + [-100] * (
            width - length
        ), "Next-token target is not shifted exactly once"
        expected_valid = [True] * length + [False] * (width - length)
        assert (
            stream["valid"][owner].tolist()
            == target["valid"][owner].tolist()
            == expected_valid
        ), "Terminal or padding eligibility differs"
        sequences.append(sequence)
    return sequences


def selected_labels(batch):
    stream, target = batch.inputs["history"], batch.targets["next_token"]
    assert target["values"].shape == target["valid"].shape == stream["valid"].shape
    positions, labels, invalid, padding = [], [], 0, 0
    for owner, padded in enumerate(batch.metadata["is_padding"].tolist()):
        for index, (present, valid, label) in enumerate(
            zip(
                stream["valid"][owner].tolist(),
                target["valid"][owner].tolist(),
                target["values"][owner].tolist(),
                strict=True,
            )
        ):
            if padded:
                padding += 1
            elif not present or not valid or label == -100:
                invalid += 1
            else:
                positions.append((owner, index))
                labels.append(label)  # Kumo-aligned IDs; no second shift or remap.
    return positions, labels, invalid, padding


def verify_loss(batch, result):
    """Scalar log-sum-exp CE over each already-aligned, eligible target."""
    positions, labels, invalid, padding = selected_labels(batch)
    assert result.logits.shape[:2] == batch.targets["next_token"]["values"].shape
    logits = result.logits.detach().double().tolist()
    terms = []
    for (owner, index), label in zip(positions, labels, strict=True):
        row = logits[owner][index]
        assert 0 <= label < len(row) and all(math.isfinite(value) for value in row)
        peak = max(row)
        terms.append(
            peak
            + math.log(math.fsum(math.exp(value - peak) for value in row))
            - row[label]
        )
    total = math.fsum(terms)
    mean = total / len(terms) if terms else None
    assert result.weight == len(terms), "Wrong eligible-token denominator"
    assert math.isclose(result.loss.item(), mean or 0.0, rel_tol=2e-6, abs_tol=2e-7), (
        "Wrong aligned next-token loss"
    )
    return {
        "loss_sum": total,
        "eligible": len(terms),
        "invalid": invalid,
        "padding": padding,
        "mean_loss": mean,
    }
