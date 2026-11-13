# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a written task back in a fresh process, and tie model weights to it.

`Task.to_artifact` gives everything a later process needs to rebuild the same
batches: the contract, the fitted state, the task's identity and its manifest.
`Artifact.save` writes them with a manifest that records every file's digest;
`Artifact.load` reads them back into the ordinary path, so the dataset and
engine may differ from the ones the artifact was written with::

    task.to_artifact().save("artifacts/run-001")
    ...
    artifact = Artifact.load("artifacts/run-001")
    with prepare(artifact.contract, dataset, engine, inference=True) as plan:
        task = plan.materialize(anchors, fitted=artifact.fitted_state)

The artifact is the only on-disk form of fitted state. Its input half is
reused under a contract with the same inputs and its own targets through
``fitted=Artifact.load(directory).fitted_state.inputs``.

Model weights are the user's, saved beside the artifact. `save_weights` records
which contract and fitted state they were trained against, and `load_weights`
refuses them under any other.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import torch

from kumo.pytorch.assembly import (
    ARTIFACT_DIGESTS,
    _fingerprint,
    _InputState,
    _json_text,
    artifact_digests,
    fitted_target_names,
    restore_fitted_state,
)
from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["Artifact", "FitRecord", "FittedState", "load_weights", "save_weights"]

_FILES = ("contract.json", "fitted-state.json", "task.json", "manifest.json")
_WEIGHTS_VERSION = 1


class FitRecord(TypedDict):
    """Where one half of a fitted state was fitted.

    Fitting takes no seed: the state is a function of these alone. The record
    is informational: reuse with ``fitted=`` keeps it, ``Artifact.load`` does
    not check it, and fitted-state equality ignores it. ``statistics`` (inputs
    only) names the encoder cutoff, in epoch nanoseconds, the DiskGraph
    ``col_stats/`` files that encoders over stored columns were fitted from,
    and, with ``Batching.length_input`` set, ``token_lengths``: for each event
    table that input reads, ``{"constant": ..., "file": ...}``, its per-event
    cost of 1 per non-``Text`` column and the ``col_stats/`` file of its
    ``Text`` columns' per-node pieces, ``None`` without ``Text``.
    """

    contract: str
    dataset: str
    anchors: str
    fit_split: str | list[str | None]
    statistics: NotRequired[dict[str, Any]]


@dataclass(frozen=True, eq=False)
class FittedState(Mapping[str, Any]):
    """State fitted for one task, reusable by another.

    Reads as the versioned JSON it is written as (``state["inputs"]``). Split
    into the part that shapes inputs and the part that shapes targets,
    because they are reused differently: a fine-tuning task keeps a pretrained
    task's input vocabularies and fits its own targets. ``state.inputs`` is that
    input-only half.

    Two states are equal when they hold the same fitted values for the same
    contract (for input-only states, the same inputs); the informational `fit`
    record is not compared. Compared with a plain mapping, only the fitted
    values are.

    :param input_fingerprint: `DataLoaderSpec.input_fingerprint` of the
        contract the state was fitted for. Reuse is refused under any other.
    :param contract_fingerprint: the full contract, required to match when the
        target half is reused too.
    """

    input_fingerprint: str
    contract_fingerprint: str
    _input_state: _InputState
    _target_encoders: Mapping[str, Any] | None
    _serialized: Mapping[str, Any]
    _fit: Mapping[str, FitRecord | None] = dataclass_field(default_factory=dict)

    @property
    def inputs(self) -> FittedState:
        """This state without its targets, for reuse under different targets."""

        return dataclass_replace(self, _target_encoders=None)

    @property
    def has_targets(self) -> bool:
        """Whether this is complete state rather than the input half alone."""
        return self._target_encoders is not None

    @property
    def fit(self) -> dict[str, FitRecord | None]:
        """Where each half was fitted, kept through reuse; ``None`` when unknown.

        Informational: reuse and weights are checked against the state itself, never this.
        """

        halves = ("inputs", "targets") if self.has_targets else ("inputs",)
        return {half: self._fit.get(half) for half in halves}

    @property
    def fingerprint(self) -> str:
        """Identity of this state as written: what model weights are tied to."""

        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """The fitted values, as `Artifact.save` writes them beside `fit`."""

        state = dict(self._serialized)
        if self._target_encoders is None:
            state.pop("targets", None)
            state["encoders"] = {
                name: value
                for name, value in state["encoders"].items()
                if name in self._input_state.encoders
            }
            state["spaces"] = {
                name: value
                for name, value in state["spaces"].items()
                if name in self._input_state.spaces
            }
            state["field_names"] = {
                name: value
                for name, value in state["field_names"].items()
                if name in self._input_state.field_names
            }
        return state

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FittedState):
            return super().__eq__(other)
        return _compared(self) == _compared(other)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _written(state: FittedState) -> dict[str, Any]:
    """The fitted-state file: the values and, outside every fingerprint, `fit` and the
    fingerprints of the contract it was fitted for."""

    return {
        **state.to_dict(),
        "fit": state.fit,
        "fitted_for": {"inputs": state.input_fingerprint, "contract": state.contract_fingerprint},
    }


def _compared(state: FittedState) -> object:
    """What equality compares: the written file without its informational `fit`, and an
    input-only state without the full contract it was fitted under."""

    written = {key: value for key, value in _written(state).items() if key != "fit"}
    if not state.has_targets:
        written["fitted_for"] = {"inputs": state.input_fingerprint}
    return json.loads(json.dumps(written, sort_keys=True))


@dataclass(frozen=True)
class Artifact:
    """A task's contract, fitted state and identity: what `Artifact.save` writes.

    `Task.to_artifact` builds one in memory; `Artifact.load` reads one back from
    disk, equal to the writing task's.

    :param contract: the contract the task was materialized from; prepare this
        one, so batches carry the training contract's fingerprint.
    :param bound_contract: the same contract with the encodings that the fitted
        state records pinned, as `Task.bound_contract` pins them.
    :param fitted_state: the task's complete fitted state, for ``fitted=``.
    :param fingerprints: the task's identity as written (`Task.fingerprints`).
    :param output_schema: the task's output schema as written.
    :param manifest: the manifest as written, without its file digests.
    """

    contract: DataLoaderSpec
    bound_contract: DataLoaderSpec
    fitted_state: FittedState
    fingerprints: Mapping[str, str]
    output_schema: Mapping[str, Any]
    manifest: Mapping[str, Any]

    def save(self, path: str | Path) -> None:
        """Write the contract, fitted state, identity and manifest; `Artifact.load` reads them.

        The manifest written also records each file's SHA-256 digest (`artifact_digests`).

        Files: ``contract.json``; ``fitted-state.json`` (fitted values, the
        fingerprints the state was fitted for, and the ``fit`` record);
        ``task.json`` (fingerprints and output schema); ``manifest.json``
        (summaries, fingerprints and digests). The digests catch files not
        written together; they are not a signature.
        """

        identity = {"fingerprints": dict(self.fingerprints), "output_schema": self.output_schema}
        texts = {
            "contract.json": self.contract.to_json() + "\n",
            "fitted-state.json": _json_text(_written(self.fitted_state)),
            "task.json": _json_text(identity),
            "manifest.json": _json_text(self.manifest),
        }
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            if name != "manifest.json":
                (output / name).write_text(text)
        (output / "manifest.json").write_text(
            _json_text({**self.manifest, ARTIFACT_DIGESTS: artifact_digests(texts)})
        )

    @classmethod
    def load(cls, path: str | Path) -> Artifact:
        """Read a directory written by `Artifact.save`.

        Nothing is fitted and no data is read. Every file must match the digest the
        manifest records for it, and the fitted state must rebuild exactly under the
        contract's declared encodings.

        :raises FileNotFoundError: when one of the four files is missing.
        :raises ValueError: naming the file and, where it applies, the field: a
            digest mismatch, or a manifest without digests; state
            fitted for another contract; target state for other targets; state
            that does not serialize back exactly, or an ID space numbering other
            IDs than its members produce; ``manifest.json`` or ``task.json``
            disagreeing with the contract and state.
        """

        root = Path(path)
        missing = [name for name in _FILES if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{root} is not a complete task artifact: missing {missing}; "
                "write it with Artifact.save"
            )
        texts = {name: (root / name).read_text() for name in _FILES}
        manifest: dict[str, Any] = json.loads(texts["manifest.json"])
        recorded = manifest.pop(ARTIFACT_DIGESTS, None)
        actual = artifact_digests(texts)
        if not isinstance(recorded, dict):
            raise ValueError(
                f"{root / 'manifest.json'} records no file digests; write it with Artifact.save"
            )
        for name in _FILES:
            if recorded.get(name) != actual[name]:
                raise ValueError(
                    f"{root / name} does not match the digest manifest.json records for it; "
                    "the artifact's files were not written together"
                )
        contract = DataLoaderSpec.from_json(texts["contract.json"])
        serialized, fit, fitted_for = _split_fit(root / "fitted-state.json")
        task: dict[str, Any] = json.loads(texts["task.json"])
        _require_fitted_for(root / "fitted-state.json", fitted_for, contract)
        _require_targets(root / "fitted-state.json", serialized, contract)
        try:
            fitted, rebuilt, bound = _restored(contract, serialized, fit)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{root}: {error}") from error
        if "lengths" in manifest:
            # written before the length summary was dropped; its fingerprint covers it
            old = {key: value for key, value in rebuilt.items() if key != "artifact_fingerprint"}
            old["lengths"] = manifest["lengths"]
            _require_same(
                root / "manifest.json", manifest, {**old, "artifact_fingerprint": _fingerprint(old)}
            )
            del manifest["lengths"]
            manifest["artifact_fingerprint"] = rebuilt["artifact_fingerprint"]
        _require_same(root / "manifest.json", manifest, rebuilt)
        _require_same(root / "task.json", task, _task_identity(contract, serialized, rebuilt, task))
        return cls(
            contract=contract,
            bound_contract=bound,
            fitted_state=fitted,
            fingerprints=dict(task["fingerprints"]),
            output_schema=task["output_schema"],
            manifest=manifest,
        )


def _split_fit(
    path: Path,
) -> tuple[dict[str, Any], dict[str, FitRecord | None], dict[str, Any] | None]:
    """The fitted values and, beside them, the informational ``fit`` and the fingerprints
    of the contract the state was fitted for."""

    serialized = json.loads(path.read_text())
    if not isinstance(serialized, dict):
        raise ValueError(f"{path} is not a fitted-state object")
    fit = serialized.pop("fit", {})
    serialized.pop("lengths", None)
    fitted_for = serialized.pop("fitted_for", None)
    if not isinstance(fit, dict) or any(
        not isinstance(record, (dict, type(None))) for record in fit.values()
    ):
        raise ValueError(f"{path} has a malformed fit record; it must map each half to an object")
    return serialized, fit, fitted_for


def _require_fitted_for(path: Path, fitted_for: object, contract: DataLoaderSpec) -> None:
    """Refuse a state fitted for another contract."""

    if not isinstance(fitted_for, dict):
        raise ValueError(
            f"{path} does not record the contract it was fitted for; write it with Artifact.save"
        )
    expected = {"inputs": contract.input_fingerprint, "contract": contract.fingerprint}
    for key, fingerprint in expected.items():
        if fitted_for.get(key) != fingerprint:
            raise ValueError(
                f"{path} was fitted under a different contract: its {key} fingerprint is "
                f"{fitted_for.get(key)}, this contract's is {fingerprint}; read it with "
                "the contract it was fitted under"
            )


def _require_targets(path: Path, serialized: Mapping[str, Any], contract: DataLoaderSpec) -> None:
    """Refuse target state for other targets than the contract's fitted ones."""

    if not isinstance(serialized.get("targets"), dict):
        raise ValueError(f"{path} has no target state object; an artifact holds complete state")
    names = sorted(serialized["targets"])
    expected = fitted_target_names(contract.batch)
    if names != expected:
        raise ValueError(
            f"{path} holds target state for {names!r}, but this contract's targets "
            f"need it for {expected}; nothing is refitted to fill the gap"
        )


def _restored(
    contract: DataLoaderSpec,
    serialized: Mapping[str, Any],
    fit: Mapping[str, FitRecord | None],
) -> tuple[FittedState, dict[str, Any], DataLoaderSpec]:
    """The fitted state, the manifest it gives, and the contract with it pinned."""

    input_state, target_encoders, manifest, bound = restore_fitted_state(contract, serialized)
    state = FittedState(
        input_fingerprint=contract.input_fingerprint,
        contract_fingerprint=contract.fingerprint,
        _input_state=input_state,
        _target_encoders=target_encoders,
        _serialized=serialized,
        _fit=fit,
    )
    return state, manifest, bound


def _task_identity(
    contract: DataLoaderSpec,
    serialized: Mapping[str, Any],
    manifest: Mapping[str, Any],
    written: Mapping[str, Any],
) -> dict[str, Any]:
    """What ``task.json`` must say, given the contract and fitted state.

    The dataset and anchors fingerprints aren't recomputable without data, so they
    are taken as written. An inference task's output schema lists no targets.
    """

    inputs = {name: value["summary"] for name, value in manifest["inputs"].items()}
    targets = written.get("output_schema", {}).get("targets")
    schema = {"inputs": inputs, "targets": {} if targets == {} else dict(manifest["targets"])}
    recorded = written.get("fingerprints", {})
    fingerprints = {
        "contract": contract.fingerprint,
        "inputs": contract.input_fingerprint,
        "dataset": recorded.get("dataset"),
        "anchors": recorded.get("anchors"),
        "input_state": _fingerprint({k: v for k, v in serialized.items() if k != "targets"}),
        "target_state": _fingerprint(serialized["targets"]),
        "output_schema": _fingerprint(schema),
    }
    fingerprints["task"] = _fingerprint(fingerprints)
    return {"fingerprints": fingerprints, "output_schema": schema}


def _require_same(path: Path, written: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse, naming the first field of ``path`` that the contract and state don't give."""

    expected = json.loads(json.dumps(expected, sort_keys=True, default=str))
    for key in sorted(written.keys() | expected.keys()):
        left, right = written.get(key), expected.get(key)
        if left == right:
            continue
        field = key
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            field += "." + next(
                str(k) for k in sorted(left.keys() | right.keys()) if left.get(k) != right.get(k)
            )
        raise ValueError(
            f"{path} field {field} is not what the contract and fitted state give; "
            "the artifact's files were not written together"
        )


def save_weights(
    path: str | Path,
    model_state: Mapping[str, torch.Tensor],
    fitted: FittedState,
) -> None:
    """Save model weights with the contract and fitted state they were trained against.

    The file is an ordinary `torch.save` of ``{"format_version", "contract_fingerprint",
    "fitted_state_fingerprint", "model"}``, readable without this package.

    :param path: file to write.
    :param model_state: the model's ``state_dict()``.
    :param fitted: the fitted state the weights were trained against.
    """

    torch.save(
        {
            "format_version": _WEIGHTS_VERSION,
            "contract_fingerprint": fitted.contract_fingerprint,
            "fitted_state_fingerprint": fitted.fingerprint,
            "model": dict(model_state),
        },
        Path(path),
    )


def load_weights(
    path: str | Path,
    fitted: FittedState,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load weights saved by `save_weights`, refusing them under other fitted state.

    Weights index embedding tables by the IDs the fitted state assigns, so
    weights trained against a different contract or a different fit are
    refused rather than silently misread.
    """

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format_version") != (
        _WEIGHTS_VERSION
    ):
        raise ValueError(f"{path} was not written by save_weights")
    expected = {
        "contract_fingerprint": fitted.contract_fingerprint,
        "fitted_state_fingerprint": fitted.fingerprint,
    }
    for key, fingerprint in expected.items():
        if checkpoint.get(key) != fingerprint:
            raise ValueError(
                f"{path} was trained against {key} {checkpoint.get(key)}, but this state "
                f"has {fingerprint}; load the artifact those weights were saved with"
            )
    return dict(checkpoint["model"])
