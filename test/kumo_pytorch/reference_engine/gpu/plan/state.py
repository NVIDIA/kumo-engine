"""A task's fitted state in upstream Kumo PyTorch's own form: what `test.kumo_pytorch.reference_engine.gpu` fitted, written as upstream writes it.

`written_state` spells this backend's fitted encoders, ID spaces, Composite
vocabularies and target state as upstream's ``fitted-state.json`` spells them
(``format_version`` 2; ``assembly.PreparedBatchPlan.fitted_state``), and
`manifest` gives upstream's manifest of it, whose artifact fingerprint every
batch carries. `as_upstream` hands the written state to upstream's own
``restore_fitted_state``, the function ``Artifact.load`` reads a written state
with: it rebuilds upstream's encoders, ID spaces and bindings from it, refuses
it unless they write it back unchanged, and gives the manifest and the bound
contract. So a task's `FittedState`, manifest, fingerprints and artifact are
upstream's, and an artifact written here loads in upstream's code, for any
engine.

`reused` goes the other way, for ``materialize(fitted=...)``: a prior task's
upstream state (any engine's, or an artifact's) as this backend's encoders,
Composite IDs, ID spaces and target state, applied unchanged.

A task needs only the manifest to batch, so that is made when it is fitted;
upstream's objects are rebuilt when first asked for. `fingerprint` is
upstream's fingerprint (``assembly._fingerprint``), the same digest, without
upstream's checks for objects a written state never holds: with a vocabulary of
a million values the state is hashed several times.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.artifact import FitRecord, FittedState
    from kumo.pytorch.encoding import FittedSpace
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import CompositeVocabulary
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import GpuEncoder
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import TargetState

__all__ = ["Fitted", "as_upstream", "fingerprint", "manifest", "reused", "written_state"]


@dataclass(frozen=True)
class Fitted:
    """A task's fitted state as upstream holds it.

    :param state: upstream's `FittedState` (``Task.fitted_state``).
    :param manifest: upstream's manifest of it (``Task.manifest``).
    :param bound_contract: the contract with the fitted encodings pinned (``Task.bound_contract()``).
    """

    state: FittedState
    manifest: dict[str, Any]
    bound_contract: DataLoaderSpec


def written_state(contract: DataLoaderSpec, encoders: Mapping[str, GpuEncoder],
                  composites: Mapping[str, CompositeVocabulary], spaces: Mapping[str, FittedSpace],
                  targets: TargetState | None,
                  field_names: Mapping[str, Mapping[tuple[str, ...], int]] | None = None) -> dict[str, Any]:
    """The fitted values as upstream writes them: every encoder's state (``FittedEncoder.state_dict``), every
    ID space's, the field-name maps, each input's bindings (field, encoder, ID space; a Composite input's
    outer vocabulary) and each fitted target's (bindings, thresholds, groups); none without `targets` (an
    inference task's from an input-only state). `field_names`: each field-name dictionary's identities and
    IDs (`test.kumo_pytorch.reference_engine.gpu.encoding.field_names`), by identity."""

    from kumo.pytorch.assembly import _encoder_identity, _identity, fitted_target_names
    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        Encode,
        Graph,
        MatchingGroupTarget,
        StructuredTarget,
    )
    from kumo.pytorch.contract.batch.encodings import UseEncoding
    from kumo.pytorch.encoding import FittedEncoder, json_value

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import graph_identity

    spec = contract.batch
    inputs: dict[str, Any] = {}
    for name, value in spec.inputs.items():
        representation = value.representation
        if isinstance(representation, Graph):
            inputs[name] = {"node_fields": {
                table: [[use.field, graph_identity(name, table, use), use.id_space] for use in group.fields]
                for table, group in representation.node_fields.items()}}
            continue
        inputs[name] = {"bindings": [[use.field, _encoder_identity(name, use), use.id_space]
                                     for use in representation.fields]}
        if name in composites:  # its outer IDs, by ID (a supplied vocabulary: as given)
            inputs[name]["composite"] = json_value(tuple(composites[name].entries()))
    fitted_targets: dict[str, Any] = {}
    for name in fitted_target_names(spec) if targets is not None else ():
        target = spec.targets[name]
        if isinstance(target, MatchingGroupTarget):
            uses = []
        elif isinstance(target, (StructuredTarget, AlignedEventTarget)):
            uses = list(target.fields)
            if isinstance(target, AlignedEventTarget):
                uses += [Encode(field=label.name, encoding=label.encoding) for label in target.derived]
        else:  # a label (LabelTarget): one field, "value"
            uses = [Encode(field="value", encoding=target.encoding, id_space=target.id_space)]
        groups = targets.groups.get(name)
        fitted_targets[name] = {
            "bindings": [[use.field, _identity("shared", use.encoding.name) if isinstance(use.encoding, UseEncoding)
                          else _identity("targets", name, use.field), use.id_space] for use in uses],
            "thresholds": dict(targets.thresholds.get(name, {})),
            "groups": json_value(() if groups is None else tuple(groups.entries())),
        }
    return {
        "format_version": 2,
        "encoders": {identity: FittedEncoder(identity=identity, config=encoder.config, dtype=encoder.dtype,
                                             entries=encoder.entries, boundaries=encoder.boundaries,
                                             mean=encoder.mean, scale=encoder.scale,
                                             tokenizer_json=encoder.tokenizer_json).state_dict()
                     for identity, encoder in sorted(encoders.items())},
        "spaces": {name: space.state_dict() for name, space in sorted(spaces.items())},
        "field_names": {name: [[list(identity), index] for identity, index in sorted(entries.items())]
                        for name, entries in (field_names or {}).items()},
        "inputs": inputs,
        "targets": fitted_targets,
    }


def as_upstream(contract: DataLoaderSpec, written: dict[str, Any],
                fit: Mapping[str, FitRecord | None], *, has_targets: bool = True) -> Fitted:
    """Upstream's fitted state, manifest and bound contract, rebuilt by upstream from `written` (refused
    unless it writes `written` back unchanged), with `lengths` as the length summary and `fit` as where each
    half was fitted; without `has_targets`, the input half alone (an inference task's from an input-only
    state, as upstream keeps it)."""

    from kumo.pytorch.artifact import FittedState
    from kumo.pytorch.assembly import restore_fitted_state

    input_state, target_encoders, manifest, bound = restore_fitted_state(contract, written)
    state = FittedState(input_fingerprint=contract.input_fingerprint, contract_fingerprint=contract.fingerprint,
                        _input_state=input_state, _target_encoders=dict(target_encoders) if has_targets else None,
                        _serialized=written, _fit=dict(fit))
    return Fitted(state, manifest, bound)


def manifest(contract: DataLoaderSpec, written: dict[str, Any]) -> dict[str, Any]:
    """Upstream's manifest of the written state (``PreparedBatchPlan.manifest``): each input's encodings (an
    encoder's state without its tokenizer, and its ID space) and each fitted target's, the ID spaces' sizes and
    roles, the length summary, and the fingerprints of the schema, contract, fitted state and the manifest itself
    (the artifact fingerprint)."""

    from kumo.pytorch.contract.batch import Graph

    def described(identity: str) -> dict[str, Any]:
        return {key: value for key, value in written["encoders"][identity].items() if key != "tokenizer_json"}

    inputs = {}
    for name, value in contract.batch.inputs.items():
        record = written["inputs"][name]
        if isinstance(value.representation, Graph):
            inputs[name] = {"summary": {table: {field: described(identity) for field, identity, _ in bindings}
                                        for table, bindings in record["node_fields"].items()}}
        else:
            inputs[name] = {"summary": {field: {**described(identity), "id_space": space}
                                        for field, identity, space in record["bindings"]}}
    result = {
        "format_version": 2,
        "schema_fingerprint": fingerprint(contract.schema),
        "contract_fingerprint": contract.fingerprint,
        "inputs": inputs,
        "targets": {name: {field: described(identity) for field, identity, _ in target["bindings"]}
                    for name, target in written["targets"].items()},
        "spaces": {name: {"size": space["size"], "roles": dict(space["roles"])}
                   for name, space in written["spaces"].items()},
        "fitted_state_fingerprint": fingerprint(written),
    }
    result["artifact_fingerprint"] = fingerprint(result)
    return result


def fingerprint(value: Any) -> str:
    """Upstream's fingerprint of `value` (``assembly._fingerprint``): the SHA-256 of its canonical JSON, floats
    spelled exactly (``{"$float": hex}``). Plain JSON values are spelled here; anything else as upstream spells
    it."""

    import gc
    import hashlib
    import json

    paused = gc.isenabled()  # millions of short-lived lists and dicts, none in a cycle: no collection needed
    gc.disable()
    try:
        plain = _plain(value)
    finally:
        if paused:
            gc.enable()
    return "sha256:" + hashlib.sha256(json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _plain(value: Any) -> Any:
    """Upstream's ``_jsonable`` for strings, integers, booleans, None, floats, lists, tuples and dicts, by exact
    type; anything else (a dataclass, a tensor, a mapping of another type) by ``_jsonable`` itself."""

    kind = type(value)
    if kind is str or kind is int or kind is bool or value is None:
        return value
    if kind is list or kind is tuple:
        return [_plain(item) for item in value]
    if kind is dict:
        return {str(key): _plain(item) for key, item in value.items()}
    if kind is float:
        return {"$float": value.hex()}
    from kumo.pytorch.assembly import _jsonable

    return _jsonable(value)


def reused(contract: DataLoaderSpec, input_state: Any, target_states: Mapping[str, Any] | None,
           ) -> tuple[dict[str, GpuEncoder], dict[str, CompositeVocabulary], dict[str, FittedSpace], TargetState | None]:
    """A prior task's fitted state as this backend's: upstream's input state (``FittedState._input_state``:
    encoders, ID spaces, each input's bindings and Composite IDs) and, when the targets are reused too, its
    target state (bindings, thresholds, matching groups). Every encoder, ID and space is applied as fitted.
    Without target state, the targets' part is None (they are fitted here); with it, the ID spaces are upstream
    ``fit_plan``'s: every declared space, numbered as the state numbers it."""

    from dataclasses import replace

    from kumo.pytorch.assembly import _allocate_spaces

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import CompositeVocabulary, component_specs
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import reused_encoder
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import TargetState, _group_specs

    encoders = {identity: reused_encoder(encoder) for identity, encoder in input_state.encoders.items()}
    composites = {name: CompositeVocabulary.from_supplied(
        component_specs(contract.batch.inputs[name].representation, name, encoders), tuple(prepared.composite.items()))
        for name, prepared in input_state.prepared_inputs.items() if prepared.composite is not None}
    spaces = dict(input_state.spaces)
    if target_states is None:
        return encoders, composites, spaces, None
    targets = TargetState()
    for name, state in target_states.items():
        for binding in state.bindings.values():
            encoders.setdefault(binding.encoder.identity, reused_encoder(binding.encoder))
            if binding.id_space is not None:
                spaces[binding.id_space] = binding.destination
        if state.thresholds is not None:
            targets.thresholds[name] = dict(state.thresholds)
        if state.groups is not None:
            specs = _group_specs(contract, contract.batch.targets[name], encoders)
            targets.groups[name] = replace(CompositeVocabulary.from_supplied(specs, tuple(state.groups.items())),
                                           unknown=-1)
    return encoders, composites, {**_allocate_spaces(contract.batch, {}, {}), **spaces}, targets
