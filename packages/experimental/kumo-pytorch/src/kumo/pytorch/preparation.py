# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a contract to a dataset and an engine.

Everything checkable happens here, before a single row is read: the contract is
internally valid, the dataset binds every table the schema declares, the engine
reads those source kinds, and the engine executes the operations the contract
uses. An unsupported contract is refused here rather than failing part-way
through a run.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any, overload

from kumo.pytorch.anchors import (
    ALL_ANCHORS,
    AllAnchors,
    AnchorTable,
    RequestAnchors,
    SplitSelection,
    anchor_issues,
    describe_selection,
    split_selection,
)
from kumo.pytorch.artifact import FitRecord, FittedState
from kumo.pytorch.assembly import PreviewResult, _fingerprint, _InputState, fit_plan
from kumo.pytorch.column_stats import (
    LATEST_ANCHOR,
    _LatestAnchor,
    cutoff_nanos,
    fit_from_stats,
    table_column_encoders,
)
from kumo.pytorch.contract.batch.encodings import Text
from kumo.pytorch.contract.sample import SampleRequest
from kumo.pytorch.data.records import cutoff_to_epoch
from kumo.pytorch.dataset import Dataset, TableSource
from kumo.pytorch.encoding import FittedEncoder
from kumo.pytorch.engine import AnchorTask, Engine, SampleSource
from kumo.pytorch.spec import (
    DataLoaderSpec,
    ValidationError,
    ValidationIssue,
    sample_without_targets,
)
from kumo.pytorch.task import Task

__all__ = ["Plan", "check", "prepare", "preview"]

logger = logging.getLogger(__name__)


def check(
    contract: DataLoaderSpec,
    dataset: Dataset,
    engine: Engine,
    *,
    anchors: AnchorTable | None = None,
    inference: bool = False,
) -> tuple[ValidationIssue, ...]:
    """Return everything wrong with this combination, without raising.

    Useful for reporting all problems at once, and for tests that assert a
    particular refusal rather than that something failed. With ``anchors``, the
    anchor binding is checked too, exactly as `Plan.materialize` checks it.

    ``inference=True`` checks what inference runs: the engine is asked about the
    inputs, not about targets inference never computes.
    The whole contract must still be well formed.
    """

    return _check(contract, dataset, engine, anchors=anchors, inference=inference)[1]


def _check(
    contract: DataLoaderSpec,
    dataset: Dataset,
    engine: Engine,
    *,
    anchors: AnchorTable | None = None,
    inference: bool = False,
) -> tuple[DataLoaderSpec, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    issues.extend(_binding_issues(contract, dataset, engine))
    try:
        # Keep reporting declaration issues for a bad binding when types are
        # already known, but never read metadata through a missing binding.
        contract = contract.resolve(dataset=None if issues else dataset)
    except (ValueError, KeyError, OSError) as error:
        if not issues:
            issues.append(
                ValidationIssue(
                    "contract.types",
                    str(error),
                    "supply compatible column metadata and calculation types",
                )
            )
        return contract, tuple(issues)
    checked = _inference_view(contract) if inference else contract
    try:
        checked.validate()
    except ValidationError as error:
        issues.extend(error.issues)
    if inference:
        try:
            contract.validate()
        except ValidationError as error:
            issues.extend(issue for issue in error.issues if issue not in issues)
    preflight = getattr(engine, "check_contract", None)
    if preflight is not None:
        issues.extend(preflight(checked))
    if anchors is not None:
        issues.extend(anchor_issues(contract, anchors))
        issues.extend(_source_issues(engine, {"": anchors.source}, "anchors.source"))
        anchor_preflight = getattr(engine, "check_anchors", None)
        if anchor_preflight is not None:
            issues.extend(anchor_preflight(anchors))
    return contract, tuple(issues)


def _inference_view(contract: DataLoaderSpec) -> DataLoaderSpec:
    """The contract as inference runs it: no row-computed target, and no batch target."""

    sample = sample_without_targets(contract.sample)
    return replace(contract, sample=sample, batch=replace(contract.batch, targets={}))


class Plan:
    """A contract bound to a dataset through an engine, before any anchors.

    Holds nothing that depends on which examples will be drawn, so one plan
    serves several anchor tables — a training set, a backfill, a batch to
    score. `materialize` binds anchors and returns the `Task` that is iterated.

    Owns the opened binding: close it with ``plan.close()`` or use the plan as a
    context manager. Tasks read through it and are usable while it is open.

    ``inference`` is true for a plan prepared with ``prepare(..., inference=True)``:
    its targets were not checked against the engine, so it only materializes
    inference tasks.
    """

    def __init__(
        self,
        contract: DataLoaderSpec,
        dataset: Dataset,
        engine: Engine,
        source: SampleSource,
        *,
        inference: bool = False,
    ) -> None:
        self.contract = contract
        self.dataset = dataset
        self.engine = engine
        self.inference = inference
        self._source: SampleSource | None = source
        self._dataset_fingerprint = _fingerprint(dataset)

    def materialize(
        self,
        anchors: AnchorTable | Sequence[SampleRequest],
        *,
        fit_split: str | Collection[str | None] | AllAnchors | None = None,
        fitted: FittedState | None = None,
        inference: bool | None = None,
        encoder_cutoff: str | dt.datetime | None = LATEST_ANCHOR,
    ) -> Task:
        """Bind anchors, fit what is still unfitted, and return the task.

        ``anchors`` is an `AnchorTable` when the anchors exist as data, or
        `SampleRequest` values built in code.

        ``fit_split`` says which anchors fitting may read, and is never implied
        when the anchors carry splits — so held-out or unassigned anchors cannot
        reach fitted state without an explicit choice:

        - anchors that declare splits require it: a split name, a set of names
          (``None`` in the set selects anchors with no split), or `ALL_ANCHORS`;
        - anchors that declare none fit on every anchor, and a named selector
          is refused because there is nothing to select.

        ``fitted`` reuses a prior task's state instead: all of it, or
        ``state.inputs`` alone, in which case only the targets are fitted here.

        ``inference=True`` makes every batch an inference batch: no targets are
        computed or emitted, so no row a target would read is read, and masking
        is off, and the last partial batch is kept unless ``drop_last`` says
        otherwise. It never fits, so it requires ``fitted`` state: the complete
        state, or ``state.inputs`` alone, since inference reads no target state.
        It defaults to the plan's mode; an inference plan refuses ``inference=False``.

        Encoders over stored table columns fit on every table row created at or
        before ``encoder_cutoff``, whatever ``fit_split`` selects: by default the
        latest anchor time ``fit_split`` selects; ``None`` reads every row.

        Reused state applies unchanged: nothing refits, grows or renumbers.
        Complete state requires an equal `DataLoaderSpec.fingerprint`;
        ``state.inputs`` requires an equal `DataLoaderSpec.input_fingerprint`,
        and new targets cannot add members to a reused final ID space.
        """

        if inference is None:
            inference = self.inference
        elif self.inference and not inference:
            raise ValueError(
                "this plan was prepared for inference (prepare(..., inference=True)), so "
                f"its targets were not checked against engine {self.engine.name!r}; to "
                "train, prepare the contract without inference=True"
            )
        if inference and fitted is None:
            raise ValueError(
                "inference never fits: pass fitted= with the training task's state, "
                "for example Artifact.load(directory).fitted_state"
            )
        if inference and fit_split is not None:
            raise ValueError("fit_split has nothing to fit in inference")
        cutoff = encoder_cutoff if encoder_cutoff is LATEST_ANCHOR else cutoff_nanos(encoder_cutoff)
        source = self._require_open()
        if isinstance(anchors, AnchorTable):
            issues = check(
                self.contract,
                self.dataset,
                self.engine,
                anchors=anchors,
                inference=self.inference,
            )
            if issues:
                raise ValidationError(issues)
            declares_splits = anchors.split is not None
            task_anchors: AnchorTask = self.engine.materialize_anchors(anchors)
            anchors_fingerprint = _fingerprint(anchors)
        else:
            requests = tuple(anchors)
            declares_splits = any(request.split is not None for request in requests)
            task_anchors = RequestAnchors(requests)
            anchors_fingerprint = _fingerprint(requests)
        try:
            selection = split_selection(fit_split)
            input_state, target_encoders = reused_state(self.contract, fitted, selection)
            # inference reads no target state, and keeps an input-only state input-only
            input_only = inference and fitted is not None and not fitted.has_targets
            if input_only:
                target_encoders = {}
            fitting: Sequence[SampleRequest] = ()
            record: FitRecord | None = None
            if input_state is None or target_encoders is None:
                chosen = self._fit_selection(selection, declares_splits)
                fitting = task_anchors.fetch(chosen, range(task_anchors.count(chosen)))
                record = {
                    "contract": self.contract.fingerprint,
                    "dataset": self._dataset_fingerprint,
                    "anchors": anchors_fingerprint,
                    "fit_split": describe_selection(chosen),
                }
            pinned, statistics = (
                ({}, {})
                if input_state is not None
                else self._pinned_from_stats(source, fitting, cutoff)
            )
            reused = {} if fitted is None else fitted.fit
            fit = {
                "inputs": ({**record, "statistics": statistics} if statistics else record)
                if input_state is None
                else reused.get("inputs"),
                "targets": record if target_encoders is None else reused.get("targets"),
            }
            assembler = fit_plan(
                self.contract.sample,
                self.contract.batch,
                fitting,
                source,
                schema_fingerprint=_fingerprint(self.contract.schema),
                contract_fingerprint=self.contract.fingerprint,
                contract=self.contract,
                input_state=input_state,
                target_encoders=target_encoders,
                length_summary=None if fitted is None else fitted._lengths,
                inference=inference,
                pinned=pinned,
            )
        except BaseException:
            task_anchors.close()
            raise
        return Task(
            contract=self.contract,
            assembler=assembler,
            anchors=task_anchors,
            dataset_fingerprint=self._dataset_fingerprint,
            anchors_fingerprint=anchors_fingerprint,
            fit=fit,
            has_targets=not input_only,
        )

    def _pinned_from_stats(
        self,
        source: SampleSource,
        fitting: Sequence[SampleRequest],
        cutoff: int | None | _LatestAnchor,
    ) -> tuple[dict[str, FittedEncoder], dict[str, Any]]:
        column_stats = getattr(source, "column_stats", None)
        column_text = getattr(source, "column_text", None)
        encoders = {
            identity: encoder
            for identity, encoder in table_column_encoders(self.contract).items()
            if (column_text if isinstance(encoder.config, Text) else column_stats) is not None
        }
        if not encoders or not fitting:
            return {}, {}
        latest = (
            max(cutoff_to_epoch(request.as_of_time) for request in fitting) * 10**9
            if self.contract.requires_anchor_time
            else None
        )
        if cutoff is LATEST_ANCHOR:
            cutoff = latest
        elif latest is not None and (cutoff is None or cutoff > latest):
            logger.warning(
                "encoder_cutoff %s is after the latest fitting anchor %s: encoders learn "
                "from rows the training anchors cannot see",
                "None (every row)" if cutoff is None else _iso(cutoff),
                _iso(latest),
            )
        started = time.perf_counter()
        requests = [item for encoder in encoders.values() for item in encoder.requests]
        stats = column_stats(requests, cutoff) if requests else None
        pinned, files = {}, list(stats.files) if stats else []
        for identity, encoder in encoders.items():
            if isinstance(encoder.config, Text):
                tokenizer, path = column_text(encoder, cutoff)
                pinned[identity] = FittedEncoder(
                    identity=identity,
                    config=encoder.config,
                    dtype=encoder.dtype,
                    tokenizer_json=tokenizer,
                )
                files.extend([path] if path else [])
            else:
                pinned[identity] = fit_from_stats(identity, encoder, stats)
        logger.info(
            "pinned %d encoders from DiskGraph column statistics (cutoff %s) in %.2fs",
            len(pinned),
            "none" if cutoff is None else _iso(cutoff),
            time.perf_counter() - started,
        )
        return pinned, {"cutoff": cutoff, "col_stats": files}

    @staticmethod
    def _fit_selection(selection: SplitSelection | None, declares_splits: bool) -> SplitSelection:
        if selection is None:
            if declares_splits:
                raise ValueError(
                    "these anchors declare splits, so fitting needs to be told which to read: "
                    'pass fit_split="train", a set of split names (None selects anchors with '
                    "no split), or ALL_ANCHORS; or reuse existing state with fitted="
                )
            return ALL_ANCHORS
        if not declares_splits and not isinstance(selection, AllAnchors):
            raise ValueError(
                "these anchors declare no splits, so there is nothing to select: omit "
                "fit_split, and fitting reads every anchor"
            )
        return selection

    def _require_open(self) -> SampleSource:
        if self._source is None:
            raise ValueError("this plan is closed")
        return self._source

    def close(self) -> None:
        """Release the engine binding. Idempotent."""

        source, self._source = self._source, None
        if source is not None:
            source.close()

    def __enter__(self) -> Plan:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def reused_state(
    contract: DataLoaderSpec,
    fitted: FittedState | None,
    fit_split: SplitSelection | None,
) -> tuple[_InputState | None, Mapping[str, Any] | None]:
    """The parts of ``fitted`` that ``contract`` may reuse; ``None`` for a part to fit.

    Input state is reused under the same inputs, target state only under the
    same whole contract.
    """

    if fitted is None:
        return None, None
    if fitted.input_fingerprint != contract.input_fingerprint:
        raise ValueError(
            "fitted state was fitted for different inputs; it can only be reused "
            "by a contract whose inputs, schema and sample are unchanged "
            f"(state {fitted.input_fingerprint}, this contract "
            f"{contract.input_fingerprint})"
        )
    if not fitted.has_targets:
        return fitted._input_state, None if contract.batch.targets else {}
    if fitted.contract_fingerprint != contract.fingerprint:
        raise ValueError(
            "fitted target state belongs to a different contract; reuse "
            "fitted.inputs and let this task fit its own targets "
            f"(state {fitted.contract_fingerprint}, this contract "
            f"{contract.fingerprint})"
        )
    if fit_split is not None:
        raise ValueError("fit_split has nothing to fit when fitted state is complete")
    return fitted._input_state, fitted._target_encoders


def prepare(
    contract: DataLoaderSpec,
    dataset: Dataset,
    engine: Engine,
    *,
    inference: bool = False,
) -> Plan:
    """Validate, then bind the dataset through the engine.

    Every check that does not need data happens first, so an unexecutable
    combination is refused before a row is read. Anchors, and everything fitted
    on them, arrive later through `Plan.materialize`.

    ``inference=True`` prepares a plan for inference only, checked as
    ``check(..., inference=True)`` checks it: an engine that runs the inputs but
    not the targets can score the contract. Its tasks are inference tasks.
    """

    contract, issues = _check(contract, dataset, engine, inference=inference)
    if issues:
        raise ValidationError(issues)
    if getattr(engine, "encoder_cutoffs", ()) and not inference:
        encoders = table_column_encoders(contract).values()
        requests = [item for encoder in encoders for item in encoder.requests]
        source = engine.open(dataset, contract.schema, column_stats=requests)
    else:
        source = engine.open(dataset, contract.schema)
    return Plan(contract, dataset, engine, source, inference=inference)


@overload
def preview(
    contract: DataLoaderSpec,
    dataset: Dataset,
    engine: Engine,
    requests: Sequence[SampleRequest],
    *,
    seed: int = 0,
) -> PreviewResult: ...


@overload
def preview(
    task: Task,
    requests: Sequence[SampleRequest],
    *,
    seed: int = 0,
) -> PreviewResult: ...


def preview(
    contract_or_task: DataLoaderSpec | Task,
    dataset_or_requests: Dataset | Sequence[SampleRequest],
    engine: Engine | None = None,
    requests: Sequence[SampleRequest] | None = None,
    *,
    seed: int = 0,
) -> PreviewResult:
    """Build up to eight raw samples and one model-facing batch.

    A contract preview temporarily fits on the preview requests and closes the
    binding it opens; do not treat those encodings as training state. A task
    preview reuses the task's fitted state and never changes it.

    Accepts at most eight requests and ignores batching limits so they appear
    together. Returns a `PreviewResult`: raw samples, the encoded batch, and an
    encoding summary.

    Called as ``preview(task, requests)`` or
    ``preview(contract, dataset, engine, requests)``.
    """

    if isinstance(contract_or_task, Task):
        if isinstance(dataset_or_requests, Dataset) or engine is not None or requests is not None:
            raise TypeError("task preview is preview(task, requests, seed=...)")
        return contract_or_task.preview(dataset_or_requests, seed=seed)

    if not isinstance(dataset_or_requests, Dataset) or engine is None or requests is None:
        raise TypeError(
            "contract preview is preview(contract, dataset, engine, requests, seed=...)"
        )
    with prepare(contract_or_task, dataset_or_requests, engine) as plan:
        # a contract preview fits on its own requests, whatever splits they carry
        return plan.materialize(requests, fit_split=ALL_ANCHORS).preview(requests, seed=seed)


def _binding_issues(
    contract: DataLoaderSpec,
    dataset: Dataset,
    engine: Engine,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    declared = [table.name for table in contract.schema.tables]
    missing = dataset.missing_for(declared)
    if missing:
        issues.append(
            ValidationIssue(
                "dataset",
                f"schema declares {sorted(missing)} but the dataset binds no source for them",
                f"add a source for each, for example Parquet('/warehouse/{missing[0]}')",
            )
        )
    unknown = sorted(set(dataset.tables) - set(declared))
    if unknown:
        issues.append(
            ValidationIssue(
                "dataset",
                f"dataset binds {unknown}, which the schema does not declare",
                "remove the binding, or declare the table in DataSchema",
            )
        )
    issues.extend(_source_issues(engine, dataset.tables, "dataset"))
    return issues


def _source_issues(
    engine: Engine, sources: Mapping[str, TableSource], path: str
) -> tuple[ValidationIssue, ...]:
    check_sources = getattr(engine, "check_sources", None)
    return () if check_sources is None else tuple(check_sources(sources, path))


def _iso(nanos: int) -> str:
    return (
        dt.datetime(1970, 1, 1, tzinfo=dt.UTC) + dt.timedelta(microseconds=nanos // 1_000)
    ).isoformat()
