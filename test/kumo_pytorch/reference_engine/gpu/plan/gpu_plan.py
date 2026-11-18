# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`GpuPlan`: upstream Kumo PyTorch's `Plan` on the GPU, which materializes tasks."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from kumo.pytorch.anchors import AnchorTable
    from kumo.pytorch.contract.sample import SampleRequest
    from kumo.pytorch.dataset import Dataset
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

from test.kumo_pytorch.reference_engine.gpu.plan.check import NAME, _Engine
from test.kumo_pytorch.reference_engine.gpu.plan.gpu_task import GpuTask
from test.kumo_pytorch.reference_engine.gpu.plan.groups import (
    _build_bytes,
    _measured,
)

__all__ = ['GpuPlan']


class GpuPlan:
    """A contract bound to a dataset on the GPU, before any anchors: upstream's `Plan`, here.

    Holds the loaded tables; one plan serves several anchor tables (training,
    a backfill, a batch to score). Close it, or use it as a context manager.
    """

    def __init__(self, contract: DataLoaderSpec, dataset: Dataset, engine: _Engine,
                 tables: dict[str, GpuTable], index: Any, sampler: Any, *, inference: bool = False) -> None:
        self.contract = contract
        self.dataset = dataset
        self.inference = inference
        self._engine = engine
        self._tables: dict[str, GpuTable] | None = tables
        self.index = index  # the entity table's rows by primary key
        self.sampler = sampler  # computes entity values for anchors (test.kumo_pytorch.reference_engine.gpu.sampling)
        #: The GPU memory building one call's batches may hold at once (`_in_groups`); `prepare` sets it.
        self.build_bytes = _build_bytes(None, None)
        self.model_bytes: int | None = None  # the model's measured peak, given `prepare(model_bytes=)`

    @property
    def tables(self) -> dict[str, GpuTable]:
        """The loaded tables: only the ones the contract reads, with only those columns."""

        if self._tables is None:
            raise ValueError("this plan is closed")
        return self._tables

    def materialize(self, anchors: AnchorTable | Sequence[SampleRequest], *,
                    fit_split: str | Collection[str | None] | Any | None = None,
                    fitted: Any | None = None, inference: bool | None = None,
                    fit_memory_limit: int | None = None, fit_spill_dir: str | Path | None = None,
                    encoder_cutoff: Any = ..., values: Any = None) -> GpuTask:
        """Materialize; fit_memory_limit explicitly opts into experimental split fitting storage.

        Limits are integer bytes, not a process RSS cap. Incoming chunks, SQLite
        caches and final encoder arrays are additional workspace. None preserves
        the original fitting path; a missing spill directory disables disk use.
        """
        if values is not None:
            raise NotImplementedError("materialize(values=...) is not supported by the GPU backend; use source tables and anchors")
        if fit_memory_limit is None:
            if fit_spill_dir is not None:
                raise ValueError("fit_spill_dir requires fit_memory_limit")
            return self._materialize(anchors, fit_split=fit_split, fitted=fitted, inference=inference, encoder_cutoff=encoder_cutoff)
        from test.kumo_pytorch.reference_engine.gpu.split.fit_storage import (
            FitBudget,
        )

        with FitBudget(fit_memory_limit, fit_spill_dir) as budget:
            if not hasattr(self, "layout"):
                raise ValueError("fit_memory_limit currently requires a split plan")
            return self._materialize(anchors, fit_split=fit_split, fitted=fitted, inference=inference,
                                     fit_budget=budget, encoder_cutoff=encoder_cutoff)

    def _materialize(self, anchors: AnchorTable | Sequence[SampleRequest], *,
                     fit_split: str | Collection[str | None] | Any | None = None,
                     fitted: Any | None = None, inference: bool | None = None,
                     fit_budget: Any | None = None, encoder_cutoff: Any = ...) -> GpuTask:
        """Bind anchors with upstream's checks and fit-split rules, and return the task.

        ``fitted`` reuses a prior task's state (upstream's `FittedState`, from any engine or an artifact)
        instead of fitting: all of it, or ``state.inputs``, in which case only the targets are fitted here.
        What is reused is applied unchanged, and refused as upstream refuses it (other inputs, or target state of
        another contract).
        """

        from test.kumo_pytorch.reference_engine.gpu.data.anchors import (
            GpuAnchors,
        )
        from test.kumo_pytorch.reference_engine.gpu.encoding.field_names import (
            field_maps,
        )
        from test.kumo_pytorch.reference_engine.gpu.plan.state import (
            manifest,
            reused,
            written_state,
        )

        import cupy

        from kumo.pytorch.anchors import AnchorTable, split_selection
        from kumo.pytorch.assembly import _check_reused_target_spaces
        from kumo.pytorch.column_stats import LATEST_ANCHOR, cutoff_nanos
        from kumo.pytorch.preparation import Plan, check, reused_state
        from kumo.pytorch.spec import ValidationError

        # Validate explicit cutoffs even when all fitted state is reused.
        if encoder_cutoff is not ... and encoder_cutoff is not LATEST_ANCHOR:
            cutoff_nanos(encoder_cutoff)
        self.tables  # noqa: B018 - a closed plan is refused before anything else
        if inference is None:
            inference = self.inference
        elif self.inference and not inference:
            raise ValueError(
                "this plan was prepared for inference (prepare(..., inference=True)), so its targets "
                f"were not checked against engine {NAME!r}; to train, prepare the contract without "
                "inference=True"
            )
        if inference and fitted is None:
            raise ValueError(
                "inference never fits: pass fitted= with the training task's state, "
                "for example Artifact.load(directory).fitted_state"
            )
        if inference and fit_split is not None:
            raise ValueError("fit_split has nothing to fit in inference")
        if isinstance(anchors, AnchorTable):
            issues = check(self.contract, self.dataset, self._engine, anchors=anchors, inference=self.inference)
            if issues:
                raise ValidationError(issues)
            declares_splits = anchors.split is not None
            bound, source = GpuAnchors.from_table(anchors), anchors
        else:
            requests = tuple(anchors)
            declares_splits = any(request.split is not None for request in requests)
            bound, source = GpuAnchors.from_requests(requests), requests
        try:
            selection = split_selection(fit_split)
            # what `fitted` gives (None: a half fitted here), refused as upstream refuses it
            input_state, target_states = reused_state(self.contract, fitted, selection)
            # inference reads no target state, and keeps an input-only state input-only
            input_only = inference and fitted is not None and not fitted.has_targets
            if input_only:
                target_states = {}
            chosen = None  # the anchors fitting reads, when anything is fitted here
            if input_state is None or target_states is None:
                chosen = Plan._fit_selection(selection, declares_splits)
                if not bound.count(chosen):
                    raise ValueError("at least one fitting request is required")
            # field-name dictionaries: the declarations' (nothing to fit), or the reused state's
            names = field_maps(self.contract) if input_state is None else \
                {name: dict(entries) for name, entries in input_state.field_names.items()}
            pinned, statistics = {}, None
            if input_state is None:
                from test.kumo_pytorch.reference_engine.gpu.encoding.table_stats import (
                    pin_table_encoders,
                )
                pinned, statistics = pin_table_encoders(self.contract, self.dataset, bound, bound.rows(chosen),
                                                        encoder_cutoff, loaded=self.tables)
                encoders, composites, spaces, targets = self._fit_inputs(bound, bound.rows(chosen), None, names, pinned=pinned, **({"budget": fit_budget} if fit_budget is not None else {}))
            elif target_states is None:  # the inputs' state reused, the targets fitted here
                _check_reused_target_spaces(self.contract.batch, input_state)
                encoders, composites, spaces, targets = self._fit_inputs(
                    bound, bound.rows(chosen), reused(self.contract, input_state, None)[:3], names,
                    **({"budget": fit_budget} if fit_budget is not None else {}))
            else:
                encoders, composites, spaces, targets = reused(self.contract, input_state, target_states)
            if chosen is None and fitted.has_targets:  # the complete state, as given: written as it is
                written = fitted.to_dict()
            else:
                written = written_state(self.contract, encoders, composites, spaces, None if input_only else targets,
                                        names)
            described = manifest(self.contract, written)
        except BaseException:
            bound.close()
            raise
        cupy.cuda.Device().synchronize()  # loading and fitting done before the caller's own streams read
        # where each reused half was fitted, as `fitted` records it; the others are fitted here
        reused_fit = {half: fitted.fit.get(half) for half, state in (("inputs", input_state), ("targets", target_states))
                      if state is not None}
        return self._task(bound, chosen, encoders, composites, spaces, targets, written, described,
                          source=source, reused_fit=reused_fit, has_targets=not input_only, inference=inference,
                          field_maps=names, statistics=statistics)

    def _fit_inputs(self, bound: GpuAnchors, rows: Any, reused: Any, names: dict[str, Any], *, pinned=None) -> tuple[Any, ...]:
        """Fit the inputs' encoders (unless `reused`) and the targets' state on the anchors at `rows`."""

        from test.kumo_pytorch.reference_engine.gpu.batching.fit import (
            fit_inputs,
        )

        return fit_inputs(self.contract, self.sampler, self.index, bound, rows, reused=reused, field_maps=names, pinned=pinned)

    def _measure_fit(self, bound: GpuAnchors, rows: Any, encoders: Any, composites: Any, spaces: Any, targets: Any,
                     names: dict[str, Any]) -> dict[str, Any]:
        """Each fitted sample's length in ``Batching.length_input`` (`_measured`)."""

        from test.kumo_pytorch.reference_engine.gpu.batching.build import (
            build_batches,
        )

        import cupy

        build = partial(build_batches, sampler=self.sampler, index=self.index, encoders=encoders,
                        composites=composites, spaces=spaces, target_state=targets, anchors=bound,
                        maximum=None, drop_last=False, model=False, fingerprint="", field_maps=names)
        return _measured(build, self.contract, rows, cupy.zeros(rows.size, dtype=bool), self.build_bytes)

    def _task(self, *arguments: Any, **options: Any) -> GpuTask:
        return GpuTask(self, *arguments, **options)

    def close(self) -> None:
        """Free the plan's tables. Idempotent."""

        self._tables = None

    def __enter__(self) -> GpuPlan:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()
