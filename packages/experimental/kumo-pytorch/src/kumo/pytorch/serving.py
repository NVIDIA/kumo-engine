# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build model-input batches for Parquet anchors over a prepared DiskGraph index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.artifact import Artifact
from kumo.pytorch.contract.batch import Batch
from kumo.pytorch.contract.sample import SampleRequest
from kumo.pytorch.data.records import Scalar
from kumo.pytorch.dataset import Parquet
from kumo.pytorch.engines.diskgraph import DiskGraphEngine
from kumo.pytorch.engines.native_anchors import NativeAnchors
from kumo.pytorch.preparation import Plan, reused_state
from kumo.pytorch.serving_rows import (
    ColumnRole,
    ColumnRows,
    RowLimit,
    RowsJoin,
    ServingRowsSpec,
    TableRoute,
    TableRows,
)

__all__ = [
    "ColumnRole",
    "ColumnRows",
    "RowLimit",
    "RowsJoin",
    "ServingAssembler",
    "ServingRows",
    "ServingRowsSpec",
    "ServingValue",
    "TableRoute",
    "TableRows",
]

ServingValue = Scalar | None
"""One supplied cell: a value of the column's declared type, or ``None``."""

ServingRows = Mapping[str, Sequence[Mapping[str, ServingValue]] | pa.Table]
"""Supplied rows by table name: row mappings or a ``pyarrow.Table`` per table."""


class ServingAssembler:
    """Build inference batches with the artifact's fitted encoders on a prepared index.

    Prepare the artifact's contract with `DiskGraphEngine` and keep that plan open
    for the lifetime of the server. Each call supplies a Parquet anchor table with
    the entities and prediction times to score. Serving calls sharing a plan run
    serially. The caller owns the plan and closes it after the calls finish.

    :param artifact: the complete training artifact.
    :param plan: a prepared DiskGraph plan for the artifact's contract. Without a
        plan, the artifact's `rows_spec` can still be inspected.
    """

    def __init__(self, artifact: Artifact, *, plan: Plan | None = None) -> None:
        self.contract = artifact.contract
        self._fitted = artifact.fitted_state
        if not self._fitted.has_targets:
            raise ValueError("serving needs the artifact's complete fitted state")
        reused_state(self.contract, self._fitted, None)
        if plan is not None:
            if not isinstance(plan.engine, DiskGraphEngine):
                raise ValueError("serving requires a prepared DiskGraph plan")
            if plan.contract.fingerprint != self.contract.fingerprint:
                raise ValueError("serving plan and artifact have different contracts")
        self._plan = plan
        self.rows_spec = ServingRowsSpec.from_contract(self.contract)

    def assemble(
        self,
        anchors: AnchorTable | ServingRows | None = None,
        requests: Sequence[SampleRequest] | None = None,
        *,
        batch_size: int | None = None,
        seed: int = 0,
        rows: ServingRows | None = None,
    ) -> list[Batch]:
        """Read Parquet anchors and return native DiskGraph inference batches.

        :param anchors: the anchor table for this call, bound to local Parquet.
        :param batch_size: maximum samples per batch. Defaults to the contract's
            sample limit, or the number of anchors if no limit is set.
        :param seed: the batch iterator seed, as in `Task.batches`.
        :param rows: legacy row-only serving calls are refused.
        :param requests: legacy Python requests are refused.
        """
        if rows is not None or requests is not None or isinstance(anchors, Mapping):
            error = NotImplementedError("native serving assembly is unsupported")
            error.add_note(
                "Row-only calls require migration to a DiskGraph plan and Parquet anchors."
            )
            raise error
        if not isinstance(anchors, AnchorTable) or not isinstance(anchors.source, Parquet):
            raise ValueError("serving requires a Parquet AnchorTable")
        if self._plan is None:
            raise ValueError("serving requires a prepared DiskGraph plan")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        with (
            self._plan._serving_lock,
            self._plan.materialize(anchors, fitted=self._fitted, inference=True) as task,
        ):
            if not isinstance(task._anchors, NativeAnchors):
                raise NotImplementedError("native serving assembly is unsupported")
            task._anchors.allow_request_fallback = False
            if batch_size is None:
                batch_size = self.contract.batch.batching.max_samples or max(1, task.count())
            return list(task.batches(batch_size=batch_size, seed=seed, output_format="model"))
