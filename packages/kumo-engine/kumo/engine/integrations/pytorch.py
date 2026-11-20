# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Stage connector, training and prediction tables for kumo-pytorch
(experimental)."""

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kumo.engine._scratch import get_scratch_dir
from kumo.engine.connector._staging import stage_source_table
from kumo.engine.connector.source_table import SourceTable

if TYPE_CHECKING:
    from kumo.engine.pquery import (
        PredictionTable,
        PredictiveQuery,
        TrainingTable,
    )
    from kumo.pytorch import api


@dataclass(frozen=True)
class StagedTable:
    r"""One table staged by `stage_tables`."""
    #: Directory of the table's Parquet parts.
    path: Path
    #: The connector's data version when staged, if it reports one.
    version: Optional[str]


@dataclass(frozen=True)
class StagedTables:
    r"""Tables staged by one `stage_tables` call, keyed by the names passed
    to it."""
    root: Path
    tables: Mapping[str, StagedTable]

    def source(self, name: str) -> 'api.Parquet':
        r"""Table ``name`` as a kumo-pytorch source, for a `Dataset` or an
        `AnchorTable`."""
        return _api().Parquet(path=str(self.tables[name].path))

    def dataset(self) -> 'api.Dataset':
        r"""Every table as a kumo-pytorch `Dataset`."""
        return _api().Dataset(
            tables={name: self.source(name)
                    for name in self.tables})

    def remove(self) -> None:
        r"""Deletes the staged files."""
        shutil.rmtree(self.root)


def stage_tables(
    tables: Mapping[str, SourceTable],
    *,
    root: Optional[Union[str, Path]] = None,
) -> StagedTables:
    r"""Stages each table through its connector as local Parquet.

    Each call writes a new directory, so tables staged earlier stay as they
    were. Columns have the types the connector stages, which can differ from
    the source's.

    Args:
        tables: the name kumo-pytorch uses for each table, to
            ``connector["TABLE"]`` or ``connector.sql(...)``.
        root: local directory to stage under; defaults to the scratch
            directory.
    """
    _api()  # Fail before staging if kumo-pytorch is missing.
    for name, table in tables.items():
        if not isinstance(table, SourceTable):
            raise TypeError(f'{name!r} must be a connector table, not '
                            f'{type(table).__name__}.')
    staged = {}
    with _staging_directory('tables-', root) as directory:
        for index, (name, table) in enumerate(tables.items()):
            # Numbered directories keep table names out of paths. Passing the
            # catalog's dtypes stops object stores casting stored Parquet
            # types, e.g. decimals.
            path, version = stage_source_table(
                table, directory / f'{index:03d}',
                dtypes={column.name: column.dtype
                        for column in table.columns})
            staged[name] = StagedTable(path, version)
    return StagedTables(directory, staged)


def stage_training_table(
    table: 'TrainingTable',
    pquery: 'PredictiveQuery',
    *,
    target: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
) -> 'api.AnchorTable':
    r"""Stages a training table's rows as a kumo-pytorch `AnchorTable`.

    Each call writes a new directory with the columns `TrainingTable.export`
    writes, with ``SPLIT`` as ``"train"``, ``"val"`` or ``"test"``; pass
    ``fit_split='train'`` to `Plan.materialize`.

    Whether anchors carry a prediction time depends on the query and the
    entity table's time column. Without a prediction time, the contract must
    declare every table ``NON_TEMPORAL``. A weighted table's ``WEIGHT`` is in
    ``batch.metadata['request_metadata']``.

    Args:
        table: training table whose exported rows become anchors.
        pquery: the query the table was generated from, on that graph or a
            snapshot of it.
        target: binds ``TARGET1`` to this ``ExternalTarget`` name, which the
            contract declares with ``TARGET1``'s type; omitted binds no target.
        root: local directory to stage under; defaults to the scratch
            directory's ``pytorch`` subdirectory.
    """
    api = _api()
    from kumo_core.pquery import resolve_query_forecast
    from kumo_core.training.export import (
        SPLIT_COLUMN,
        STAGE_SPLITS,
        iter_export_parts,
    )

    # The generation plan's forecast length can make the task forecasting.
    task_type = resolve_query_forecast(pquery._parsed, pquery._native_graph,
                                       table.generation_plan).task_type
    _check_stageable(table, pquery, task_type)
    as_of_time = _anchor_time(pquery)
    split_names = pa.array(sorted(STAGE_SPLITS, key=STAGE_SPLITS.__getitem__))
    weighted = False
    with _staging_directory('anchors-', root) as directory:
        with tempfile.TemporaryDirectory(dir=directory.parent) as export:
            for index, part in enumerate(
                    iter_export_parts(str(table.staging_dir), export)):
                rows = pq.read_table(part)
                weighted = 'WEIGHT' in rows.column_names
                rows = rows.set_column(
                    rows.schema.get_field_index(SPLIT_COLUMN), SPLIT_COLUMN,
                    pc.take(split_names, rows[SPLIT_COLUMN]))
                pq.write_table(rows, directory / f'part-{index:05d}.parquet')
        return api.AnchorTable(source=api.Parquet(path=str(directory)),
                               entity_id='ENTITY1', as_of_time=as_of_time,
                               split=SPLIT_COLUMN,
                               targets={target: 'TARGET1'} if target else {},
                               metadata=('WEIGHT', ) if weighted else ())


def stage_prediction_table(
    table: 'PredictionTable',
    pquery: 'PredictiveQuery',
    *,
    root: Optional[Union[str, Path]] = None,
) -> 'api.AnchorTable':
    r"""Stages a prediction table's scored rows as a kumo-pytorch
    `AnchorTable`.

    Each call writes a new directory. Rows a custom table skipped are not
    included, and sample positions index the scored rows. Anchors have no
    split or target; use them with
    ``plan.materialize(anchors, fitted=..., inference=True)``.

    Whether anchors carry a prediction time depends on the query and the
    entity table's time column. Without a prediction time, the contract must
    declare every table ``NON_TEMPORAL``.

    Args:
        table: prediction table whose scored rows become anchors.
        pquery: the query the table was generated from, on that graph or a
            snapshot of it.
        root: local directory to stage under; defaults to the scratch
            directory's ``pytorch`` subdirectory.
    """
    api = _api()
    # Only forecasting prediction tables keep compiler options.
    task_type = ('forecasting'
                 if table.compiler_options else pquery._parsed.task_type)
    _check_stageable(table, pquery, task_type)
    as_of_time = _anchor_time(pquery)
    path = Path(table.table_path)
    parts = sorted(path.glob('*.parquet')) if path.is_dir() else (path, )
    with _staging_directory('predictions-', root) as directory:
        for index, part in enumerate(parts):
            shutil.copyfile(part, directory / f'part-{index:05d}.parquet')
        return api.AnchorTable(source=api.Parquet(path=str(directory)),
                               entity_id='ENTITY1', as_of_time=as_of_time)


@contextmanager
def _staging_directory(prefix: str,
                       root: Optional[Union[str, Path]]) -> Iterator[Path]:
    parent = Path(root) if root is not None else get_scratch_dir() / 'pytorch'
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        yield directory
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _check_stageable(table: Union['TrainingTable', 'PredictionTable'],
                     pquery: 'PredictiveQuery', task_type: str) -> None:
    from kumoapi.task import TaskType

    if table.query != pquery.query:
        raise ValueError('The table was not generated from pquery.query.')
    task = TaskType(task_type)
    if task.is_link_pred or task.is_multilabel or task == TaskType.FORECASTING:
        raise ValueError(f'Staging {task.value} tables is not supported; '
                         'only tasks with one scalar label are.')


def _anchor_time(pquery: 'PredictiveQuery') -> Optional[str]:
    from kumo_core.pquery import query_is_temporal

    temporal = query_is_temporal(pquery._parsed, pquery._native_graph)
    return 'TIMESTAMP' if temporal else None


def _api() -> Any:
    try:
        from kumo.pytorch import api
    except ImportError as error:
        raise ImportError(
            'Staging tables for kumo-pytorch needs the kumo-pytorch '
            'package.') from error
    return api
