# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowflake Experiment Tracking for `Trainer.fit`, `Trainer.predict` and
monitoring (``pip install 'kumo-engine[snowflake]'``)."""

import json
import logging
import posixpath
import re
import tempfile
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from kumo_core.instrumentation import format_error
from kumo_core.sources.sql import resolve_snowflake_identifier
from kumo_core.summaries import (
    Confusion,
    Curve,
    Dataset,
    Histogram,
    Scalar,
    Series,
    Summary,
    Table,
    Tag,
)

from kumo.engine._events import Event
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    RunRecord,
    by_step,
    record_failure,
)
from kumo.engine.connector._extras import import_provider

logger = logging.getLogger(__name__)

_MAX_METRICS = 200
_MAX_PARAMS = 1000
# Metric name budgets: per-epoch metrics give way first, then trial summaries.
_EPOCH_METRICS = _MAX_METRICS - 60
_SUMMARY_METRICS = _MAX_METRICS - 30
_FINISH_PARAMS = 20


class SnowflakeTracking:
    r"""Records `Trainer.fit` in Snowflake Experiment Tracking as one run;
    trial attempts are logged in it under ``trial-<n>-attempt-<m>/`` names and
    tags are logged as params. Snowflake has no dataset inputs, so a fit's
    source tables are not recorded as inputs. The Snowpark session's
    parameters are not changed. As ``kumo.MLOps(tracking=...)``, it also
    records each prediction run and each monitoring call as one run named
    after it; ``run_name`` and ``tags`` apply to fit runs only.

    Args:
        source: a ``SnowflakeConnector``, whose identity opens the session
            and whose Snowflake Model Registry ``kumo.MLOps(registry=...)``
            registers models in; or a Snowpark session, which tracks only.
            It needs CREATE EXPERIMENT on the schema, or MODIFY on an
            existing experiment.
        experiment_name: experiment for the fit run, created if missing.
        database: `None` uses the session's current database.
        schema_name: `None` uses the session's current schema, or ``PUBLIC``
            when `database` is given or the session has none.
        run_name: name of the fit run. `None` names it after its registry
            model and UTC start, as ``fit_users_churn_20261027_0618``; a
            taken name gets a random suffix.
        tags: extra tags for the fit run.
        native_model_lineage: keep the run open as a context for the whole
            fit, so models registered meanwhile are linked to it.

    ``run_id`` is the most recent tracked run's name (Snowflake runs are
    identified by name), once it has started.
    """

    platform = 'snowflake'
    run_url = None

    def __init__(
        self,
        source: Any,
        *,
        experiment_name: str,
        database: Optional[str] = None,
        schema_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
        native_model_lineage: bool = False,
    ) -> None:
        from kumo.engine.connector.base import Connector

        self._module = import_provider('snowflake.ml.experiment', 'snowflake')
        if source is None:
            raise ValueError('SnowflakeTracking takes a SnowflakeConnector '
                             'or a Snowpark session.')
        self.connector = source if isinstance(source, Connector) else None
        self.session = (source.session()
                        if self.connector is not None else source)
        self.experiment_name = experiment_name
        self.database = database
        self.schema = schema_name
        self.run_name = run_name
        self.tags = dict(tags or {})
        self.native_model_lineage = native_model_lineage
        self.run_id: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self._tracking: Any = None
        self._run: Any = None
        self._run_context = ExitStack()
        self._trial_tags: Dict[Tuple[int, int], Set[str]] = {}
        self._metrics: Set[str] = set()
        self._params = 0
        self._start_tags: Set[str] = set()
        self._artifacts: Set[str] = set()
        self._dropped_metrics: Set[str] = set()
        self._dropped_params = 0

    def load_model(self, location: Any, **kwargs: Any) -> Any:
        r"""The `Trainer` of the Snowflake Model Registry version
        ``location`` names, through this tracker's connector."""
        from kumo.engine.trainer.trainer import Trainer

        if self.connector is None:
            raise ValueError('Loading a registered model needs '
                             'SnowflakeTracking(connector, ...).')
        return Trainer.load_from_connector(self.connector, location=location,
                                           **kwargs)

    def check(self, registry: Any = None) -> None:
        r"""Raises before any work unless the session can see the
        experiment's schema (one metadata query, no warehouse)."""
        database = self.database or self.session.get_current_database()
        schema = self.schema or (None if self.database else
                                 self.session.get_current_schema())
        if database is None:
            return
        name = resolve_snowflake_identifier(schema or 'PUBLIC')
        try:
            rows = self.session.sql(
                f"SHOW SCHEMAS LIKE '{name.replace(chr(39), chr(39) * 2)}' "
                f"IN DATABASE {database}").collect()
        except Exception as error:
            raise RuntimeError(
                f'Snowflake Experiment Tracking cannot use {database}: '
                f'{type(error).__name__}: {error}') from error
        if not rows:
            raise ValueError(
                f'Snowflake Experiment Tracking needs schema {database}.'
                f'{schema or "PUBLIC"}, which does not exist or this role '
                f'cannot see.')

    def start(self, record: RunRecord) -> None:
        self._reset()
        self.run_id = None
        cls = self._module.ExperimentTracking
        tracking = getattr(cls, '_instance', None)
        if tracking is None:
            tracking = cls(self.session)
        elif (not hasattr(tracking, '_run') or tracking._run is not None
              or getattr(tracking, '_session', None) is not self.session):
            raise RuntimeError(
                f'Snowflake ExperimentTracking is in use in this process, by '
                f'an active run or another Snowpark session, so this '
                f'{record.tags.get("kumo.kind")} run is not tracked in '
                f'Snowflake; end that run, or pass SnowflakeTracking the '
                f'session it uses')
        self._tracking = tracking
        database = self.database or self.session.get_current_database()
        schema = self.schema or (None if self.database else
                                 self.session.get_current_schema())
        experiment = tracking.set_experiment(self.experiment_name,
                                             database_name=database,
                                             schema_name=schema or 'PUBLIC')
        fit = record.tags.get('kumo.kind') == 'fit'
        self._run = tracking.start_run(
            self._new_run_name(
                tracking, experiment.name,
                self.run_name if fit and self.run_name else
                _run_identifier(record.name or 'fit')))
        self.run_id = str(self._run.name)
        if self.native_model_lineage:
            self._run_context.enter_context(self._run)
        self._start_tags = set(record.tags)
        self._log_params({**record.params, **record.tags})
        self._log_artifacts(record.artifacts)

    def log_prediction_run(self, record: RunRecord) -> None:
        self.start(record)
        self.finish(record, record_failure(record))

    def log_monitoring(self, record: RunRecord) -> None:
        self.start(record)
        self.finish(record, None)

    def on_trial(self, record: RunRecord, event: Event) -> None:
        if self._run is None:
            return
        prefix = f'{record.name}/'
        key = (event.trial or 0, int(event.params['attempt']))
        if event.name == 'trial_started':
            self._trial_tags[key] = set(record.tags)
            self._log_params({prefix + k: v
                              for k, v in {**record.params,
                                           **record.tags}.items()})
        elif event.name == 'epoch_ended':
            self._log_metrics({prefix + k: v
                               for k, v in event.metrics.items()},
                              _EPOCH_METRICS, step=event.epoch or 0)
        elif event.name == 'trial_ended':
            self._log_metrics({prefix + k: v
                               for k, v in record.metrics.items()},
                              _SUMMARY_METRICS)
            logged = self._trial_tags.pop(key, set())
            self._log_params({prefix + k: v
                              for k, v in record.tags.items()
                              if k not in logged})

    def finish(self, record: Optional[RunRecord],
               error: Optional[BaseException]) -> None:
        status = 'FAILED'
        try:
            if self._run is not None and record is not None:
                for step, metrics in by_step(record.steps):
                    self.log_metrics(metrics, step)
                self._log_metrics(record.metrics, _MAX_METRICS)
                self._log_params({k: v
                                  for k, v in record.tags.items()
                                  if k not in self._start_tags}, _MAX_PARAMS)
                self._log_artifacts(record.artifacts)
                self.log_summaries(record.summaries)
                self._log_dropped()
                if error is None:
                    status = 'FINISHED'
        finally:
            try:
                if self._run is not None:
                    self._tracking.end_run(status=status)
            except Exception as error:
                logger.warning('Could not end the Snowflake run %s: %s',
                               status, format_error(error), exc_info=True)
            finally:
                self._run_context.close()
                self._reset()

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if self._run is not None:
            self._log_metrics(metrics, _EPOCH_METRICS, step=step)

    def _new_run_name(self, tracking: Any, experiment: Any,
                      run_name: Optional[str]) -> Optional[str]:
        if run_name is None:
            return None
        # start_run(name) resumes, or raises on, any run matching name as a
        # SQL LIKE pattern and cannot create an existing run, so pick a name
        # that is neither.
        existing = [
            str(row['name'])
            for row in tracking._sql_client.show_runs_in_experiment(
                experiment_name=experiment)
        ]
        short = uuid.uuid4().hex
        for name in (run_name, _suffixed(run_name, short[:8]),
                     _suffixed(run_name, short)):
            if not _taken(name, existing):
                break
        else:
            name = None
        if name != run_name:
            logger.warning(
                'Snowflake run %s already exists in experiment %s; recording '
                'this run as %s', run_name, self.experiment_name,
                name or 'a generated run name')
        return name

    def log_summaries(self, summaries: Sequence[Summary]) -> None:
        r"""Logs each summary on the run through its type's call in
        `_SINKS`: values as metrics within the name budget, tables as JSON
        artifacts; Snowflake has no datasets."""
        if self._run is not None:
            for summary in summaries:
                _SINKS[type(summary)](self, summary)

    def _log_scalar(self, summary: Scalar) -> None:
        self._log_metrics({summary.name: summary.value}, _MAX_METRICS)

    def _log_tag(self, summary: Tag) -> None:
        self._log_params({summary.name: summary.value})

    def _log_series(self, summary: Series) -> None:
        self._log_points({summary.name: summary.points})

    def _log_curve(self, summary: Curve) -> None:
        self._log_points({summary.x.name: summary.x.points,
                          summary.y.name: summary.y.points})

    def _log_histogram(self, summary: Histogram) -> None:
        self._log_points({
            f'{summary.name}/count': tuple(enumerate(summary.counts)),
            f'{summary.name}/lower_edge': tuple(enumerate(summary.edges, 1)),
        })

    def _log_table(self, summary: Table) -> None:
        self._log_artifacts({f'{summary.name}.json': json.dumps({
            'columns': list(summary.columns),
            'data': [list(row) for row in summary.rows],
        }, default=str)})

    def _log_confusion(self, summary: Confusion) -> None:
        self._log_table(Table(
            summary.name, ('actual', *summary.labels),
            tuple((label, *row)
                  for label, row in zip(summary.labels, summary.counts))))

    def _log_points(self, series: Mapping[str, Sequence[Tuple[int, float]]]
                    ) -> None:
        for step, metrics in by_step(series):
            self._log_metrics(metrics, _SUMMARY_METRICS, step=step)

    def _log_artifacts(self, artifacts: Mapping[str, str]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, text in artifacts.items():
                if name in self._artifacts:
                    continue
                path = Path(directory) / posixpath.basename(name)
                path.write_text(text, encoding='utf-8')
                self._tracking.log_artifact(
                    str(path), posixpath.dirname(name) or None)
                self._artifacts.add(name)

    def _log_dropped(self) -> None:
        dropped = {
            DROPPED_METRICS: len(self._dropped_metrics),
            DROPPED_PARAMS: self._dropped_params,
        }
        dropped = {name: count for name, count in dropped.items() if count}
        if dropped:
            self._tracking.log_params(dropped)

    def _log_params(self, params: Mapping[str, str],
                    limit: int = _MAX_PARAMS - _FINISH_PARAMS) -> None:
        kept = {}
        for name, value in params.items():
            if '$$' in name or '$$' in value or self._params >= limit:
                self._dropped_params += 1
            else:
                kept[name] = value
                self._params += 1
        if kept:
            self._tracking.log_params(kept)

    def _log_metrics(self, metrics: Mapping[str, float], limit: int,
                     step: int = 0) -> None:
        kept = {}
        for name, value in metrics.items():
            if '$$' in name:
                self._dropped_metrics.add(name)
                continue
            if name not in self._metrics:
                if len(self._metrics) >= limit:
                    self._dropped_metrics.add(name)
                    continue
                self._metrics.add(name)
            kept[name] = value
        if kept:
            self._tracking.log_metrics(kept, step=step)


_SINKS: Dict[type, Callable[[SnowflakeTracking, Any], None]] = {
    Scalar: SnowflakeTracking._log_scalar,
    Tag: SnowflakeTracking._log_tag,
    Series: SnowflakeTracking._log_series,
    Curve: SnowflakeTracking._log_curve,
    Histogram: SnowflakeTracking._log_histogram,
    Table: SnowflakeTracking._log_table,
    Confusion: SnowflakeTracking._log_confusion,
    Dataset: lambda tracker, summary: None,
}


def _run_identifier(name: str) -> str:
    r"""``name`` as an unquoted identifier: snowflake-ml puts a run's name
    into its artifact URIs, where a quoted name is not found."""
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    return name if re.match(r'[A-Za-z_]', name) else f'_{name}'


def _suffixed(name: str, suffix: str) -> str:
    if len(name) > 1 and name[0] == name[-1] == '"':
        return f'{name[:-1]}_{suffix}"'
    return f'{name}_{suffix}'


def _taken(name: str, existing: List[str]) -> bool:
    resolved = resolve_snowflake_identifier(name)
    regex = ''.join('.*' if c == '%' else '.' if c == '_' else re.escape(c)
                    for c in name)
    return any(
        other == resolved
        or re.fullmatch(regex, other, re.IGNORECASE | re.DOTALL) is not None
        for other in existing)
