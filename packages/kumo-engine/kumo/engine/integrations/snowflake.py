# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowflake Experiment Tracking for `Trainer.fit`, `Trainer.predict` and
monitoring (``pip install 'kumo-engine[snowflake]'``)."""

import logging
import posixpath
import re
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from kumo_core.instrumentation import format_error
from kumo_core.sources.sql import resolve_snowflake_identifier
from kumo_core.summaries import Summary

from kumo.engine import _card
from kumo.engine._events import Event
from kumo.engine._summary_tables import route, table_json
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    RunRecord,
    by_step,
    model_metrics,
    record_failure,
    run_name,
    versioned,
)
from kumo.engine.connector._extras import import_provider

logger = logging.getLogger(__name__)

_MAX_METRICS = 200
_MAX_PARAMS = 1000
# Metric name budgets: per-epoch metrics give way first, then trial summaries.
_EPOCH_METRICS = _MAX_METRICS - 60
_SUMMARY_METRICS = _MAX_METRICS - 30
_FINISH_PARAMS = 20
_CARD_FILE = 'run_card.md'
_MAX_RUNS = 500
_WARN_RUNS = 450


class SnowflakeTracking:
    r"""Records `Trainer.fit` in Snowflake Experiment Tracking as one run
    with the selected trial's epochs; each trial of a search adds its
    params, status and best tuned validation metric under ``TRIAL_<n>/``
    names (``TRIAL_<n>_ATTEMPT_<m>/`` for a retry). Tags are logged as
    params. Snowflake has no dataset inputs, so a fit's
    source tables are not recorded as inputs. The Snowpark session's
    parameters are not changed. As ``kumo.MLOps(tracking=...)``, it also
    records each prediction run and each monitoring call as one run;
    ``tags`` apply to fit runs only.

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
        run_name: replaces ``<MODEL>_V<version>`` in each run's name:
            ``FIT_<RUN_NAME>``, ``PREDICT_<RUN_NAME>_<as-of date>``, and
            ``DRIFT_`` or ``EVAL_`` for a monitoring call. `None` names a
            fit ``FIT_<MODEL>_<YYYYMMDD_HHMMSS>`` (its UTC start) and the
            others ``PREDICT_<MODEL>_V<version>_<as-of date>``
            (``<MODEL>_<bundle id[:8]>`` for a model not registered). A
            taken name gets ``_2``, ``_3`` and so on.
        tags: extra tags for the fit run.
        native_model_lineage: keep the run open as a context for the whole
            fit, so models registered meanwhile are linked to it.

    ``run_id`` is the most recent tracked run's name (Snowflake runs are
    identified by name), once it has started. An experiment holds at most
    500 runs: from 450 each run warns, and at 500 runs are not tracked.
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
        self.logged_run_name: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self._tracking: Any = None
        self._run: Any = None
        self._run_context = ExitStack()
        self._tune = 'loss'
        self._metrics: Set[str] = set()
        self._values: Dict[str, float] = {}
        self._started = _now()
        self._params = 0
        self._start_tags: Set[str] = set()
        self._artifacts: Set[str] = set()
        # The run's notes and brief tables, for its description.
        self._card: List[Summary] = []
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
        self.run_id = self.logged_run_name = None
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
        existing = [
            str(row['name'])
            for row in tracking._sql_client.show_runs_in_experiment(
                experiment_name=experiment.name)
        ]
        self._check_room(len(existing), record.tags.get('kumo.kind'))
        self._started = getattr(record, 'started', None) or _now()
        self._run = tracking.start_run(self._new_run_name(
            existing, _snowflake_name(record, self._started, self.run_name)))
        self.run_id = self.logged_run_name = str(self._run.name)
        if self.native_model_lineage:
            self._run_context.enter_context(self._run)
        self._start_tags = set(record.tags)
        self._tune = record.tags.get('kumo.tune_metric', 'loss')
        self._log_params({**record.params, **record.tags,
                          **_model_param(record)})
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
        attempt = int(event.params['attempt'])
        prefix = f'TRIAL_{event.trial or 0}' + (
            f'_ATTEMPT_{attempt}/' if attempt > 1 else '/')
        if event.name == 'trial_started':
            self._log_params({prefix + k: v for k, v in record.params.items()})
        elif event.name == 'trial_ended':
            best = f'val/best_{self._tune}'
            self._log_metrics({prefix + best: value
                               for name, value in record.metrics.items()
                               if name == best}, _SUMMARY_METRICS)
            self._log_params({prefix + k: v for k, v in record.tags.items()
                              if k in ('kumo.status', 'kumo.error')})

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
                finished = getattr(record, 'finished', None) or _now()
                self._log_metrics({'duration_seconds': (
                    finished - self._started).total_seconds()}, _MAX_METRICS)
                self._log_card(record)
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

    def _check_room(self, runs: int, kind: Optional[str]) -> None:
        if runs >= _MAX_RUNS:
            raise RuntimeError(
                f'Snowflake experiment {self.experiment_name} holds {runs} '
                f'runs, the most it can ({_MAX_RUNS}), so this {kind} run is '
                f'not tracked; track into a new experiment, e.g. '
                f'SnowflakeTracking(..., experiment_name='
                f"'{self.experiment_name}_2')")
        if runs >= _WARN_RUNS:
            logger.warning(
                'Snowflake experiment %s holds %d of the %d runs it can; '
                'track into a new experiment before it is full',
                self.experiment_name, runs, _MAX_RUNS)

    def _new_run_name(self, existing: List[str], run_name: str) -> str:
        # start_run(name) resumes, or raises on, any run matching name as a
        # SQL LIKE pattern and cannot create an existing run, so pick a name
        # that is neither.
        name, number = run_name, 1
        while _taken(name, existing):
            number += 1
            name = f'{run_name}_{number}'
        if number > 1:
            logger.warning(
                'Snowflake run %s already exists in experiment %s; recording '
                'this run as %s', run_name, self.experiment_name, name)
        return name

    def log_summaries(self, summaries: Sequence[Summary]) -> None:
        r"""Logs ``summaries``; charts become tables, and notes and brief
        tables go on the card."""
        if self._run is None:
            return
        routed = route(summaries, charts=False)
        self._card += routed.card
        self._log_metrics({s.name: s.value for s in routed.scalars},
                          _MAX_METRICS)
        self._log_params(routed.tags)
        self._log_artifacts({f'{table.name}.json': table_json(table)
                             for table in routed.tables})

    def _log_card(self, record: RunRecord) -> None:
        r"""Writes ``run_card.md``: the run's heading and sentence, the
        fit's query and holdout metrics, its notes and brief tables, and its
        files."""
        tags = record.tags
        kind = tags.get('kumo.kind', '')
        try:
            intro = _card.intro(kind, tags, record.params, record.description,
                                tags.get('kumo.fit_run_id'))
            files = sorted(name for name in self._artifacts
                           if name.endswith('.json'))
            self._log_artifacts({_CARD_FILE: '\n'.join([
                *_card.heading(kind, self._values, tags),
                *_card.body(intro, [*_card.holdout(kind, model_metrics(record), tags),
                                    *self._card], files)])})
        except Exception as error:
            logger.warning('Could not write the card of Snowflake run %s: %s',
                           self.run_id, format_error(error))

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
            self._values.update(kept)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _snowflake_name(record: RunRecord, started: datetime,
                    named: Optional[str] = None) -> str:
    r"""The run's name (see `SnowflakeTracking`) as an upper-case
    identifier, for a fit started at ``started``; the tracker's ``named``
    run name replaces the model and version."""
    tags = record.tags
    kind = tags.get('kumo.kind')
    model = getattr(record, 'model_label', None) or 'model'
    if kind == 'fit':
        return _identifier(run_name('fit', named or (
            f"{model}-{started.astimezone(timezone.utc):%Y%m%d-%H%M%S}")))
    version = tags.get('kumo.model_version')
    core = named or (versioned(model, version) if version else
                     f"{model}-{tags.get('kumo.bundle_id', '')[:8]}")
    suffix = record.name_suffix or (
        f"-{tags['kumo.as_of']}" if tags.get('kumo.as_of') else '')
    word = (kind if kind != 'monitoring' else
            'eval' if tags.get('kumo.monitor') == 'evaluate' else 'drift')
    return _identifier(run_name(word, core, suffix))


def _model_param(record: RunRecord) -> Dict[str, str]:
    r"""``kumo.model``, the registered version a prediction run scored
    with, as ``<model>/<version>``."""
    tags = record.tags
    if (tags.get('kumo.kind') != 'prediction' or not tags.get('kumo.model_name')
            or not tags.get('kumo.model_version')):
        return {}
    return {'kumo.model':
            f"{tags['kumo.model_name']}/{tags['kumo.model_version']}"}


def _identifier(name: str) -> str:
    r"""``name``, which starts with its kind, as an unquoted identifier:
    snowflake-ml puts a run's name into its artifact URIs, where a quoted
    name is not found."""
    return re.sub(r'[^A-Za-z0-9_]+', '_', name).upper()


def _taken(name: str, existing: List[str]) -> bool:
    regex = ''.join('.' if c == '_' else re.escape(c) for c in name)
    return any(re.fullmatch(regex, other, re.IGNORECASE | re.DOTALL)
               for other in existing)
