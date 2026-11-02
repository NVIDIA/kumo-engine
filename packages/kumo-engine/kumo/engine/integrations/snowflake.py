# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowflake Experiment Tracking for `Trainer.fit`."""

import logging
import posixpath
import re
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from kumo_core.instrumentation import format_error
from kumo_core.sources.sql import resolve_snowflake_identifier
from kumo_core.summaries import Summary

from kumo.engine import _card
from kumo.engine._events import Event
from kumo.engine._run_names import record_name
from kumo.engine._summary_tables import route, table_json
from kumo.engine._tracking import (
    DROPPED_METRICS,
    DROPPED_PARAMS,
    RunRecord,
    SingleRunHooks,
    TrackingBase,
    by_step,
    model_metrics,
)
from kumo.engine.connector._extras import import_provider

logger = logging.getLogger(__name__)

_MAX_METRICS = 200
_MAX_PARAMS = 1000
_EPOCH_METRICS = _MAX_METRICS - 60
_SUMMARY_METRICS = _MAX_METRICS - 30
_FINISH_PARAMS = 20
_CARD_FILE = 'run_card.md'
_MAX_RUNS = 500
_WARN_RUNS = 450


class SnowflakeTracking(SingleRunHooks, TrackingBase):
    r"""Records each fit, prediction and monitoring call as a Snowflake run.

    Args:
        source: a ``SnowflakeConnector``, or a Snowpark session (tracking only).
        native_model_lineage: link models registered during the fit to its run.
        serving_options: ``ModelOutputConfig.serving_options`` for registered
            versions.
    """

    platform = 'snowflake'

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
        serving_options: Optional[Mapping[str, str]] = None,
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
        self.serving_options = dict(serving_options or {}) or None
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
        self._card: List[Summary] = []
        self._dropped_metrics: Set[str] = set()
        self._dropped_params = 0

    def load_model(self, location: Any, **kwargs: Any) -> Any:
        r"""The `Trainer` registered at ``location``."""
        from kumo.engine.trainer.trainer import Trainer

        if self.connector is None:
            raise ValueError('Loading a registered model needs '
                             'SnowflakeTracking(connector, ...).')
        return Trainer.load_from_connector(self.connector, location=location,
                                           **kwargs)

    def _location(self) -> Tuple[Optional[str], str]:
        database = self.database or self.session.get_current_database()
        schema = self.schema or (None if self.database else
                                 self.session.get_current_schema())
        return database, schema or 'PUBLIC'

    def check(self, registry: Any = None) -> None:
        r"""Raises unless the session can see the experiment's schema."""
        database, schema = self._location()
        if database is None:
            return
        name = resolve_snowflake_identifier(schema)
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
                f'Snowflake schema {database}.{schema} does not exist or this '
                f'role cannot see it.')

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
                f'Snowflake ExperimentTracking is already in use in this '
                f'process, so this {record.tags.get("kumo.kind")} run is not '
                f'tracked; end that run or pass its session.')
        self._tracking = tracking
        database, schema = self._location()
        experiment = tracking.set_experiment(self.experiment_name,
                                             database_name=database,
                                             schema_name=schema)
        existing = [
            str(row['name'])
            for row in tracking._sql_client.show_runs_in_experiment(
                experiment_name=experiment.name)
        ]
        self._check_room(len(existing), record.tags.get('kumo.kind'))
        self._started = record.started or _now()
        self._run = tracking.start_run(self._new_run_name(
            existing, _identifier(record_name(record, self.run_name))))
        self.run_id = self.logged_run_name = str(self._run.name)
        if self.native_model_lineage:
            self._run_context.enter_context(self._run)
        self._start_tags = set(record.tags)
        self._tune = record.tags.get('kumo.tune_metric', 'loss')
        self._log_params({**record.params, **record.tags,
                          **_model_param(record)})
        self._log_artifacts(record.artifacts)

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
                self._log_summaries(record.summaries)
                finished = record.finished or _now()
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
        # start_run matches name with LIKE.
        name, number = run_name, 1
        while _taken(name, existing):
            number += 1
            name = f'{run_name}_{number}'
        if number > 1:
            logger.warning(
                'Snowflake run %s already exists in experiment %s; recording '
                'this run as %s', run_name, self.experiment_name, name)
        return name

    def _log_summaries(self, summaries: Sequence[Summary]) -> None:
        routed = route(summaries, charts=False)
        self._card += routed.card
        self._log_metrics({s.name: s.value for s in routed.scalars},
                          _MAX_METRICS)
        self._log_params(routed.tags)
        self._log_artifacts({**routed.files, **{
            f'{table.name}.json': table_json(table)
            for table in routed.tables}})

    def _log_card(self, record: RunRecord) -> None:
        try:
            card = _card.run_card(record.tags, record.params, self._values,
                                  model_metrics(record), record.description,
                                  self._card, record.tags.get('kumo.fit_run_id'))
            files = sorted(name for name in self._artifacts
                           if name.endswith('.json'))
            self._log_artifacts({_CARD_FILE: card.text(files)})
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


def _model_param(record: RunRecord) -> Dict[str, str]:
    r"""The scoring version as kumo.model=<model>/<version>."""
    tags = record.tags
    if (tags.get('kumo.kind') != 'prediction' or not tags.get('kumo.model_name')
            or not tags.get('kumo.model_version')):
        return {}
    return {'kumo.model':
            f"{tags['kumo.model_name']}/{tags['kumo.model_version']}"}


def _identifier(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', name).upper()


def _taken(name: str, existing: List[str]) -> bool:
    regex = ''.join('.' if c == '_' else re.escape(c) for c in name)
    return any(re.fullmatch(regex, other, re.IGNORECASE | re.DOTALL)
               for other in existing)
