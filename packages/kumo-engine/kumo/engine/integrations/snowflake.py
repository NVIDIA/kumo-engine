# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Snowflake Experiment Tracking for `Trainer.fit`
(``pip install 'kumo-engine[snowflake]'``)."""

import logging
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from kumo_core.sources.sql import resolve_snowflake_identifier

from kumo.engine._events import Event
from kumo.engine._tracking import DROPPED_METRICS, DROPPED_PARAMS, RunRecord
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
    parameters are not changed.

    Args:
        session: Snowpark session; needs CREATE EXPERIMENT on the schema, or
            MODIFY on an existing experiment.
        experiment_name: experiment for the fit run, created if missing.
        database: `None` uses the session's current database.
        schema: `None` uses the session's current schema, or ``PUBLIC`` when
            `database` is given or the session has none.
        run_name: name of the fit run. `None` lets Snowflake pick one; a
            taken name gets a suffix from the kumo run id.
        tags: extra tags for the fit run.
        native_model_lineage: keep the run open as a context for the whole
            fit, so models registered meanwhile are linked to it.

    ``run_id`` is the latest fit run's name (Snowflake runs are identified
    by name), once it has started.
    """

    def __init__(
        self,
        session: Any,
        *,
        experiment_name: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, str]] = None,
        native_model_lineage: bool = False,
    ) -> None:
        self._module = import_provider('snowflake.ml.experiment', 'snowflake')
        self.session = session
        self.experiment_name = experiment_name
        self.database = database
        self.schema = schema
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

    def start(self, record: RunRecord) -> None:
        self._reset()
        self.run_id = None
        cls = self._module.ExperimentTracking
        tracking = getattr(cls, '_instance', None)
        if tracking is None:
            tracking = cls(self.session)
        elif (not hasattr(tracking, '_run') or tracking._run is not None
              or getattr(tracking, '_session', None) is not self.session):
            logger.warning('Snowflake ExperimentTracking is in use in this '
                           'process; this fit is not tracked in Snowflake')
            return
        self._tracking = tracking
        database = self.database or self.session.get_current_database()
        schema = self.schema or (None if self.database else
                                 self.session.get_current_schema())
        experiment = tracking.set_experiment(self.experiment_name,
                                             database_name=database,
                                             schema_name=schema or 'PUBLIC')
        self._run = tracking.start_run(
            self._new_run_name(tracking, experiment.name,
                               record.tags['kumo.run_id']))
        self.run_id = str(self._run.name)
        if self.native_model_lineage:
            self._run_context.enter_context(self._run)
        self._start_tags = set(record.tags)
        self._log_params({**record.params, **record.tags})
        self._log_artifacts(record.artifacts)

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
                self._log_metrics(record.metrics, _MAX_METRICS)
                self._log_params({k: v
                                  for k, v in record.tags.items()
                                  if k not in self._start_tags}, _MAX_PARAMS)
                self._log_artifacts(record.artifacts)
                self._log_dropped()
                if error is None:
                    status = 'FINISHED'
        finally:
            try:
                if self._run is not None:
                    self._tracking.end_run(status=status)
            except Exception:
                logger.warning('Could not end the Snowflake run %s', status,
                               exc_info=True)
            finally:
                self._run_context.close()
                self._reset()

    def _new_run_name(self, tracking: Any, experiment: Any,
                      run_id: str) -> Optional[str]:
        if self.run_name is None:
            return None
        # start_run(name) resumes, or raises on, any run matching name as a
        # SQL LIKE pattern and cannot create an existing run, so pick a name
        # that is neither.
        existing = [
            str(row['name'])
            for row in tracking._sql_client.show_runs_in_experiment(
                experiment_name=experiment)
        ]
        short = run_id.rsplit('-', 1)[-1]
        for name in (self.run_name, _suffixed(self.run_name, short[:8]),
                     _suffixed(self.run_name, short)):
            if not _taken(name, existing):
                break
        else:
            name = None
        if name != self.run_name:
            logger.warning(
                'Snowflake run %s already exists in experiment %s; recording '
                'this fit as %s', self.run_name, self.experiment_name,
                name or 'a generated run name')
        return name

    def _log_artifacts(self, artifacts: Mapping[str, str]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, text in artifacts.items():
                if name in self._artifacts:
                    continue
                path = Path(directory) / name
                path.write_text(text, encoding='utf-8')
                self._tracking.log_artifact(str(path))
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
