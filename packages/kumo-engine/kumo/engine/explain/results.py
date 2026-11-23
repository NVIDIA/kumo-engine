# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy access to retained training and prediction explanation artifacts."""

import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kumo_core.io import filesystem
from kumo_core.prediction_explanation_result import (
    PredictionExplanationCoverage,
)
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.gradient import GraphGradientScore
from kumoapi.explain.subgraph import Subgraph
from typing_extensions import Self

from kumo.engine.explain._artifacts import (
    artifact_root,
    list_samples,
    mapped_sample_ids,
    read_cohorts,
    read_gradient,
    read_sample,
    resolve_run,
    sample_paths,
)
from kumo.engine.explain._prediction_manifest import read_record
from kumo.engine.explain.errors import (
    ExplanationUnavailableError,
    InvalidRunError,
)
from kumo.engine.explain.types import (
    ArtifactAvailability,
    CohortKey,
    SampleReference,
    UnavailableReason,
)

if TYPE_CHECKING:
    from kumo_core.prediction_explanation_result import (
        PredictionExplanationResult,
        ProducerOutcome,
    )
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class _CohortExplanations(ABC):
    @abstractmethod
    def _cohort_path(self) -> str:
        """Resolve an eligible cohort producer's artifact."""

    @abstractmethod
    def _collector_root(self) -> str:
        """Resolve an eligible sampled-graph producer's directory."""

    @abstractmethod
    def _sample_ids(self) -> frozenset[str] | None:
        """Restrict prediction graphs to saved selection mappings."""

    def column_analysis(self) -> ColumnAnalysisOutput:
        """Return validated cohort predictions, populations and targets."""
        return read_cohorts(self._cohort_path())

    def cohort_keys(self) -> tuple[CohortKey, ...]:
        return tuple(
            (cohort.table_name, cohort.hop, cohort.col_name)
            for cohort in self.column_analysis().cohorts
        )

    def classes(self, key: CohortKey) -> tuple[str, ...]:
        """List recorded class labels; binary/regression vectors return empty."""
        cohort = self.column_analysis()[key]
        return tuple(cohort.predictions) if isinstance(cohort.predictions, dict) else ()

    def cohort(self, key: CohortKey, *, class_name: str | None = None) -> ColumnCohorts:
        """Select a cohort, explicitly choosing among recorded class labels.

        The shared cohort object retains every class. ``class_name`` validates
        selection without replacing the original dictionaries or score values.
        Binary classification records only a positive-class prediction vector,
        so it has no class dictionary to select; regression also uses vectors.
        """
        cohort = self.column_analysis()[key]
        if isinstance(cohort.predictions, dict):
            if class_name is None:
                raise ValueError(
                    'Select a classification class explicitly; '
                    'use classes(key) to list labels.'
                )
            if class_name not in cohort.predictions:
                raise KeyError(f'Unknown classification class {class_name!r}')
        elif class_name is not None:
            raise ValueError('Prediction vectors do not have class labels')
        return cohort

    def samples(self) -> tuple[SampleReference, ...]:
        """List collector IDs and optional metadata without loading graphs."""
        root, ids = self._collector_root(), self._sample_ids()
        return tuple(
            item
            for item in list_samples(root)
            if ids is None or item.artifact_id in ids
        )

    def sample(self, artifact_id: str) -> Subgraph:
        """Load one eligible graph by its collector ID."""
        root, ids = self._collector_root(), self._sample_ids()
        if ids is not None and artifact_id not in ids:
            raise ExplanationUnavailableError(
                f'Sample {artifact_id!r} is not a completed artifact'
            )
        return read_sample(root, artifact_id)

    def plot_cohort(
        self, key: CohortKey, *, class_name: str | None = None
    ) -> tuple['Figure', tuple['Axes', 'Axes']]:
        """Return prediction/target and population panels without displaying."""
        from kumo.engine.explain.plotting import plot_cohort

        return plot_cohort(
            self.cohort(key, class_name=class_name), class_name=class_name
        )


@dataclass(frozen=True)
class TrainingExplanations(_CohortExplanations):
    """Retained TEST-stage training diagnostics, without model computation.

    Scores preserve recorded values, including producer normalization and CSV
    rounding. Cohort variation and gradient scores are different quantities;
    neither is causal evidence. Sampled graphs describe bounded training
    context and cannot explain arbitrary entities. Keep the training run
    separately: published model bundles do not retain these artifacts.
    """

    run_root: str | None
    trial_root: str | None
    _reason: UnavailableReason = 'unknown'
    _failure: str | None = None

    @classmethod
    def load(cls, run_root: str | Path) -> Self:
        """Open exactly one retained run, including its selected TEST refit.

        Accepts the run containing ``models/<task>/automl/best_trainer.json``.
        Multiple selected-task records and model-only bundles are rejected.
        Older runs cannot establish why a particular artifact is absent.
        """
        root, trial = resolve_run(run_root)
        return cls(root, trial)

    @classmethod
    def from_trial(
        cls,
        run_root: str | Path,
        trial_root: str | Path,
        *,
        disabled: bool = False,
        failure: str | None = None,
    ) -> Self:
        """Attach an already selected trial during fit, without reading files."""
        reason: UnavailableReason = (
            'disabled' if disabled else 'failed' if failure else 'unknown'
        )
        return cls(artifact_root(run_root), artifact_root(trial_root), reason, failure)

    @classmethod
    def unavailable(cls, *, reason: UnavailableReason = 'model_only') -> Self:
        """Represent a restored model whose training artifacts are unknown."""
        return cls(None, None, reason)

    def _path(self, name: str) -> str | None:
        if self.trial_root is None:
            return None
        return posixpath.join(self.trial_root, name)

    @property
    def availability(self) -> dict[str, ArtifactAvailability]:
        """Inspect artifact presence without parsing graph bodies or data."""
        gradient = self._path('explain_test/scores.csv')
        cohorts = self._path('column_analysis/test/column_analysis.json')
        samples = self._path('subgraph_collector/test/subgraph_json')
        states = {}
        for kind, path in (('gradient', gradient), ('cohorts', cohorts)):
            present = path is not None and filesystem.exists(path)
            states[kind] = ArtifactAvailability(
                present, path, None if present else self._reason, self._failure
            )
        present = samples is not None and bool(sample_paths(samples))
        states['samples'] = ArtifactAvailability(
            present, samples, None if present else self._reason, self._failure
        )
        return states

    def _require(self, relative: str) -> str:
        path = self._path(relative)
        if path is None:
            raise ExplanationUnavailableError(
                f'Training explanations unavailable ({self._reason}); '
                'retain and open the training run separately.'
            )
        return path

    def gradient_scores(self) -> GraphGradientScore:
        """Return table/column gradient scores in their shared typed schema."""
        return read_gradient(self._require('explain_test/scores.csv'))

    def _cohort_path(self) -> str:
        return self._require('column_analysis/test/column_analysis.json')

    def _collector_root(self) -> str:
        return self._require('subgraph_collector/test')

    def _sample_ids(self) -> frozenset[str] | None:
        return None

    def plot_gradient(self, *, top_k: int = 20) -> tuple['Figure', 'Axes']:
        """Return a ranked gradient bar figure and axes without displaying it."""
        from kumo.engine.explain.plotting import plot_gradient

        return plot_gradient(self.gradient_scores(), top_k=top_k)


@dataclass(frozen=True)
class PredictionExplanations(_CohortExplanations):
    """Lazy cohorts and sampled graphs from one retained prediction run.

    These diagnostics describe a bounded sample of the current prediction
    inputs, without observed targets. Keep the run separately from exported
    predictions and published model bundles.
    """

    run_root: str | None
    _reason: UnavailableReason = 'unknown'
    _result: 'PredictionExplanationResult | None' = None
    _run_id: str | None = None

    @classmethod
    def load(cls, run_root: str | Path) -> Self:
        """Reopen a validated retained run with its recorded producer outcomes.

        Supply the engine run root, whose ``xai/`` directory holds outputs.
        Paths resolve beneath this root, including when a run has been copied.
        """
        root = artifact_root(run_root)
        try:
            record = read_record(root)
        except (OSError, ValueError) as error:
            raise InvalidRunError(
                f'Invalid retained prediction explanations at {root}: {error}'
            ) from error
        if record.state == 'running':
            return cls(root, 'failed', record.result, record.run_id)
        return cls.from_run(
            root, enabled=record.enabled, result=record.result, run_id=record.run_id
        )

    @classmethod
    def from_run(
        cls,
        run_root: str | Path,
        *,
        enabled: bool = False,
        result: 'PredictionExplanationResult | None' = None,
        run_id: str | None = None,
    ) -> Self:
        """Attach generation status without reading any artifact bodies."""
        reason: UnavailableReason = 'unknown' if enabled else 'disabled'
        return cls(artifact_root(run_root), reason, result, run_id)

    @classmethod
    def unavailable(cls) -> Self:
        """An exported prediction has no retained explanation-run reference."""
        return cls(None)

    def _path(self, relative: str) -> str | None:
        return (
            None
            if self.run_root is None
            else posixpath.join(self.run_root, 'xai', relative)
        )

    @property
    def run_id(self) -> str | None:
        """The recorded prediction identity, independent of the retained path."""
        return self._run_id

    @property
    def coverage(self) -> PredictionExplanationCoverage:
        """Observed generation counts and the published checkpoint identity."""
        return (
            self._result.coverage
            if self._result is not None
            else PredictionExplanationCoverage()
        )

    def _producer(
        self, kind: Literal['cohorts', 'samples']
    ) -> 'ProducerOutcome | None':
        if self._result is None:
            return None
        return self._result.cohorts if kind == 'cohorts' else self._result.samples

    def _availability(
        self, kind: Literal['cohorts', 'samples']
    ) -> ArtifactAvailability:
        relative = (
            'column_analysis/pred/column_analysis.json'
            if kind == 'cohorts'
            else 'subgraph_collector/pred/subgraph_json'
        )
        path = self._path(relative)
        outcome = self._producer(kind)
        reason: UnavailableReason | None = self._reason
        if outcome is not None and outcome.status != 'complete':
            reason = 'unsupported' if outcome.status == 'unsupported' else 'failed'
        error = outcome.error if outcome else None
        if (
            outcome is None
            or outcome.status != 'complete'
            or reason in {'disabled', 'failed'}
            or path is None
        ):
            return ArtifactAvailability(False, path, reason, error)
        if kind == 'samples':
            collector = self._path('subgraph_collector/pred')
            ids = frozenset() if collector is None else mapped_sample_ids(collector)
            present = any(
                filesystem.exists(posixpath.join(path, f'{id}.json')) for id in ids
            )
        else:
            present = filesystem.exists(path)
        return ArtifactAvailability(present, path, None if present else reason, error)

    @property
    def availability(self) -> dict[str, ArtifactAvailability]:
        """Only completed producers expose their present artifacts."""
        return {
            'cohorts': self._availability('cohorts'),
            'samples': self._availability('samples'),
        }

    def _require_producer(self, kind: Literal['cohorts', 'samples']) -> None:
        outcome = self._producer(kind)
        if (
            outcome is None
            or outcome.status != 'complete'
            or self._reason in {'disabled', 'failed'}
        ):
            reason = (
                ('unsupported' if outcome.status == 'unsupported' else 'failed')
                if outcome is not None
                else self._reason
            )
            error = outcome.error if outcome is not None else None
            raise ExplanationUnavailableError(
                f'Prediction {kind} unavailable ({reason})'
                + (f': {error}' if error else '')
            )

    def _cohort_path(self) -> str:
        self._require_producer('cohorts')
        path = self._path('column_analysis/pred/column_analysis.json')
        if path is None:
            raise ExplanationUnavailableError('Prediction run is unavailable')
        return path

    def _collector_root(self) -> str:
        self._require_producer('samples')
        path = self._path('subgraph_collector/pred')
        if path is None:
            raise ExplanationUnavailableError('Prediction run is unavailable')
        return path

    def _sample_ids(self) -> frozenset[str]:
        return mapped_sample_ids(self._collector_root())
