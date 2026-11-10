# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy access to explanation artifacts from one selected training trial."""

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kumo_core.io import filesystem
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.gradient import GraphGradientScore
from kumoapi.explain.subgraph import Subgraph
from typing_extensions import Self

from kumo.engine.explain._artifacts import (
    artifact_root,
    list_samples,
    read_cohorts,
    read_gradient,
    read_sample,
    resolve_run,
    sample_paths,
)
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
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _CohortArtifacts:
    """One concrete view of cohort and sampled-graph producer paths."""

    cohort_path: str
    collector_root: str

    def column_analysis(self) -> ColumnAnalysisOutput:
        """Return validated cohort predictions, populations and targets."""
        return read_cohorts(self.cohort_path)

    def cohort_keys(self) -> tuple[CohortKey, ...]:
        return tuple((cohort.table_name, cohort.hop, cohort.col_name)
                     for cohort in self.column_analysis().cohorts)

    def classes(self, key: CohortKey) -> tuple[str, ...]:
        """List recorded class labels; binary/regression vectors return empty."""
        cohort = self.column_analysis()[key]
        return (tuple(cohort.predictions)
                if isinstance(cohort.predictions, dict) else ())

    def cohort(self, key: CohortKey, *,
               class_name: str | None = None) -> ColumnCohorts:
        """Select a cohort, explicitly choosing among recorded class labels.

        The shared cohort object retains every class. ``class_name`` validates
        selection without replacing the original dictionaries or score values.
        Binary classification records only a positive-class prediction vector,
        so it has no class dictionary to select; regression also uses vectors.
        """
        cohort = self.column_analysis()[key]
        if isinstance(cohort.predictions, dict):
            if class_name is None:
                raise ValueError('Select a classification class explicitly; '
                                 'use classes(key) to list labels.')
            if class_name not in cohort.predictions:
                raise KeyError(f'Unknown classification class {class_name!r}')
        elif class_name is not None:
            raise ValueError('Prediction vectors do not have class labels')
        return cohort

    def samples(self) -> tuple[SampleReference, ...]:
        """List collector IDs and optional metadata without loading graphs."""
        return list_samples(self.collector_root)

    def sample(self, artifact_id: str) -> Subgraph:
        """Load one graph by collector ID, independently of mapping metadata."""
        return read_sample(self.collector_root, artifact_id)

    def plot_cohort(self, key: CohortKey, *, class_name: str | None = None
                    ) -> tuple['Figure', tuple['Axes', 'Axes']]:
        """Return prediction/target and population panels without displaying."""
        from kumo.engine.explain.plotting import plot_cohort

        return plot_cohort(self.cohort(key, class_name=class_name),
                           class_name=class_name)


@dataclass(frozen=True)
class TrainingExplanations:
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
    def from_trial(cls, run_root: str | Path, trial_root: str | Path, *,
                   disabled: bool = False, failure: str | None = None) -> Self:
        """Attach an already selected trial during fit, without reading files."""
        reason: UnavailableReason = ('disabled' if disabled else
                                     'failed' if failure else 'unknown')
        return cls(artifact_root(run_root), artifact_root(trial_root), reason,
                   failure)

    @classmethod
    def unavailable(cls, *,
                    reason: UnavailableReason = 'model_only') -> Self:
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
                present, path, None if present else self._reason, self._failure)
        present = samples is not None and bool(sample_paths(samples))
        states['samples'] = ArtifactAvailability(
            present, samples, None if present else self._reason, self._failure)
        return states

    def _require(self, relative: str) -> str:
        path = self._path(relative)
        if path is None:
            raise ExplanationUnavailableError(
                f'Training explanations unavailable ({self._reason}); '
                'retain and open the training run separately.')
        return path

    def gradient_scores(self) -> GraphGradientScore:
        """Return table/column gradient scores in their shared typed schema."""
        return read_gradient(self._require('explain_test/scores.csv'))

    def _cohort_artifacts(self) -> _CohortArtifacts:
        return _CohortArtifacts(
            self._require('column_analysis/test/column_analysis.json'),
            self._require('subgraph_collector/test'))

    def column_analysis(self) -> ColumnAnalysisOutput:
        """Return validated cohort predictions, populations and targets."""
        return self._cohort_artifacts().column_analysis()

    def cohort_keys(self) -> tuple[CohortKey, ...]:
        return self._cohort_artifacts().cohort_keys()

    def classes(self, key: CohortKey) -> tuple[str, ...]:
        """List recorded class labels; binary/regression vectors return empty."""
        return self._cohort_artifacts().classes(key)

    def cohort(self, key: CohortKey, *,
               class_name: str | None = None) -> ColumnCohorts:
        """Select a cohort, explicitly choosing among recorded class labels."""
        return self._cohort_artifacts().cohort(key, class_name=class_name)

    def samples(self) -> tuple[SampleReference, ...]:
        """List collector IDs and optional metadata without loading graphs."""
        return self._cohort_artifacts().samples()

    def sample(self, artifact_id: str) -> Subgraph:
        """Load one graph by collector ID, independently of mapping metadata."""
        return self._cohort_artifacts().sample(artifact_id)

    def plot_gradient(self, *, top_k: int = 20) -> tuple['Figure', 'Axes']:
        """Return a ranked gradient bar figure and axes without displaying it."""
        from kumo.engine.explain.plotting import plot_gradient

        return plot_gradient(self.gradient_scores(), top_k=top_k)

    def plot_cohort(self, key: CohortKey, *, class_name: str | None = None
                    ) -> tuple['Figure', tuple['Axes', 'Axes']]:
        """Return prediction/target and population panels without displaying."""
        return self._cohort_artifacts().plot_cohort(key, class_name=class_name)


@dataclass(frozen=True)
class PredictionExplanations:
    """Lazy cohorts and sampled graphs from one retained prediction run.

    These diagnostics describe a bounded sample of the current prediction
    inputs, without observed targets. Keep the run separately from exported
    predictions and published model bundles.
    """

    run_root: str | None
    _reason: UnavailableReason = 'unknown'
    _failure: str | None = None
    _cohorts_supported: bool | None = None
    _samples_supported: bool | None = None

    @classmethod
    def load(cls, run_root: str | Path) -> Self:
        """Reopen a retained prediction run; absent artifact causes are unknown.

        Supply the engine run root, whose ``xai/`` directory holds outputs.
        Paths resolve beneath this root, including when a run has been copied.
        """
        root = artifact_root(run_root)
        if not filesystem.exists(root):
            raise InvalidRunError(f'Prediction run is not retained: {root}')
        try:
            # fsspec lists a file as itself rather than raising NotADirectory.
            if any(artifact_root(item) == root
                   for item in filesystem.listdir(root)):
                raise ValueError('run root is a file')
        except (OSError, ValueError) as error:
            raise InvalidRunError(
                f'Expected a prediction run directory: {root}') from error
        return cls(root)

    @classmethod
    def from_run(cls, run_root: str | Path, *, enabled: bool = False,
                 failure: str | None = None,
                 cohorts_supported: bool | None = None,
                 samples_supported: bool | None = None) -> Self:
        """Attach generation status without reading any artifact bodies."""
        reason: UnavailableReason = ('disabled' if not enabled else
                                     'failed' if failure else 'unknown')
        return cls(artifact_root(run_root), reason, failure,
                   cohorts_supported, samples_supported)

    @classmethod
    def unavailable(cls) -> Self:
        """An exported prediction has no retained explanation-run reference."""
        return cls(None)

    def _path(self, relative: str) -> str | None:
        return (None if self.run_root is None else
                posixpath.join(self.run_root, 'xai', relative))

    @property
    def availability(self) -> dict[str, ArtifactAvailability]:
        """Inspect presence; support and failures do not hide partial outputs."""
        states = {}
        paths = (
            ('cohorts', 'column_analysis/pred/column_analysis.json',
             self._cohorts_supported),
            ('samples', 'subgraph_collector/pred/subgraph_json',
             self._samples_supported),
        )
        for kind, relative, supported in paths:
            path = self._path(relative)
            present = path is not None and (
                bool(sample_paths(path)) if kind == 'samples'
                else filesystem.exists(path))
            reason: UnavailableReason | None = (
                None if present else 'unsupported' if supported is False
                else self._reason)
            states[kind] = ArtifactAvailability(
                present, path, reason, self._failure)
        return states

    def _cohort_artifacts(self) -> _CohortArtifacts:
        cohort = self._path('column_analysis/pred/column_analysis.json')
        collector = self._path('subgraph_collector/pred')
        if cohort is None or collector is None:
            raise ExplanationUnavailableError(
                'Prediction explanations unavailable; retain and open '
                'the prediction run separately.')
        return _CohortArtifacts(cohort, collector)

    def column_analysis(self) -> ColumnAnalysisOutput:
        """Return validated cohort predictions, populations and targets."""
        return self._cohort_artifacts().column_analysis()

    def cohort_keys(self) -> tuple[CohortKey, ...]:
        return self._cohort_artifacts().cohort_keys()

    def classes(self, key: CohortKey) -> tuple[str, ...]:
        """List recorded class labels; binary/regression vectors return empty."""
        return self._cohort_artifacts().classes(key)

    def cohort(self, key: CohortKey, *,
               class_name: str | None = None) -> ColumnCohorts:
        """Select a cohort, explicitly choosing among recorded class labels."""
        return self._cohort_artifacts().cohort(key, class_name=class_name)

    def samples(self) -> tuple[SampleReference, ...]:
        """List collector IDs and optional metadata without loading graphs."""
        return self._cohort_artifacts().samples()

    def sample(self, artifact_id: str) -> Subgraph:
        """Load one graph by collector ID, independently of mapping metadata."""
        return self._cohort_artifacts().sample(artifact_id)

    def plot_cohort(self, key: CohortKey, *, class_name: str | None = None
                    ) -> tuple['Figure', tuple['Axes', 'Axes']]:
        """Return prediction and sampled population panels without displaying."""
        return self._cohort_artifacts().plot_cohort(key, class_name=class_name)
