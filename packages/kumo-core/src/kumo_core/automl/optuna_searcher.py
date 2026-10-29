# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import posixpath
import warnings
from io import BytesIO
from typing import Any

import numpy as np
import optuna
from kumo_core.automl.random_searcher import RandomSearcher
from kumo_core.automl.searcher import PlanT, Searcher
from kumo_core.train.io import makedirs, write_binary

logger = logging.getLogger(__name__)


class OptunaSearcher(Searcher[PlanT]):
    """Sample model plans with Optuna's ask-and-tell interface."""
    def __init__(
        self,
        space: PlanT,
        sampler_name: str,
        max_attempts: int = 50,
        **kwargs: Any,
    ) -> None:
        warnings.filterwarnings(
            "ignore",
            message=".*plot_.*",
            category=optuna.exceptions.ExperimentalWarning,
        )
        super().__init__(space, max_attempts)
        direction = "maximize" if self.tune_mode == "max" else "minimize"
        with warnings.catch_warnings():
            # Multivariate TPE is a deliberate choice, but Optuna flags the
            # argument as experimental on every sampler construction.
            warnings.filterwarnings(
                "ignore",
                message="Argument ``multivariate`` is an experimental feature",
                category=optuna.exceptions.ExperimentalWarning,
            )
            sampler = getattr(optuna.samplers, sampler_name)(**kwargs)
        self.study = optuna.create_study(direction=direction, sampler=sampler)
        warnings.filterwarnings("ignore", ".*Choices for a categorical.*")
        self.fallback_sampler = RandomSearcher(space, max_attempts)

    def sample_instance(self) -> PlanT:
        self.current_trial = self.study.ask()
        return super().sample_instance()

    def update(self, cfg: PlanT, performance: list[float]) -> None:
        eval_fn = {"min": min, "max": max}
        best_performance = eval_fn[self.tune_mode](performance)
        self.study.tell(self.current_trial, best_performance)
        super().update(cfg, performance)

    def suggest(self) -> PlanT:
        for _ in range(self.max_attempts):
            cfg = self.sample_instance()
            if cfg not in [prev_cfg for prev_cfg, _ in self.state]:
                return cfg
        for _ in range(self.fallback_sampler.max_attempts):
            self.current_trial = self.study.ask()
            cfg = self.fallback_sampler.sample_instance()
            if cfg not in [prev_cfg for prev_cfg, _ in self.state]:
                return cfg
        return cfg

    def sample_value(self, key: str, value: list[Any]) -> Any:
        try:
            for choice in value:
                hash(choice)
        except TypeError:
            # Optuna requires categorical choices to be hashable (it uses
            # them as trial-param dict keys during analysis/plotting), but
            # search-space values may be unhashable (e.g. dataclasses,
            # lists). Sample a unique string label so analysis plots remain
            # meaningful, then map it back to the original value.
            choice_labels = [
                f"{index}: {choice}" for index, choice in enumerate(value)
            ]
            choice_label = self.current_trial.suggest_categorical(
                key, choice_labels)
            return value[choice_labels.index(choice_label)]
        return self.current_trial.suggest_categorical(key, value)

    def analyze(self, path: str) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )

        matplotlib.use("Agg")
        makedirs(path)
        complete_trials = [
            trial for trial in self.study.trials
            if trial.state is optuna.trial.TrialState.COMPLETE
        ]
        if not complete_trials:
            return

        def axes_savefig(
            axes: matplotlib.axes.Axes | np.ndarray,
            output_path: str,
            tight_layout: bool = True,
        ) -> None:
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()[0]
            fig = axes.get_figure()
            if tight_layout:
                fig.tight_layout()
            try:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                write_binary(output_path, buf.getvalue())
            except FileNotFoundError:
                logger.exception("Analysis output path does not exist: %s",
                                 output_path)
            finally:
                plt.close(fig)

        # Save (and close) each plot as soon as it's created, rather than
        # creating every plot_slice figure up front and only closing them
        # afterward -- search spaces routinely have 15-30+ tunable params,
        # so holding all of their figures open at once trips matplotlib's
        # open-figure warning well before any of them get closed.
        try:
            optimization_history_axes = plot_optimization_history(self.study)
        except Exception:
            logger.exception("Optuna search analysis failed")
            return
        axes_savefig(
            optimization_history_axes,
            posixpath.join(path, "optimization_history.png"),
        )

        if len(complete_trials) > 1:
            try:
                param_importances_axes = plot_param_importances(self.study)
            except Exception:
                logger.exception(
                    "Optuna search analysis failed (param_importances)")
            else:
                axes_savefig(param_importances_axes,
                             posixpath.join(path, "param_importances.png"))

            slice_path = posixpath.join(path, "slice")
            makedirs(slice_path)
            # A conditional param may not appear until a later trial (e.g.
            # nested search spaces), so union params across all trials
            # rather than reading only complete_trials[0].params.
            all_params = dict.fromkeys(param for trial in complete_trials
                                       for param in trial.params)
            for param in all_params:
                values = {
                    trial.params[param]
                    for trial in complete_trials if param in trial.params
                }
                if len(values) <= 1:
                    # A slice plot shows how the objective varies with this
                    # param; if every trial sampled the same value there's
                    # nothing to show (and optuna's own xlim padding divides
                    # by a zero range for a constant param, which is what
                    # trips matplotlib's identical-xlim warning).
                    continue
                try:
                    plot_slice_axes = plot_slice(self.study, [param])
                except Exception:
                    logger.exception(
                        "Optuna search analysis failed (slice: %s)", param)
                    continue
                axes_savefig(
                    plot_slice_axes,
                    posixpath.join(slice_path, f"{param}.png"),
                    tight_layout=False,
                )


class TPESearcher(OptunaSearcher[PlanT]):
    """Optuna searcher using the Tree-structured Parzen Estimator."""
    def __init__(
        self,
        space: PlanT,
        max_attempts: int = 50,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("n_startup_trials", 4)
        kwargs.setdefault("multivariate", True)
        major, minor = (int(part) for part in optuna.__version__.split(".")[:2])
        if (major, minor) < (4, 9):
            # Optuna 4.9 made False the default and deprecated the argument.
            kwargs.setdefault("warn_independent_sampling", False)
        super().__init__(
            space,
            sampler_name="TPESampler",
            max_attempts=max_attempts,
            **kwargs,
        )


class NSGAIISearcher(OptunaSearcher[PlanT]):
    """Optuna searcher using the NSGA-II sampler."""
    def __init__(
        self,
        space: PlanT,
        max_attempts: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            space,
            sampler_name="NSGAIISampler",
            max_attempts=max_attempts,
            **kwargs,
        )
