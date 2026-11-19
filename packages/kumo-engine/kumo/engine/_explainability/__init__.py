# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-training explainability widget for one trained model.

The data side reads the Kumo files a training run leaves on disk and computes
every number; the front end in ``static/`` only draws them. Open it with
``TrainingResult.visualize()``.
"""

from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from kumo.engine import _widget
from kumo.engine._explainability import _cards, _files, _render, _run
from kumo.engine._explainability._snapshot import Snapshot

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult


class ExplainabilityView:
    """The explainability widget of one trained model.

    Displaying it in a notebook cell renders the widget; ``html()`` returns
    the same self-contained HTML for saving to a file.
    """
    def __init__(self, files: _files.RunFiles) -> None:
        self._files = files
        self._widget: Any = None

    @cached_property
    def data(self) -> dict[str, Any]:
        """Every number and text the widget draws."""
        return _cards.build(_run.load(self._files))

    @property
    def model_type(self) -> str:
        return self.data['model_type']

    @cached_property
    def _snapshot(self) -> Snapshot:
        return Snapshot(self.data)

    def html(self) -> str:
        """The widget as one self-contained HTML fragment."""
        return _widget.render_widget(
            **_render.specification(self.data, self._snapshot.html()))

    def as_widget(self) -> Any:
        """Return a cached anywidget view, for hosts that need one."""
        if self._widget is None or getattr(self._widget, 'comm', 1) is None:
            self._widget = _widget.render_anywidget(
                fallback_mime=self._snapshot.mime,
                **_render.specification(self.data, self._snapshot.html()))
        return self._widget

    def _repr_html_(self) -> str:
        return self.html()

    def _repr_svg_(self) -> str:
        return self._snapshot.svg()

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(self._snapshot.plain())

    def _ipython_display_(self) -> None:
        if not _widget.is_snowflake_workspace():
            raise NotImplementedError
        from IPython.display import display

        try:
            widget = self.as_widget()
        except ImportError:
            bundle = self._snapshot.mime()
            hint = ('Interactive display requires ipywidgets and anywidget '
                    'version 0.9 or later in the 0.x series.')
            bundle['text/html'] += f'<p>{hint}</p>'
            bundle['text/plain'] += '\n' + hint
            display(bundle, raw=True)
        else:
            display(widget)

    def __repr__(self) -> str:
        return f'<ExplainabilityView {self._files.bundle}>'


def for_result(result: 'TrainingResult') -> ExplainabilityView:
    files = _files.locate(str(result.bundle_dir), result.explanations.run_root,
                          result.explanations.trial_root)
    return ExplainabilityView(
        replace(
            files, holdout_path=str(result.holdout_path) if result.holdout_path
            else None, holdout_truncated=result.holdout_truncated))


def open_run(run_root: str) -> ExplainabilityView:
    """The widget for a retained run whose bundle is in the run directory."""
    return ExplainabilityView(_files.open_run(run_root))
