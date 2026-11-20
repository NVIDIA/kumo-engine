# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-training explainability widget for one trained model.

The data side reads the Kumo files a training run leaves on disk and computes
the chart values; the front end in ``static/`` draws them and derives the
cutoff statistics from the transported counts. Open it with
``TrainingResult.visualize()``, or ``open_run`` for a retained run.
"""

import json
import posixpath
from functools import cache, cached_property
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from kumo.engine import _widget
from kumo.engine._explainability import (
    _classification,
    _columns,
    _metrics,
    _ranking,
    _subgraphs,
    _training,
)
from kumo.engine._explainability._forecasting import forecast_card
from kumo.engine._explainability._regression import predicted_card
from kumo.engine._explainability._run import (
    FORECASTING,
    LINK_PREDICTION,
    Run,
    load,
    mapping,
    read_json,
)
from kumo.engine._explainability._snapshot import Snapshot

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult

# The cards of the Evaluation, Training and Explanation tabs, in order. A card
# that does not apply to the model type or whose inputs are missing is left
# out; a tab with no card is left out.
TABS = (('evaluation',
         (('metrics', _metrics.card), ('cutoff', _classification.cutoff_card),
          ('scores', _classification.scores_card), ('ranking', _ranking.card),
          ('predicted', predicted_card), ('forecast', forecast_card))),
        ('training', (('history', _training.history_card),
                      ('training_table', _training.table_card))),
        ('explanation', (('subgraphs', _subgraphs.card), ('importance',
                                                          _columns.importance),
                         ('groups', _columns.groups))))

# The page around a standalone widget, as the widget's own HTML export sets it.
PAGE_STYLE = ('html{color-scheme:light dark}body{margin:clamp(8px,2vw,24px);'
              'background:light-dark(white,#1a1c1f)}'
              '.kumo-widget{max-width:1200px;margin:auto}')


def build(run: Run) -> dict[str, Any]:
    """The cards and header of one run, as the front end reads them."""
    tabs = []
    for tab_id, spec in TABS:
        cards = [
            dict(id=card_id, **card) for card_id, make in spec
            if (card := make(run)) is not None
        ]
        if cards:
            tabs.append(dict(id=tab_id, label=tab_id.capitalize(), cards=cards))
    return dict(model_type=run.model_type, task_type=run.task_type,
                entities=run.entities, header=_metrics.header(run), tabs=tabs)


@cache
def _static(name: str) -> str:
    return (files(__name__) / 'static' / name).read_text(encoding='utf-8')


def script(model_type: str) -> str:
    """The shared front end, the model type's own cards, then the mount."""
    names = [
        'charts.js', 'shell.js', 'epoch-history.js', 'training-table.js',
        'exports.js'
    ]
    names.append('ranking.js' if model_type ==
                 LINK_PREDICTION else f'{model_type}.js')
    if model_type != FORECASTING:
        names.append('subgraphs.js')
    # The widget layer runs the script inside one function; keep it from
    # closing its own <script> element.
    code = '\n'.join(_static(name) for name in names).replace('</', '<\\/')
    return (
        'function renderExplainability({host,widgetId,data,signal,onDispose}, saved) {\n'
        'const pageCSS = ' + json.dumps(PAGE_STYLE) + ';\n' +
        'const exportCSS = ' + json.dumps(_static('widget.css')) + ';\n' +
        code + '''
const fallback = host.querySelector(':scope > .kx-static');
onDispose(() => {
  host.querySelector(':scope > .kx')?.remove();
  fallback.hidden = false;
});
const root = KX.mount(host, data, signal, onDispose);
fallback.hidden = true;
widgetExports(root, {host, data, signal, onDispose, saved, css: exportCSS,
  source: renderExplainability.toString(), render: renderExplainability});
}
renderExplainability({host,widgetId,data,signal,onDispose});
''')


class ExplainabilityView:
    """The explainability widget of one trained model.

    Displaying it in a notebook cell renders the widget; ``html()`` returns
    the same self-contained HTML for saving to a file.
    """
    def __init__(self, result: 'TrainingResult') -> None:
        self._result = result
        self._widget: Any = None

    @cached_property
    def data(self) -> dict[str, Any]:
        """The payload the front end draws."""
        return build(load(self._result))

    @property
    def model_type(self) -> str:
        return self.data['model_type']

    @cached_property
    def _snapshot(self) -> Snapshot:
        return Snapshot(self.data)

    def _specification(self) -> dict[str, Any]:
        return dict(body=self._snapshot.html(), style=_static('widget.css'),
                    script=script(self.model_type), data=self.data,
                    class_name='kumo-explainability')

    def html(self) -> str:
        """The widget as one self-contained HTML fragment."""
        return _widget.render_widget(**self._specification())

    def page(self, title: str = 'Training report') -> str:
        """The widget as a standalone, interactive HTML page.

        The page loads nothing from outside itself. Open it in a browser
        or an artifact viewer that permits scripted HTML.
        """
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,'
                f'initial-scale=1"><title>{escape(title)}</title>'
                f'<style>{PAGE_STYLE}</style></head><body>{self.html()}'
                '</body></html>')

    def save(self, path: Union[str, Path],
             title: str = 'Training report') -> Path:
        """Write :meth:`page` to ``path`` and return the path."""
        path = Path(path)
        path.write_text(self.page(title), encoding='utf-8')
        return path

    def as_widget(self) -> Any:
        """Return a cached anywidget view, for hosts that need one."""
        if self._widget is None or getattr(self._widget, 'comm', 1) is None:
            self._widget = _widget.render_anywidget(
                fallback_mime=self._snapshot.mime, **self._specification())
        return self._widget

    def _repr_html_(self) -> str:
        return self.html()

    def _repr_svg_(self) -> str:
        return self._snapshot.svg()

    def _repr_pretty_(self, printer: Any, _cycle: bool) -> None:
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
        return f'<ExplainabilityView {self._result.bundle_dir}>'


def open_run(run_root: str) -> ExplainabilityView:
    """The widget of a retained run whose bundle is ``<run>/published-model``.

    A bundle on object storage is copied to scratch first, as
    ``Trainer.load_from_connector`` does.
    """
    from kumo_core.io import filesystem

    from kumo.engine.explain import TrainingExplanations
    from kumo.engine.trainer._spec import read_bundle_artifact_paths
    from kumo.engine.trainer.trainer import (
        TrainingResult,
        _fetch_dir,
        _listed_holdout,
    )

    explanations = TrainingExplanations.load(run_root)
    bundle = posixpath.join(str(explanations.run_root), 'published-model')
    if not filesystem.is_local(bundle):
        local = _fetch_dir('fetched-run-')
        filesystem.copy(bundle, str(local))
        bundle = str(local)
    paths, bundle_id = read_bundle_artifact_paths(Path(bundle))
    holdout, truncated = _listed_holdout(Path(bundle))
    return ExplainabilityView(
        TrainingResult(
            mapping(read_json(posixpath.join(bundle, 'metrics.json'))), paths,
            Path(bundle), holdout_path=holdout, holdout_truncated=truncated,
            bundle_id=bundle_id, explanations=explanations))
