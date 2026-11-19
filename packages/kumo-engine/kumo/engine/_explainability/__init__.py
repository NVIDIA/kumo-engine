# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-training explainability widget for one trained model.

The data side reads the Kumo files a training run leaves on disk and computes
every number; the front end in ``static/`` only draws them. Open it with
``TrainingResult.visualize()``.
"""

import json
from functools import cache, cached_property
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from kumo.engine import _widget
from kumo.engine._explainability import (
    _classification,
    _columns,
    _metrics,
    _subgraphs,
)
from kumo.engine._explainability._forecasting import forecast_card
from kumo.engine._explainability._regression import predicted_card
from kumo.engine._explainability._run import (
    FORECASTING,
    LINK_PREDICTION,
    Run,
    load,
)
from kumo.engine._explainability._snapshot import Snapshot

if TYPE_CHECKING:
    from kumo.engine.trainer.trainer import TrainingResult

# The cards of the Evaluation and Explanation tabs, in order. A card that does
# not apply to the model type or whose inputs are missing is left out; a tab
# with no card is left out.
TABS = (('evaluation',
         (('metrics', _metrics.card), ('cutoff', _classification.cutoff_card),
          ('scores', _classification.scores_card),
          ('predicted', predicted_card), ('forecast', forecast_card))),
        ('explanation', (('subgraphs', _subgraphs.card), ('importance',
                                                          _columns.importance),
                         ('groups', _columns.groups))))


def build(run: Run) -> dict[str, Any]:
    """Every number and text the widget draws for one run."""
    tabs = []
    for tab_id, spec in TABS:
        cards = [
            dict(id=card_id, **card) for card_id, make in spec
            if (card := make(run)) is not None
        ]
        if cards:
            tabs.append(dict(id=tab_id, label=tab_id.capitalize(), cards=cards))
    return dict(model_type=run.model_type, entities=run.entities,
                header=_metrics.header(run), tabs=tabs)


@cache
def _static(name: str) -> str:
    return (files(__name__) / 'static' / name).read_text(encoding='utf-8')


def script(model_type: str) -> str:
    """The shared front end, the model type's own cards, then the mount."""
    names = ['charts.js', 'shell.js', 'epoch-history.js', 'exports.js']
    if model_type != LINK_PREDICTION:
        names.append(f'{model_type}.js')
    if model_type != FORECASTING:
        names.append('subgraphs.js')
    # The widget layer runs the script inside one function; keep it from
    # closing its own <script> element.
    code = '\n'.join(_static(name) for name in names).replace('</', '<\\/')
    return (
        'function renderExplainability({host,widgetId,data,signal,onDispose}, saved) {\n'
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
        """Every number and text the widget draws."""
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
        return f'<ExplainabilityView {self._result.bundle_dir}>'
