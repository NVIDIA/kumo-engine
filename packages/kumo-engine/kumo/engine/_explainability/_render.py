# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook rendering: the computed data plus the static front end, through
the same widget layer as the graph visualization."""

import json
from functools import cache
from importlib.resources import files as resource_files
from typing import Any

from kumo.engine._explainability._run import (
    CLASSIFICATION,
    FORECASTING,
    REGRESSION,
)

# Shared front end first, then the file of the model type's own cards.
_SHARED = ('charts.js', 'shell.js', 'epoch-history.js', 'exports.js')
_TYPE_SCRIPT = {
    CLASSIFICATION: 'classification.js',
    REGRESSION: 'regression.js',
    FORECASTING: 'forecasting.js',
}


@cache
def _static(name: str) -> str:
    return (resource_files('kumo.engine._explainability') / 'static' /
            name).read_text(encoding='utf-8')


def style() -> str:
    return _static('widget.css')


def script(model_type: str) -> str:
    names: list[str] = list(_SHARED)
    if model_type in _TYPE_SCRIPT:
        names.append(_TYPE_SCRIPT[model_type])
    if model_type != FORECASTING:
        names.append('subgraphs.js')
    code = '\n'.join(_static(name) for name in names)
    # The widget layer runs the script inside one function; keep it from
    # closing its own <script> element.
    code = code.replace('</', '<\\/')
    return (
        'function renderExplainability({host,widgetId,data,signal,onDispose}, saved) {\n'
        'const exportCSS = ' + json.dumps(style()) + ';\n' + code + '''
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


def specification(data: dict[str, Any], snapshot: str) -> dict[str, Any]:
    """Use the same markup, behavior and cleanup for both notebook transports."""
    return dict(body=snapshot, style=style(), script=script(data['model_type']),
                data=data, class_name='kumo-explainability')
