# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Notebook display of a retained training diagnostics snapshot."""

from html import escape

from kumo.engine import _widget


class TrainingVisualization:
    """One bounded snapshot with independent interactive notebook views."""

    def __init__(self, payload: dict):
        self._payload = payload
        self._widget = None

    def __repr__(self) -> str:
        return ('TrainingVisualization: display in a trusted notebook to '
                'inspect retained training diagnostics.')

    def _viewer(self) -> dict:
        from kumo.engine.explain._viewer import viewer_spec

        return viewer_spec(self._payload)

    def _repr_html_(self) -> str:
        return _widget.render_widget(**self._viewer())

    def as_widget(self):
        """Use the host's optional widget runtime, caching the model."""
        if self._widget is None or self._widget.comm is None:
            try:
                self._widget = _widget.render_anywidget(**self._viewer())
            except ImportError as error:
                raise ImportError(
                    'Interactive training diagnostics require anywidget>=0.9,<1 '
                    'and ipywidgets. Restore compatible notebook runtime '
                    'packages, then rerun the display cell.') from error
        return self._widget

    def _ipython_display_(self):
        if not _widget.is_snowflake_workspace():
            raise NotImplementedError
        from IPython.display import display

        from kumo.engine.explain._viewer import static_summary

        try:
            widget = self.as_widget()
        except ImportError as error:
            display({'text/html': static_summary(self._payload) +
                     f'<p>{escape(str(error))}</p>',
                     'text/plain': str(error)}, raw=True)
        else:
            display(widget)
