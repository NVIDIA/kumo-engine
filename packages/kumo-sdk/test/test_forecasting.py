# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import plotly.io as pio

from kumoai_studio.utils.forecasting import ForecastVisualizer


def test_forecast_visualization_preserves_values_and_entity_selection(
        monkeypatch):
    data = pd.DataFrame({
        "ENTITY": ["b", "a", "b", "a"],
        "TIMESTAMP":
        pd.to_datetime(
            ["2026-01-02", "2026-01-01", "2026-01-01", "2026-01-02"]),
        "TARGET": [4.0, 10.0, 2.0, 12.0],
        "TARGET_PRED": [3.0, 9.0, 1.0, 14.0],
    })
    visualizer = ForecastVisualizer(data)
    shown = []
    monkeypatch.setattr(type(visualizer.fig), "show",
                        lambda figure: shown.append(figure))

    visualizer.visualize()

    assert shown == [visualizer.fig]
    traces = visualizer.fig.data
    assert len(traces) == 8
    assert list(traces[0].y) == [2.0, 4.0]
    assert list(traces[1].y) == [1.0, 3.0]
    assert list(traces[2].y) == list(traces[3].x) == [1.0, 1.0]
    assert list(traces[6].y) == list(traces[7].x) == [1.0, -2.0]
    assert [trace.visible for trace in traces] == [True] * 4 + [False] * 4
    buttons = visualizer.fig.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == ["b", "a"]
    assert list(buttons[1].args[0]["visible"]) == [False] * 4 + [True] * 4
    restored = pio.from_json(visualizer.fig.to_json())
    assert len(restored.data) == 8
    assert "Plotly.newPlot" in restored.to_html(include_plotlyjs=True)
