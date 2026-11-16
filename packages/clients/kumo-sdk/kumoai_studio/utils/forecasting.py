# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ForecastVisualizer:
    r"""A tool for quickly visualizing forecast results from the holdout
    dataframe of a forecasting job.

    .. code-block:: python

        import kumoai_studio

        # Retrieve job results from a training training job. Note
        # that the job ID passed here must be in a completed state:
        job_result = kumoai_studio.TrainingJob("trainingjob-...").result()

        # Read the holdout table as a Pandas DataFrame:
        holdout_df = job_result.holdout_df()

        # Pass holdout table to ForecastVisualizer and visualize results
        holdout_forecast = kumoai_studio.utils.ForecastVisualizer(holdout_df)
        holdout_forecast.visualize()
    """
    def __init__(self, holdout_df: pd.DataFrame) -> None:
        # Sort the holdout dataframe and extract unique entities:
        self.forecast = holdout_df.sort_values(['ENTITY', 'TIMESTAMP'])
        self.entities = holdout_df['ENTITY'].unique().tolist()

        self.fig = self._initialize_subplot()
        self.buttons: List[Dict] = []
        self.plot_config = {
            'target': {
                'color': 'blue',
                'name': 'TARGET'
            },
            'prediction': {
                'color': 'red',
                'name': 'TARGET_PRED'
            },
            'residuals': {
                'color': 'green',
                'name': 'Residuals'
            },
            'residuals_time': {
                'color': 'orange',
                'name': 'Residuals Over Time'
            }
        }

    @staticmethod
    def _initialize_subplot() -> go.Figure:
        r"""Initializes the subplot structure with three rows:
        Row 1: Line plot of actual forecast vs predicted
        Row 2: Line plot of residuals overtime
        Row 3: Histogram distribution of residuals
        """
        return make_subplots(
            rows=3,
            cols=1,
            specs=[[{
                "type": "scatter"
            }], [{
                "type": "scatter"
            }], [{
                "type": "xy"
            }]],
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Forecast vs Actual', 'Residuals Over Time',
                            'Residuals Distribution'),
        )

    def _create_time_series_trace(
        self,
        data: pd.Series,
        entity: str,
        trace_type: str,
        visibility: bool,
    ) -> go.Scatter:
        r"""Create a time series trace for either target or prediction"""
        config = self.plot_config[trace_type]
        return go.Scatter(
            x=data["TIMESTAMP"],
            y=data[config['name']],
            name=f"{entity} - {config['name']}",
            mode="lines",
            line=dict(color=config['color']),
            visible=visibility,
            opacity=0.75,
        )

    def _create_residuals_time_trace(
        self,
        data: pd.Series,
        entity: str,
        visibility: bool,
    ) -> go.Scatter:
        r"""Create a time series trace for residuals over time"""
        residuals = data["TARGET"] - data["TARGET_PRED"]
        return go.Scatter(
            x=data["TIMESTAMP"],
            y=residuals,
            name=f"{entity} - Residuals Over Time",
            mode="lines+markers",
            line=dict(color=self.plot_config['residuals_time']['color']),
            visible=visibility,
            opacity=0.75,
        )

    def _create_residuals_hist_trace(
        self,
        data: pd.Series,
        entity: str,
        visibility: bool,
    ) -> go.Histogram:
        r"""Create a histogram trace for residuals distribution."""
        residuals = data["TARGET"] - data["TARGET_PRED"]
        return go.Histogram(
            x=residuals,
            name=f"{entity} - Residuals Distribution",
            marker=dict(color=self.plot_config['residuals']['color']),
            visible=visibility,
            opacity=0.75,
            nbinsx=30,
        )

    def _create_button(self, index: int, entity: str) -> None:
        r"""Create visibility toggle button for an entity."""
        # target, prediction, residuals time, and residuals hist:
        num_traces_per_entity = 4
        total_traces = len(self.entities) * num_traces_per_entity

        # Build visibility list with this entity's traces visible:
        visible: list[bool] = [False] * total_traces
        base_index = index * num_traces_per_entity
        for i in range(num_traces_per_entity):
            visible[base_index + i] = True

        button = dict(label=entity, method="update", args=[{
            "visible": visible
        }])
        self.buttons.append(button)

    def _create_traces(self) -> None:
        """Create all traces for the visualization."""
        for i, entity in enumerate(self.entities):
            entity_data = self.forecast.loc[self.forecast.ENTITY == entity]

            # First entity's traces are visible by default:
            visibility = (i == 0)

            # Create traces
            trace_target = self._create_time_series_trace(
                entity_data, entity, 'target', visibility)
            trace_pred = self._create_time_series_trace(
                entity_data, entity, 'prediction', visibility)
            trace_residuals_time = self._create_residuals_time_trace(
                entity_data, entity, visibility)
            trace_residuals_hist = self._create_residuals_hist_trace(
                entity_data, entity, visibility)

            # Add traces to appropriate subplots
            self.fig.add_trace(trace_target, row=1, col=1)
            self.fig.add_trace(trace_pred, row=1, col=1)
            self.fig.add_trace(trace_residuals_time, row=2, col=1)
            self.fig.add_trace(trace_residuals_hist, row=3, col=1)

            self._create_button(i, entity)

    def _update_layout(self) -> None:
        r"""Update the figure layout with all necessary configurations."""
        self.fig.update_layout(
            updatemenus=[
                dict(active=0, buttons=self.buttons, direction="down", pad={
                    "r": 10,
                    "t": 10
                }, showactive=True, x=1, xanchor="left", y=1.07, yanchor="top")
            ],
            title="Forecast Results by Department",
            height=1000,  # Increased height to accommodate third plot
            width=1300,
            showlegend=True,
            hovermode='x unified')

        # Update axis labels and add zero reference line for residuals
        self.fig.update_xaxes(title_text="Timestamp", row=1, col=1)
        self.fig.update_xaxes(title_text="Timestamp", row=2, col=1)
        self.fig.update_xaxes(title_text="Residual Value", row=3, col=1)

        self.fig.update_yaxes(title_text="Patient Volume", row=1, col=1)
        self.fig.update_yaxes(title_text="Residual Value", row=2, col=1)
        self.fig.update_yaxes(title_text="Frequency", row=3, col=1)

        # Add zero reference line for residuals time series
        self.fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            row=2,
            col=1,
        )

    def visualize(self) -> None:
        r"""Generate and display the complete visualization."""
        self._create_traces()
        self._update_layout()
        self.fig.show()
