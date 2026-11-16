// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ScatterPointData = {
    x: number;
    y: number;
};

export type ScatterLineData = {
    coordinates: Array<{ x: number; y: number }>;
    lineColor: string;
    lineWidth?: number;
};

export type HeatMapData = {
    z: number[][];
    x: (string | number)[];
    y: (string | number)[];
};

export type HeatMapLineData = {
    x: number[];
    y: number[];
};

export type HeatMapDataPoint =
    | [number, number, number]
    | {
          value: [number, number, number];
          itemStyle: {
              color: string;
              borderColor: string;
          };
      };

export type ChartPlotPoints = {
    color: string;
    value: Array<number>;
    label: string;
};

export enum jobStatus {
    Queued = "QUEUED",
    Running = "RUNNING",
    Done = "DONE",
    Failed = "FAILED",
    Cancelled = "CANCELLED",
    Untrained = "UNTRAINED",
    None = ""
}

export type QueryProgressStats = {
    xAxisLabels: Array<string | number>;
    datasets: Array<ChartPlotPoints>;
    xLabel: string;
    yLabel: string;
    mode: string;
};

export type ScalarChartData = Record<
    string, // train, val, test, ...
    Record<
        string, // experiment key: "0", "1", ...
        { xData: (number | string)[]; yData: (number | null)[] }
    >
>;

export type BarChartData = {
    value: number;
    itemStyle: {
        color: string | { image: string | HTMLCanvasElement; repeat: string };
        borderColor: string;
        borderWidth: number;
    };
};
export type TargetHistogram = {
    bins?: Array<number>;
    counts?: Array<number>;
    bin_width?: number;
    num_bins?: number;
};

export const lineTypes: Array<string> = [
    "solid", // Default solid line
    "dotted", // Dotted line
    "dashed" // Dashed line
];
