// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

export const useChartConfig = () => {
    return useMemo(() => {
        const root = document.documentElement;
        const computedStyle = getComputedStyle(root);

        const getCSSVar = (varName: string) => {
            return computedStyle.getPropertyValue(varName).trim();
        };

        const borders = {
            tooltipBorderRadius:
                getCSSVar("--chart-tooltip-border-radius") || "4",
            tooltipBorderWidth: getCSSVar("--chart-tooltip-border-width") || "1"
        };

        const colors = {
            primary: getCSSVar("--chart-primary") || "#3b82f6",
            primaryDark: getCSSVar("--chart-primary-dark") || "#134e4a",
            secondary: getCSSVar("--chart-secondary") || "#10b981",
            ideal: getCSSVar("--chart-ideal") || "#3b82f6",
            axisLine: getCSSVar("--chart-axis-line") || "#3b82f6",
            horizontalLine:
                getCSSVar("--chart-horizontal-axis-line") || "#e5e7eb",
            axisName: getCSSVar("--chart-axis-name") || "#3b82f6",
            axisLabel: getCSSVar("--chart-axis-label") || "#3b82f6",
            tooltipBackground:
                getCSSVar("--chart-tooltip-background") || "#ffffff",
            tooltipForeground:
                getCSSVar("--chart-tooltip-foreground") || "#1e293b",
            tooltipBorderColor:
                getCSSVar("--chart-tooltip-border") || "#f1f5f9",
            borderColor: getCSSVar("--muted-foreground") || "#6e84a3",
            barBackground: getCSSVar("--chart-bar-background") || "#e5e5e5",
            barBorderColor: getCSSVar("--chart-bar-border") || "#404040",
            primaryLight: getCSSVar("--chart-primary-light") || "#14b8a6",
            lime400: getCSSVar("--chart-lime-400") || "#a3e635",
            lime700: getCSSVar("--chart-lime-700") || "#4d7c0f", //olive
            amber800: getCSSVar("--chart-amber-800") || "#92400e", //brown
            red600: getCSSVar("--chart-red-600") || "#dc2626",
            orange400: getCSSVar("--chart-orange-400") || "#fb923c",
            yellow300: getCSSVar("--chart-yellow-300") || "#fde047",
            green600: getCSSVar("--chart-green-600") || "#16a34a",
            blue600: getCSSVar("--chart-blue-600") || "#2563eb",
            fuschia700: getCSSVar("--chart-fuschia-700") || "#a21caf", //purple
            slate400: getCSSVar("--chart-slate-400") || "#94a3b8",
            slate500: getCSSVar("--chart-slate-500") || "#64748b",
            violet300: getCSSVar("--chart-violet-300") || "#C4B5FD",
            violet500: getCSSVar("--chart-violet-500") || "#8B5CF6",
            rose300: getCSSVar("--chart-rose-300") || "#FDA4AF",
            rose400: getCSSVar("--chart-rose-400") || "#FB7185",
            pink600: getCSSVar("--chart-pink-600") || "#db2777"
        };

        const fonts = {
            font: getCSSVar("--font-sans") || "NVIDIA Sans, sans-serif",
            axisNameFontSize:
                getCSSVar("--chart-axis-name-font-size") || "14px",
            axisNameFontWeight:
                getCSSVar("--chart-axis-name-font-weight") || "500",
            axisLabelFontSize:
                getCSSVar("--chart-axis-label-font-size") || "12px",
            axisLabelFontWeight:
                getCSSVar("--chart-axis-label-font-weight") || "400",
            tooltipFontSize: getCSSVar("--chart-tooltip-font-size") || "14px",
            tooltipFontWeight: getCSSVar("--chart-tooltip-font-weight") || "600"
        };

        const paddings = {
            tooltipPaddingX: getCSSVar("--chart-tooltip-padding-x") || "8",
            tooltipPaddingY: getCSSVar("--chart-tooltip-padding-y") || "8"
        };

        return {
            borders,
            colors,
            fonts,
            paddings
        };
    }, []);
};
