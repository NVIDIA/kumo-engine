// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ScatterPointData } from "@/types/chart";

/**
 * Checks if the chart is a ROC curve based on axis titles
 * @param xAxisTitle The title of the x-axis
 * @param yAxisTitle The title of the y-axis
 * @returns boolean indicating if the chart is a ROC curve
 */
export const isRocCurve = (xAxisTitle: string, yAxisTitle: string): boolean => {
    return (
        yAxisTitle === "True Positive Rate" &&
        xAxisTitle === "False Positive Rate"
    );
};

/**
 * Checks if the data points only contain all (0,0)s and (1,1)s
 * @param points Array of data points to check
 * @returns boolean indicating if the points only contain all (0,0)s and (1,1)s
 */
export const hasOnlyZeroOnePoints = (
    points: Array<ScatterPointData>
): boolean => {
    for (let i = 0; i < points.length; i++) {
        if (points[i].x !== 0 && points[i].x !== 1) {
            return false;
        }
        if (points[i].y !== 0 && points[i].y !== 1) {
            return false;
        }
    }
    return true;
};
