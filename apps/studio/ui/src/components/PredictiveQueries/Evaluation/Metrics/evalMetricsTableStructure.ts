// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import { ColumnDef, useWindowDimensions } from "@kumo-ai/kumo-ui-catalog";
import { TrainingJobResults } from "gen/v0/types";

import { EvaluationMetricsColumns } from "components/PredictiveQueries/Evaluation/Metrics/EvaluationMetrics";

import { TaskType } from "types/predictiveQueries";
import { Metric } from "types/stats";

import { parseNumbers } from "utils/Helpers/numberHelpers";

const MACRO_AVG = "(Macro Avg)";

/**
 * @param prevHeaders - Previous header columns array
 * @param newHeaders - New header columns array
 * @returns New headers with Select column preserved from previous headers
 */
const preserveSelectColumn = <T extends { [key: string]: string | number }>(
    prevHeaders: Array<ColumnDef<T>>,
    newHeaders: Array<ColumnDef<T>>
): Array<ColumnDef<T>> => {
    const prevSelectColumn = prevHeaders.find((col) => col.id === "selection");

    if (!prevSelectColumn) {
        return newHeaders;
    }

    return newHeaders.map((col) =>
        col.id === "selection" ? prevSelectColumn : col
    );
};

function metricValue(value?: string | number | null) {
    if (value !== undefined && value !== null) {
        const parsed = parseNumbers(value, { decimals: 3 });
        return parsed !== null ? parsed : "-";
    }
    return "-";
}

function allowedMetric(value?: string | null) {
    return (
        value &&
        value?.toLowerCase() !== "loss" &&
        value?.toLowerCase() !== "locality_score"
    );
}

/**
 * Strips display suffixes added by transformMetrics so raw metric names
 * (e.g. "f1") match transformed result names (e.g. "F1 (Macro Avg)").
 */
const splits: Array<{
    name: string;
    key: "test_metrics" | "val_metrics" | "train_metrics";
}> = [
    { name: "Holdout", key: "test_metrics" },
    { name: "Validation", key: "val_metrics" },
    { name: "Training", key: "train_metrics" }
];

export const useEvalMetricsAsTableStructure = ({
    trainingResults,
    taskType,
    tuneMetric,
    downloadClicked,
    onDownloadClick
}: {
    trainingResults?: TrainingJobResults;
    taskType: TaskType;
    tuneMetric: string;
    downloadClicked?: boolean;
    onDownloadClick?: () => void;
}) => {
    const [headers, setHeaders] = useState<
        Array<ColumnDef<{ [key: string]: string | number }>>
    >([]);

    const [rows, setRows] = useState<Array<{ [key: string]: unknown }>>([]);

    const updatedMetrics = useRef(false);
    const enableKRef = useRef(false);

    const { width } = useWindowDimensions();
    const widthRef = useRef(width);
    widthRef.current = width;

    const parseMetricsToTable = useCallback(
        (results: TrainingJobResults) => {
            type rowType = { [key: string]: string | number | null };
            const dataX = [
                ...(results?.test_metrics ?? []),
                ...(results?.train_metrics ?? []),
                ...(results?.val_metrics ?? [])
            ];
            const uniqueNames = [...new Set(dataX.map((item) => item.name))]
                .filter((m) => allowedMetric(m))
                .sort();

            const rows: Array<rowType> = [];
            const keyCheck: { [key: string]: number } = {};
            let enableK = false;
            uniqueNames.forEach((key, i) => {
                let metricName = key;
                let topK: string | undefined = "";
                let showBorder = false;

                if (
                    metricName?.includes("@") &&
                    !metricName?.includes(MACRO_AVG)
                ) {
                    metricName = metricName.split("@")[0];
                    showBorder =
                        i + 1 < uniqueNames.length &&
                        metricName !== uniqueNames[i + 1].split("@")[0];
                    if (keyCheck[metricName]) {
                        metricName = "";
                    } else {
                        keyCheck[metricName] = 1;
                    }
                    topK = key.split("@").pop();
                    enableK = true;
                } else {
                    showBorder =
                        i + 1 < uniqueNames.length &&
                        metricName !== uniqueNames[i + 1];
                }
                const tempRow: { [key: string]: string | number } = {
                    "metric_name": `${metricName?.toUpperCase()}`,
                    "holdout": "-",
                    "training": "-",
                    "validation": "-",
                    "compare_key": `${key?.toUpperCase()}`
                };

                if (topK) {
                    tempRow["k"] = `${topK}`;
                }

                if (key.toUpperCase() === tuneMetric) {
                    tempRow["cellBgColor"] = "p-0 m-0";
                }
                if (!showBorder) {
                    tempRow["rowClassName"] = "border-none";
                }

                for (const split of splits) {
                    if (results?.[split.key]) {
                        const foundMetric = (
                            results[split.key] as Metric[]
                        )?.find((m) => `${m?.name}` === key);
                        tempRow[split.name.toLowerCase()] = metricValue(
                            foundMetric?.val
                        );
                    }
                }

                rows.push(tempRow);
            });

            // Create headers
            enableKRef.current = enableK;
            const headers = EvaluationMetricsColumns({
                tuneMetric,
                enableK: enableK,
                downloadClicked,
                onDownloadClick,
                windowWidth: widthRef.current
            });
            updatedMetrics.current = true;
            setHeaders(headers);
            setRows(rows);
        },
        [tuneMetric, downloadClicked, onDownloadClick]
    );

    useEffect(() => {
        if (trainingResults && updatedMetrics.current === false) {
            parseMetricsToTable(trainingResults as TrainingJobResults);
        }
    }, [parseMetricsToTable, taskType, trainingResults]);

    // Rebuild headers when downloadClicked changes
    useEffect(() => {
        if (updatedMetrics.current) {
            setHeaders((prevHeaders) => {
                const newHeaders = EvaluationMetricsColumns({
                    tuneMetric,
                    enableK: enableKRef.current,
                    downloadClicked,
                    onDownloadClick,
                    windowWidth: width
                });

                return preserveSelectColumn(prevHeaders, newHeaders);
            });
        }
    }, [downloadClicked, onDownloadClick, tuneMetric, width]);

    return { rows, headers };
};
