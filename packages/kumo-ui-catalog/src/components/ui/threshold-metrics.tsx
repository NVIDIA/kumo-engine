// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import { Table, TableBody, TableCell, TableRow } from "./table";
import { BinaryClassificationMetrics } from "@/types/threshold-metrics";
import { cn } from "@/lib/utils";

interface ThresholdMetricTableProps {
    data?: BinaryClassificationMetrics;
    currentPosition: number;
    title: string;
}

export const ThresholdMetricTable: React.FC<ThresholdMetricTableProps> = ({
    data,
    currentPosition,
    title
}) => {
    const renderPrecision = () => {
        const tp = data?.true_positive[currentPosition] ?? 0;
        const fp = data?.false_positive[currentPosition] ?? 0;
        const precision = data?.precision[currentPosition];

        if (tp === 0 && fp === 0 && precision === 1) {
            return "N/A";
        }
        return precision?.toFixed(2);
    };

    const labelCellClass =
        "w-44 px-3 py-2 text-sm font-normal leading-tight text-slate-500 !border !border-slate-400";
    const valueCellClass =
        "w-38 px-3 py-2 text-sm font-bold leading-tight text-gray-800 !font-['Source_Code_Pro'] !border !border-slate-400";

    return (
        <div className="inline-flex flex-col items-start justify-end gap-4">
            <div className="justify-start text-sm font-bold leading-none text-slate-500">
                {title?.toUpperCase()}
            </div>
            <Table className="w-fit border-collapse">
                <TableBody>
                    {[
                        ["Precision", renderPrecision()],
                        ["Recall", data?.recall[currentPosition]?.toFixed(2)],
                        [
                            "Accuracy",
                            data?.accuracy[currentPosition]?.toFixed(2)
                        ],
                        [
                            "Predicted Positive Rate",
                            data?.predicted_pos_rate[currentPosition]?.toFixed(
                                2
                            )
                        ],
                        ["F1", data?.f1[currentPosition]?.toFixed(2)]
                    ].map(([label, value]) => (
                        <TableRow
                            key={label}
                            className="!border-1 !border-slate-400 hover:bg-white group-hover:bg-white">
                            <TableCell
                                className={cn(
                                    labelCellClass,
                                    "hover:bg-white group-hover:bg-white"
                                )}>
                                {label}
                            </TableCell>
                            <TableCell
                                className={cn(
                                    valueCellClass,
                                    "hover:bg-white group-hover:bg-white"
                                )}>
                                {value}
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </div>
    );
};
