// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ColumnDef,
    Info,
    Tooltip,
    TooltipContent,
    TooltipTrigger
} from "@kumo-ai/kumo-ui-catalog";

import { TransposedRow } from "components/PredictiveQueries/Training/SplitStats/trainingDataStats";

import { predictionWindowNonSplitPopoverMessage } from "utils/Constants";

const tooltipMessages: Record<string, string> = {
    "Prediction window": predictionWindowNonSplitPopoverMessage,
    "Split distribution":
        "Percentage distribution of rows across training, validation, and holdout splits",
    "Total rows": "Total number of data points (samples) used in this split",
    "Unique entities": "Number of unique entities in this split"
};

const MetricCell = ({ metricName }: { metricName: string }) => {
    const tooltipMessage = tooltipMessages[metricName];

    return (
        <span className="flex flex-row items-center text-title-primary text-base leading-5">
            {metricName}
            {tooltipMessage && (
                <Tooltip
                    id={`tooltip-${metricName.toLowerCase().replace(/\s+/g, "-")}`}>
                    <TooltipTrigger className="overflow-visible">
                        <span className="ml-1">
                            <Info className="size-3.5 text-title-primary" />
                        </span>
                    </TooltipTrigger>
                    <TooltipContent
                        className="max-w-100 text-justify"
                        arrow={false}
                        side={"top"}
                        sideOffset={5}>
                        {tooltipMessage}
                    </TooltipContent>
                </Tooltip>
            )}
        </span>
    );
};

export const SplitStatsColumns = ({
    windowWidth
}: {
    windowWidth: number;
}): ColumnDef<TransposedRow>[] => {
    const INNER_WINDOW = windowWidth * 0.8;
    const total = INNER_WINDOW - 250 - 100;
    const COLUMN_SIZE = total / 3;
    return [
        {
            accessorKey: "metric",
            size: 250,
            header: () => (
                <span className="text-title-secondary text-base">
                    Data Statistics
                </span>
            ),
            cell: ({ row }: { row: { original: TransposedRow } }) => (
                <MetricCell metricName={row.original.metric} />
            )
        },
        {
            accessorKey: "holdout",
            size: COLUMN_SIZE,
            header: () => (
                <span className="text-pink-600 text-base">Holdout</span>
            ),
            cell: ({ row }: { row: { original: TransposedRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.holdout}
                </div>
            )
        },
        {
            accessorKey: "validation",
            size: COLUMN_SIZE,
            header: () => (
                <span className="text-title-secondary text-base">
                    Validation
                </span>
            ),

            cell: ({ row }: { row: { original: TransposedRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.validation}
                </div>
            )
        },
        {
            accessorKey: "training",
            size: COLUMN_SIZE,
            header: () => (
                <span className="text-title-secondary text-base">Training</span>
            ),
            cell: ({ row }: { row: { original: TransposedRow } }) => (
                <div className="text-title-primary text-base">
                    {row.original.training}
                </div>
            )
        }
    ];
};
