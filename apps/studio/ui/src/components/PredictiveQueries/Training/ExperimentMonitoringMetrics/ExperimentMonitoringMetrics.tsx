// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import {
    Card,
    CardHeader,
    DropdownCard,
    ScalarChart,
    Skeleton,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useTensorBoardStats } from "hooks/jobs/training/eval/tensorBoard";

/**
 * Predictive Query Tensorboard training data stats component to represent loss
 * curves/validation in UI
 */
export interface ExperimentMonitoringMetricsProps {
    jobId: string;
    trainingStatus: string;
    tuneMetric?: string;
}

const ExperimentMonitoringMetrics = ({
    jobId,
    trainingStatus,
    tuneMetric
}: ExperimentMonitoringMetricsProps) => {
    const { tags, loading, chartStats } = useTensorBoardStats(
        jobId,
        trainingStatus
    );

    const tagOptions = useMemo(() => {
        const options =
            tags?.map((tag) => ({
                label: tag?.toUpperCase(),
                value: tag
            })) || [];

        // Move tuneMetric to the top of the array if it exists
        if (tuneMetric && options.length > 0) {
            const tuneMetricIndex = options.findIndex(
                (option) => option.value === tuneMetric
            );
            if (tuneMetricIndex > 0) {
                // Move tuneMetric to the beginning using splice
                const [tuneMetricOption] = options.splice(tuneMetricIndex, 1);
                options.unshift(tuneMetricOption);
            }
        }

        return options;
    }, [tags, tuneMetric]);

    if (!tags || tags?.length === 0) {
        return null;
    }

    if (loading && tags.length < 1)
        return (
            <Card className="p-0 gap-2 mb-0">
                <CardHeader className="px-6 py-3 !mb-0" bottomSeparator={true}>
                    <Title
                        variant="tertiary-heading"
                        children={"Experiment Monitoring Metrics"}
                    />
                </CardHeader>
                <Skeleton className="m-2.5 h-12.5" />
            </Card>
        );

    return (
        <DropdownCard
            title="Experiment Monitoring Metrics"
            options={tagOptions || []}
            placeholder="Metrics">
            {({ value }) => {
                const selectedTag = value as string;
                return (
                    <>
                        {/* Conditionally render content or loading centered spinner */}
                        {tags.length < 1 || !selectedTag ? (
                            <Skeleton className="m-2.5 h-12.5" />
                        ) : (
                            <>
                                {/* Charts Row */}
                                {selectedTag && chartStats[selectedTag] && (
                                    <div className="overflow-y-auto">
                                        <ScalarChart
                                            id={"scalar-chart-" + selectedTag}
                                            tag={selectedTag}
                                            xAxisLabel={"Epoch"}
                                            yAxisLabel={selectedTag}
                                            dataDict={chartStats[selectedTag]}
                                        />
                                    </div>
                                )}
                            </>
                        )}
                    </>
                );
            }}
        </DropdownCard>
    );
};

export default WithErrorBoundary(ExperimentMonitoringMetrics);
