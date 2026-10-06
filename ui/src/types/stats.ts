// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Inferred from kumo/data/utils.py:get_target_distribution, must be kept

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";

import { jobStatus } from "utils/Constants";

// in sync
export type ClassificationTargetDistribution = {
    count: {
        [k: string]: number;
    };
    percentage: {
        [k: string]: number;
    };
};

// Inferred from kumo/data/utils.py:get_target_distribution, must be kept
// in sync
export type RegressionTargetDistribution = {
    target_mean: number;
    target_min: number;
    target_percentile_25: number;
    target_median: number;
    target_percentile_75: number;
    target_max: number;
    target_mode: number;
    target_mode_count: number;
    target_mode_percentage: number;
    histogram: {
        bins: Array<number>;
        counts: Array<number>;
        bin_width: number;
        num_bins: number;
    };

    [key: string]:
        | number
        | {
              bins: Array<number>;
              counts: Array<number>;
              bin_width: number;
              num_bins: number;
          };
};

export type TrainTableGenStats = {
    num_expected_iterations: number;
    num_finished_iterations: number;
    iteration_data?: TimeFrameStats[];
};

export type TimeFrameStats = {
    max_date: string;
    min_date: string;
    anchor_date: string;
    isActive?: boolean;
    node_degree_stats?: {
        avg_entity_degree: number;
        avg_target_degree: number;
        min_entity_degree: number;
        max_entity_degree: number;
        min_target_degree: number;
        max_target_degree: number;
    };
    num_entities: number;
    num_targets: number;
    num_train_rows: number;
    num_links?: number;
    train_time?: string;

    target_distribution:
        | ClassificationTargetDistribution
        | RegressionTargetDistribution;

    [key: string]:
        | string
        | number
        | boolean
        | object
        | undefined
        | ClassificationTargetDistribution
        | RegressionTargetDistribution; // used two extra fields
};

export type TrainingTimeFrameStatsData = {
    allStats: TrainTableGenStats | null;
    tableData: {
        data: TimeFrameStats[];
        columns: ColumnDef<TimeFrameStats, number>[];
    };
    selectedStats: TimeFrameStats | null;
    finishedIterations: number;
    expectedIterations: number;
    receivedResponse: boolean;
    isTemporal: boolean;
    queryStatus: jobStatus;
};

export type SummaryStats = [string, number];

export type DistributionDate = [string, string, number];

export type DistributionPoints = [number, number];

export type DistributionCategory = [string, number];

export type DistributionCounts = {
    total: number;
    missing_percentage: number;
    num_unique: number;
    invalid_percentage: number;
};

export type Statistics = {
    counts: DistributionCounts;
    summary_stats: Array<SummaryStats>;
    distribution_points: Array<DistributionPoints> | null;
    distribution_category: Array<DistributionCategory> | null;
    distribution_date?: Array<DistributionDate> | null;
    bp_stats: {
        [key: string]: { [key: string]: Array<number> | number };
    } | null;
    approx: boolean;
};

export type Metric = {
    [key: string]: number | string | undefined;
    name?: string;
    val?: number;
};

export type XaiJsonData = {
    [key: string]: (string | number)[] | XaiJsonData;
};

export type XaiTableData = {
    name: string;
    fq_name: string;
    table_row_names?: string;
    type: string;
    kind: string;
    variance_score: string | number;
    gradient_score: string | number;
};

export type XaiCartData = {
    labels: (string | number)[];
    plots_truth: (string | number)[];
    plots_pred: (string | number)[];
    population_ratio: (string | number)[];
    global_avg_score: number;
};

export type XaiChartPlots = {
    yAxisTitle: string;
    allDataTruth: (string | number)[];
    allDataPrediction: (string | number)[];
    allLabels: (string | number)[];
    allPopulationRatio: (string | number)[];
    activeDataTruth: (string | number)[];
    activeLabels: (string | number)[];
    activePopulationRatio: (string | number)[];
    globalAvgScore: string | number;
    activeDataPrediction: (string | number)[];
};

export type XaiCategoryDropDowns = {
    name: string;
    options: string[]; // populate later
    selected: string | null; // populate later
};
