// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type BinaryClassificationMetrics = {
    precision: Array<number>;
    recall: Array<number>;
    accuracy: Array<number>;
    predicted_pos_rate: Array<number>;
    false_positive_rate?: Array<number>;
    f1: Array<number>;
    ppr_ideal?: number;
    num_samples?: number;
    true_positive: Array<number>;
    true_negative?: Array<number>;
    false_positive: Array<number>;
    false_negative?: Array<number>;
    total_positive_actual?: number;
    total_negative_actual?: number;
    total_positive_pred?: Array<number>;
    total_negative_pred?: Array<number>;
    threshold?: Array<number>;
    //properties defined for UI
    rocData?: Array<xyCoordinates>;
    prData?: Array<xyCoordinates>;
    gainData?: Array<xyCoordinates>;
    confMatrix?: Array<Array<ConfusionMatrixCell>[]>;
    threshLen?: number;
    random_guess?: number;
};

export type xyCoordinates = { x: number | string; y: number | string };

export type ConfusionMatrixCell = { count: number; percentage: number };
