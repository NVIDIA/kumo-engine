// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Formats a user-friendly failure message for table generation jobs (single child).
 *
 * Used by: TrainingTableGenOverview, PredictionTableGenOverview
 *
 * @param jobType - Type of table generation job
 * @param childType - Name of the failed child (defaults to 'Graph Snapshot')
 * @returns Formatted message explaining the child failed
 *
 * @example
 * formatTableGenFailureMessage('training-table')
 * // "This training table generation job failed because Graph Snapshot failed"
 *
 * formatTableGenFailureMessage('prediction-table')
 * // "This prediction table generation job failed because Graph Snapshot failed"
 */
export function formatTableGenFailureMessage(
    jobType: "training-table" | "prediction-table",
    childType: string = "Graph Snapshot"
): string {
    const tableTypeName =
        jobType === "training-table"
            ? "training table generation"
            : "prediction table generation";

    return `This ${tableTypeName} job failed because ${childType} failed`;
}

/**
 * Formats a user-friendly cancellation message for table generation jobs (single child).
 *
 * Used by: TrainingTableGenOverview, PredictionTableGenOverview
 *
 * @param jobType - Type of table generation job
 * @param childType - Name of the cancelled child (defaults to 'Graph Snapshot')
 * @returns Formatted message explaining the child was cancelled
 *
 * @example
 * formatTableGenCancellationMessage('training-table')
 * // "This training table generation job failed because Graph Snapshot was cancelled"
 *
 * formatTableGenCancellationMessage('prediction-table')
 * // "This prediction table generation job failed because Graph Snapshot was cancelled"
 */
export function formatTableGenCancellationMessage(
    jobType: "training-table" | "prediction-table",
    childType: string = "Graph Snapshot"
): string {
    const tableTypeName =
        jobType === "training-table"
            ? "training table generation"
            : "prediction table generation";

    return `This ${tableTypeName} job failed because ${childType} was cancelled`;
}
