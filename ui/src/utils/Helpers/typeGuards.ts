// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TaskType } from "types/predictiveQueries";

/**
 * Checks if a value is valid (not null, undefined, or empty string).
 */
export function isValid<T>(data: T | undefined | null): boolean {
    let retValue = false;
    if (
        data !== undefined &&
        data !== null &&
        data?.toString()?.trim() !== ""
    ) {
        retValue = true;
    }
    return retValue;
}

/**
 * Checks if a task type is a link prediction type.
 */
export function isLinkPredictionType(taskType: string): boolean {
    return [
        TaskType.LINK_PREDICTION.toString(),
        TaskType.STATIC_LINK_PREDICTION.toString(),
        TaskType.TEMPORAL_LINK_PREDICTION.toString()
    ].includes(taskType);
}

/**
 * Checks if a table ID is a snapshot (contains '@').
 */
export function tableIdIsSnapshot(id: string): boolean {
    return id?.includes("@");
}

export type Nullish = undefined | null;

export type Nullable<T> = T | Nullish;

export const isNullish = <T>(value: Nullable<T>): value is Nullish =>
    value === undefined || value === null;
