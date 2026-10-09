// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JobScalarTagsTrialTags } from "gen/v0/types";

import { getValues } from "./objectHelpers";

/**
 * Check if a tag is valid for display.
 */
export const isValidTag = (tag: string) =>
    (tag?.startsWith("val_") || tag?.startsWith("train_")) &&
    !tag?.includes("_per_label_");

/**
 * Filter tags from trial tags based on valid tag criteria.
 */
export const filterTags = (
    trails: string[],
    trialTags: JobScalarTagsTrialTags
) => {
    const tmpTags = trails.map((t) => getValues(trialTags[t] ?? {}));
    const uniqueTags = [...new Set(tmpTags.flat())]
        .filter((t) => isValidTag(t))
        .sort();
    return uniqueTags;
};

/**
 * Filter tags for display, removing internal tags like pquery_id.
 */
export const filterTagsForDisplay = (
    obj: Record<string, string> | undefined
): Record<string, string> | undefined => {
    if (!obj) return undefined;

    return Object.fromEntries(
        Object.entries(obj).filter(([key]) => key !== "pquery_id")
    );
};
