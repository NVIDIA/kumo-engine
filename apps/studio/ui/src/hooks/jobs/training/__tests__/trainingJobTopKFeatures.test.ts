// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/react";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { useTrainingJobTopKFeatures } from "../trainingJobTopKFeatures";

const JOB_ID = "training-job-123";
const ENTITY_ID = "entity-456";
const CATEGORY = "random";

describe("useTrainingJobTopKFeatures", () => {
    it.each([
        {
            desc: "no IDs provided",
            jobId: undefined,
            entityId: undefined,
            expectedFeatureCount: undefined,
            globalIsEmpty: true
        },
        {
            desc: "jobId only (entityId missing)",
            jobId: JOB_ID,
            entityId: undefined,
            expectedFeatureCount: undefined,
            globalIsEmpty: true
        },
        {
            desc: "both IDs provided",
            jobId: JOB_ID,
            entityId: ENTITY_ID,
            expectedFeatureCount: 3,
            globalIsEmpty: false
        }
    ])(
        "Should return correct features and global map when $desc",
        async ({ jobId, entityId, expectedFeatureCount, globalIsEmpty }) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingJobTopKFeatures({
                    jobId,
                    entityId,
                    category: CATEGORY
                })
            );

            await waitFor(() => expect(result.current.isLoading).toBe(false));
            expect(result.current.isError).toBe(false);

            if (expectedFeatureCount === undefined) {
                expect(result.current.features).toBeUndefined();
            } else {
                expect(result.current.features).toHaveLength(
                    expectedFeatureCount
                );
            }

            if (globalIsEmpty) {
                expect(result.current.global).toEqual({});
            } else {
                expect(
                    Object.keys(result.current.global).length
                ).toBeGreaterThan(0);
            }
        }
    );
});
