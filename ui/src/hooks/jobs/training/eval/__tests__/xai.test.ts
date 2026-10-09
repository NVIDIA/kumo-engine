// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { removeFromLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { waitFor } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { storageKeys } from "utils/Constants";

import { useXai } from "../xai";

const TRAINING_JOB_ID = "training-job-123";
const BP_JOB_ID = "bp-job-123";
const QUERY_ID = "query-123";

describe("useXai", () => {
    it("does not fetch without a project", () => {
        removeFromLocalStorage(storageKeys.selectedProject);

        renderWithQueryClient(() =>
            useXai({
                trainingJobId: undefined,
                queryId: QUERY_ID,
                bpjobId: undefined
            })
        );

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it.each([
        {
            desc: "no IDs provided",
            trainingJobId: undefined,
            queryId: undefined,
            bpjobId: undefined,
            dataIsUndefined: true
        },
        {
            desc: "trainingJobId provided",
            trainingJobId: TRAINING_JOB_ID,
            queryId: undefined,
            bpjobId: undefined,
            dataIsUndefined: false
        },
        {
            desc: "bpjobId only",
            trainingJobId: undefined,
            queryId: undefined,
            bpjobId: BP_JOB_ID,
            dataIsUndefined: false
        },
        {
            desc: "trainingJobId overrides bpjobId",
            trainingJobId: TRAINING_JOB_ID,
            queryId: undefined,
            bpjobId: BP_JOB_ID,
            dataIsUndefined: false
        },
        {
            desc: "trainingJobId overrides queryId (eval disabled)",
            trainingJobId: TRAINING_JOB_ID,
            queryId: QUERY_ID,
            bpjobId: undefined,
            dataIsUndefined: false
        }
    ])(
        "Should return correct data when $desc",
        async ({ trainingJobId, queryId, bpjobId, dataIsUndefined }) => {
            const { result } = renderWithQueryClient(() =>
                useXai({ trainingJobId, queryId, bpjobId })
            );

            await waitFor(() => expect(result.current.isLoading).toBe(false));
            if (dataIsUndefined) {
                expect(result.current.data).toBeUndefined();
            } else {
                expect(result.current.data).not.toBeUndefined();
            }
            expect(result.current.isError).toBe(false);
        }
    );
});
