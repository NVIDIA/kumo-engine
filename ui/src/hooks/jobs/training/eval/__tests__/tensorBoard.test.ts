// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/react";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { jobStatus } from "utils/Constants";

import { useTensorBoardStats } from "../tensorBoard";

const JOB_ID = "tensorboard-job-123";

describe("useTensorBoardStats", () => {
    it.each([
        { desc: "jobId is empty", jobId: "", status: jobStatus.Running },
        { desc: "status is Failed", jobId: JOB_ID, status: jobStatus.Failed },
        {
            desc: "status is Cancelled",
            jobId: JOB_ID,
            status: jobStatus.Cancelled
        }
    ])(
        "Should return empty tags and loading true when $desc",
        ({ jobId, status }) => {
            const { result } = renderWithQueryClient(() =>
                useTensorBoardStats(jobId, status)
            );

            expect(result.current.tags).toEqual([]);
            expect(result.current.loading).toBe(true);
            expect(result.current.chartStats).toEqual({});
        }
    );

    it("Should load tags and chartStats when jobId is valid and status is Running", async () => {
        const { result } = renderWithQueryClient(() =>
            useTensorBoardStats(JOB_ID, jobStatus.Running)
        );

        await waitFor(() => expect(result.current.loading).toBe(false));
        expect(result.current.tags.length).toBeGreaterThan(0);
        expect(Object.keys(result.current.chartStats).length).toBeGreaterThan(
            0
        );
    });
});
