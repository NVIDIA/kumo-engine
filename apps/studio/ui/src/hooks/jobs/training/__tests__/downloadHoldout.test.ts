// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/react";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { useDownloadHoldout } from "../downloadHoldout";

const TRAINING_JOB_ID = "training-job-123";
const PQ_ID = "pq-123";

describe("useDownloadHoldout", () => {
    it.each([
        {
            desc: "enabled is false (default)",
            id: TRAINING_JOB_ID,
            isTrainingJob: true,
            enabled: false,
            expectedIsError: false
        },
        {
            desc: "id is empty with enabled=true",
            id: "",
            isTrainingJob: true,
            enabled: true,
            expectedIsError: false
        },
        {
            desc: "isTrainingJob=true with valid id and enabled=true",
            id: TRAINING_JOB_ID,
            isTrainingJob: true,
            enabled: true,
            expectedIsError: false
        },
        {
            desc: "isTrainingJob=false (PQ) with valid id and enabled=true",
            id: PQ_ID,
            isTrainingJob: false,
            enabled: true,
            expectedIsError: true
        }
    ])(
        "Should return correct state when $desc",
        async ({ id, isTrainingJob, enabled, expectedIsError }) => {
            const { result } = renderWithQueryClient(() =>
                useDownloadHoldout({ id, isTrainingJob, enabled })
            );

            await waitFor(() =>
                expect(result.current.isError).toBe(expectedIsError)
            );
            expect(result.current.presignedUrl).toBe("");
            expect(result.current.refetch).toBeTypeOf("function");
        }
    );
});
