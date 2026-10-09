// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from "@testing-library/react";
import { renderWithQueryClient } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { useTrainingJobNodePredRfm } from "../trainingJobNodePredRfm";

const JOB_ID = "training-job-123";
const ENTITY_ID = "entity-456";

describe("useTrainingJobNodePredRfm", () => {
    it.each([
        {
            desc: "both IDs empty",
            jobId: "",
            entityId: "",
            nodePredRfmDefined: false,
            globalDefined: false
        },
        {
            desc: "jobId only (entityId empty)",
            jobId: JOB_ID,
            entityId: "",
            nodePredRfmDefined: false,
            globalDefined: true
        },
        {
            desc: "both IDs provided",
            jobId: JOB_ID,
            entityId: ENTITY_ID,
            nodePredRfmDefined: true,
            globalDefined: true
        }
    ])(
        "Should return correct nodePredRfm and globalNodePredRfm when $desc",
        async ({ jobId, entityId, nodePredRfmDefined, globalDefined }) => {
            const { result } = renderWithQueryClient(() =>
                useTrainingJobNodePredRfm(jobId, entityId)
            );

            await waitFor(() =>
                expect(result.current.isNodePredRfmLoading).toBe(false)
            );

            if (nodePredRfmDefined) {
                expect(result.current.nodePredRfm).toBeDefined();
            } else {
                expect(result.current.nodePredRfm).toBeUndefined();
            }

            if (globalDefined) {
                expect(result.current.globalNodePredRfm).toBeDefined();
            } else {
                expect(result.current.globalNodePredRfm).toBeUndefined();
            }
        }
    );
});
