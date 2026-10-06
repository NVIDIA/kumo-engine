// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { initializeOutputs } from "utils/batchPredictions";
import { outputStorage } from "utils/Constants";

describe("initializeOutputs", () => {
    const jobId = "trainingjob-09ab19cc90784a01a7407ca5081ee652";
    const artifact = "PREDICTIONS";
    const storage = outputStorage.S3;

    it.each([
        {
            description: "Should create a compatible output for `null`",
            output: null
        },
        {
            description: "Should create a compatible output",
            output: {
                "PREDICTIONS": {
                    "artifact_type": "PREDICTIONS",
                    "directory": "",
                    "table_schema": "",
                    "table_name": "59f46_predictions",
                    "extension": ".parquet",
                    "errorInfo": {
                        "color": "",
                        "info": ""
                    }
                }
            }
        }
    ])("$description", ({ output }) => {
        const result = initializeOutputs(jobId, artifact, storage, output);

        expect(result[artifact]).toMatchObject({
            artifact_type: artifact,
            table_name: "09ab1_predictions",
            extension: ".parquet"
        });
    });
});
