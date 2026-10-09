// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KumoCoreTypesEnumsTaskType } from "gen/v0/types";
import {
    BatchPredictionJobResponse,
    KumoapiJobsPredictionArtifactType
} from "gen/v1/types";

import { TrainingJobResults } from "types/predictiveQueries";

import {
    getOutputArtifactStatsFromResults,
    parseUnlinkedPredictionJobs
} from "../batchPredictionsConverter";

import {
    mockOutputArtifactStatsFromResults,
    mockParseUnlinkedPredictionJobs
} from "./MockResponse/mockResponses";

describe("getOutputArtifactStatsFromResults", () => {
    it.each([
        {
            description:
                "Should return aggregated stats from output_artifact_stats",
            mockResults: mockOutputArtifactStatsFromResults,
            expectedNumRows: 30,
            expectedNumCols: 5,
            expectedSizeBytes: 3072,
            expectedOutputPaths: ["s3://file1", "s3://file2"]
        },
        {
            description:
                "Should fall back to legacy fields when output_artifact_stats is missing",
            mockResults: {
                output_paths: ["s3://file1"],
                num_rows: 100,
                num_cols: 10,
                size_bytes: 4096,
                key_cols: [],
                output_artifact_stats: null
            } as TrainingJobResults,
            expectedNumRows: 100,
            expectedNumCols: 10,
            expectedSizeBytes: 4096,
            expectedOutputPaths: ["s3://file1"]
        },
        {
            description: "Should handle empty output_artifact_stats gracefully",
            mockResults: {
                output_artifact_stats: [],
                output_paths: [],
                num_rows: 0,
                num_cols: 0,
                size_bytes: 0,
                key_cols: []
            } as TrainingJobResults,
            expectedNumRows: 0,
            expectedNumCols: 0,
            expectedSizeBytes: 0,
            expectedOutputPaths: []
        }
    ])(
        "$description",
        ({
            mockResults,
            expectedNumRows,
            expectedNumCols,
            expectedSizeBytes,
            expectedOutputPaths
        }) => {
            const result = getOutputArtifactStatsFromResults(mockResults);

            expect(result.totalNumRows).toBe(expectedNumRows);
            expect(result.totalNumCols).toBe(expectedNumCols);
            expect(result.totalSizeBytes).toBe(expectedSizeBytes);
            expect(result.outputPaths).toEqual(expectedOutputPaths);
        }
    );
});

describe("parseUnlinkedPredictionJobs", () => {
    const createModifiedJob = (
        overrides: Partial<BatchPredictionJobResponse>
    ): BatchPredictionJobResponse => {
        return {
            ...mockParseUnlinkedPredictionJobs[0],
            ...overrides
        };
    };

    it("Should parse a basic job prediction correctly", () => {
        const result = parseUnlinkedPredictionJobs(
            mockParseUnlinkedPredictionJobs,
            "project-test"
        );
        expect(result.length).toBe(1);
        const job = result[0];
        expect(job.id).toBe("job-123");
        expect(job.predictiveQueryId).toBe("PQ123");
        expect(job.numRows).toBe("100");
        expect(job.keyColumns).toBe("id");
        expect(job.runtime).toBe("30m");
        expect(job.downloadSupported).toBe(false);
    });

    it("Should preserve the backend download capability", () => {
        const result = parseUnlinkedPredictionJobs(
            [createModifiedJob({ download_supported: true })],
            "project-test"
        );

        expect(result[0].downloadSupported).toBe(true);
    });

    it("Should handle missing execution results gracefully", () => {
        const jobWithMissingResults = createModifiedJob({
            execution: {
                ...mockParseUnlinkedPredictionJobs[0].execution,
                results: undefined
            }
        });

        const result = parseUnlinkedPredictionJobs(
            [jobWithMissingResults],
            "project-test"
        );
        expect(result.length).toBe(1);
        expect(result[0].numRows).toBe(null);
        expect(result[0].numCols).toBe("N/A");
        expect(result[0].size).toBe("N/A");
    });

    it("Should parse binary classification thresholds correctly", () => {
        const binaryJob = createModifiedJob({
            config: {
                ...mockParseUnlinkedPredictionJobs[0].config,
                task_type: KumoCoreTypesEnumsTaskType.BINARY_CLASSIFICATION,
                binary_classification_threshold: 0.75
            }
        });

        const result = parseUnlinkedPredictionJobs([binaryJob], "project-test");
        expect(result.length).toBe(1);
        const job = result[0];

        expect(job.threshold).toBe(0.75);
        expect(job.threshold_lable).toBe(true);
    });

    it("Should include Snowflake connector and table in outputsList", () => {
        const snowflakeOutputJob = createModifiedJob({
            config: {
                ...mockParseUnlinkedPredictionJobs[0].config,
                outputs: [
                    {
                        artifact_type:
                            KumoapiJobsPredictionArtifactType.PREDICTIONS,
                        storage_type: "SNOWFLAKE" as const,
                        connector_id: "sf_conn_1",
                        table_name: "predictions_table",
                        extra_fields: []
                    }
                ]
            }
        });

        const result = parseUnlinkedPredictionJobs(
            [snowflakeOutputJob],
            "project-test"
        );
        expect(result[0].outputsList).toContain("sf_conn_1.predictions_table");
    });

    it.each(["GCS", "ADLS"] as const)(
        "Should include %s file paths in outputsList",
        (storageType) => {
            const filePath = "exports/predictions.parquet";
            const result = parseUnlinkedPredictionJobs(
                [
                    createModifiedJob({
                        config: {
                            ...mockParseUnlinkedPredictionJobs[0].config,
                            outputs: [
                                {
                                    artifact_type:
                                        KumoapiJobsPredictionArtifactType.PREDICTIONS,
                                    storage_type: storageType,
                                    connector_id: "connector-1",
                                    file_path: filePath,
                                    extra_fields: []
                                }
                            ]
                        }
                    })
                ],
                "project-test"
            );
            expect(result[0].outputsList).toContain(filePath);
        }
    );

    it("Should include stacktrace if failure_info has stacktrace", () => {
        const jobWithStacktrace = createModifiedJob({
            execution: {
                ...mockParseUnlinkedPredictionJobs[0].execution,
                failure_info: {
                    message: "Something went wrong",
                    stacktrace: "Error: stacktrace details here"
                }
            }
        });

        const result = parseUnlinkedPredictionJobs(
            [jobWithStacktrace],
            "project-test"
        );
        expect(result[0].failureInfo).toBe("Something went wrong");
        expect(result[0].stacktrace).toBe("Error: stacktrace details here");
    });

    it("Should map kumo_version to kumoVersion", () => {
        const jobWithVersion = createModifiedJob({ kumo_version: "1.2.3" });
        const result = parseUnlinkedPredictionJobs(
            [jobWithVersion],
            "project-test"
        );
        expect(result[0].kumoVersion).toBe("1.2.3");
    });

    it("Should leave kumoVersion undefined when kumo_version is absent", () => {
        const result = parseUnlinkedPredictionJobs(
            mockParseUnlinkedPredictionJobs,
            "project-test"
        );
        expect(result[0].kumoVersion).toBeUndefined();
    });
});
