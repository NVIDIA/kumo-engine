// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    TrainingJobOutput,
    TrainingWorkflowResponse,
    TrialProgress
} from "gen/v0/types";
import { EditableQuery, KumoapiJobsPredictionArtifactType } from "gen/v1/types";
import { describe, expect, it } from "vitest";

import mockWorkflowResponse from "../../../components/Jobs/Training/__tests__/MockResponses/TrainingWorkflow.test.json";

import { TaskType } from "types/predictiveQueries";

import {
    convertResultsToTrainingJobResults,
    generateProgressStatsforChart,
    getFiltersFromBpOverride,
    parseDateTimePeriods,
    parseTrainingWorkflowResponse,
    timePeriodsFromQueryResponse
} from "../predictiveQueriesConverter";

describe("getFiltersFromBpOverride", () => {
    it("Should return empty overrides when editable_fields is empty", () => {
        const result = getFiltersFromBpOverride({
            editable_fields: [],
            text: ""
        });
        expect(result).toEqual({
            entity_filter_override: "",
            temporal_entity_filter_override: "",
            target_filter_override: ""
        });
    });

    it("Should return entity filter when there is only one Entity filter field", () => {
        const response: EditableQuery = {
            text: "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
            editable_fields: [
                {
                    field_name: "Entity filter",
                    text: "entity_text",
                    position: [83, 83],
                    full_filter_format_string: " WHERE {0}",
                    empty_filter_default: "",
                    bold_text_position: [83, 83]
                }
            ],
            validation_response: { warnings: [], errors: [], info_items: [] }
        };

        const result = getFiltersFromBpOverride(response);
        expect(result).toEqual({
            entity_filter_override: "entity_text",
            temporal_entity_filter_override: "",
            target_filter_override: ""
        });
    });

    it("Should return target filter when there is only one Target filter field", () => {
        const response: EditableQuery = {
            text: "PREDICT COUNT(FINANCE_TRANS.*, 0, 30, days) = 0 FOR EACH FINANCE_ACCOUNT.ACCOUNT_ID",
            editable_fields: [
                {
                    field_name: "Target filter",
                    text: "target_text",
                    position: [0, 0],
                    full_filter_format_string: "",
                    empty_filter_default: "",
                    bold_text_position: [0, 0]
                }
            ]
        };

        const result = getFiltersFromBpOverride(response);
        expect(result).toEqual({
            entity_filter_override: "",
            temporal_entity_filter_override: "",
            target_filter_override: "target_text"
        });
    });

    it("Should return both entity and target filter when multiple fields exist", () => {
        const response: EditableQuery = {
            text: "",
            editable_fields: [
                {
                    field_name: "Entity filter",
                    text: "entity_text",
                    position: [0, 0],
                    full_filter_format_string: "",
                    empty_filter_default: "",
                    bold_text_position: [0, 0]
                },
                {
                    field_name: "Target filter",
                    text: "target_text",
                    position: [0, 0],
                    full_filter_format_string: "",
                    empty_filter_default: "",
                    bold_text_position: [0, 0]
                }
            ]
        };

        const result = getFiltersFromBpOverride(response);
        expect(result).toEqual({
            entity_filter_override: "entity_text",
            temporal_entity_filter_override: "",
            target_filter_override: "target_text"
        });
    });

    it("Should extract only valid(Entity filter, Target filter) filter overrides", () => {
        const response: EditableQuery = {
            text: "",
            editable_fields: [
                {
                    field_name: "Random filter",
                    text: "random",
                    position: [0, 0],
                    full_filter_format_string: "",
                    empty_filter_default: "",
                    bold_text_position: [0, 0]
                },
                {
                    field_name: "Entity filter",
                    text: "entity_text",
                    position: [0, 0],
                    full_filter_format_string: "",
                    empty_filter_default: "",
                    bold_text_position: [0, 0]
                }
            ]
        };

        const result = getFiltersFromBpOverride(response);
        expect(result).toEqual({
            entity_filter_override: "entity_text",
            temporal_entity_filter_override: "",
            target_filter_override: ""
        });
    });
});

describe("convertResultsToTrainingJobResults", () => {
    it.each([
        {
            description: "Should return undefined if results is null",
            input: null
        },
        {
            description: "Should return undefined if results is undefined",
            input: undefined
        }
    ])("$description", ({ input }) => {
        expect(convertResultsToTrainingJobResults(input)).toBeUndefined();
    });

    it("Should transform f1, precision, and recall metric names with @0.5 for binary classification", () => {
        const input = {
            test_metrics: [
                { name: "f1", val: 0.9 },
                { name: "precision", val: 0.8 },
                { name: "recall", val: 0.7 }
            ],
            key_cols: [],
            output_artifact_stats: [
                {
                    artifact_type:
                        KumoapiJobsPredictionArtifactType.PREDICTIONS,
                    num_rows: 100,
                    num_cols: 5,
                    size_bytes: 1024,
                    output_path: "/tmp/predictions.csv"
                }
            ],
            num_rows: 0,
            num_cols: 0,
            size_bytes: 0,
            output_path: "",
            output_paths: []
        };

        const result = convertResultsToTrainingJobResults(
            input,
            TaskType.BINARY_CLASSIFICATION
        );

        expect(result?.test_metrics).toEqual([
            { name: "F1@0.5 (Macro Avg)", val: 0.9 },
            { name: "PRECISION (Macro Avg)", val: 0.8 },
            { name: "RECALL (Macro Avg)", val: 0.7 }
        ]);
    });

    it("Should transform f1 metric name WITHOUT @0.5 for multiclass classification", () => {
        const input = {
            test_metrics: [
                { name: "f1", val: 0.9 },
                { name: "precision", val: 0.8 },
                { name: "recall", val: 0.7 }
            ],
            key_cols: [],
            output_artifact_stats: [
                {
                    artifact_type:
                        KumoapiJobsPredictionArtifactType.PREDICTIONS,
                    num_rows: 100,
                    num_cols: 5,
                    size_bytes: 1024,
                    output_path: "/tmp/predictions.csv"
                }
            ],
            num_rows: 0,
            num_cols: 0,
            size_bytes: 0,
            output_path: "",
            output_paths: []
        };

        const result = convertResultsToTrainingJobResults(
            input,
            TaskType.MULTICLASS_CLASSIFICATION
        );

        expect(result?.test_metrics).toEqual([
            { name: "F1 (Macro Avg)", val: 0.9 },
            { name: "PRECISION (Macro Avg)", val: 0.8 },
            { name: "RECALL (Macro Avg)", val: 0.7 }
        ]);
    });

    it("Should leave unrelated metric names unchanged", () => {
        const input = {
            train_metrics: [
                { name: "accuracy", val: 0.95 },
                { name: "loss", val: 0.1 }
            ],
            test_metrics: [],
            val_metrics: [],
            key_cols: [],
            output_artifact_stats: [
                {
                    artifact_type:
                        KumoapiJobsPredictionArtifactType.PREDICTIONS,
                    num_rows: 100,
                    num_cols: 5,
                    size_bytes: 1024,
                    output_path: "/tmp/predictions.csv"
                }
            ],
            num_rows: 0,
            num_cols: 0,
            size_bytes: 0,
            output_path: "",
            output_paths: []
        };

        const result = convertResultsToTrainingJobResults(input);

        expect(result?.train_metrics).toEqual([
            { name: "accuracy", val: 0.95 },
            { name: "loss", val: 0.1 }
        ]);
    });

    it("Should apply transformation to all metric groups if present (multilabel with @0.5)", () => {
        const input = {
            test_metrics: [{ name: "f1", val: 0.9 }],
            train_metrics: [{ name: "precision", val: 0.8 }],
            val_metrics: [{ name: "recall", val: 0.7 }],
            key_cols: [],
            output_artifact_stats: [
                {
                    artifact_type:
                        KumoapiJobsPredictionArtifactType.PREDICTIONS,
                    num_rows: 100,
                    num_cols: 5,
                    size_bytes: 1024,
                    output_path: "/tmp/predictions.csv"
                }
            ],
            num_rows: 0,
            num_cols: 0,
            size_bytes: 0,
            output_path: "",
            output_paths: []
        };

        const result = convertResultsToTrainingJobResults(
            input,
            TaskType.MULTILABEL_CLASSIFICATION
        );

        expect(result).toEqual(
            expect.objectContaining({
                test_metrics: [{ name: "F1@0.5 (Macro Avg)", val: 0.9 }],
                train_metrics: [{ name: "PRECISION (Macro Avg)", val: 0.8 }],
                val_metrics: [{ name: "RECALL (Macro Avg)", val: 0.7 }]
            })
        );
    });
});

describe("parseTrainingWorkflowResponse", () => {
    it("Should parse a completed training job correctly", () => {
        const result = parseTrainingWorkflowResponse(
            mockWorkflowResponse as unknown as TrainingWorkflowResponse
        );

        // Basic structure
        expect(result).toBeDefined();
        expect(result.id).toBe("trainingjob-9f8b259b4f764c8f932ad5912a926ec2");
        expect(result.status).toBe("DONE");
        expect(result.statusKey).toBe("Progress");
        expect(result.queryYaml).toContain("PREDICT");
    });

    it("Should map important metadata fields", () => {
        const result = parseTrainingWorkflowResponse(
            mockWorkflowResponse as unknown as TrainingWorkflowResponse
        );

        expect(result.taskType).toBe(TaskType.BINARY_CLASSIFICATION);
        expect(result.queryType).toBe("temporal");
        expect(result.trainingStatusMessage).toContain("successfully");
        expect(result.graphDefinition).toHaveProperty("tables");
        expect(result.tags).toHaveProperty("pquery_id");
    });

    it("Should provide results and metrics when available", () => {
        const result = parseTrainingWorkflowResponse(
            mockWorkflowResponse as unknown as TrainingWorkflowResponse
        );

        expect(result.results).toBeTruthy();
        expect(result.results?.train_metrics?.length).toBeGreaterThan(0);
        expect(result.queryProgressStats).toBeTruthy();
    });

    it("Should format dates and runtime correctly", () => {
        const result = parseTrainingWorkflowResponse(
            mockWorkflowResponse as unknown as TrainingWorkflowResponse
        );

        expect(typeof result.dateCreated).toBe("string");
        expect(typeof result.dateUpdated).toBe("string");
        expect(result.runtime).not.toBe("-");
    });

    it("Should handle empty status (status === '')", () => {
        const result = parseTrainingWorkflowResponse({
            ...mockWorkflowResponse,
            training_job: {
                execution: {
                    log: { detail: "" }
                }
            }
        } as unknown as TrainingWorkflowResponse);

        expect(result.status).toBe("");
        expect(result.statusKey).toBe("Progress");
    });

    it("Should calculate total training time when start_time and last_updated_at exist", () => {
        const start = new Date(Date.now() - 60_000).toISOString();
        const end = new Date().toISOString();
        const result = parseTrainingWorkflowResponse({
            ...mockWorkflowResponse,
            training_job: {
                ...mockWorkflowResponse.training_job,
                execution: {
                    ...mockWorkflowResponse.training_job.execution,
                    status: "DONE",
                    start_time: start,
                    last_updated_at: end,
                    results: {
                        total_iterations: 5,
                        total_training_time_sec: 60
                    },
                    log: { detail: "done" }
                }
            }
        } as unknown as TrainingWorkflowResponse);

        expect(result.estTimings!.text).toBe("Total Training Time");
        expect(result.estTimings!.value).toBe("01m");
        expect(result.estExperiments!.value).toBe("5");
    });

    it("Should use the execution wall-clock duration", () => {
        const result = parseTrainingWorkflowResponse({
            ...mockWorkflowResponse,
            training_job: {
                ...mockWorkflowResponse.training_job,
                execution: {
                    ...mockWorkflowResponse.training_job.execution,
                    status: "DONE",
                    start_time: "2026-07-16T18:00:00.000Z",
                    last_updated_at: "2026-07-16T18:35:00.000Z",
                    results: {
                        total_iterations: 5,
                        total_training_time_sec: 300
                    },
                    log: { detail: "done" }
                }
            }
        } as unknown as TrainingWorkflowResponse);

        expect(result.estTimings!.text).toBe("Total Training Time");
        expect(result.estTimings!.value).toBe("35m");
    });

    it("Should compute estimated timings and experiments when status is Running and progress is valid", () => {
        const start = new Date(Date.now() - 2 * 60_000).toISOString();
        const estimatedEnd = new Date(Date.now() + 5 * 60_000).toISOString();

        const result = parseTrainingWorkflowResponse({
            ...mockWorkflowResponse,
            training_job: {
                ...mockWorkflowResponse.training_job,
                execution: {
                    status: "RUNNING",
                    start_time: start,
                    log: { detail: "running" },
                    progress: {
                        start_time: start,
                        estimated_end_time: estimatedEnd,
                        iterations_completed: 2,
                        iterations_remaining: 3,
                        trial_progress: [
                            {
                                trial_id: "trial-1",
                                best_metric: 0.85,
                                iterations: 2
                            },
                            {
                                trial_id: "trial-2",
                                best_metric: 0.9,
                                iterations: 1
                            }
                        ],
                        metric_mode: "max"
                    }
                }
            }
        } as unknown as TrainingWorkflowResponse);

        expect(result.status).toBe("RUNNING");
        expect(result.estTimings).toBeDefined();
        expect(result.estTimings!.value).toMatch(/minutes/);
        expect(result.estExperiments?.value).toBe("2 of 5");
        expect(result.queryProgressStats).toBeNull();
    });
});

describe("generateProgressStatsforChart", () => {
    it("Should return proper stats when data is valid", () => {
        const result = generateProgressStatsforChart(
            [
                {
                    trial_id: "0",
                    best_train_metric: { name: "AUROC", val: 0.9948 },
                    best_validation_metric: { name: "AUROC", val: 0.9986 }
                },
                {
                    trial_id: "1",
                    best_train_metric: { name: "AUROC", val: 0.9764 },
                    best_validation_metric: { name: "AUROC", val: 0.9983 }
                }
            ],
            "test-mode"
        );

        expect(result).toEqual({
            xAxisLabels: [1, 2],
            datasets: [
                {
                    color: "#ff6666",
                    value: [0.9986, 0.9983],
                    label: "Best validation metric"
                }
            ],
            xLabel: "Experiments",
            yLabel: "AUROC",
            mode: "test-mode"
        });
    });

    it.each([
        {
            description: "Should return null if data is null",
            input: null as unknown as TrialProgress[],
            mode: ""
        },
        {
            description: "Should return null if data is not an array",
            input: {} as unknown as TrialProgress[],
            mode: ""
        },
        {
            description:
                "Should return null if best_validation_metric is missing",
            input: [
                {
                    trial_id: "0",
                    best_train_metric: { name: "AUROC", val: 0.9948 }
                }
            ],
            mode: ""
        }
    ])("$description", ({ input, mode }) => {
        const result = generateProgressStatsforChart(input, mode);
        expect(result).toBeNull();
    });

    it("Should skip invalid validation metric values", () => {
        const result = generateProgressStatsforChart(
            [
                {
                    trial_id: "0",
                    best_train_metric: { name: "AUROC", val: 0.9948 },
                    best_validation_metric: { name: "AUROC", val: null }
                },
                {
                    trial_id: "1",
                    best_train_metric: { name: "AUROC", val: 0.9764 },
                    best_validation_metric: { name: "AUROC", val: 0.9983 }
                }
            ],
            ""
        );

        expect(result).toEqual({
            xAxisLabels: [1],
            datasets: [
                {
                    color: "#ff6666",
                    value: [0.9983],
                    label: "Best validation metric"
                }
            ],
            xLabel: "Experiments",
            yLabel: "AUROC",
            mode: ""
        });
    });
});

describe("timePeriodsFromQueryResponse", () => {
    const createTrainingJob = (
        timePeriods: Record<string, unknown>,
        splitSizes: Partial<Record<string, number>>
    ): TrainingJobOutput => {
        return {
            results: {
                time_periods: timePeriods,
                split_sizes: splitSizes
            }
        } as unknown as TrainingJobOutput;
    };

    it("Should return correct trained and evaluated data when all periods are valid", () => {
        const trainingJob = {
            results: {
                time_periods: {
                    train: { start: "2024-01-01", end: "2024-01-10" },
                    val: { start: "2024-01-02", end: "2024-01-08" },
                    test: { start: "2024-02-01", end: "2024-02-05" }
                },
                split_sizes: {
                    train: 100,
                    val: 20,
                    test: 30
                }
            }
        } as unknown as TrainingJobOutput;

        const result = timePeriodsFromQueryResponse(trainingJob);

        expect(result.exists).toBe(true);
        expect(result.trainedOnData).toContain("2024");
        expect(result.trainedOnData).toContain("120 Samples");
        expect(result.evaluatedOnData).toContain("30 Samples");
        expect(result.trainSplitSize).toBe(120);
        expect(result.testSplitSize).toBe(30);
    });

    it("Should handle pseudo times (only sample counts shown)", () => {
        const trainingJob = {
            results: {
                time_periods: {
                    train: {
                        start: "2024-01-01",
                        end: "2024-01-10",
                        is_pseudo: true
                    },
                    val: { start: "2024-01-02", end: "2024-01-08" },
                    test: { start: "2024-02-01", end: "2024-02-05" }
                },
                split_sizes: { train: 50, val: 10, test: 15 }
            }
        } as unknown as TrainingJobOutput;

        const result = timePeriodsFromQueryResponse(trainingJob);

        expect(result.exists).toBe(false);
        expect(result.trainSplitSize).toBe(60);
        expect(result.testSplitSize).toBe(15);
    });

    it("Should return minimal data when fields are missing", () => {
        const trainingJob = {
            results: { split_sizes: { test: 5 } }
        } as unknown as TrainingJobOutput;

        const result = timePeriodsFromQueryResponse(trainingJob);

        expect(result.exists).toBe(false);
        expect(result.trainedOnData).toBeNull();
        expect(result.evaluatedOnData).toBeNull();
        expect(result.trainSplitSize).toBeNull();
        expect(result.testSplitSize).toBe(5);
    });

    it.each([
        {
            description: "Should pick val.start when only val.start is valid",
            timePeriods: {
                val: { start: "2024-01-05", end: "2024-01-06" },
                train: { start: "", end: "" }
            },
            splitSizes: { val: 10, test: 5 },
            expectedDateContains: ["2024-01-05"],
            expectedTrainSplit: 10
        },
        {
            description:
                "Should pick train.start when only train.start is valid",
            timePeriods: {
                val: { start: "", end: "" },
                train: { start: "2024-01-03", end: "2024-01-04" }
            },
            splitSizes: { train: 20, test: 10 },
            expectedDateContains: ["2024-01-03"],
            expectedTrainSplit: 20
        },
        {
            description:
                "Should pick the earlier start and later end when both val and train are present",
            timePeriods: {
                val: { start: "2024-01-02", end: "2024-01-09" },
                train: { start: "2024-01-03", end: "2024-01-08" }
            },
            splitSizes: { train: 5, val: 15, test: 2 },
            expectedDateContains: ["2024-01-02", "2024-01-09"],
            expectedTrainSplit: 20
        },
        {
            description:
                "Should fall back to val.end only when train.end missing",
            timePeriods: {
                val: { start: "2024-01-02", end: "2024-01-07" },
                train: { start: "2024-01-02" }
            },
            splitSizes: { val: 5 },
            expectedDateContains: ["2024-01-07"],
            expectedTrainSplit: 5
        },
        {
            description:
                "Should fall back to train.end only when val.end missing",
            timePeriods: {
                val: { start: "2024-01-02" },
                train: { start: "2024-01-01", end: "2024-01-06" }
            },
            splitSizes: { train: 5 },
            expectedDateContains: ["2024-01-06"],
            expectedTrainSplit: 5
        },
        {
            description:
                "Should use only val split size when train split is invalid",
            timePeriods: {
                val: { start: "2024-01-01", end: "2024-01-02" }
            },
            splitSizes: { val: 11 },
            expectedDateContains: [],
            expectedTrainSplit: 11
        }
    ])(
        "$description",
        ({
            timePeriods,
            splitSizes,
            expectedDateContains,
            expectedTrainSplit
        }) => {
            const trainingJob = createTrainingJob(timePeriods, splitSizes);
            const result = timePeriodsFromQueryResponse(trainingJob);

            expectedDateContains.forEach((dateStr) => {
                expect(result.trainedOnData).toContain(dateStr);
            });
            expect(result.trainSplitSize).toBe(expectedTrainSplit);
        }
    );

    it("Should return trainedOnData and evaluatedOnData with date ranges when time periods exist", () => {
        const trainingJob = {
            results: {
                time_periods: {
                    train: { start: "2024-01-01", end: "2024-01-05" },
                    test: { start: "2024-02-01", end: "2024-02-02" }
                },
                split_sizes: { train: 10, test: 7 }
            }
        } as unknown as TrainingJobOutput;

        const result = timePeriodsFromQueryResponse(trainingJob);

        expect(result.trainedOnData).toBe(
            "2024-01-01 UTC - 2024-01-05 UTC (10 Samples)"
        );
        expect(result.evaluatedOnData).toBe("- - - (7 Samples)");
    });
});

describe("parseDateTimePeriods", () => {
    it.each([
        {
            description:
                "Should return date+time when start and end are on the same date",
            startInput: "2024-01-01T10:15:30",
            endInput: "2024-01-01T12:45:00",
            expectedStart: "2024-01-01 10:15 UTC",
            expectedEnd: "2024-01-01 12:45 UTC"
        },
        {
            description:
                "Should return only date when start and end are on different dates",
            startInput: "2024-01-01T10:15:30",
            endInput: "2024-01-02T12:45:00",
            expectedStart: "2024-01-01 UTC",
            expectedEnd: "2024-01-02 UTC"
        },
        {
            description:
                "Should handle inputs without time portion (only date)",
            startInput: "2024-01-01",
            endInput: "2024-01-10",
            expectedStart: "2024-01-01 UTC",
            expectedEnd: "2024-01-10 UTC"
        },
        {
            description: "Should handle empty strings",
            startInput: "",
            endInput: "",
            expectedStart: " UTC",
            expectedEnd: " UTC"
        }
    ])(
        "$description",
        ({ startInput, endInput, expectedStart, expectedEnd }) => {
            const result = parseDateTimePeriods(startInput, endInput);
            expect(result).toEqual({
                start: expectedStart,
                end: expectedEnd
            });
        }
    );
});
