// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ClassificationEvaluationResponse } from "gen/v0/types";
import { describe, it } from "vitest";

import mockResponse from "components/PredictiveQueries/Evaluation/BinaryClassification/__tests__/MockResponses/BinaryClassificationResponse.json";

import { TaskType } from "types/predictiveQueries";

import { evalOutputFromBinaryClassification } from "utils/evaluation";
import { getKeys } from "utils/Helpers/objectHelpers";

describe("evalOutputFromBinaryClassification", () => {
    const properties = [
        "rocData",
        "prData",
        "gainData",
        "confMatrix",
        "random_guess",
        "threshLen"
    ];

    it.each([
        {
            description:
                "Should parse response and return data for BINARY_CLASSIFICATION",
            taskType: TaskType.BINARY_CLASSIFICATION,
            expectedConfMatrixLength: "greater than 0"
        },
        {
            description:
                "Should parse response and should not return data for `confMatrix` - except binary query",
            taskType: "other",
            expectedConfMatrixLength: "equal to 0"
        }
    ])("$description", ({ taskType, expectedConfMatrixLength }) => {
        const { defClass, classOptions, data } =
            evalOutputFromBinaryClassification(mockResponse, taskType);

        // Check default class
        expect(defClass).toBe("True");

        const classes = getKeys(mockResponse.classes);
        const options = classes.map((c) => ({ label: c, value: c }));

        // Check class options
        expect(classOptions).toMatchObject(options);

        // Check properties for all classes
        classes.forEach((c) => {
            properties.forEach((p) => {
                expect(data[c]).toHaveProperty(p);
            });
        });

        // Check confMatrix length based on task type
        classes.forEach((c) => {
            if (expectedConfMatrixLength === "greater than 0") {
                expect(data[c]["confMatrix"]?.length).toBeGreaterThan(0);
                expect(data[c]["rocData"]?.length).toBeGreaterThan(0);
                expect(data[c]["prData"]?.length).toBeGreaterThan(0);
                expect(data[c]["gainData"]?.length).toBeGreaterThan(0);
            } else {
                expect(data[c]["confMatrix"]?.length).toBe(0);
            }
        });
    });

    it("Should handle empty classes gracefully", () => {
        const emptyResp = { classes: {} };
        const result = evalOutputFromBinaryClassification(
            emptyResp,
            TaskType.BINARY_CLASSIFICATION
        );
        expect(result.defClass).toBe("");
        expect(result.classOptions).toEqual([]);
    });

    it("Should handle invalid data values (null/undefined) in metrics", () => {
        const respWithInvalidData = {
            classes: {
                "True": {
                    threshold: [0.1, 0.2],
                    false_positive_rate: [null, 0.2],
                    predicted_pos_rate: [0.1, null],
                    recall: [0.5, null],
                    precision: [null, 0.8],
                    true_negative: [10, 20],
                    false_positive: [5, 10],
                    false_negative: [3, 6],
                    true_positive: [15, 30],
                    num_samples: 100,
                    total_positive_actual: 50,
                    accuracy: [0.8, 0.9],
                    ppr_ideal: [0.5, 0.6],
                    f1: [0.7, 0.8],
                    total_negative_actual: 50,
                    random_guess: 0.5,
                    threshLen: 1
                }
            }
        } as unknown as ClassificationEvaluationResponse;

        const { data } = evalOutputFromBinaryClassification(
            respWithInvalidData,
            TaskType.BINARY_CLASSIFICATION
        );

        // When data is invalid, those points should be skipped
        expect(data["True"]["rocData"]?.length).toBeLessThan(2);
        expect(data["True"]["prData"]?.length).toBeLessThan(2);
        expect(data["True"]["gainData"]?.length).toBeLessThan(2);
    });

    it("Should handle negative values in metrics", () => {
        const respWithNegativeData = {
            classes: {
                "True": {
                    threshold: [0.1, 0.2],
                    false_positive_rate: [-1, 0.2],
                    predicted_pos_rate: [0.1, 0.3],
                    recall: [-1, 0.8],
                    precision: [0.9, -1],
                    true_negative: [10, 20],
                    false_positive: [5, 10],
                    false_negative: [3, 6],
                    true_positive: [15, 30],
                    num_samples: 100,
                    total_positive_actual: 50,
                    accuracy: [0.8, 0.9],
                    ppr_ideal: [0.5, 0.6],
                    f1: [0.7, 0.8],
                    total_negative_actual: 50,
                    random_guess: 0.5,
                    threshLen: 1
                }
            }
        } as unknown as ClassificationEvaluationResponse;

        const { data } = evalOutputFromBinaryClassification(
            respWithNegativeData,
            TaskType.BINARY_CLASSIFICATION
        );

        // Check that negative values are converted to null in chart data
        const rocDataWithNulls = data["True"]["rocData"]?.filter(
            (point) => point.x === null || point.y === null
        );
        expect(rocDataWithNulls?.length).toBeGreaterThan(0);

        const prDataWithNulls = data["True"]["prData"]?.filter(
            (point) => point.x === null || point.y === null
        );
        expect(prDataWithNulls?.length).toBeGreaterThan(0);
    });
});
