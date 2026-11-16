// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ClassificationEvaluationResponse } from "gen/v0/types";

import {
    ExtendedBinaryClassificationMetrics,
    TaskType
} from "types/predictiveQueries";

import { getKeys } from "./Helpers/objectHelpers";
import { isValid } from "./Helpers/typeGuards";

/**
 * Creating confusion matrix array from true/false positive/negative values
 * and returns value(array of array of object)
 * @param data eval data
 * @param pos Position - current index of slider (0 to 100)
 * @return `[[{count:0, percentage: 0},{count: 0, percentage: 0}],
 *   [{count:0, percentage: 0},{count: 0, percentage: 0}]]`
 */
function getConfusionMatrix(
    data: ExtendedBinaryClassificationMetrics,
    pos: number
) {
    if (!data) return [];
    let perc1 = (data.true_negative[pos] / data.num_samples) * 100;
    let perc2 = (data.false_positive[pos] / data.num_samples) * 100;
    const r1 = [
        //Row 1
        { count: data.true_negative[pos], percentage: perc1 },
        { count: data.false_positive[pos], percentage: perc2 }
    ];
    perc1 = (data.false_negative[pos] / data.num_samples) * 100;
    perc2 = (data.true_positive[pos] / data.num_samples) * 100;
    const r2 = [
        //Row 2
        { count: data.false_negative[pos], percentage: perc1 },
        { count: data.true_positive[pos], percentage: perc2 }
    ];
    return [r1, r2];
}

export function evalOutputFromBinaryClassification(
    respData: ClassificationEvaluationResponse,
    taskType: string
) {
    const classes = getKeys(respData?.classes || {});
    const data: { [key: string]: ExtendedBinaryClassificationMetrics } = {};
    const defClass = classes?.length > 0 ? classes[0] : "";
    const classOptions: {
        label: string;
        value: string;
    }[] = [];

    classes?.forEach((clas) => {
        classOptions.push({ label: clas, value: clas });

        const classData = respData?.classes[
            clas
        ] as ExtendedBinaryClassificationMetrics;
        data[clas] = classData;
        data[clas].rocData = [];
        data[clas].prData = [];
        data[clas].gainData = [];
        data[clas].confMatrix = [];

        //Find the number of indexes based on len(threshold[])
        const dataLen = data[clas].threshold?.length as number;
        data[clas].threshLen = dataLen - 1;
        for (let i = 0; i < dataLen; i++) {
            if (
                isValid(data[clas].false_positive_rate[i]) &&
                isValid(data[clas].predicted_pos_rate[i])
            ) {
                //Constructing data for `ROC Curve` chart
                const x =
                    data[clas].false_positive_rate[i] < 0
                        ? null
                        : data[clas].false_positive_rate[i].toFixed(2);
                const y =
                    data[clas].recall[i] < 0
                        ? null
                        : data[clas].recall[i].toFixed(2);
                data[clas].rocData.push({
                    x: x as string,
                    y: y as string
                });
            }
            if (
                isValid(data[clas].recall[i]) &&
                isValid(data[clas].precision[i])
            ) {
                //Constructing data for `Precision-Recall Curve` chart
                const x1 =
                    data[clas].recall[i] < 0
                        ? null
                        : data[clas].recall[i].toFixed(2);
                const y1 =
                    data[clas].precision[i] < 0
                        ? null
                        : data[clas].precision[i].toFixed(2);
                data[clas].prData.push({
                    x: x1 as string,
                    y: y1 as string
                });
            }
            if (
                isValid(data[clas].recall[i]) &&
                isValid(data[clas].predicted_pos_rate[i])
            ) {
                //Constructing data for `Cumulative Gain Curve` chart
                const x2 =
                    data[clas].predicted_pos_rate[i] < 0
                        ? null
                        : data[clas].predicted_pos_rate[i].toFixed(2);
                const y2 =
                    data[clas].recall[i] < 0
                        ? null
                        : data[clas].recall[i].toFixed(2);
                data[clas].gainData.push({
                    x: x2 as string,
                    y: y2 as string
                });
            }

            data[clas].random_guess =
                data[clas].total_positive_actual / data[clas].num_samples;

            if (taskType === TaskType.BINARY_CLASSIFICATION) {
                data[clas].confMatrix.push(getConfusionMatrix(data[clas], i));
            }
        }
    });

    return { defClass, data, classOptions };
}
