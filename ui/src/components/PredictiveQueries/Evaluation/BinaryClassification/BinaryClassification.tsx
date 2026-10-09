// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Alert,
    Card,
    ChartLabel,
    ConfusionMetricTable,
    Description,
    GridCol,
    GridRow,
    Label,
    ScatterPointData,
    ScatterPointLineChart,
    Skeleton,
    Slider,
    Switch,
    ThresholdMetricTable,
    Title
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useGetEvalStats } from "hooks/jobs/training/eval/binaryClassification";

import { DropdownOption } from "types/inputs";
import {
    ConfusionMatrixCell,
    ExtendedBinaryClassificationClasses,
    ExtendedBinaryClassificationMetrics,
    TaskType
} from "types/predictiveQueries";

import { evalOutputFromBinaryClassification } from "utils/evaluation";
import { isObjectNotEmpty } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";

interface BinaryClassificationEvaluationDefProps {
    data: ExtendedBinaryClassificationMetrics;
    selectedClass: DropdownOption;
    width: number;
    currentPosition: number;
    taskType: TaskType;
    matrixLabels: React.JSX.Element[] | string[];
    step: number;
}

const BinaryClassificationEvaluationDef = ({
    data,
    selectedClass,
    width,
    currentPosition,
    taskType,
    matrixLabels
}: BinaryClassificationEvaluationDefProps) => {
    const datalength = (data?.gainData?.length as number) - 1;

    return (
        <>
            <GridRow>
                <GridCol xs={6} sm={6} md={6} className="w-full">
                    {taskType === TaskType.BINARY_CLASSIFICATION && (
                        <>
                            <Title
                                variant="tertiary-heading"
                                children={"Confusion Matrix"}
                            />
                            <ConfusionMetricTable
                                classes={matrixLabels}
                                values={
                                    data?.confMatrix?.[
                                        currentPosition
                                    ] as ConfusionMatrixCell[][]
                                }
                                sumsRight={[
                                    data?.total_negative_actual,
                                    data?.total_positive_actual
                                ]}
                                sumsBottom={[
                                    data?.total_negative_pred[currentPosition],
                                    data?.total_positive_pred[currentPosition]
                                ]}
                                title={""}
                            />
                        </>
                    )}
                </GridCol>
                <GridCol xs={6} sm={6} md={6} className="w-full">
                    <Title
                        variant="tertiary-heading"
                        children={"Threshold Metrics"}
                    />
                    <ThresholdMetricTable
                        data={data}
                        currentPosition={currentPosition}
                        title={""}
                    />
                </GridCol>
            </GridRow>
            <GridRow className="flex flex-wrap justify-between gap-1">
                <GridCol>
                    <Title variant="tertiary-heading" children={"ROC Curve"} />
                    <div className="h-87.5">
                        <ScatterPointLineChart
                            id={`roc-${selectedClass?.value}`}
                            scatterPointData={
                                data?.rocData as ScatterPointData[]
                            }
                            yAxisLabel="True Positive Rate"
                            xAxisLabel="False Positive Rate"
                            currentIndex={currentPosition}
                            width={width / 5 + 140}
                            height={width / 6 + 110}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        {
                                            x: datalength,
                                            y: datalength
                                        }
                                    ],
                                    lineColor: "var(--color-zinc-200)",
                                    lineWidth: 1
                                }
                            ]}
                        />
                    </div>
                    <Spacer height={4} />
                    <ChartLabel
                        iconClassName="bg-teal-800"
                        text={"Predictive Query Performance"}
                    />
                    <ChartLabel
                        iconClassName="bg-kumo-grey-100"
                        text={"Random Guesses"}
                    />
                    <Spacer height={4} />
                </GridCol>
                <GridCol>
                    <Title
                        variant="tertiary-heading"
                        children={"Precision VS Recall"}
                    />
                    <div className="h-87.5">
                        <ScatterPointLineChart
                            id={`pr-${selectedClass?.value}`}
                            scatterPointData={
                                data?.prData as ScatterPointData[]
                            }
                            yAxisLabel="Precision"
                            xAxisLabel="Recall"
                            currentIndex={currentPosition}
                            width={width / 5 + 140}
                            height={width / 6 + 110}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        {
                                            x: 0,
                                            y:
                                                (data?.random_guess as number) *
                                                datalength
                                        },
                                        {
                                            x: datalength,
                                            y:
                                                (data?.random_guess as number) *
                                                datalength
                                        }
                                    ],
                                    lineColor: "var(--color-zinc-200)",
                                    lineWidth: 1.5
                                }
                            ]}
                        />{" "}
                    </div>
                    <Spacer height={4} />
                    <ChartLabel
                        iconClassName="bg-teal-800"
                        text={"Predictive Query Performance"}
                    />
                    <ChartLabel
                        iconClassName="bg-kumo-grey-100"
                        text={"Random Guesses"}
                    />
                </GridCol>
                <GridCol>
                    <Title
                        variant="tertiary-heading"
                        children={"Cumulative Gain"}
                    />
                    <div className="h-87.5">
                        <ScatterPointLineChart
                            id={`gain-${selectedClass.value}`}
                            scatterPointData={
                                data?.gainData as ScatterPointData[]
                            }
                            yAxisLabel="True Positive Rate"
                            xAxisLabel="Rate of Positive Predictions"
                            currentIndex={currentPosition}
                            width={width / 5 + 140}
                            height={width / 6 + 110}
                            scatterLineData={[
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        {
                                            x: datalength,
                                            y: datalength
                                        }
                                    ],
                                    lineColor: "var(--color-lime-300)"
                                },
                                {
                                    coordinates: [
                                        { x: 0, y: 0 },
                                        {
                                            x: data?.ppr_ideal * datalength,
                                            y: datalength
                                        },
                                        {
                                            x: datalength,
                                            y: datalength
                                        }
                                    ],
                                    lineColor: "var(--chart-ideal)"
                                }
                            ]}
                        />
                    </div>
                    <Spacer height={4} />

                    <ChartLabel
                        iconClassName="bg-pink-600"
                        text={"Ideal Case"}
                    />
                    <ChartLabel
                        iconClassName="bg-teal-800"
                        text={"Lift Curve"}
                    />
                    <ChartLabel iconClassName="bg-lime-400" text={"Baseline"} />
                </GridCol>
            </GridRow>
        </>
    );
};
const BinaryClassificationEvaluation = WithErrorBoundary(
    BinaryClassificationEvaluationDef
);

/**
 * Classification Chart component for Gain Chart with ROC curve,
 * Precision and confusion matrix.
 *
 * This component will invoke the API(`v0/eval/{jobId}/binary_classification`) in
 * `componentDidMount`, restructurize the response, sets it to state variable
 * and renders it in UI
 *
 * @param query_id Predictive query id
 * @param taskType Task type
 * @param height Screen height
 * @param width Screen width
 */
export interface BinaryClassificationProps {
    queryId: string;
    trainingJobId?: string;
    taskType: TaskType;
    width: number;
    handleError: (err: Error, title?: string, isWarning?: boolean) => void;
}

const BinaryClassification = ({
    queryId,
    trainingJobId,
    taskType,
    width,
    handleError
}: BinaryClassificationProps) => {
    const [isLoading, setLoading] = useState<boolean>(true);
    const [evalData, setEvalData] =
        useState<ExtendedBinaryClassificationClasses | null>(null);
    const [selectedClass, setSelectedClass] = useState<DropdownOption | null>(
        null
    );
    const [classOptions, setClassOptions] = useState<Array<DropdownOption>>([]);
    const [sliderPoint, setSliderPoint] = useState<Array<number>>([0.0]);
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [threshold, setThreshold] = useState<number>(0);
    const [threshLen, setThreshLen] = useState<number>(100);
    const matrixLabels = useMemo(() => {
        return ["False", "True"];
    }, []);

    const { evaluationResponse, isDataLoading } = useGetEvalStats(
        trainingJobId,
        queryId,
        handleError,
        setLoading
    );

    /**
     * Re-structurizing response data for metrics chart and
     * matrixes
     * @param respData
     */
    useEffect(() => {
        if (!evaluationResponse) return;

        const { defClass, classOptions, data } =
            evalOutputFromBinaryClassification(evaluationResponse, taskType);
        const recastedData = data as ExtendedBinaryClassificationClasses;

        const defClassKey =
            defClass as keyof ExtendedBinaryClassificationClasses;
        if (
            isValid(defClassKey) &&
            isObjectNotEmpty(recastedData[defClassKey])
        ) {
            const pos = (recastedData[defClassKey]?.threshLen as number) / 2;
            const threshold = recastedData[defClassKey]?.threshold?.[pos];
            setSelectedClass({ label: defClass, value: defClass });
            setClassOptions(classOptions);
            setEvalData(recastedData);
            setSliderPoint([1 / 2]);
            setCurrentIndex(pos);
            setThreshLen(recastedData[defClassKey]?.threshLen as number);
            setThreshold(threshold as number);
            setLoading(false);
        }
    }, [evaluationResponse, taskType]);

    const Loading = isLoading || isDataLoading;

    /**
     * DropDown event handler for class change
     * @param e event
     */
    const classSelectionChange = useCallback(
        (e: DropdownOption) => {
            // Place the position approximately at the middle of threshold array of
            // this newly selected class.
            const defClassKey =
                e?.value as keyof ExtendedBinaryClassificationClasses;
            if (
                isValid(defClassKey) &&
                isObjectNotEmpty(evalData?.[defClassKey])
            ) {
                const pos = (evalData?.[defClassKey]?.threshLen as number) / 2;
                const threshold = evalData?.[defClassKey]?.threshold?.[pos];
                setSelectedClass(e);
                setSliderPoint([1 / 2]);
                setCurrentIndex(pos);
                setThreshLen(evalData?.[defClassKey]?.threshLen as number);
                setThreshold(threshold as number);
            }
        },
        [evalData]
    );

    /**
     * Slider slide event handler
     * @param sliderPoint slider position
     */
    const sliderRangeUpdate = useCallback(
        (sliderPoint: number[]) => {
            const defClassKey =
                selectedClass?.value as keyof ExtendedBinaryClassificationClasses;
            if (
                isValid(defClassKey) &&
                isObjectNotEmpty(evalData?.[defClassKey])
            ) {
                const pos = Math.round(sliderPoint[0] * threshLen);
                const threshold = evalData?.[defClassKey]?.threshold?.[pos];
                setSliderPoint(sliderPoint);
                setCurrentIndex(pos);
                setThreshold(threshold as number);
            }
        },
        [evalData, selectedClass?.value, threshLen]
    );

    let data = null;
    const defClassKey =
        selectedClass?.value as keyof ExtendedBinaryClassificationClasses;

    if (isValid(defClassKey) && isObjectNotEmpty(evalData?.[defClassKey])) {
        data = evalData?.[defClassKey];
    }

    return (
        <>
            <Card className="rounded-sm mb-1 shadow-none p-6">
                <Title
                    variant="tertiary-heading"
                    children={"Classification Tradeoffs & Cutoff"}
                />
                {Loading && <Skeleton className="mt-2.5 h-12.5" />}
                {!Loading && !data && (
                    <Alert
                        variant="default"
                        enableClose={false}
                        className="mt-4">
                        {"No data found or failed to get Evaluation Stats!"}
                    </Alert>
                )}

                {data && data.precision && (
                    <>
                        <GridRow className="gap-0 mt-4">
                            <GridCol xs={4} sm={4} md={4} lg={3}>
                                <Description
                                    size="sm"
                                    variant="secondary"
                                    className="font-medium py-0  pb-4 pt-2">
                                    {"Classification threshold for "}
                                </Description>
                                {classOptions?.length > 1 && (
                                    <div className="flex">
                                        <div className="flex items-center space-x-2">
                                            <Switch
                                                id="classification-toggle"
                                                checked={
                                                    selectedClass?.value ===
                                                    classOptions[0]?.value
                                                }
                                                onCheckedChange={() => {
                                                    classSelectionChange(
                                                        (selectedClass?.value ===
                                                        classOptions[0]?.value
                                                            ? classOptions[1]
                                                            : classOptions[0]) as DropdownOption
                                                    );
                                                }}
                                            />
                                            <Label
                                                htmlFor="controlled-switch"
                                                className="justify-start ml-2 text-sm font-medium leading-none text-green-700 pt-0.5">
                                                {selectedClass?.value
                                                    ?.toString()
                                                    ?.toLocaleUpperCase()}
                                            </Label>
                                        </div>
                                    </div>
                                )}
                            </GridCol>
                            <GridCol xs={8} sm={8} md={8} lg={9}>
                                <div className="pt-2 mt-3 font-['Source_Code_Pro']">
                                    <Slider
                                        step={1 / (data?.threshLen ?? 1)}
                                        min={0}
                                        max={1}
                                        value={sliderPoint}
                                        onValueChange={sliderRangeUpdate}
                                        defaultValue={Array.of(threshold)}
                                    />
                                </div>
                            </GridCol>
                        </GridRow>
                    </>
                )}
            </Card>
            {data && data.precision && (
                <Card className="rounded-sm mb-1 shadow-none px-6 py-4">
                    <BinaryClassificationEvaluation
                        step={
                            1 / (data?.threshLen as number) > 0
                                ? (data?.threshLen as number)
                                : 1
                        }
                        data={data}
                        selectedClass={selectedClass as DropdownOption}
                        width={width}
                        currentPosition={currentIndex}
                        taskType={taskType}
                        matrixLabels={matrixLabels}
                    />
                </Card>
            )}
        </>
    );
};
export default WithErrorBoundary(BinaryClassification);
