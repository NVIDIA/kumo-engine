// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { default as React, useEffect, useMemo } from "react";

import { ScrollArea, Skeleton } from "@kumo-ai/kumo-ui-catalog";

import MPCodeEditor from "components/CodeEditor/MPCodeEditor";
import { CODE_EDITOR_SETUP } from "components/PredictiveQueries/Write/ModelPlannerEditor";

import { useGetModelPlans } from "hooks/jobs/training/modelPlan";

import { DropdownOption } from "types/inputs";
import { TrainingJobID } from "types/training_job";

import { jobStatus } from "utils/Constants";

export interface ModelPlannerProps {
    trainingJobId: TrainingJobID;
    bestExperimentIndex: number;
    jobStatus: jobStatus;
    setOptions: React.Dispatch<React.SetStateAction<DropdownOption[]>>;
    selectedOption: DropdownOption;
    setSelectedOption: React.Dispatch<React.SetStateAction<DropdownOption>>;
    fallbackPlan?: string | null;
}

const ModelPlanner = ({
    trainingJobId,
    bestExperimentIndex,
    jobStatus,
    setOptions,
    selectedOption,
    setSelectedOption,
    fallbackPlan
}: ModelPlannerProps) => {
    const { plans: modelPlans, options: advancedOptions } = useGetModelPlans(
        trainingJobId,
        jobStatus,
        bestExperimentIndex
    );

    useEffect(() => {
        const options =
            advancedOptions?.length > 0 ? advancedOptions : modelPlans;
        if (options?.length > bestExperimentIndex) {
            setSelectedOption({
                value: bestExperimentIndex.toString(),
                label: `Experiment ${bestExperimentIndex + 1} (BEST)`
            });
        } else if (options?.length > 0) {
            const d = options.length - 1;
            setSelectedOption({
                value: d.toString(),
                label: `Experiment ${options.length}`
            });
        }
    }, [
        modelPlans,
        setOptions,
        advancedOptions,
        bestExperimentIndex,
        setSelectedOption
    ]);

    useEffect(() => {
        const optionsArray =
            advancedOptions?.length > 0 ? advancedOptions : modelPlans;
        const options = optionsArray?.map((val, idx) => {
            return {
                value: idx.toString(),
                label:
                    `Experiment ${idx + 1}` +
                    (idx === bestExperimentIndex ? " (BEST)" : "")
            };
        });
        if (options?.length > 0) {
            setOptions(options);
        }
    }, [advancedOptions, bestExperimentIndex, modelPlans, setOptions]);

    const currentValue = useMemo(() => {
        return advancedOptions?.length > 0
            ? (advancedOptions[Number(selectedOption.value)] ?? "")
            : (modelPlans[Number(selectedOption.value)] ?? "");
    }, [advancedOptions, modelPlans, selectedOption.value]);

    const displayValue: string =
        (selectedOption?.value ? currentValue : fallbackPlan) ?? "";

    return (
        <div>
            {selectedOption?.value || fallbackPlan ? (
                <ScrollArea
                    type="always"
                    className="border rounded-sm w-full !h-102.5">
                    <MPCodeEditor
                        language="YAML"
                        value={displayValue}
                        basicSetup={CODE_EDITOR_SETUP}
                        onChange={() => {}}
                        editable={false}
                        hideBorder
                        height="100%"
                        backgroundColor="#F4F6F8"
                    />
                </ScrollArea>
            ) : (
                <Skeleton className="w-100 h-85" />
            )}
        </div>
    );
};

export default ModelPlanner;
