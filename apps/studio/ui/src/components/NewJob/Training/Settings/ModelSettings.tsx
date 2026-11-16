// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import { RunModeSelection } from "components/PredictiveQueries/Write/AdvancedOptionsSection";

import { DropdownOption } from "types/inputs";
import { CreateTrainingJobArgs } from "types/jobs";

export interface ModelSettingsProps {
    createJobArgs: CreateTrainingJobArgs;
    changeRunMode: (value: DropdownOption) => void;
}

const ModelSettings: React.FC<ModelSettingsProps> = ({
    createJobArgs,
    changeRunMode
}) => {
    return (
        <div className="flex flex-col gap-4">
            <RunModeSelection
                selectedRunMode={{
                    value: createJobArgs.run_mode?.valueOf(),
                    label: createJobArgs.run_mode?.toUpperCase()
                }}
                onRunModeChange={(newVal: DropdownOption) => {
                    changeRunMode(newVal);
                }}
            />
        </div>
    );
};

export default WithErrorBoundary(ModelSettings);
