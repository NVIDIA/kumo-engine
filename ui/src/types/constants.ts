// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type TableDataTypes = {
    [key: string]: string[];
};

export enum SDKObjectType {
    Graph = "graph",
    Table = "table",
    TrainingJob = "training job workflow",
    PredictionJob = "prediction job workflow"
}
type CtaLevel = {
    value: string;
    color: string;
};

type CtaLevels = {
    Critical: CtaLevel;
    Warning: CtaLevel;
    Info: CtaLevel;
};
export const ctaLevels: CtaLevels = {
    Critical: { value: "CRITICAL", color: "red" },
    Warning: { value: "WARNING", color: "orange" },
    Info: { value: "INFO", color: "grey" }
};
