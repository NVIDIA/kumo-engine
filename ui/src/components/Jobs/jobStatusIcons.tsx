// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    AutoRenew,
    BrightnessAlert,
    CheckCircle,
    NextPlan,
    TabClose
} from "@kumo-ai/kumo-ui-catalog";

import { jobStatus } from "utils/Constants";

export const jobStatusIconMap: Record<string, React.ReactNode> = {
    [jobStatus.Cancelled]: <TabClose className="fill-text-title-primary" />,
    [jobStatus.Done]: <CheckCircle className="fill-green-700" />,
    [jobStatus.Failed]: <BrightnessAlert className="fill-red-500" />,
    [jobStatus.Queued]: <NextPlan className="fill-yellow-700" />,
    [jobStatus.Running]: <AutoRenew className="fill-violet-700 animate-spin" />
};
