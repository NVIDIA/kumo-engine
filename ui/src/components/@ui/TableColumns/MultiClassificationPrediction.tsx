// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";

export const MultiClassificationPredictionColumns = (): ColumnDef<{
    label: string;
    value: string;
}>[] => {
    return [
        {
            id: "label",
            accessorKey: "label",
            header: "Label"
        },
        {
            id: "value",
            accessorKey: "value",
            header: "Value"
        }
    ];
};
