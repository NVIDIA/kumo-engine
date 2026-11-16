// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnDef } from "@kumo-ai/kumo-ui-catalog";
import type { KumoGroupDBEntry } from "gen/v0/types";
export const adminGroupsColumns = (): ColumnDef<KumoGroupDBEntry>[] => [
    {
        header: "Groups",
        accessorKey: "display_name",
        enableSorting: true,
        size: 250
    },
    {
        header: "Description",
        accessorKey: "description",
        size: 400
    }
];
