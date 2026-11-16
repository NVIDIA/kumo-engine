// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { DataTable } from "@kumo-ai/kumo-ui-catalog";
import { APIKeySummary } from "gen/v0/types";

import { AdminApiKeysTableColumns } from "components/@ui/TableColumns/ApiKeysTable";

export interface ApiKeysTableProps {
    data: APIKeySummary[];
    searchStr: string;
    onDeleteClick: (id: string) => void;
    isLoading: boolean;
    showUserColumn?: boolean;
}

export const ApiKeysTable = ({
    data,
    onDeleteClick,
    searchStr,
    isLoading,
    showUserColumn = false
}: ApiKeysTableProps) => {
    const filteredData = useMemo(
        () =>
            data.filter((key) =>
                key.name.toLowerCase().includes(searchStr.toLowerCase())
            ),
        [data, searchStr]
    );
    const columns = AdminApiKeysTableColumns(onDeleteClick);

    return (
        <DataTable
            id="api-keys-table"
            columns={columns}
            data={filteredData}
            enableGlobalFilter={false}
            isLoading={isLoading}
            columnsVisibility={
                showUserColumn ? {} : { owner_display_name: false }
            }
        />
    );
};
