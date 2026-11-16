// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from "react";

import { DataTable, EmptyTable } from "@kumo-ai/kumo-ui-catalog";

import { useTrainingJobTopKFeatures } from "hooks/jobs/training/trainingJobTopKFeatures";

export interface Top3FeaturesProps {
    entityId?: string;
    jobId?: string;
    category: string;
}

export const Top3Features: React.FC<Top3FeaturesProps> = ({
    entityId,
    jobId,
    category
}) => {
    const { features, global, isLoading } = useTrainingJobTopKFeatures({
        jobId,
        entityId,
        category
    });

    const columns = [
        {
            id: "table",
            header: "Table",
            accessorKey: "table",
            meta: { className: "text-title-primary" }
        },
        {
            id: "column",
            header: "Column",
            accessorKey: "column",
            meta: { className: "text-title-primary" }
        },
        {
            id: "value",
            header: "Value",
            accessorKey: "value",
            meta: { className: "text-title-primary" }
        },
        {
            id: "mode",
            header: "Mode",
            accessorKey: "mode",
            meta: { className: "text-title-primary" }
        }
    ];

    const formattedData = useMemo(() => {
        return (
            features?.map((feature) => {
                const key = `${feature?.table_name}.${feature?.name}`;
                return {
                    table: feature?.table_name || "—",
                    column: feature?.name || "—",
                    value: feature?.value ?? "—",
                    mode: global[key]?.value ?? "—"
                };
            }) ?? []
        );
    }, [features, global]);

    return (
        <>
            {formattedData?.length > 0 || isLoading ? (
                <DataTable
                    id="top3-features-table"
                    columns={columns}
                    isLoading={isLoading}
                    disableRowSelectionHover={true}
                    data={formattedData as Array<object>}
                    enableGlobalFilter={false}
                    pageCount={
                        formattedData?.length > 0
                            ? Math.ceil(formattedData?.length / 10)
                            : 1
                    }
                    defaultPageSize={10}
                />
            ) : (
                <EmptyTable
                    id={"top3-features-empty-table"}
                    label="No top features found!"
                />
            )}
        </>
    );
};
