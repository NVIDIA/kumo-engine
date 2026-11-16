// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import { Card, Select, Title } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import StatisticsTable from "components/@ui/TablePlus/StatisticsTable";

import { useSnapshotColumnStats } from "hooks/snapshot/graphSnapshot";

import { ColumnStatsdownOption } from "types/inputs";

import { tableIdIsSnapshot } from "utils/Helpers/typeGuards";

export interface ColumnSummaryWithStatsProps {
    tables: ColumnStatsdownOption[];
}

const ColumnSummaryWithStats = ({ tables }: ColumnSummaryWithStatsProps) => {
    const [selectedTable, setSelectedTable] =
        useState<ColumnStatsdownOption | null>(null);

    const selId: string = selectedTable?.value as string;
    const isSnapshotId = tableIdIsSnapshot(selId);

    useEffect(() => {
        if (tables?.length > 0 && selectedTable === null) {
            setSelectedTable(tables[0]);
        }
    }, [selectedTable, tables]);

    const { columnStats, fetchingSamples } = useSnapshotColumnStats(selId);

    if (!tables?.length) return null;

    return (
        <Card className="p-6 gap-6">
            <div className="flex justify-between items-center gap-2">
                <Title
                    variant="tertiary-heading"
                    children={"Column Statistics"}
                />
                <div className="min-w-45">
                    <Select
                        id="select-table"
                        options={tables?.map((item) => ({
                            value: item?.value as string,
                            label: item?.label as string
                        }))}
                        selectedValues={
                            selectedTable?.value
                                ? [selectedTable?.value as string]
                                : []
                        }
                        onValuesChange={(newValue) => {
                            const currSel = tables?.find((item) =>
                                newValue.includes(item?.value as string)
                            );
                            if (currSel) {
                                setSelectedTable(currSel);
                            }
                        }}
                        search={false}
                        mode="single"
                        placeholder={"Tables"}
                        variant="secondary"
                        size="sm"
                        isInValid={false}
                    />
                </div>
            </div>
            <StatisticsTable
                loading={fetchingSamples}
                search={false}
                columnStats={columnStats}
                primaryKey={selectedTable?.pkey}
                timeCol={selectedTable?.timeCol}
                endTimeCol={selectedTable?.endTimeCol}
                hiddenColumns={["statistics.counts.total"]}
                isSnapShot={isSnapshotId}
            />
        </Card>
    );
};
export default WithErrorBoundary(ColumnSummaryWithStats);
