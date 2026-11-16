// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import Spacer from "react-spacer";

import { DataTable } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { OutputTableColumns } from "../TableColumns/OutputTable";

export interface OutputTableProps {
    title?: string;
    cols: Array<{
        name: string;
        colType: string;
        samples?: Array<number | string>;
    }>;
    loading: boolean;
    areSamplesAvailable: boolean;
}

const OutputTable: React.FC<OutputTableProps> = ({
    title,
    cols,
    loading,
    areSamplesAvailable = false
}) => {
    const columns = OutputTableColumns(areSamplesAvailable);
    return (
        <>
            {title && (
                <label
                    className="mb-0 text-base font-semibold leading-6 text-title-primary"
                    data-testid="output-table-title">
                    {title}
                </label>
            )}
            <Spacer height={8} />
            <DataTable
                id="output-stats-table"
                columns={columns}
                disableRowSelectionHover={true}
                data={cols}
                isLoading={loading}
                defaultPageSize={50}
                enableMultiSort={true}
                enableGlobalFilter={false}
            />
        </>
    );
};

export default WithErrorBoundary(OutputTable);
