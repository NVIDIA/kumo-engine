// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useState } from "react";

import {
    ColumnDef,
    DataTable,
    GridCol,
    Select
} from "@kumo-ai/kumo-ui-catalog";

import { PreprocessingColumns } from "components/Tables/Connect/PreprocessingColumns";

import {
    useColumnRowSelection,
    useColumnTypeOptions
} from "hooks/tables/preprocess";

import { DropdownOption } from "types/inputs";
import { ColumnArgs } from "types/tables";

export interface ColumnSelectionTableProps {
    columns: Array<ColumnArgs>;
    selected: Array<string>;
    showPreview: boolean;
    onSelectColumns: (selectedRows: Array<ColumnArgs>) => void;
    columnDTypes: DropdownOption[];
    columnSTypes: DropdownOption[];
    onCreateTimeColSelect: (
        rowIndex: number,
        cellIndex: number,
        value: ColumnArgs
    ) => void;
    onEndTimeColSelect: (
        rowIndex: number,
        cellIndex: number,
        value: ColumnArgs
    ) => void;
    onSelectPrimaryKey: (index: number, rowData: ColumnArgs) => void;
    onSetColumnDataType: (e: DropdownOption, rowIndex: number) => void;
    onSetColumnSType: (e: DropdownOption, rowIndex: number) => void;
    isLoading: boolean;
}

export const ColumnSelectionTable = ({
    selected,
    showPreview,
    onSelectColumns,
    columnDTypes,
    columnSTypes,
    onCreateTimeColSelect,
    onEndTimeColSelect,
    onSelectPrimaryKey,
    onSetColumnDataType,
    onSetColumnSType,
    columns,
    isLoading
}: ColumnSelectionTableProps) => {
    const [preservedPageIndex, setPreservedPageIndex] = useState(0);
    const defaultPageSize = 50;

    const { dTypeOptions, sTypeOptions } = useColumnTypeOptions(columnDTypes);

    const {
        paginatedData,
        adjustedRowSelectionState,
        handleRowSelectionChange
    } = useColumnRowSelection(
        selected,
        columns,
        isLoading,
        onSelectColumns,
        preservedPageIndex,
        defaultPageSize
    );

    const columnList: ColumnDef<ColumnArgs, string>[] = useMemo(
        () =>
            PreprocessingColumns({
                columnDTypes,
                columnSTypes,
                dTypeOptions,
                showPreview,
                onCreateTimeColSelect,
                onEndTimeColSelect,
                onSelectPrimaryKey,
                onSetColumnDataType,
                onSetColumnSType,
                preservedPageIndex,
                defaultPageSize,
                selected,
                sTypeOptions
            }),
        [
            columnDTypes,
            columnSTypes,
            dTypeOptions,
            showPreview,
            onCreateTimeColSelect,
            onEndTimeColSelect,
            onSelectPrimaryKey,
            onSetColumnDataType,
            onSetColumnSType,
            preservedPageIndex,
            selected,
            sTypeOptions
        ]
    );
    return (
        <DataTable
            id="preprocessing"
            data={paginatedData}
            columns={columnList}
            isLoading={columns?.length < 1 && isLoading}
            pageCount={
                columns?.length > 0
                    ? Math.ceil(columns?.length / defaultPageSize)
                    : 1
            }
            defaultPageSize={defaultPageSize}
            enableMultiSort={true}
            enableGlobalFilter={true}
            showBodyCellRightBorder={true}
            manualRowSelectionState={adjustedRowSelectionState}
            setManualRowSelectionState={handleRowSelectionChange}
            manualPagination={true}
            manualPageIndex={preservedPageIndex}
            handleManualPageChange={(newPage: number) =>
                setPreservedPageIndex(newPage)
            }
        />
    );
};

export const ChangeColumnType = ({
    onChange,
    selected,
    options,
    dType = false,
    isDisabled = false
}: {
    onChange: (e: DropdownOption) => void;
    selected: DropdownOption;
    options: Array<DropdownOption>;
    dType?: boolean;
    isDisabled?: boolean;
}) => {
    return (
        <GridCol xs={12} sm={12} className="w-37.5">
            <Select
                id={`select-${dType ? "d" : "s"}-type`}
                placeholder={dType ? "Data Type" : "Semantic Type"}
                search={false}
                mode="single"
                variant="nooutline"
                sortOptions={false}
                popoverClassName={dType ? "w-40" : "w-125"}
                selectedValues={
                    !selected || selected?.value === ""
                        ? undefined
                        : Array.of(selected?.value as string)
                }
                options={options?.map((item) => {
                    return {
                        value: item?.value as string,
                        label: item?.label as string,
                        labelDescription: item?.subLabel as string,
                        disabled: item?.isDisabled === true ? true : false
                    };
                })}
                onValuesChange={(newValue) => {
                    const currSel = options?.find((item) =>
                        newValue.includes(item?.value as string)
                    );
                    onChange(currSel as DropdownOption);
                }}
                disabled={isDisabled}
            />
        </GridCol>
    );
};
