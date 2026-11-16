// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import { act, renderHook, waitFor } from "@testing-library/react";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { vi } from "vitest";

import { DropdownOption } from "types/inputs";
import { ColumnArgs } from "types/tables";

import { tableDataTypes, tableSemanticTypes } from "utils/Constants";

import {
    useColumnManagement,
    useColumnMetadataSync,
    useColumnRowSelection,
    useColumnTypeOptions
} from "../preprocess";

beforeEach(() => {
    vi.clearAllMocks();
});

describe("useColumnTypeOptions", () => {
    const renderUseColumnTypeOptions = (columnDTypes: DropdownOption[]) => {
        return renderHook(() => useColumnTypeOptions(columnDTypes));
    };

    it("Should generate dType options for all table data types", () => {
        const { result } = renderUseColumnTypeOptions([]);

        expect(result.current.dTypeOptions).toHaveLength(tableDataTypes.length);
        result.current.dTypeOptions.forEach((option, index) => {
            expect(option.value).toBe(tableDataTypes[index]);
        });
    });

    it("Should label dType options with capitalized first letter", () => {
        const { result } = renderUseColumnTypeOptions([]);

        const boolOption = result.current.dTypeOptions.find(
            (o) => o.value === "bool"
        );
        expect(boolOption?.label).toBe("Bool");

        const floatOption = result.current.dTypeOptions.find(
            (o) => o.value === "float64"
        );
        expect(floatOption?.label).toBe("Float64");
    });

    it.each([
        {
            description: "int dtype",
            dtype: "int",
            enabledStypes: ["numerical", "categorical", "ID"]
        },
        {
            description: "string dtype",
            dtype: "string",
            enabledStypes: [
                "categorical",
                "multicategorical",
                "ID",
                "text",
                "sequence",
                "image",
                "timestamp"
            ]
        },
        {
            description: "date dtype",
            dtype: "date",
            enabledStypes: ["timestamp"]
        },
        {
            description: "bool dtype",
            dtype: "bool",
            enabledStypes: ["categorical"]
        },
        {
            description: "float64 dtype",
            dtype: "float64",
            enabledStypes: ["numerical", "categorical", "ID"]
        },
        {
            description: "floatlist dtype",
            dtype: "floatlist",
            enabledStypes: ["sequence"]
        },
        {
            description: "timedelta dtype",
            dtype: "timedelta",
            enabledStypes: ["numerical", "categorical"]
        }
    ])(
        "Should enable correct stypes for $description",
        ({ dtype, enabledStypes }) => {
            const { result } = renderUseColumnTypeOptions([
                { value: dtype, label: dtype }
            ]);

            const stypeRow = result.current.sTypeOptions[0];

            stypeRow.forEach((option) => {
                const shouldBeEnabled = enabledStypes.includes(
                    option.value as string
                );
                expect(option.isDisabled).toBe(!shouldBeEnabled);
            });
        }
    );

    it("Should label sequence stype as Embedding", () => {
        const { result } = renderUseColumnTypeOptions([
            { value: "string", label: "String" }
        ]);

        const sequenceOption = result.current.sTypeOptions[0].find(
            (o) => o.value === "sequence"
        );
        expect(sequenceOption?.label).toBe("Embedding");
    });

    it("Should generate sType options for all semantic types", () => {
        const { result } = renderUseColumnTypeOptions([
            { value: "string", label: "String" }
        ]);

        expect(result.current.sTypeOptions[0]).toHaveLength(
            tableSemanticTypes.length
        );
    });

    it("Should include subLabel from dataTypeDescription", () => {
        const { result } = renderUseColumnTypeOptions([
            { value: "string", label: "String" }
        ]);

        const numericalOption = result.current.sTypeOptions[0].find(
            (o) => o.value === "numerical"
        );
        expect(numericalOption?.subLabel).toBe(
            "Numbers and fractions whose values relative to each other are important (eg amounts)."
        );
    });

    it("Should return empty sTypeOptions when columnDTypes is empty", () => {
        const { result } = renderUseColumnTypeOptions([]);

        expect(result.current.sTypeOptions).toHaveLength(0);
    });

    it("Should generate sType options per column when multiple dtypes provided", () => {
        const { result } = renderUseColumnTypeOptions([
            { value: "int", label: "Int" },
            { value: "date", label: "Date" }
        ]);

        expect(result.current.sTypeOptions).toHaveLength(2);

        const intStypes = result.current.sTypeOptions[0];
        const idOption = intStypes.find((o) => o.value === "ID");
        expect(idOption?.isDisabled).toBe(false);

        const dateStypes = result.current.sTypeOptions[1];
        const timestampOption = dateStypes.find((o) => o.value === "timestamp");
        expect(timestampOption?.isDisabled).toBe(false);

        const dateIdOption = dateStypes.find((o) => o.value === "ID");
        expect(dateIdOption?.isDisabled).toBe(true);
    });
});

describe("useColumnMetadataSync", () => {
    const createColumn = (
        name: string,
        overrides: Partial<ColumnArgs> = {}
    ): ColumnArgs => ({
        name,
        stype: "numerical",
        dtype: "float64",
        isPrimary: false,
        isTimeColumn: false,
        isEndTimeColumn: false,
        ...overrides
    });

    const createTableArgs = (
        pkey?: string,
        timeCol?: string
    ): KumoapiTableTableDefinition =>
        ({
            cols: [],
            pkey: pkey ?? null,
            time_col: timeCol ?? null,
            source_table: {
                connector_id: "test",
                source: { mode: "relative", path: "test" },
                data_source_type: "S3",
                file_type: "PARQUET"
            }
        }) as KumoapiTableTableDefinition;

    const renderSync = (
        columns: ColumnArgs[],
        tableArgs: KumoapiTableTableDefinition,
        columnSTypes: DropdownOption[] = columns.map((c) => ({
            value: c.stype,
            label: c.stype
        }))
    ) => {
        const setColumns = vi.fn();
        renderHook(() =>
            useColumnMetadataSync(columns, columnSTypes, tableArgs, setColumns)
        );
        return { setColumns };
    };

    it("Should set isTimeColumn flag when time_col matches a column", () => {
        const columns = [
            createColumn("id"),
            createColumn("created_at", { stype: "timestamp" })
        ];
        const tableArgs = createTableArgs(undefined, "created_at");

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).toHaveBeenCalledWith(
            expect.arrayContaining([
                expect.objectContaining({
                    name: "created_at",
                    isTimeColumn: true
                })
            ])
        );
    });

    it("Should set isPrimary flag and stype to ID when pkey matches a column", () => {
        const columns = [
            createColumn("id", { stype: "numerical" }),
            createColumn("value")
        ];
        const tableArgs = createTableArgs("id");

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).toHaveBeenCalledWith(
            expect.arrayContaining([
                expect.objectContaining({
                    name: "id",
                    isPrimary: true,
                    stype: "ID"
                })
            ])
        );
    });

    it("Should not call setColumns when columns are already synced", () => {
        const columns = [
            createColumn("id", { isPrimary: true, stype: "ID" }),
            createColumn("created_at", {
                isTimeColumn: true,
                stype: "timestamp"
            })
        ];
        const tableArgs = createTableArgs("id", "created_at");

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).not.toHaveBeenCalled();
    });

    it("Should not call setColumns when columns are empty", () => {
        const tableArgs = createTableArgs("id");

        const { setColumns } = renderSync([], tableArgs);

        expect(setColumns).not.toHaveBeenCalled();
    });

    it("Should set both isPrimary and isTimeColumn when both pkey and time_col are set", () => {
        const columns = [
            createColumn("id", { stype: "numerical" }),
            createColumn("created_at", { stype: "timestamp" })
        ];
        const tableArgs = createTableArgs("id", "created_at");

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).toHaveBeenCalledWith(
            expect.arrayContaining([
                expect.objectContaining({
                    name: "id",
                    isPrimary: true,
                    stype: "ID"
                }),
                expect.objectContaining({
                    name: "created_at",
                    isTimeColumn: true
                })
            ])
        );
    });

    it("Should not call setColumns when pkey does not match any column", () => {
        const columns = [createColumn("id"), createColumn("value")];
        const tableArgs = createTableArgs("nonexistent");

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).not.toHaveBeenCalled();
    });

    it("Should not call setColumns when neither pkey nor time_col is set", () => {
        const columns = [createColumn("id"), createColumn("value")];
        const tableArgs = createTableArgs();

        const { setColumns } = renderSync(columns, tableArgs);

        expect(setColumns).not.toHaveBeenCalled();
    });
});

describe("useColumnManagement multicategorical delimiter", () => {
    const renderManagement = (column: ColumnArgs) =>
        renderHook(() => {
            const [columns, setColumns] = useState([column]);
            const [columnDTypes, setSelectedDType] = useState<DropdownOption[]>(
                [{ value: column.dtype, label: column.dtype }]
            );
            const [columnSTypes, setSelectedSType] = useState<DropdownOption[]>(
                [{ value: column.stype, label: column.stype }]
            );
            const [tableArgs, setTableArgs] =
                useState<KumoapiTableTableDefinition>({
                    cols: [
                        {
                            name: column.name,
                            stype: column.stype,
                            dtype: column.dtype,
                            multicategorical_delimiter:
                                column.multicategorical_delimiter
                        }
                    ],
                    source_table: {
                        connector_id:
                            "connector_00000000000000000000000000000000",
                        file_type: "PARQUET",
                        source: { mode: "relative", path: "items" }
                    }
                } as KumoapiTableTableDefinition);
            const [pkPrevSType, setPkPrevSType] = useState<DropdownOption>({
                value: "",
                label: ""
            });
            const [invalid, setInvalidError] = useState({
                title: "",
                description: ""
            });
            const [selected, setSelected] = useState(["true"]);
            const [allCols, setAllCols] = useState(["true"]);
            const callbacks = useColumnManagement(
                columns,
                columnDTypes,
                columnSTypes,
                tableArgs,
                pkPrevSType,
                setColumns,
                setSelectedDType,
                setSelectedSType,
                setTableArgs,
                setPkPrevSType,
                setInvalidError,
                allCols,
                setSelected,
                setAllCols
            );
            return { callbacks, columns, tableArgs, invalid, selected };
        });

    it("clears proof when semantic type changes away", () => {
        const { result } = renderManagement({
            name: "tags",
            stype: "multicategorical",
            dtype: "string",
            multicategorical_delimiter: ","
        });

        act(() =>
            result.current.callbacks.onSetColumnSType(
                { value: "categorical", label: "Categorical" },
                0
            )
        );

        expect(result.current.columns[0].multicategorical_delimiter).toBeNull();
        expect(
            result.current.tableArgs.cols[0].multicategorical_delimiter
        ).toBeNull();
    });

    it("clears scalar proof when dtype becomes a native list", () => {
        const { result } = renderManagement({
            name: "tags",
            stype: "multicategorical",
            dtype: "string",
            multicategorical_delimiter: "|"
        });

        act(() =>
            result.current.callbacks.onSetColumnDataType(
                { value: "stringlist", label: "Stringlist" },
                0
            )
        );

        expect(result.current.columns[0].multicategorical_delimiter).toBeNull();
        expect(
            result.current.tableArgs.cols[0].multicategorical_delimiter
        ).toBeNull();
    });

    it("rejects a native-list to scalar transition without delimiter proof", () => {
        const { result } = renderManagement({
            name: "tags",
            stype: "multicategorical",
            dtype: "stringlist"
        });

        act(() =>
            result.current.callbacks.onSetColumnDataType(
                { value: "string", label: "String" },
                0
            )
        );

        expect(result.current.columns[0].dtype).toBe("stringlist");
        expect(result.current.tableArgs.cols[0].dtype).toBe("stringlist");
        expect(result.current.invalid.title).toBe(
            "Multicategorical delimiter required"
        );
        expect(result.current.invalid.description).toContain(
            "requires a delimiter proven from its source sample"
        );
    });

    it("rejects scalar multicategorical selection without inferred proof", () => {
        const { result } = renderManagement({
            name: "tags",
            stype: "categorical",
            dtype: "string"
        });

        act(() =>
            result.current.callbacks.onSetColumnSType(
                { value: "multicategorical", label: "Multicategorical" },
                0
            )
        );

        expect(result.current.columns[0].stype).toBe("categorical");
        expect(result.current.invalid.title).toBe(
            "Multicategorical delimiter required"
        );
    });
});

describe("useColumnRowSelection", () => {
    const createColumns = (count: number): ColumnArgs[] =>
        Array.from({ length: count }, (_, i) => ({
            name: `col_${i}`,
            stype: "numerical",
            dtype: "float64"
        }));

    const onSelectColumns = vi.fn();

    const renderRowSelection = (
        selected: string[],
        columns: ColumnArgs[],
        pageIndex = 0,
        pageSize = 5,
        isLoading = false
    ) => {
        return renderHook(() =>
            useColumnRowSelection(
                selected,
                columns,
                isLoading,
                onSelectColumns,
                pageIndex,
                pageSize
            )
        );
    };

    it.each([
        {
            description: "first page",
            pageIndex: 0,
            expectedLength: 5,
            expectedFirstName: "col_0"
        },
        {
            description: "second page",
            pageIndex: 1,
            expectedLength: 5,
            expectedFirstName: "col_5"
        }
    ])(
        "Should return paginated data for $description",
        ({ pageIndex, expectedLength, expectedFirstName }) => {
            const columns = createColumns(10);
            const selected = columns.map(() => "true");

            const { result } = renderRowSelection(
                selected,
                columns,
                pageIndex,
                5
            );

            expect(result.current.paginatedData).toHaveLength(expectedLength);
            expect(result.current.paginatedData[0].name).toBe(
                expectedFirstName
            );
        }
    );

    it("Should initialize row selection state from selected array", async () => {
        const columns = createColumns(3);
        const selected = ["true", "false", "true"];

        const { result } = renderRowSelection(selected, columns);

        await waitFor(() => {
            expect(result.current.rowSelectionState).toEqual({
                0: true,
                1: false,
                2: true
            });
        });
    });

    it("Should return empty row selection when loading", () => {
        const columns = createColumns(3);
        const selected = ["true", "true", "true"];

        const { result } = renderRowSelection(selected, columns, 0, 5, true);

        expect(result.current.rowSelectionState).toEqual({});
    });

    it("Should handle partial last page", () => {
        const columns = createColumns(7);
        const selected = columns.map(() => "true");

        const { result } = renderRowSelection(selected, columns, 1, 5);

        expect(result.current.paginatedData).toHaveLength(2);
        expect(result.current.paginatedData[0].name).toBe("col_5");
        expect(result.current.paginatedData[1].name).toBe("col_6");
    });

    it("Should compute adjustedRowSelectionState for current page", async () => {
        const columns = createColumns(10);
        const selected = [
            "true",
            "false",
            "true",
            "false",
            "false",
            "true",
            "false",
            "true",
            "false",
            "false"
        ];

        const { result } = renderRowSelection(selected, columns, 1, 5);

        await waitFor(() => {
            expect(result.current.adjustedRowSelectionState).toEqual({
                0: true,
                2: true
            });
        });
    });

    it("Should update selection via handleRowSelectionChange", async () => {
        const columns = createColumns(5);
        const selected = ["true", "true", "false", "true", "false"];

        const { result } = renderRowSelection(selected, columns, 0, 5);

        await waitFor(() => {
            expect(result.current.rowSelectionState[2]).toBeFalsy();
        });

        act(() => {
            result.current.handleRowSelectionChange({
                0: true,
                1: true,
                2: true,
                3: true,
                4: false
            });
        });

        await waitFor(() => {
            expect(result.current.rowSelectionState[2]).toBe(true);
        });
    });

    it("Should return empty paginated data when columns are empty", () => {
        const { result } = renderRowSelection([], []);

        expect(result.current.paginatedData).toHaveLength(0);
        expect(result.current.rowSelectionState).toEqual({});
    });
});
