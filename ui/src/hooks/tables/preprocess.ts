// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { RowSelectionState } from "@kumo-ai/kumo-ui-catalog";
import { ValidationWarning } from "gen/v0/types";
import { useCreateV1TablesInferMetadata } from "gen/v1/api/tables/tables";
import { Column } from "gen/v1/types/column";
import { Dtype as KumoapiTypingDtype } from "gen/v1/types/dtype";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { Stype as KumoapiTypingStype } from "gen/v1/types/stype";

import { useGetColumns, useGetSampleRows } from "hooks/connector/connector";
import { useProjectId } from "hooks/project/projects";

import { CustomError } from "types/common";
import { DropdownOption } from "types/inputs";
import { ColumnArgs, FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import {
    dataTypeDescription,
    errorStatusCode,
    numSampleRows,
    tableDataTypes,
    tableDataTypesToSTypesMapping,
    tableSemanticTypes
} from "utils/Constants";
import { parseErrorMessage, prepareCustomError } from "utils/errorHandler";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import {
    assignSampleRows,
    canInferStatsBasedMetadata,
    parseColsAndSmplRows,
    RecommendedStype,
    reconcileStypeRecommendations,
    toApiColumn,
    updateWithRecommendeStype
} from "utils/tables";
interface UseTablePreprocessingParams {
    connectorId: string;
    selectedSourceTableName: string;
    tableArgs: KumoapiTableTableDefinition;
    flowType: FlowType;
    connectorNotExists: boolean;
    selectFromTableArgs?: boolean;
    cancelFetchSampleCal: boolean;
    setTableArgs: React.Dispatch<
        React.SetStateAction<KumoapiTableTableDefinition>
    >;
    setUnnamedConSourceTable: React.Dispatch<React.SetStateAction<string>>;
    setLoading: (v: boolean) => void;
    handleError: HandleErrorFunc;
}

interface UseTablePreprocessingReturn {
    columns: Array<ColumnArgs>;
    columnSTypes: Array<DropdownOption>;
    columnDTypes: Array<DropdownOption>;
    selected: Array<string>;
    allCols: Array<string>;
    errors: {
        columnWarnings: { warnings?: ValidationWarning[] } | null;
        sampleRowError: string;
        invalidKeyError: { title: string; description: string };
    };
    newStypes: {
        name: string;
        newStype: string;
        multicategoricalDelimiter?: "," | "|" | ";" | "\t" | null;
    }[];
    pkPrevSType: DropdownOption;
    isLoading: boolean;
    recommendedSTypes: Array<string>;
    onSetColumnDataType: (e: DropdownOption, rowIndex: number) => void;
    onSetColumnSType: (e: DropdownOption, rowIndex: number) => void;
    onSelectColumns: (selectedRows: ColumnArgs[]) => void;
    onSelectPrimaryKey: (index: number, rowData: ColumnArgs) => void;
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
    applyRecommendedTypes: (checkUserUpdated: boolean) => void;
    setColumns: React.Dispatch<React.SetStateAction<Array<ColumnArgs>>>;
    setSelectedSType: React.Dispatch<React.SetStateAction<DropdownOption[]>>;
    setSelectedDType: React.Dispatch<React.SetStateAction<DropdownOption[]>>;
    setPkPrevSType: React.Dispatch<React.SetStateAction<DropdownOption>>;
    setInvalidError: React.Dispatch<
        React.SetStateAction<{ title: string; description: string }>
    >;
    setNewStypes: React.Dispatch<
        React.SetStateAction<
            {
                name: string;
                newStype: string;
                multicategoricalDelimiter?: "," | "|" | ";" | "\t" | null;
            }[]
        >
    >;
    setSampleRowError: React.Dispatch<React.SetStateAction<string>>;
    isView: boolean;
    skipSampleFetch: () => void;
}

export const useTablePreprocessing = ({
    connectorId,
    selectedSourceTableName,
    tableArgs,
    flowType,
    selectFromTableArgs,
    cancelFetchSampleCal,
    setTableArgs,
    setUnnamedConSourceTable,
    setLoading,
    handleError
}: UseTablePreprocessingParams): UseTablePreprocessingReturn => {
    const [columnSTypes, setSelectedSType] = useState<DropdownOption[]>([]);
    const [columnDTypes, setSelectedDType] = useState<DropdownOption[]>([]);
    const [invalidKeyError, setInvalidError] = useState<{
        title: string;
        description: string;
    }>({ title: "", description: "" });
    const [sampleRowError, setSampleRowError] = useState("");

    const [fetchColumns, setFetchColumns] = useState(false);

    // Using `state` and `ref` because can't get actual state inside of
    // `setTimeout` or `setInterval` and can't render the UI with
    // by updating `ref`
    const [skipSamples, setSkipSamples] = useState<boolean>(
        flowType === FlowType.EDIT
    );
    const skipRef = useRef(flowType === FlowType.EDIT ? 1 : 0);
    // Creating `ref` checking value inside `fetchSampleRows` is because
    // we can't get actual value of state inside of `setTimeout` or
    // `setInterval`. Added `useEffect` to update `cancelSampleRef`
    const cancelSampleRef = useRef(cancelFetchSampleCal);

    const [fetchSampleRows, setFetchSampleRows] = useState(false);
    const [selected, setSelected] = useState<Array<string>>([]);
    //Stores all columns visible, or hidden due to search.
    const [allCols, setAllCols] = useState<Array<string>>([]);

    const [columns, setColumns] = useState<Array<ColumnArgs>>([]);
    const [newStypes, setNewStypes] = useState<
        {
            name: string;
            newStype: string;
            multicategoricalDelimiter?: "," | "|" | ";" | "\t" | null;
        }[]
    >([]);

    const [pkPrevSType, setPkPrevSType] = useState<DropdownOption>({
        label: "",
        value: ""
    });

    const sourceTableName = useRef("");

    // Stats-based stype inference (kumo-ai/kumo#30131): the connector
    // `columns`/`sample` previews rendered above are stats-FREE, while the
    // SDK's `infer_metadata` defaults are stats-BASED, so the two authoring
    // paths diverge on identical data. The wizard requests the stats-based
    // inference for the selected source and reconciles it into the same
    // recommendation pipeline the sample fetch uses (auto-apply on create,
    // offer via the banner on edit). The request is best-effort: on failure
    // the stats-free preview stypes stand, as before.
    const projectId = useProjectId() ?? "";
    const [statsBasedCols, setStatsBasedCols] = useState<Column[] | null>(null);
    // Ref mirror so the sample-rows effect can reconcile against the latest
    // stats-based result without re-running on its arrival.
    const statsBasedColsRef = useRef<Column[] | null>(null);
    const sampleRecsRef = useRef<RecommendedStype[]>([]);
    const statsAppliedRef = useRef(false);
    // Sequence token of the latest inference dispatch. A response whose
    // dispatch-time token no longer matches is a stale answer for a
    // previously selected source and must be discarded, or rapid source
    // switches would stamp another table's stypes onto same-named columns.
    // The token, not the request's `source_table`, identifies the request
    // because the wizard mutates one shared `source_table` object in place
    // across selections (`applySelectedSourceTable`); the source-switch
    // reset bumps it so an in-flight request is invalidated even when the
    // new source dispatches no inference of its own.
    const statsRequestSeqRef = useRef(0);

    const { mutate: requestStatsBasedMetadata } =
        useCreateV1TablesInferMetadata();

    useEffect(() => {
        cancelSampleRef.current = cancelFetchSampleCal;
    }, [cancelFetchSampleCal]);

    useEffect(() => {
        const sourceChanged =
            selectedSourceTableName !== "" &&
            sourceTableName?.current !== selectedSourceTableName;
        if (connectorId && sourceChanged) {
            sourceTableName.current = selectedSourceTableName as string;
            setSkipSamples(flowType === FlowType.EDIT);
            skipRef.current = flowType === FlowType.EDIT ? 1 : 0; // Start column/samples fetch
            setLoading(true);
            setFetchColumns(true);
            setFetchSampleRows(false);
            setNewStypes([]);
            setSelectedSType([]);
            setSelectedDType([]);
            setColumns([]);
            setSelected([]);
            setAllCols([]);
            setStatsBasedCols(null);
            statsBasedColsRef.current = null;
            sampleRecsRef.current = [];
            statsAppliedRef.current = false;
            statsRequestSeqRef.current += 1;
        }
    }, [
        connectorId,
        fetchColumns,
        flowType,
        selectedSourceTableName,
        setLoading
    ]);

    useEffect(() => {
        if (isValid(sampleRowError) && columns?.length > 0) {
            if (columns[0]?.isSampleAvailable === false) {
                const cols = [...columns];
                // Updating columns with `isSampleAvailable` with true and sample
                // with null,  This is the case where sample rows fetch failed and
                // it is not possible to get updated info of `columns` in
                // catch block of `fetchSampleRows`
                const updatedData = cols?.map((c) => {
                    return {
                        ...c,
                        isSampleAvailable: true,
                        sampleRows: null,
                        nonEmpty: null
                    } as ColumnArgs;
                });
                skipRef.current = 0;
                setColumns(updatedData);
            }
        }
    }, [columns, sampleRowError]);

    // Sync column metadata (time column and primary key) with table args
    useColumnMetadataSync(columns, columnSTypes, tableArgs, setColumns);

    const handleGetColumnsError = useCallback(
        (err: Error) => {
            setFetchColumns(false);
            setLoading(false);
            setColumns([]);
            setNewStypes([]);
            setSelectedSType([]);
            setSelectedDType([]);
            setFetchSampleRows(false);
            const message = parseErrorMessage(err as CustomError);
            const error: CustomError = prepareCustomError(
                message,
                (err as { response?: { status?: number } })?.response?.status ??
                    errorStatusCode.UnProcessableEntityError
            );
            handleError(error, "Failed to load columns");
        },
        [handleError, setLoading]
    );

    const handleGetSampleRowsError = useCallback((error: Error) => {
        setSkipSamples(true);
        setFetchSampleRows(false);
        setSampleRowError(parseErrorMessage(error as CustomError));
    }, []);

    const params = useMemo(() => {
        return {
            connectorId: connectorId ?? ("" as string),
            params: {
                source_table_name: selectedSourceTableName ?? ("" as string),
                num_rows: numSampleRows
            }
        };
    }, [connectorId, selectedSourceTableName]);

    const {
        columns: tableCols,
        isLoading: fetching,
        sourceTableName: unSourceTableName,
        columnWarnings
    } = useGetColumns(params, handleGetColumnsError, setLoading, true);

    useEffect(() => {
        if (!fetching && tableCols && tableCols.length > 0 && fetchColumns) {
            setLoading(false);
            setFetchColumns(false);
            const tmpArgs = { ...tableArgs };
            setSampleRowError("");

            const {
                sourceColumns,
                cols,
                selectedColumns,
                primaryKeyCol,
                timeColumn,
                endTimeColumn
            } = parseColsAndSmplRows(
                tableCols,
                tmpArgs.cols,
                flowType,
                tmpArgs.pkey as string,
                tmpArgs.time_col as string,
                tmpArgs.end_time_col as string
            );

            const sTypeOptions: DropdownOption[] = [];
            const dTypeOptions = [];
            for (const column of sourceColumns) {
                sTypeOptions.push({
                    value: column?.stype,
                    label: column?.sTypeLabel,
                    origValue: column?.stype
                });
                dTypeOptions.push({
                    value: column?.dtype,
                    label: column?.dTypeLabel,
                    origValue: column?.dtype
                });
            }
            if (unSourceTableName) {
                setUnnamedConSourceTable(unSourceTableName);
            }
            setSelectedSType(sTypeOptions);
            setSelectedDType(dTypeOptions);
            setColumns(sourceColumns);
            if (selectFromTableArgs) {
                const colNames: Array<string> = tableArgs?.cols?.map(
                    (c) => c.name as string
                );
                const sel: Array<string> = sourceColumns?.map((c) => {
                    if (c?.name && colNames.indexOf(c?.name) > -1) {
                        return "true";
                    } else {
                        return "false";
                    }
                });

                setSelected(sel);
                setAllCols(sel);
            } else {
                setSelected(selectedColumns);
                setAllCols(selectedColumns);
            }
            tmpArgs.cols = cols;
            tmpArgs.pkey = primaryKeyCol;
            tmpArgs.time_col = timeColumn;
            tmpArgs.end_time_col = endTimeColumn;
            setTableArgs(tmpArgs);
            setFetchSampleRows(true);
            // Ask the backend for the SDK's stats-based inference of this
            // source; `cols: []` mirrors an SDK `infer_metadata` call so both
            // authoring paths start from the same defaults.
            if (canInferStatsBasedMetadata(tmpArgs.source_table)) {
                statsRequestSeqRef.current += 1;
                const requestSeq = statsRequestSeqRef.current;
                requestStatsBasedMetadata(
                    {
                        data: { cols: [], source_table: tmpArgs.source_table },
                        params: { project_id: projectId }
                    },
                    {
                        onSuccess: (response) => {
                            if (requestSeq !== statsRequestSeqRef.current) {
                                return;
                            }
                            const cols = response?.data?.cols ?? null;
                            statsBasedColsRef.current = cols;
                            setStatsBasedCols(cols);
                        },
                        onError: () => {
                            if (requestSeq !== statsRequestSeqRef.current) {
                                return;
                            }
                            // Best-effort enhancement: keep the stats-free
                            // preview stypes when the deeper inference is
                            // unavailable.
                            statsBasedColsRef.current = null;
                            setStatsBasedCols(null);
                        }
                    }
                );
            }
        }
    }, [
        fetchColumns,
        fetching,
        flowType,
        projectId,
        requestStatsBasedMetadata,
        selectFromTableArgs,
        setLoading,
        setTableArgs,
        setUnnamedConSourceTable,
        tableArgs,
        tableCols,
        unSourceTableName
    ]);

    const isViewValue = false;

    const shouldFetchSampleRows = fetchSampleRows && !cancelFetchSampleCal;

    const { columnsWithRows, isLoading: loadingSampleRows } = useGetSampleRows(
        params,
        handleGetSampleRowsError,
        shouldFetchSampleRows
    );

    useEffect(() => {
        if (columnsWithRows && !loadingSampleRows) {
            if (
                (columnsWithRows?.rows?.length as number) > 0 &&
                fetchSampleRows
            ) {
                setFetchSampleRows(false);
                const [colWithSamples, columnsWithNewSTypes] = assignSampleRows(
                    columnsWithRows.columns ?? [],
                    columns,
                    columnsWithRows?.rows as string[][]
                );
                // Keep the raw sample recommendations and reconcile them
                // with the stats-based inference (which wins per column) so
                // the outcome is order-independent across the two fetches.
                sampleRecsRef.current = columnsWithNewSTypes;
                if (skipRef?.current !== 1) {
                    const { newArgs, updatedColumn, updatedSelectStype } =
                        updateWithRecommendeStype(
                            columns,
                            columnSTypes,
                            tableArgs,
                            reconcileStypeRecommendations(
                                statsBasedColsRef.current,
                                columnsWithNewSTypes,
                                columns
                            ),
                            /*checkUserUpdated:*/ true,
                            /*markUserUpdated:*/ false
                        );
                    setTableArgs((args) => ({
                        ...args,
                        cols: newArgs?.cols,
                        time_col: newArgs?.time_col,
                        end_time_col: newArgs?.end_time_col
                    }));
                    setSelectedSType(updatedSelectStype);
                    setColumns(updatedColumn);
                } else {
                    setColumns(colWithSamples);
                    setNewStypes(
                        reconcileStypeRecommendations(
                            statsBasedColsRef.current,
                            columnsWithNewSTypes,
                            colWithSamples
                        )
                    );
                }
                skipRef.current = 0;
                setSkipSamples(true);
            }
        }
    }, [
        fetchSampleRows,
        columnSTypes,
        columns,
        columnsWithRows,
        loadingSampleRows,
        setTableArgs,
        tableArgs
    ]);

    // Apply the stats-based inference once it arrives (kumo-ai/kumo#30131):
    // on create the reconciled stypes become the defaults (columns the user
    // already retyped are left alone); on edit they are offered through the
    // existing recommendation banner instead of silently rewriting a saved
    // table.
    useEffect(() => {
        if (
            !statsBasedCols ||
            statsAppliedRef.current ||
            !(columns?.length > 0) ||
            // A source-switch reset clears the ref synchronously before the
            // queued `statsBasedCols` clear lands; a mismatch is that stale
            // window, and latching here would drop the next table's stats.
            statsBasedCols !== statsBasedColsRef.current
        ) {
            return;
        }
        statsAppliedRef.current = true;
        const recommendations = reconcileStypeRecommendations(
            statsBasedCols,
            sampleRecsRef.current,
            columns
        );
        if (flowType === FlowType.EDIT) {
            setNewStypes(recommendations);
            return;
        }
        if (recommendations.length < 1) {
            return;
        }
        const { newArgs, updatedColumn, updatedSelectStype } =
            updateWithRecommendeStype(
                columns,
                columnSTypes,
                tableArgs,
                recommendations,
                /*checkUserUpdated:*/ true,
                /*markUserUpdated:*/ false
            );
        setTableArgs((args) => ({
            ...args,
            cols: newArgs?.cols,
            time_col: newArgs?.time_col,
            end_time_col: newArgs?.end_time_col
        }));
        setSelectedSType(updatedSelectStype);
        setColumns(updatedColumn);
    }, [
        statsBasedCols,
        columns,
        columnSTypes,
        flowType,
        setTableArgs,
        tableArgs
    ]);

    const skipSampleFetch = () => {
        skipRef.current = 1;
        setSkipSamples(true);
    };

    const {
        onSetColumnDataType,
        onSetColumnSType,
        onSelectColumns,
        onSelectPrimaryKey,
        onCreateTimeColSelect,
        onEndTimeColSelect
    } = useColumnManagement(
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

    // Recommended types logic
    const recommendedSTypes = useMemo(() => {
        const recommended: Array<string> = [];

        newStypes?.forEach((col) => {
            const cIndex = columns?.findIndex(
                (column) =>
                    column?.name === col?.name &&
                    column?.stype !== col?.newStype
            );
            if (cIndex !== -1) {
                recommended.push(
                    "Column '" +
                        col.name +
                        "' to '" +
                        capitalizeFirstLetter(col?.newStype) +
                        "'"
                );
            }
        });
        return recommended;
    }, [columns, newStypes]);

    const applyRecommendedTypes = useCallback(
        (checkUserUpdated: boolean) => {
            if (!newStypes || newStypes?.length < 1) {
                return;
            }
            const { newArgs, updatedColumn, updatedSelectStype } =
                updateWithRecommendeStype(
                    columns,
                    columnSTypes,
                    tableArgs,
                    newStypes,
                    checkUserUpdated
                );

            setTableArgs((args: KumoapiTableTableDefinition) => ({
                ...args,
                cols: newArgs?.cols,
                time_col: newArgs?.time_col,
                end_time_col: newArgs?.end_time_col
            }));
            setSelectedSType(updatedSelectStype);
            setColumns(updatedColumn);
            setNewStypes([]);
        },
        [columns, columnSTypes, newStypes, setTableArgs, tableArgs]
    );

    return {
        columns,
        columnSTypes,
        columnDTypes,
        selected,
        allCols,
        errors: { sampleRowError, invalidKeyError, columnWarnings },
        newStypes,
        pkPrevSType,
        isLoading: (fetchSampleRows || fetchColumns) && !skipSamples,
        recommendedSTypes,
        onSetColumnDataType,
        onSetColumnSType,
        onSelectColumns,
        onSelectPrimaryKey,
        onCreateTimeColSelect,
        onEndTimeColSelect,
        applyRecommendedTypes,
        setColumns,
        setSelectedSType,
        setSelectedDType,
        setPkPrevSType,
        setInvalidError,
        setNewStypes,
        setSampleRowError,
        isView: isViewValue,
        skipSampleFetch
    };
};

/** Restore the previous primary-key column's semantic type when it changes. */
function restorePreviousPkeyType(
    columnsList: Array<ColumnArgs>,
    tmpSelectedSType: Array<DropdownOption>,
    tmpArgs: KumoapiTableTableDefinition,
    cIndex: number,
    pkPrevSType: DropdownOption
): void {
    columnsList[cIndex].isPrimary = false;
    if (
        pkPrevSType.value !== "" &&
        pkPrevSType.value !== tmpSelectedSType[cIndex].value
    ) {
        tmpSelectedSType[cIndex] = pkPrevSType;
        columnsList[cIndex].stype = pkPrevSType?.value as string;
        if (pkPrevSType.value !== "multicategorical") {
            columnsList[cIndex].multicategorical_delimiter = null;
        }
        columnsList[cIndex].sTypeLabel =
            pkPrevSType?.label === "Embedding"
                ? "Sequence"
                : pkPrevSType?.label;
        columnsList[cIndex].userUpdatedType = true;
        const clIndex = tmpArgs?.cols?.findIndex(
            (col) => col?.name === columnsList[cIndex]?.name
        );
        if (clIndex !== -1) {
            tmpArgs.cols[clIndex].stype =
                pkPrevSType?.value as KumoapiTypingStype;
            if (pkPrevSType.value !== "multicategorical") {
                tmpArgs.cols[clIndex].multicategorical_delimiter = null;
            }
        }
    }
}

/** Mark a column as the new primary key: set ID stype, clear time/end-time flags. */
function applyNewPkeyColumn(
    columnsList: Array<ColumnArgs>,
    tmpSelectedSType: Array<DropdownOption>,
    tmpArgs: KumoapiTableTableDefinition,
    index: number,
    pkStype: DropdownOption
): void {
    columnsList[index].isPrimary = true;
    if (columnsList[index]?.isTimeColumn) {
        columnsList[index].isTimeColumn = false;
        tmpArgs.time_col = "";
    }
    if (columnsList[index]?.isEndTimeColumn) {
        columnsList[index].isEndTimeColumn = false;
        tmpArgs.end_time_col = "";
    }
    if (tmpSelectedSType[index]?.value !== "ID") {
        tmpSelectedSType[index] = pkStype;
        columnsList[index].stype = pkStype?.value as string;
        columnsList[index].multicategorical_delimiter = null;
        columnsList[index].sTypeLabel = pkStype?.label;
        columnsList[index].userUpdatedType = true;
        const clIndex = tmpArgs?.cols?.findIndex(
            (col) => col?.name === columnsList[index]?.name
        );
        if (clIndex !== -1) {
            tmpArgs.cols[clIndex].stype = pkStype?.value as KumoapiTypingStype;
            tmpArgs.cols[clIndex].multicategorical_delimiter = null;
        }
    }
}

function lacksRequiredMulticategoricalDelimiter(
    option: DropdownOption,
    column: ColumnArgs
): boolean {
    return (
        option.value === "multicategorical" &&
        column.dtype === "string" &&
        !column.multicategorical_delimiter
    );
}

function applyTimestampDtype(
    option: DropdownOption,
    rowIndex: number,
    columns: ColumnArgs[],
    dtypes: DropdownOption[],
    setSelectedDType: React.Dispatch<
        React.SetStateAction<Array<DropdownOption>>
    >
): void {
    if (option.value !== "timestamp" || dtypes[rowIndex].value === "date")
        return;
    const updatedDtypes = [...dtypes];
    updatedDtypes[rowIndex] = { value: "date", label: "Date" };
    columns[rowIndex].dtype = "date";
    setSelectedDType(updatedDtypes);
}

function applySelectedStype(
    option: DropdownOption,
    rowIndex: number,
    columns: ColumnArgs[],
    stypes: DropdownOption[],
    tableDefinition: KumoapiTableTableDefinition
): void {
    stypes[rowIndex] = { value: option.value, label: option.label };
    const column = columns[rowIndex];
    column.stype = option.value as string;
    column.sTypeLabel =
        option.label === "Embedding" ? "Sequence" : option.label;
    column.userUpdatedType = true;
    if (option.value !== "multicategorical") {
        column.multicategorical_delimiter = null;
    }

    const configuredColumn = tableDefinition.cols.find(
        (candidate) => candidate.name === column.name
    );
    if (!configuredColumn) return;
    configuredColumn.stype = option.value as KumoapiTypingStype;
    if (option.value !== "multicategorical") {
        configuredColumn.multicategorical_delimiter = null;
    }
    if (option.value === "timestamp") configuredColumn.dtype = "date";
}

export const useColumnManagement = (
    columns: Array<ColumnArgs>,
    columnDTypes: Array<DropdownOption>,
    columnSTypes: Array<DropdownOption>,
    tableArgs: KumoapiTableTableDefinition,
    pkPrevSType: DropdownOption,
    setColumns: React.Dispatch<React.SetStateAction<Array<ColumnArgs>>>,
    setSelectedDType: React.Dispatch<
        React.SetStateAction<Array<DropdownOption>>
    >,
    setSelectedSType: React.Dispatch<
        React.SetStateAction<Array<DropdownOption>>
    >,
    setTableArgs: React.Dispatch<
        React.SetStateAction<KumoapiTableTableDefinition>
    >,
    setPkPrevSType: React.Dispatch<React.SetStateAction<DropdownOption>>,
    setInvalidError: React.Dispatch<
        React.SetStateAction<{ title: string; description: string }>
    >,
    allCols: Array<string>,
    setSelected: React.Dispatch<React.SetStateAction<Array<string>>>,
    setAllCols: React.Dispatch<React.SetStateAction<Array<string>>>
) => {
    const onSetColumnDataType = useCallback(
        (e: DropdownOption, rowIndex: number) => {
            if (
                e?.value === "string" &&
                columns[rowIndex]?.stype === "multicategorical" &&
                !columns[rowIndex]?.multicategorical_delimiter
            ) {
                setInvalidError({
                    title: "Multicategorical delimiter required",
                    description:
                        "A scalar string multicategorical column requires a delimiter proven from its source sample. Keep the native string or integer list dtype, or choose another semantic type before changing to String."
                });
                return;
            }
            const tmpSelectedDataType = [...columnDTypes];
            tmpSelectedDataType[rowIndex].value = e?.value;
            tmpSelectedDataType[rowIndex].label = e?.label;
            const columnsList = [...columns];
            columnsList[rowIndex].dtype = e?.value as string;
            columnsList[rowIndex].dTypeLabel = e?.label;
            if (e?.value !== "string") {
                columnsList[rowIndex].multicategorical_delimiter = null;
            }
            const tmpArgs = { ...tableArgs };
            const clIndex = tmpArgs?.cols?.findIndex(
                (c) => c?.name === columnsList[rowIndex]?.name
            );
            if (clIndex !== -1) {
                tmpArgs.cols[clIndex].dtype = e?.value as KumoapiTypingDtype;
                if (e?.value !== "string") {
                    tmpArgs.cols[clIndex].multicategorical_delimiter = null;
                }
            }
            setTableArgs(tmpArgs);
            setSelectedDType(tmpSelectedDataType);
            setColumns(columnsList);
            setInvalidError({ title: "", description: "" });
        },
        [
            columnDTypes,
            columns,
            setTableArgs,
            tableArgs,
            setSelectedDType,
            setColumns,
            setInvalidError
        ]
    );

    const onSetColumnSType = useCallback(
        (e: DropdownOption, rowIndex: number) => {
            if (lacksRequiredMulticategoricalDelimiter(e, columns[rowIndex])) {
                setInvalidError({
                    title: "Multicategorical delimiter required",
                    description:
                        "This scalar string column has no supported delimiter inferred from its sample. Use Categorical, or store the values as a native string or integer list."
                });
                return;
            }
            const tmpSelectedSType = [...columnSTypes];
            const columnsList = [...columns];
            const tmpArgs = { ...tableArgs };
            applyTimestampDtype(
                e,
                rowIndex,
                columnsList,
                columnDTypes,
                setSelectedDType
            );
            applySelectedStype(
                e,
                rowIndex,
                columnsList,
                tmpSelectedSType,
                tmpArgs
            );
            setTableArgs(tmpArgs);
            setSelectedSType(tmpSelectedSType);
            setColumns(columnsList);
            setInvalidError({ title: "", description: "" });
        },
        [
            columnDTypes,
            columnSTypes,
            columns,
            setTableArgs,
            tableArgs,
            setSelectedDType,
            setSelectedSType,
            setColumns,
            setInvalidError
        ]
    );

    const onSelectColumns = useCallback(
        (selectedRows: ColumnArgs[]) => {
            const cols: Array<Column> = [];
            let selectedCols: Array<string> = [];
            const allColsCopy: Array<string> = [...allCols];

            const selectedColsSet = new Set(
                selectedRows?.map((obj) => obj?.name)
            );
            selectedCols = columns?.map((obj) =>
                selectedColsSet?.has(obj?.name) ? "true" : "false"
            );

            //Update allColsCopy with the selected/de-selected ones.
            selectedCols.forEach((col, index) => {
                allColsCopy[index] = col && col === "true" ? "true" : "false";
            });
            // allColsCopy now has all the selected/unselected cols. Now generate
            // the cols
            columns?.forEach((col, index) => {
                if (allColsCopy[index] === "true" && !col?.disableSelection) {
                    cols.push(toApiColumn(col));
                }
            });

            const tmpArgs = { ...tableArgs };
            tmpArgs.cols = cols;
            setSelected(selectedCols);
            setAllCols(allColsCopy);
            setTableArgs(tmpArgs);
        },
        [allCols, columns, setTableArgs, tableArgs, setSelected, setAllCols]
    );

    const onSelectPrimaryKey = useCallback(
        (index: number, rowData: ColumnArgs) => {
            const tmpArgs = { ...tableArgs };
            const pkey = tmpArgs?.pkey;
            const tmpSelectedSType = [...columnSTypes];
            const columnsList = [...columns];
            const pkStype: DropdownOption = { value: "ID", label: "ID" };
            let cIndex = -1;

            if (pkey === rowData?.name) {
                tmpArgs.pkey = "";
                cIndex = index;
            } else {
                cIndex = isValid(pkey)
                    ? columns?.findIndex((col) => col?.name === pkey)
                    : -1;
                tmpArgs.pkey = rowData?.name as string;
            }

            if (cIndex > -1) {
                // Unmark the previous primary key
                restorePreviousPkeyType(
                    columnsList,
                    tmpSelectedSType,
                    tmpArgs,
                    cIndex,
                    pkPrevSType
                );
            }

            if (tmpArgs?.pkey) {
                setPkPrevSType(tmpSelectedSType[index]);
                applyNewPkeyColumn(
                    columnsList,
                    tmpSelectedSType,
                    tmpArgs,
                    index,
                    pkStype
                );
            }

            setTableArgs(tmpArgs);
            setColumns(columnsList);
            setSelectedSType(tmpSelectedSType);
            setInvalidError({
                title: "",
                description: ""
            });
        },
        [
            columnSTypes,
            columns,
            pkPrevSType,
            setTableArgs,
            tableArgs,
            setPkPrevSType,
            setColumns,
            setSelectedSType,
            setInvalidError
        ]
    );

    const onCreateTimeColSelect = useCallback(
        (rowIndex: number, cellIndex: number, value: ColumnArgs) => {
            const tmpArgs = { ...tableArgs };
            const columnsList = [...columns];

            // Clear the previously selected time column
            if (isValid(tmpArgs?.time_col)) {
                const prevIndex = columns?.findIndex(
                    (col) => col?.name === tmpArgs?.time_col
                );
                if (prevIndex > -1) {
                    columnsList[prevIndex].isTimeColumn = false;
                    tmpArgs.time_col = "";
                }
            }

            if (tableArgs?.time_col !== value?.name) {
                if (rowIndex !== -1) {
                    columnsList[rowIndex].isTimeColumn = true;
                    tmpArgs.time_col = value?.name;

                    if (columnsList[rowIndex]?.isPrimary) {
                        columnsList[rowIndex].isPrimary = false;
                        tmpArgs.pkey = "";
                    }

                    if (columnsList[rowIndex]?.isEndTimeColumn) {
                        columnsList[rowIndex].isEndTimeColumn = false;
                        tmpArgs.end_time_col = "";
                    }
                }
            }
            setTableArgs(tmpArgs);
            setColumns(columnsList);
            setInvalidError({
                title: "",
                description: ""
            });
        },
        [columns, setTableArgs, tableArgs, setColumns, setInvalidError]
    );

    const onEndTimeColSelect = useCallback(
        (rowIndex: number, cellIndex: number, value: ColumnArgs) => {
            const tmpArgs = { ...tableArgs };
            const columnsList = [...columns];

            // Clear the previously selected end time column
            if (isValid(tmpArgs?.end_time_col)) {
                const prevEndTimeIndex = columns?.findIndex(
                    (col) => col?.name === tmpArgs?.end_time_col
                );
                if (prevEndTimeIndex > -1) {
                    columnsList[prevEndTimeIndex].isEndTimeColumn = false;
                    tmpArgs.end_time_col = "";
                }
            }

            // If the selected column is different from the current end time column
            if (tableArgs?.end_time_col !== value?.name) {
                if (rowIndex !== -1) {
                    // Set the new end time column
                    columnsList[rowIndex].isEndTimeColumn = true;
                    tmpArgs.end_time_col = value?.name;

                    if (columnsList[rowIndex]?.isPrimary) {
                        columnsList[rowIndex].isPrimary = false;
                        tmpArgs.pkey = "";
                    }

                    if (columnsList[rowIndex]?.isTimeColumn) {
                        columnsList[rowIndex].isTimeColumn = false;
                        tmpArgs.time_col = "";
                    }
                }
            }
            setTableArgs(tmpArgs);
            setColumns(columnsList);
            setInvalidError({
                title: "",
                description: ""
            });
        },
        [columns, setTableArgs, tableArgs, setColumns, setInvalidError]
    );

    return {
        onSetColumnDataType,
        onSetColumnSType,
        onSelectColumns,
        onSelectPrimaryKey,
        onCreateTimeColSelect,
        onEndTimeColSelect
    };
};

/**
 * Hook for managing column row selection state with pagination support
 * Handles conversion between selected array and row selection state
 */
export const useColumnRowSelection = (
    selected: Array<string>,
    columns: Array<ColumnArgs>,
    isLoading: boolean,
    onSelectColumns: (selectedRows: Array<ColumnArgs>) => void,
    preservedPageIndex: number,
    defaultPageSize: number
) => {
    const [rowSelectionState, setRowSelectionState] =
        useState<RowSelectionState>({});
    const initialRowSelectionState = useRef(false);
    const initializingRowSelection = useRef(false);

    // Initialize row selection state from selected array
    useEffect(() => {
        if (
            selected &&
            Array.isArray(selected) &&
            selected?.length > 0 &&
            columns?.length > 0 &&
            isLoading === false &&
            initialRowSelectionState.current === false
        ) {
            const newState: RowSelectionState = {};

            for (let i = 0; i < selected.length; i++) {
                const value = selected[i];
                newState[i] = value === "true";
            }

            // The change effect below also runs in this render with the old,
            // empty state. Skip that synthetic callback; the state update will
            // rerun it with the initialized selection.
            initializingRowSelection.current = true;
            setRowSelectionState(newState);
            initialRowSelectionState.current = true;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selected?.length]);

    // Update selected columns when row selection changes
    useEffect(() => {
        if (initializingRowSelection.current) {
            initializingRowSelection.current = false;
            return;
        }
        if (columns?.length > 0 && initialRowSelectionState.current === true) {
            const selected: ColumnArgs[] = [];

            columns?.forEach((table, index) => {
                if (rowSelectionState[index]) {
                    selected.push(table);
                }
            });

            onSelectColumns(selected);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [columns, rowSelectionState]);

    // Calculate paginated data
    const paginatedData = useMemo(() => {
        const startIndex = preservedPageIndex * defaultPageSize;
        const endIndex = startIndex + defaultPageSize;
        return columns.slice(startIndex, endIndex);
    }, [columns, preservedPageIndex, defaultPageSize]);

    // Adjust row selection state for current page
    const adjustedRowSelectionState = useMemo(() => {
        const adjusted: RowSelectionState = {};
        const startIndex = preservedPageIndex * defaultPageSize;

        paginatedData.forEach((_, pageIndex) => {
            const originalIndex = startIndex + pageIndex;
            if (rowSelectionState[originalIndex]) {
                adjusted[pageIndex] = true;
            }
        });

        return adjusted;
    }, [rowSelectionState, preservedPageIndex, defaultPageSize, paginatedData]);

    // Handle row selection changes from the table
    const handleRowSelectionChange: React.Dispatch<
        React.SetStateAction<RowSelectionState>
    > = useCallback(
        (updater) => {
            const startIndex = preservedPageIndex * defaultPageSize;

            const newSelection =
                typeof updater === "function"
                    ? updater(adjustedRowSelectionState)
                    : updater;

            const updatedGlobalSelection = { ...rowSelectionState };

            for (let i = 0; i < paginatedData.length; i++) {
                const originalIndex = startIndex + i;
                delete updatedGlobalSelection[originalIndex];
            }

            Object.entries(newSelection).forEach(([pageIndex, isSelected]) => {
                if (isSelected) {
                    const originalIndex = startIndex + parseInt(pageIndex);
                    updatedGlobalSelection[originalIndex] = true;
                }
            });

            setRowSelectionState(updatedGlobalSelection);
        },
        [
            preservedPageIndex,
            defaultPageSize,
            adjustedRowSelectionState,
            rowSelectionState,
            paginatedData.length
        ]
    );

    return {
        rowSelectionState,
        paginatedData,
        adjustedRowSelectionState,
        handleRowSelectionChange
    };
};

/**
 * Hook for generating column type options (data types and semantic types)
 */
export const useColumnTypeOptions = (columnDTypes: Array<DropdownOption>) => {
    const dTypeOptions: Array<DropdownOption> = useMemo(
        () =>
            tableDataTypes?.map((dType) => {
                return {
                    value: dType,
                    label: capitalizeFirstLetter(dType)
                };
            }),
        []
    );

    const sTypeOptions: Array<DropdownOption>[] = useMemo(
        () =>
            columnDTypes?.map((dType: DropdownOption) => {
                return tableSemanticTypes?.map((sType: string) => {
                    let disabled = true;
                    if (
                        tableDataTypesToSTypesMapping[
                            dType?.value as string
                        ]?.includes(sType)
                    ) {
                        disabled = false;
                    }

                    return {
                        value: sType,
                        label:
                            sType === "sequence"
                                ? "Embedding"
                                : capitalizeFirstLetter(sType),
                        subLabel: dataTypeDescription[sType] ?? "",
                        isDisabled: disabled
                    };
                });
            }),
        [columnDTypes]
    );

    return {
        dTypeOptions,
        sTypeOptions
    };
};

/**
 * Hook for synchronizing column metadata with table arguments
 * Updates isTimeColumn and isPrimary flags based on table args
 */
export const useColumnMetadataSync = (
    columns: Array<ColumnArgs>,
    columnSTypes: Array<DropdownOption>,
    tableArgs: KumoapiTableTableDefinition,
    setColumns: React.Dispatch<React.SetStateAction<Array<ColumnArgs>>>
) => {
    useEffect(() => {
        // Set `isTimeColumn` and `isPrimary` to true. This is when parent
        // component set `time_col`\`pkey` but not updated in column list
        if (
            columns?.length > 0 &&
            (isValid(tableArgs?.time_col) || isValid(tableArgs?.pkey))
        ) {
            const timeColIndex = columns.findIndex(
                (c) => c.name === tableArgs.time_col
            );
            const pKeyIndex = columns.findIndex(
                (c) => c.name === tableArgs.pkey
            );
            const cols = [...columns];
            let updateCols = false;

            if (timeColIndex !== -1) {
                if (!columns[timeColIndex].isTimeColumn) {
                    cols[timeColIndex].isTimeColumn = true;
                    updateCols = true;
                }
            }

            if (pKeyIndex !== -1) {
                if (!columns[pKeyIndex].isPrimary) {
                    cols[pKeyIndex].isPrimary = true;

                    const pkStype: DropdownOption = {
                        value: "ID",
                        label: "ID"
                    };
                    const tmpSelectedSType = [...columnSTypes];
                    tmpSelectedSType[pKeyIndex] = pkStype;
                    cols[pKeyIndex].stype = pkStype?.value as string;
                    cols[pKeyIndex].sTypeLabel = pkStype?.label;
                    cols[pKeyIndex].userUpdatedType = true;
                    updateCols = true;
                }
            }

            if (updateCols) {
                setColumns(cols);
            }
        }
    }, [columnSTypes, columns, tableArgs.pkey, tableArgs.time_col, setColumns]);
};
