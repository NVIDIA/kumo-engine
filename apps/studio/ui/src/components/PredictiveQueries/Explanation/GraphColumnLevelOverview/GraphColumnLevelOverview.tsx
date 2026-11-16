// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useMemo, useState } from "react";
import Spacer from "react-spacer";

import {
    Alert,
    AlertDescription,
    Button,
    Card,
    CardContent,
    CardHeader,
    CloseSmall,
    cn,
    DivergingBarChart,
    GridCol,
    GridRow,
    SearchBar,
    Select,
    Skeleton,
    Symbol,
    Tooltip,
    TooltipContent,
    TooltipTrigger,
    TruncatedTooltip
} from "@kumo-ai/kumo-ui-catalog";
import { DimensionSpec } from "gen/v0/types";
import { JSONPath } from "jsonpath-plus";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { useXaiDataTransform } from "hooks/jobs/training/eval/useXaiDataTransform";
import { useXai } from "hooks/jobs/training/eval/xai";

import { GraphNodeData, GraphXaiScores, VisualSpecAxis } from "types/explorer";
import { NA_VALUES } from "types/graphs";
import {
    XaiCartData,
    XaiCategoryDropDowns,
    XaiChartPlots,
    XaiJsonData,
    XaiTableData
} from "types/stats";

import { SUPPORT_CONTACT_MESSAGE } from "utils/Constants";
import { parseNumbers } from "utils/Helpers/numberHelpers";
import { getKeys } from "utils/Helpers/objectHelpers";
import { isValid } from "utils/Helpers/typeGuards";
import { isValidISODate, monthsDifference } from "utils/timeConverter";

//Column type string, Specific to this component only
const COL_TYPE = "COLUMN";
const COUNT_COL = "COUNT(*)";
const DEFAULT_PAGE_SIZE = 12;
/**
 * Data Key Fields of Column Analysis stats
 */
const KeyFields = {
    column: "Column Value",
    percentage: "Population Ratio",
    data: "Target Average Score",
    ratio: "Ratio Interval",
    count: "Count Interval",
    dataPred: "Predicted Average Score"
};

/**
 * Column Explanation and Column Analysis Component.
 *
 * This component will invoke the Column Explanation and Column Analysis Xai API's
 * in `useEffect`, restructurize the response, sets it to state variable
 * and renders it in UI
 *
 */
export interface GraphColumnLevelOverviewProps {
    queryId?: string;
    show: boolean;
    trainingJobId?: string;
    setXaiScore: (data: GraphXaiScores) => void;
    subGraphInfo?: GraphNodeData | null;
    predictionAnchorTime?: string;
    bpjobId?: string;
}

const GraphColumnLevelOverview = ({
    queryId,
    show,
    trainingJobId,
    setXaiScore,
    subGraphInfo,
    predictionAnchorTime,
    bpjobId
}: GraphColumnLevelOverviewProps) => {
    const [localFilteredTableData, setLocalFilteredTableData] = useState<
        XaiTableData[]
    >([]);
    const [dropDowns, setDropDowns] = useState<XaiCategoryDropDowns[]>([]);
    const [tableName, setTableName] = useState<string | null>(null);
    const [columnName, setColumnName] = useState<string | null>(null);
    const [columnStype, setColumnStype] = useState<string | null>(null);
    const [pageIndex, setPageIndex] = useState(1);

    const [barChartPlots, setBarChartPlots] = useState<XaiChartPlots | null>(
        null
    );
    const [chartErrorMsg, setChartErrorMsg] = useState<string | null>(null);

    const [selected, setSelected] = useState<XaiTableData | null>(null);
    const [search, setSearch] = useState<string>("");

    /**
     *  This method invokes REST API call and fetches the Column Analysis stats
     *  and sets it to state variable.
     */
    const {
        data: xaiResponse,
        isLoading: dataLoading,
        isError,
        error
    } = useXai({
        trainingJobId,
        queryId,
        bpjobId
    });

    // Use the extracted hook for XAI data transformation
    const {
        xaiData,
        tableData,
        filteredTableData: hookFilteredTableData,
        initialSelected,
        loading,
        getUpdatedTableData,
        sortByVarianceScore
    } = useXaiDataTransform({
        xaiResponse: xaiResponse?.data,
        isLoading: dataLoading,
        isError,
        error
    });

    // Sync local filtered data with hook data, allowing local updates
    useEffect(() => {
        if (
            hookFilteredTableData.length > 0 &&
            localFilteredTableData.length === 0
        ) {
            setLocalFilteredTableData(hookFilteredTableData);
        }
    }, [hookFilteredTableData, localFilteredTableData.length]);

    // Set initial selected when hook provides it
    useEffect(() => {
        if (initialSelected && !selected) {
            setSelected(initialSelected);
        }
    }, [initialSelected, selected]);

    // Use local state if available, otherwise use hook data
    const filteredTableData =
        localFilteredTableData.length > 0
            ? localFilteredTableData
            : hookFilteredTableData;

    /**
     * Getting array of values for either labels or plots
     * based on `array_spec`'s `col_keys` to generate chart data
     */
    const getArray = useCallback(
        (
            table_data: { [key: string]: XaiJsonData },
            dim_values: string[],
            array_spec: VisualSpecAxis
        ) => {
            const col_arrays = array_spec?.col_keys?.map((col_key: string) => {
                const col_key_vals = col_key?.split(".");
                let i = 0;
                const key = dim_values
                    ?.map((v: string) => (v === null ? col_key_vals[i++] : v))
                    ?.join(".");
                return table_data[key] as unknown as string | number[];
            });
            const num_rows = col_arrays[0]?.length;
            const result = [];
            for (let row_index = 0; row_index < num_rows; ++row_index) {
                const cols = Array.from(
                    { length: col_arrays.length },
                    (_, i) => i
                ).map((i) => col_arrays[i][row_index]);
                result.push(array_spec.get_value_fn(cols));
            }
            return result;
        },
        []
    );

    // Rules to decide how to plot multidimensional data cube.
    const createVisualizationSpec = useCallback(
        (columnStype: string, dimensions: DimensionSpec[]) => {
            const dim_selections: Array<string | null> = Array(
                dimensions?.length
            ).fill("$DROP_DOWN$");
            dim_selections[dimensions?.length - 1] = null;
            return {
                chart_type:
                    columnStype?.toLowerCase() === "timestamp"
                        ? "histogram"
                        : "bar",
                dim_selections: dim_selections,
                label_axis: {
                    col_keys: [`${KeyFields.column}`],
                    get_value_fn: (cols: Array<number | string>) => cols[0]
                },
                plot_axis_truth: {
                    col_keys: [`${KeyFields.data}`],
                    get_value_fn: (cols: Array<number | string>) =>
                        parseNumbers(cols[0], { decimals: 3 })
                            ?.toString()
                            ?.replaceAll(",", "")
                },
                plot_axis_pred: {
                    col_keys: [`${KeyFields.dataPred}`],
                    get_value_fn: (cols: Array<number | string>) =>
                        parseNumbers(cols[0], { decimals: 3 })
                            ?.toString()
                            ?.replaceAll(",", "")
                },
                plot_population_ratio: {
                    col_keys: [`${KeyFields.percentage}`],
                    get_value_fn: (cols: Array<number | string>) =>
                        ((cols[0] as number) * 100).toFixed(3)
                }
            };
        },
        []
    );

    /**
     * Creating data for Bar-Chart
     * from the stats object
     * @param {String|Array} columnName
     * @param {Object} chartData
     * @returns
     */
    const getBarChartData = useCallback(
        (columnName: string, chartData: XaiCartData) => {
            // Creating data for Bar Chart
            const plots: XaiChartPlots = {
                yAxisTitle: [COUNT_COL, "COUNT INTERVAL"].includes(
                    columnName?.toLocaleUpperCase()
                )
                    ? "Count of Facts"
                    : columnName,
                allDataTruth: chartData?.plots_truth,
                allDataPrediction: chartData?.plots_pred,
                allLabels: chartData?.labels,
                allPopulationRatio: chartData?.population_ratio,
                activeDataPrediction: chartData?.plots_pred?.slice(
                    0,
                    DEFAULT_PAGE_SIZE
                ),
                activeDataTruth: chartData?.plots_truth?.slice(
                    0,
                    DEFAULT_PAGE_SIZE
                ),
                activeLabels: chartData?.labels.slice(0, DEFAULT_PAGE_SIZE),
                activePopulationRatio: chartData?.population_ratio?.slice(
                    0,
                    DEFAULT_PAGE_SIZE
                ),
                globalAvgScore: chartData?.global_avg_score
            };
            return plots;
        },
        []
    );

    const prepareVisualizationData = useCallback(
        (
            tableName: string,
            columnName: string,
            columnStype: string,
            dimIndexUpdate: number,
            selectedOption: {
                value: string;
                label: string;
            } | null
        ) => {
            const dropdowns: Array<XaiCategoryDropDowns | null> = [
                ...dropDowns
            ];

            const cube = xaiData?.tables?.[tableName][columnName]?.cohort;
            const visual_spec = createVisualizationSpec(
                columnStype,
                cube?.dimensions as DimensionSpec[]
            );

            // Initialize dropdowns if needed.
            if (dropDowns?.length === 0) {
                cube?.dimensions?.forEach(
                    (dimension: DimensionSpec, index: number) => {
                        const dim_selection =
                            visual_spec?.dim_selections[index];
                        if (dim_selection === "$DROP_DOWN$") {
                            dropdowns.push({
                                name: dimension?.name,
                                options: [], // populate later
                                selected: null // populate later
                            });
                        } else {
                            dropdowns.push(null);
                        }
                    }
                );
            }

            // Start from the root json data cube and go through all dimensions.
            let json_data = cube?.json_data;
            let json_path = "$";
            const dim_values: Array<string | null> = [];
            const length = cube?.dimensions?.length ?? 0;
            for (let i = 0; i < length; ++i) {
                const dim_selection = visual_spec?.dim_selections[i];
                const dropdown = dropdowns[i] ?? null;
                if (dropdown && dim_selection === "$DROP_DOWN$") {
                    // Current dropdown selection to update
                    if (i === dimIndexUpdate) {
                        if (
                            selectedOption !== null &&
                            selectedOption !== undefined
                        ) {
                            dropdown.selected = selectedOption.value;
                        } else if (dropdown.selected === null) {
                            dropdown.options = getKeys(json_data);
                            dropdown.selected = dropdown.options[0];
                        }
                    } else if (i > dimIndexUpdate) {
                        dropdown.options = getKeys(json_data);
                        dropdown.selected = dropdown.options[0];
                    }
                    if (json_data && dropdown && dropdown?.selected)
                        json_data = json_data[dropdown.selected] as XaiJsonData;
                    json_path += `['${dropdown.selected}']`;
                    dim_values.push(dropdown.selected);
                } else if (dim_selection) {
                    // literal value
                    if (json_data)
                        json_data = json_data[dim_selection] as XaiJsonData;
                    json_path += `['${dim_selection}']`;
                    dim_values.push(dim_selection);
                } else {
                    // table column dimension, just use the first value
                    const column_keys = getKeys(json_data);
                    if (json_data)
                        json_data = json_data[column_keys[0]] as XaiJsonData;
                    json_path += "[*]";
                    dim_values.push(null);
                }
            }

            const table_data: { [key: string]: XaiJsonData } = {};
            (
                JSONPath({
                    path: json_path,
                    json: cube?.json_data ?? null,
                    resultType: "all"
                }) as Array<{ path: string; value: XaiJsonData }>
            ).forEach((node) => {
                const pathArray = JSONPath.toPathArray(node.path);
                table_data[pathArray.slice(1).join(".")] = node?.value;
            });

            return {
                table_data: table_data,
                dim_values: dim_values,
                visual_spec: visual_spec,
                dropdowns
            };
        },
        [createVisualizationSpec, dropDowns, xaiData?.tables]
    );

    const getGlobalAvg = (
        plotsTruth: (string | number)[],
        populationRatio: (string | number)[]
    ) => {
        // Compute population average score on the fly
        let global_avg_score = 0.0;
        let weights = 0.0;
        for (let i = 0; i < plotsTruth?.length; ++i) {
            const weight =
                populationRatio && populationRatio[i]
                    ? Number(populationRatio[i])
                    : 0;
            global_avg_score += plotsTruth[i]
                ? Number(plotsTruth[i]) * weight
                : 0;
            weights += weight;
        }
        if (global_avg_score !== 0 && weights !== 0)
            global_avg_score = global_avg_score / weights;
        return global_avg_score;
    };

    const plotCohortAnalysis = useCallback(
        (
            tableName: string,
            columnName: string,
            columnStype: string,
            dimIndexUpdate: number,
            selectedOption: {
                value: string;
                label: string;
            } | null
        ) => {
            const { table_data, dim_values, visual_spec, dropdowns } =
                prepareVisualizationData(
                    tableName,
                    columnName,
                    columnStype,
                    dimIndexUpdate,
                    selectedOption as {
                        value: string;
                        label: string;
                    }
                );
            const plots_truth = getArray(
                table_data,
                dim_values as string[],
                visual_spec?.plot_axis_truth as VisualSpecAxis
            );
            const plots_pred = getArray(
                table_data,
                dim_values as string[],
                visual_spec?.plot_axis_pred as VisualSpecAxis
            );
            const population_ratio = getArray(
                table_data,
                dim_values as string[],
                visual_spec?.plot_population_ratio
            );

            const globalAvgScore = getGlobalAvg(plots_truth, population_ratio);
            const labels = getArray(
                table_data,
                dim_values as string[],
                visual_spec?.label_axis
            );
            const chart_data: XaiCartData = {
                labels: labels,
                plots_truth: plots_truth,
                plots_pred: plots_pred,
                population_ratio: population_ratio,
                global_avg_score: globalAvgScore
            };

            const plots = getBarChartData(columnName, chart_data);
            setTableName(tableName);
            setColumnName(columnName);
            setColumnStype(columnStype);
            setBarChartPlots(plots);
            setDropDowns(dropdowns as XaiCategoryDropDowns[]);
        },
        [prepareVisualizationData, getArray, getBarChartData]
    );

    const onDimensionSelection = useCallback(
        (
            j: number,
            e: {
                value: string;
                label: string;
            }
        ) => {
            //Setting null to refresh the chart
            setBarChartPlots(null);
            setTimeout(() => {
                plotCohortAnalysis(
                    tableName as string,
                    columnName as string,
                    columnStype as string,
                    j,
                    e
                );
            }, 1);
        },
        [tableName, columnName, columnStype, plotCohortAnalysis]
    );

    /**
     * Generate Bar-chart / Drop-down options data from cohort stats.
     *
     * Checking data from `cohortStats` state variable by using table and column
     * name. If not exits then showing error message. If exists then
     * creating data for Bar-chart, drop-downs and assigning it to respective
     * state variable.
     *
     */
    const generateChartData = useCallback(
        (value: XaiTableData) => {
            //fq_name contains full name `connector_name.table_name.column_name`
            // while table name should be `connector_name.table_name`.
            let tableName = value?.fq_name;
            const columnName = value?.name;
            const columnStype = value?.type;
            const tableNameParts = tableName?.split(".");
            if (tableNameParts.at(-1) === columnName) {
                tableName = tableNameParts?.slice(0, -1).join(".");
            }

            //let tmp = this.state.cohortStats;
            // Checking if the table name and column name are exists in
            // Column Analysis stats, If not, return proper warning
            let warnMsg = "";
            if (!xaiData?.tables || !xaiData?.tables[tableName]) {
                warnMsg = `Column Analysis for ${tableName} is not supported`;
            } else if (!xaiData?.tables[tableName][columnName]?.cohort) {
                warnMsg = `Column Analysis for ${tableName}.${value.name} is not supported`;
            }
            if (isValid(warnMsg)) {
                setBarChartPlots(null);
                setChartErrorMsg(warnMsg);
                return;
            }

            plotCohortAnalysis(tableName, columnName, columnStype, 0, null);
        },
        [plotCohortAnalysis, xaiData?.tables]
    );
    // const [evalue, setEvalue] = useState<string | number>(); Maybe needed in future

    const addEnityToGraph = useCallback(
        (
            subgraph: GraphNodeData | null | undefined,
            graph?: XaiTableData,
            anchorTime?: string
        ) => {
            try {
                const extractedValues = subgraph?.cols?.map(
                    ({ name, value }) => ({
                        name,
                        value
                    })
                );
                const tableRowNamePart = graph?.table_row_names?.split(".");
                let tableNamefromEntity: string | undefined;
                if (subgraph?.tableName?.includes(".")) {
                    tableNamefromEntity = subgraph?.tableName
                        ?.split(".")?.[1]
                        ?.trim();
                } else {
                    tableNamefromEntity = subgraph?.tableName;
                }

                const tableNamefromGraph = tableRowNamePart?.[0]?.trim();

                if (tableNamefromEntity === tableNamefromGraph) {
                    const found = extractedValues?.some((item) => {
                        if (item.name === tableRowNamePart?.[1]?.trim()) {
                            let value = item?.value;

                            if (isValidISODate(value as string)) {
                                value = monthsDifference(
                                    anchorTime,
                                    value as string
                                );
                            }

                            //setEvalue(value);
                            return true;
                        } else {
                            //setEvalue(undefined);
                            return false;
                        }
                    });

                    if (!found) {
                        // setEvalue(undefined);
                    }
                } else {
                    // setEvalue(undefined);
                }
            } catch (error) {
                console.error(error);
                // setEvalue(undefined);
            }
        },
        []
    );

    /**
     * Table cell click handler.
     *
     * By clicking on the column link, this will show the BarChart/Cohort Table,
     * otherwise a click on table link updates `tableData`.
     *
     * @param rowIndex
     * @param cellIndex
     * @param value
     */
    const handleColumnClick = useCallback(
        (value?: XaiTableData) => {
            setSelected(value || null);
            setPageIndex(0);

            setChartErrorMsg(null);
            setBarChartPlots(null);
            if (value?.kind !== COL_TYPE) {
                const tData = getUpdatedTableData(value?.fq_name as string);
                const sorted = sortByVarianceScore(tData);
                setLocalFilteredTableData(sorted);
                setTableName(value?.fq_name as string);
                setColumnName("");
            } else {
                try {
                    setTimeout(() => {
                        addEnityToGraph(
                            subGraphInfo,
                            value,
                            predictionAnchorTime
                        );

                        generateChartData(value);
                    }, 10);
                } catch {
                    setChartErrorMsg(
                        `Unable to find statistics for this column. ${SUPPORT_CONTACT_MESSAGE} to resolve this issue.`
                    );
                }
            }
        },
        [
            generateChartData,
            getUpdatedTableData,
            sortByVarianceScore,
            addEnityToGraph,
            subGraphInfo,
            predictionAnchorTime
        ]
    );

    useEffect(() => {
        const xaiScores: GraphXaiScores = {};
        if (tableData?.length > 0 && xaiData?.tables) {
            tableData.forEach((element) => {
                if (element?.kind === COL_TYPE) {
                    let tableName = element?.fq_name;
                    const columnName = element?.name;
                    const columnStype = element?.type;
                    const tableNameParts = tableName?.split(".");
                    if (tableNameParts?.at(-1) === columnName) {
                        tableName = tableNameParts?.slice(0, -1)?.join(".");
                    }
                    const { table_data, dim_values, visual_spec } =
                        prepareVisualizationData(
                            tableName,
                            columnName,
                            columnStype,
                            0,
                            null
                        );

                    const plots_truth = getArray(
                        table_data,
                        dim_values as string[],
                        visual_spec?.plot_axis_truth as VisualSpecAxis
                    );
                    const population_ratio = getArray(
                        table_data,
                        dim_values as string[],
                        visual_spec?.plot_population_ratio
                    );
                    const plots_pred = getArray(
                        table_data,
                        dim_values as string[],
                        visual_spec?.plot_axis_pred as VisualSpecAxis
                    );
                    const labels = getArray(
                        table_data,
                        dim_values as string[],
                        visual_spec.label_axis
                    );
                    const globalAvgScore = getGlobalAvg(
                        plots_truth,
                        population_ratio
                    );
                    const key = element?.fq_name?.replaceAll(" ", "_");
                    if (key) {
                        xaiScores[key] = {
                            globalAvgScore: globalAvgScore,
                            labels: labels,
                            predictions: plots_pred
                        };
                    }
                }
            });
        }
        setXaiScore(xaiScores);
    }, [
        getArray,
        setXaiScore,
        prepareVisualizationData,
        tableData,
        xaiData?.tables
    ]);

    /**
     * Method to handle pagination of Bar chart
     * @param gotoIndex page index. -1: Next page, -2: Previous page, other: goto page
     */
    const onGotoPage = useCallback(
        (gotoIndex: number) => {
            let finalIndex = pageIndex;
            switch (gotoIndex) {
                case -1: //Next Page
                    finalIndex += 1;
                    break;
                case -2: //Previous Page
                    finalIndex -= 1;
                    break;
                default: //Goto given page number
                    finalIndex = gotoIndex;
                    break;
            }
            const start = finalIndex * DEFAULT_PAGE_SIZE;
            const end = start + DEFAULT_PAGE_SIZE;
            // Update the activeData field
            const plots = { ...barChartPlots };
            plots.activeDataTruth = barChartPlots?.allDataTruth?.slice(
                start,
                end
            );
            plots.activeDataPrediction =
                barChartPlots?.allDataPrediction?.slice(start, end);
            plots.activeLabels = barChartPlots?.allLabels?.slice(start, end);
            plots.activePopulationRatio =
                barChartPlots?.allPopulationRatio?.slice(start, end);
            setPageIndex(finalIndex);

            setBarChartPlots(null); //Setting null to refresh the chart

            setTimeout(() => {
                setBarChartPlots(
                    plots as React.SetStateAction<XaiChartPlots | null>
                );
            }, 1);
        },
        [barChartPlots, pageIndex]
    );

    const handleCloseChart = useCallback(() => {
        setChartErrorMsg(null);
        setBarChartPlots(null);
        setPageIndex(0);
    }, []);

    useEffect(() => {
        try {
            if (
                filteredTableData?.length > 0 &&
                barChartPlots === null &&
                chartErrorMsg === null
            ) {
                addEnityToGraph(
                    subGraphInfo,
                    filteredTableData[0],
                    predictionAnchorTime
                );
                generateChartData(filteredTableData[0]);
            }
        } catch {
            setChartErrorMsg(
                `Unable to find statistics for this column. ${SUPPORT_CONTACT_MESSAGE} to resolve this issue.`
            );
        }
    }, [
        addEnityToGraph,
        barChartPlots,
        chartErrorMsg,
        filteredTableData,
        generateChartData,
        predictionAnchorTime,
        subGraphInfo
    ]);

    useEffect(() => {
        if (subGraphInfo) {
            handleCloseChart();
        }
    }, [subGraphInfo, handleCloseChart]);

    const dropdown_menus = useMemo(
        () =>
            dropDowns
                ?.map((dropdown, index) => {
                    if (!dropdown) return null;

                    const menuOptions = dropdown.options.map((value) => ({
                        label: value,
                        value
                    }));

                    const selectedOption = menuOptions.find(
                        (opt) => opt.value === dropdown.selected
                    );

                    return (
                        <GridRow className="flex items-center justify-between me-2 gap-2">
                            <GridCol>{`${dropdown.name}:`}</GridCol>
                            <GridCol className="min-w-37.5">
                                <Select
                                    id={`select-${index}`}
                                    placeholder={`Select ${dropdown.name}`}
                                    variant="default"
                                    size="sm"
                                    popoverClassName="w-50"
                                    search={false}
                                    options={menuOptions}
                                    selectedValues={
                                        selectedOption
                                            ? [selectedOption.value]
                                            : []
                                    }
                                    onValuesChange={(values: string[]) => {
                                        const option = menuOptions.find(
                                            (opt) => opt.value === values[0]
                                        );
                                        if (option) {
                                            onDimensionSelection(index, option);
                                        }
                                    }}
                                    mode="single"
                                />
                            </GridCol>
                        </GridRow>
                    );
                })
                .filter(Boolean),
        [dropDowns, onDimensionSelection]
    );

    const columns = useMemo(() => {
        return filteredTableData.filter((a) =>
            a.fq_name?.toLocaleLowerCase()?.includes(search.toLocaleLowerCase())
        );
    }, [filteredTableData, search]);

    if (!show) {
        return null;
    }

    if (loading) {
        return <Skeleton className="w-full h-40" />;
    }

    // If explainability is not available or enabled, hide it completely
    // TODO: Find a way to check if explanations are expected
    if (!tableData || tableData?.length === 0) {
        return null;
    }

    const contributionCell = (contribution: string | number) => {
        if (contribution === null || contribution === undefined) {
            return <Symbol type="emdash" />;
        }
        return (
            <>
                {!contribution ||
                NA_VALUES.includes(contribution?.toString()?.toUpperCase()) ? (
                    contribution
                ) : (
                    <>{(Number(contribution) * 100)?.toFixed(2) + "% "}</>
                )}
            </>
        );
    };

    return (
        <>
            <GridRow xs={12} sm={12} md={12}>
                <GridCol xs={3} sm={3} md={3} lg={3} className="ps-0 pe-3">
                    <SearchBar
                        data-testid="xai-columns-search"
                        type="search"
                        placeholder="Search columns"
                        value={search}
                        onChange={(e) => {
                            setSearch(e?.target?.value || "");
                        }}
                    />
                    <Spacer height={8} />
                    {search !== "" && columns?.length < 1 && (
                        <p
                            data-testid={`no-columns`}
                            className="text-title-secondary text-s my-2 text-center w-75">
                            No columns
                        </p>
                    )}
                    <div className="max-h-150 overflow-auto">
                        {columns.map((value, i) => {
                            const isActive =
                                value.fq_name === selected?.fq_name;
                            return (
                                <div
                                    className={cn(
                                        // KUI Vertical Nav aligned selected item (matches catalog
                                        // ListView): 2px brand-green pill rail inset via ::before,
                                        // neutral pressed bg, square 2px corners, bold selected text.
                                        "relative flex overflow-hidden rounded-xs p-3 hover:cursor-pointer",
                                        isActive
                                            ? "text-title-primary before:bg-brand-green bg-kui-bg-pressed before:absolute before:bottom-1 before:left-1 before:top-1 before:w-0.5 before:rounded-full before:content-['']"
                                            : "text-title-secondary hover:bg-kumo-grey-50"
                                    )}
                                    aria-selected={isActive}
                                    key={value.fq_name}
                                    data-testid={`column-${value.fq_name}`}
                                    onClick={() => handleColumnClick(value)}>
                                    <TruncatedTooltip
                                        tooltipContent={value.fq_name}
                                        testId={`entity-name-${i}`}
                                        className={cn(
                                            "self-stretch truncate text-base",
                                            isActive
                                                ? "font-bold"
                                                : "font-normal"
                                        )}>
                                        {value.fq_name}
                                    </TruncatedTooltip>
                                    <span
                                        className={cn(
                                            "ml-1.25 text-title-secondary",
                                            isActive
                                                ? "font-bold text-title-primary"
                                                : "font-normal"
                                        )}>
                                        {contributionCell(value.variance_score)}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </GridCol>
                {barChartPlots !== null && (
                    <GridCol xs={9} sm={9} md={9} lg={9} className="ps-0 ms-0">
                        <Card className="p-0 gap-0 mb-6">
                            <CardContent className="px-0">
                                <GridRow
                                    xs={12}
                                    sm={12}
                                    md={12}
                                    lg={12}
                                    className="m-1 mt-3 border-b pb-3">
                                    <GridCol
                                        xs={6}
                                        sm={6}
                                        md={6}
                                        lg={6}
                                        className="flex items-center font-inter ps-2 font-medium text-xl leading-7">
                                        {selected?.table_row_names &&
                                        selected.table_row_names.length > 40 ? (
                                            <Tooltip
                                                id={"table-row-names-tooltip"}>
                                                <TooltipTrigger>
                                                    <span>
                                                        {selected.table_row_names.substring(
                                                            0,
                                                            40
                                                        )}
                                                        ...
                                                    </span>
                                                </TooltipTrigger>
                                                <TooltipContent
                                                    arrow={false}
                                                    side={"bottom"}
                                                    sideOffset={5}>
                                                    {selected.table_row_names}
                                                </TooltipContent>
                                            </Tooltip>
                                        ) : (
                                            selected?.table_row_names
                                        )}
                                    </GridCol>
                                    <GridCol>
                                        <span className="text-title-secondary">
                                            Hops
                                        </span>
                                        <br />
                                        {
                                            selected?.fq_name?.match(
                                                /\((\d+\s*hops?)\)/
                                            )?.[1]
                                        }
                                    </GridCol>
                                    <GridCol xs={2} sm={2} md={2} lg={2}>
                                        <span className="text-title-secondary">
                                            Type
                                        </span>
                                        <br />
                                        {selected?.type}
                                    </GridCol>
                                    <GridCol xs={3} sm={3} md={3} lg={3}>
                                        <span className="text-title-secondary">
                                            Variation of prediction
                                        </span>
                                        <br />
                                        {selected?.variance_score ? (
                                            <>
                                                {contributionCell(
                                                    selected.variance_score
                                                )}
                                            </>
                                        ) : (
                                            <Symbol type="emdash" />
                                        )}
                                    </GridCol>
                                </GridRow>

                                <DivergingBarChart
                                    id="bar-chart"
                                    xAxisLabels={barChartPlots.activeLabels}
                                    dataSetTruth={barChartPlots.activeDataTruth}
                                    dataSetPred={
                                        barChartPlots.activeDataPrediction
                                    }
                                    populationRatio={
                                        barChartPlots.activePopulationRatio
                                    }
                                    globalAvgScore={Number(
                                        barChartPlots.globalAvgScore
                                    )}
                                    totalRecords={
                                        barChartPlots.allDataTruth.length
                                    }
                                    pageIndex={pageIndex}
                                    pageSize={DEFAULT_PAGE_SIZE}
                                    onGotoPage={onGotoPage}
                                    dropdown_menus={dropdown_menus}
                                />
                            </CardContent>
                        </Card>
                    </GridCol>
                )}
            </GridRow>
            {chartErrorMsg !== null && (
                <Card className="p-0 gap-0 min-w-150">
                    <CardHeader
                        className="px-6 py-3 flex-row items-center justify-between"
                        bottomSeparator={true}>
                        <h3 className="mb-0">{columnName}</h3>
                        <div className="flex items-center gap-4">
                            {dropdown_menus}
                            <Button
                                id={"close"}
                                aria-label="Close"
                                onClick={handleCloseChart}
                                className="btn-close whitespace-nowrap"
                                variant="light-outline">
                                <CloseSmall className="!size-5" />
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent className="p-6">
                        <Alert variant="default" enableClose={false}>
                            <AlertDescription>{chartErrorMsg}</AlertDescription>
                        </Alert>
                    </CardContent>
                </Card>
            )}
        </>
    );
};

export default WithErrorBoundary(GraphColumnLevelOverview);
