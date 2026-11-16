// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from "react";

import {
    Add,
    ChevronDown,
    Close,
    cn,
    DataTable,
    Description,
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    EmptyTable,
    GridCol,
    GridRow,
    Label,
    SearchBar,
    Skeleton,
    Title
} from "@kumo-ai/kumo-ui-catalog";
import { ColumnData, HistoricalItems, NodeOutputData } from "gen/v0/types";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { LinkPredictionEntityOutput, NodeData } from "types/explorer";

const in_ground_truth_or_target_col = "in_ground_truth_or_target";

const TABLE_SUBTITLES = {
    historicalItems:
        "Items this entity interacted with before the prediction window",
    groundTruth:
        "Items this entity actually interacted with during the test window",
    predictions: "Items the model predicted this entity would interact with"
};

const LinkPrediction: React.FC<{
    entityOutput: LinkPredictionEntityOutput;
    historicalItems: HistoricalItems;
    isDataLoading: boolean;
    showGroundTruth?: boolean;
    selectedColumns?: Array<string>;
    setSelectedColumns?: (columns: Array<string>) => void;
}> = ({
    entityOutput,
    historicalItems,
    isDataLoading,
    showGroundTruth,
    selectedColumns: externalSelectedColumns,
    setSelectedColumns: externalSetSelectedColumns
}) => {
    const [transformedEntityOutput, setTransformedEntityOutput] =
        useState<LinkPredictionEntityOutput | null>(null);
    const [columns, setColumns] = useState<Array<string>>([]);
    const [localSelectedColumns, setLocalSelectedColumns] = useState<
        Array<string>
    >([]);
    const [historicalSearch, setHistoricalSearch] = useState("");
    const [groundTruthSearch, setGroundTruthSearch] = useState("");
    const [predictionsSearch, setPredictionsSearch] = useState("");

    const selectedColumns = externalSelectedColumns ?? localSelectedColumns;
    const setSelectedColumns =
        externalSetSelectedColumns ?? setLocalSelectedColumns;

    const handleToggleColumn = (col: string): void => {
        setSelectedColumns(
            selectedColumns.includes(col)
                ? selectedColumns.filter((c) => c !== col)
                : [...selectedColumns, col]
        );
    };

    useEffect(() => {
        if (entityOutput && entityOutput?.prediction?.length > 0) {
            const cols = entityOutput?.prediction[0]?.cols?.map(
                (col: ColumnData) => col?.name
            );
            const id_col_name = entityOutput?.prediction[0]?.id_name;
            if (id_col_name) {
                cols?.unshift(id_col_name);
            }
            setColumns(cols as string[]);
            if (selectedColumns?.length === 0 && cols?.length > 0) {
                setSelectedColumns([...cols.slice(0, 2)]);
            }
        }

        const adjustEntityOutput = (
            entityOutput: LinkPredictionEntityOutput
        ) => {
            const prediction = entityOutput?.prediction?.map(
                (item) =>
                    ({
                        ...item,
                        cols: [
                            ...item.cols,
                            {
                                name: in_ground_truth_or_target_col,
                                value: entityOutput?.target?.some(
                                    (targetItem) => targetItem?.id === item.id
                                ),
                                type: "boolean"
                            }
                        ]
                    }) as NodeOutputData
            );

            const target = entityOutput?.target?.map(
                (item) =>
                    ({
                        ...item,
                        cols: [
                            ...(item?.cols ?? []),
                            {
                                name: in_ground_truth_or_target_col,
                                value: entityOutput?.prediction?.some(
                                    (predictionItem) =>
                                        predictionItem?.id === item?.id
                                ),
                                type: "boolean"
                            }
                        ]
                    }) as NodeData
            );

            setTransformedEntityOutput({
                ...entityOutput,
                prediction: prediction,
                target: target as NodeOutputData[]
            });
        };

        if (entityOutput && entityOutput?.prediction && entityOutput?.target) {
            if (
                // eslint-disable-next-line no-prototype-builtins
                !entityOutput.prediction[0]?.cols[0]?.hasOwnProperty(
                    in_ground_truth_or_target_col
                )
            ) {
                adjustEntityOutput(entityOutput);
            }
        }
    }, [entityOutput, selectedColumns, setSelectedColumns]);

    const textCell = ({ getValue }: { getValue: () => unknown }) => (
        <span className="text-title-primary">{String(getValue() ?? "")}</span>
    );

    const generateHeaders = (data: NodeOutputData[]) => {
        if (data && data?.length > 0) {
            const defaultColumn = data[0]?.id_name;
            const headers =
                defaultColumn && selectedColumns?.includes(defaultColumn)
                    ? [
                          {
                              id: defaultColumn,
                              header: defaultColumn,
                              accessorKey: defaultColumn,
                              cell: textCell
                          }
                      ]
                    : [];

            const columnHeaders = data[0]?.cols
                ?.filter((col) =>
                    selectedColumns?.includes(col?.name as string)
                )
                ?.filter((col) => col?.name !== in_ground_truth_or_target_col)
                ?.map((col) => {
                    if (!col || !col?.name) {
                        return null;
                    } else if (col?.stype?.toLowerCase() === "image") {
                        return {
                            id: col?.name,
                            header: col?.name,
                            accessorKey: col?.name,
                            cell: ({
                                getValue
                            }: {
                                getValue: () => unknown;
                            }) => (
                                <>
                                    <img
                                        src={getValue() as string}
                                        alt={col?.name ?? ""}
                                        className="w-12.5"
                                    />
                                </>
                            )
                        };
                    } else {
                        return {
                            id: col?.name,
                            header: col?.name,
                            accessorKey: col?.name,
                            cell: textCell
                        };
                    }
                })
                ?.filter((col) => col !== null);
            return headers.concat(columnHeaders);
        }
        return [];
    };

    const transformData = (data: Array<NodeOutputData>) => {
        if (!data) {
            return [];
        }

        return data?.map((item: NodeOutputData) => {
            const transformedItem: { [key: string]: string } = {};
            item?.cols?.forEach((col: ColumnData) => {
                if (col?.name) {
                    transformedItem[col?.name] = col?.value as string;
                    if (
                        col?.name === in_ground_truth_or_target_col &&
                        col?.value
                    ) {
                        transformedItem["cellBgColor"] = "bg-emerald-50";
                    }
                }
            });
            if (item?.id_name && item?.id) {
                transformedItem[item?.id_name] = item?.id as string;
            }
            return transformedItem;
        });
    };

    const itemsTable = (
        data: NodeOutputData[],
        title: string,
        subtitle: string,
        isDataLoading: boolean,
        search: string,
        onSearchChange: (v: string) => void
    ) => {
        if (!data) return <></>;

        const transformed = transformData(data);
        const filteredData = search
            ? transformed.filter((row) =>
                  Object.entries(row)
                      .filter(([k]) => k !== "cellBgColor")
                      .some(([, v]) =>
                          String(v ?? "")
                              .toLowerCase()
                              .includes(search.toLowerCase())
                      )
              )
            : transformed;

        return (
            <GridCol xs={12} sm={12}>
                <div className="mb-4">
                    <div className="flex flex-col gap-2 mb-4">
                        <Title
                            variant="tertiary-heading"
                            className="font-bold text-title-primary py-0">
                            {`${title} (${isDataLoading ? "..." : data?.length} items)`}
                        </Title>
                        <Description
                            size="md"
                            variant="secondary"
                            className="py-0">
                            {subtitle}
                        </Description>
                    </div>
                    <SearchBar
                        placeholder="Search"
                        value={search}
                        variant="secondary"
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            onSearchChange(e.target.value)
                        }
                        className="w-72 h-9"
                    />
                </div>

                {isDataLoading ? (
                    <Skeleton className="w-20 h-6" />
                ) : (
                    <>
                        {data?.length > 0 ? (
                            <DataTable
                                id="prediction-explorer-table"
                                isLoading={isDataLoading}
                                columns={generateHeaders(data)}
                                data={filteredData}
                                enableGlobalFilter={false}
                                defaultPageSize={10}
                                disableRowSelectionHover={true}
                                pageCount={
                                    filteredData.length > 0
                                        ? Math.ceil(filteredData.length / 10)
                                        : 1
                                }
                            />
                        ) : (
                            <EmptyTable
                                id={"prediction-explorer-empty-table"}
                                label={
                                    data.length === 0
                                        ? `No ${title.toLowerCase()} data available for this entity.`
                                        : "Select columns to view historical items."
                                }
                                showLoading={isDataLoading}
                            />
                        )}
                    </>
                )}
            </GridCol>
        );
    };

    if (columns?.length === 0) {
        return null;
    }

    return (
        <>
            <div className="flex flex-wrap items-center gap-2 mb-6">
                <Label className="text-base text-title-secondary font-medium mb-0">
                    Columns
                </Label>
                {selectedColumns.map((col) => (
                    <div
                        key={col}
                        className="flex items-center gap-1 border border-border-secondary rounded px-2 py-1"
                        data-testid={`chip-${col}`}>
                        <span className="text-sm text-title-primary font-medium">
                            {col}
                        </span>
                        <button
                            onClick={() => handleToggleColumn(col)}
                            aria-label={`remove-chip-${col}`}
                            className="cursor-pointer text-icon-secondary hover:text-title-primary"
                            data-testid={`remove-chip-${col}`}>
                            <Close className="size-3.5" />
                        </button>
                    </div>
                ))}
                {columns.length > 0 && (
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <button className="flex items-center gap-1 border border-border-secondary rounded px-2 py-1 cursor-pointer text-title-primary">
                                <Add className="size-3.5" />
                                <span className="text-sm font-medium">
                                    Add column
                                </span>
                                <ChevronDown className="size-3.5" />
                            </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start">
                            {columns.map((col) => {
                                const isChecked = selectedColumns.includes(col);
                                return (
                                    <DropdownMenuItem
                                        key={col}
                                        onSelect={() =>
                                            handleToggleColumn(col)
                                        }>
                                        <span
                                            className={cn(
                                                "flex items-center gap-2",
                                                !isChecked &&
                                                    "text-title-secondary"
                                            )}>
                                            <input
                                                type="checkbox"
                                                checked={isChecked}
                                                readOnly
                                                className={cn(
                                                    "cursor-pointer",
                                                    isChecked &&
                                                        "[accent-color:var(--color-title-primary,#292d36)]"
                                                )}
                                            />
                                            {col}
                                        </span>
                                    </DropdownMenuItem>
                                );
                            })}
                        </DropdownMenuContent>
                    </DropdownMenu>
                )}
            </div>

            <GridRow>
                {itemsTable(
                    historicalItems?.nodes as NodeOutputData[],
                    "Historical Items",
                    TABLE_SUBTITLES.historicalItems,
                    isDataLoading,
                    historicalSearch,
                    setHistoricalSearch
                )}
                {showGroundTruth &&
                    itemsTable(
                        transformedEntityOutput?.target as NodeOutputData[],
                        "Ground Truth",
                        TABLE_SUBTITLES.groundTruth,
                        isDataLoading,
                        groundTruthSearch,
                        setGroundTruthSearch
                    )}
                {itemsTable(
                    transformedEntityOutput?.prediction as NodeOutputData[],
                    "Predictions",
                    TABLE_SUBTITLES.predictions,
                    isDataLoading,
                    predictionsSearch,
                    setPredictionsSearch
                )}
            </GridRow>
        </>
    );
};

export default WithErrorBoundary(LinkPrediction);
