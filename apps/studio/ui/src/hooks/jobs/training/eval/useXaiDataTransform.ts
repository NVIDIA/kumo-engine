// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useState } from "react";

import { XAIResponse } from "gen/v0/types";

import { NA_VALUES } from "types/graphs";
import { XaiTableData } from "types/stats";

import { getKeys } from "utils/Helpers/objectHelpers";
import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";

// Column type string, Specific to XAI data transformation
const COL_TYPE = "COLUMN";
const COUNT_COL = "COUNT(*)";

export interface UseXaiDataTransformResult {
    xaiData: XAIResponse | null;
    tableData: XaiTableData[];
    filteredTableData: XaiTableData[];
    initialSelected: XaiTableData | null;
    loading: boolean;
    getUpdatedTableData: (name: string) => XaiTableData[];
    sortByVarianceScore: (data: XaiTableData[]) => XaiTableData[];
}

export interface UseXaiDataTransformParams {
    xaiResponse: XAIResponse | null | undefined;
    isLoading: boolean;
    isError: boolean;
    error: unknown;
}

/**
 * Custom hook for transforming XAI API response data into structured table data.
 *
 * This hook handles:
 * - Re-structuring response data from Column Explanation endpoint
 * - Creating options for drop-down and table
 * - Filtering and sorting table data by variance score
 *
 * @param params - The parameters for the hook
 * @returns Transformed XAI data ready for rendering
 */
export const useXaiDataTransform = ({
    xaiResponse,
    isLoading: dataLoading,
    isError,
    error
}: UseXaiDataTransformParams): UseXaiDataTransformResult => {
    const [xaiData, setXaiData] = useState<XAIResponse | null>(null);
    const [tableData, setTableData] = useState<XaiTableData[]>([]);
    const [filteredTableData, setFilteredTableData] = useState<XaiTableData[]>(
        []
    );
    const [initialSelected, setInitialSelected] = useState<XaiTableData | null>(
        null
    );
    const [loading, setLoading] = useState<boolean>(true);

    // We (tentatively) decide to show a single table of all table columns,
    // instead of BFS navigation starting at the root(entity) table. Since the
    // decision is not final yet, we keep the boolean switch here to preserve
    // the code/logic, in case we want to revert back to BFS navigation or
    // enable both options of BFS navigation and single table view.
    const enableBfsNavigation = useCallback(() => {
        return false;
    }, []);

    const shouldIncludeTableRow = useCallback(
        (
            row: XaiTableData,
            table_fqn: string,
            navigation_tree: {
                [key: string]: string[];
            }
        ) => {
            if (enableBfsNavigation()) {
                return (
                    navigation_tree[table_fqn]?.includes(row?.fq_name) ||
                    row?.fq_name === table_fqn + "." + COUNT_COL
                );
            } else {
                return row?.kind === COL_TYPE;
            }
        },
        [enableBfsNavigation]
    );

    /**
     * Sort table data by variance score, with N/A values at the end
     */
    const sortByVarianceScore = useCallback((data: XaiTableData[]) => {
        return [...data].sort((a, b) => {
            const aScore = a.variance_score;
            const bScore = b.variance_score;

            // Handle N/A cases
            const aIsNA = NA_VALUES.includes(aScore?.toString()?.toUpperCase());
            const bIsNA = NA_VALUES.includes(bScore?.toString()?.toUpperCase());

            if (aIsNA && bIsNA) return 0;
            if (aIsNA) return 1;
            if (bIsNA) return -1;

            return Number(bScore) - Number(aScore);
        });
    }, []);

    /**
     * Re-structurizing response data from Column Explanation endpoint
     * to create options for drop-down and table
     */
    const restructurizeColExplanation = useCallback(
        (data: XAIResponse) => {
            if (
                !data?.navigation ||
                !data.tables ||
                getKeys(data?.tables).length === 0
            ) {
                console.warn(
                    "No model explanation was found. " +
                        "We are not able to find Model Explanation data, this is likely because the model was trained before Model Explanation features was enabled. Re-training the Predictive Query should most likely resolve this."
                );
                setLoading(false);
                return;
            }

            const navigation = data?.navigation;
            const rootTableName = navigation?.root_table_name;
            const tData: XaiTableData[] = [];

            // Creating Table View Data
            Object.entries(data.tables).forEach(([table_id, cols]) => {
                const table_ids = table_id?.split(".").pop();
                Object.entries(cols).forEach(([col, col_xai]) => {
                    let type = "N/A";
                    if (
                        col_xai?.col_id?.endsWith(COUNT_COL) &&
                        !col_xai?.stype
                    ) {
                        type = "Aggregation";
                    } else if (col_xai?.stype) {
                        type = capitalizeFirstLetter(col_xai.stype);
                    }
                    const tableRowNames = `${table_ids}.${col}`;
                    const withoutHops = tableRowNames?.split("(")[0];

                    tData.push({
                        name: col, // Name of table or column
                        fq_name: `${table_id}.${col}`, // FQ Name of column,
                        table_row_names: withoutHops, //name displayed in Column Contribution
                        type: type.trim(), // Type of column or table
                        kind: COL_TYPE,
                        variance_score: col_xai?.variance_score || "N/A",
                        gradient_score: col_xai?.gradient_score || "N/A"
                    });
                });
            });
            getKeys(navigation?.navigation_tree).forEach((table) => {
                if (table !== rootTableName) {
                    tData.push({
                        name: table.split(".")[1] ?? "", // Name of table
                        fq_name: table, // FQ Name of column,
                        type: "Linked Table",
                        kind: "Table",
                        variance_score: "N/A",
                        gradient_score: "N/A"
                    });
                }
            });

            if (tData.length > 0) {
                setXaiData(data);
                const filtered = tData.filter((info) =>
                    shouldIncludeTableRow(
                        info,
                        rootTableName,
                        navigation?.navigation_tree
                    )
                );
                const sorted = sortByVarianceScore(filtered);
                setFilteredTableData(sorted);
                setInitialSelected(sorted[0]);
                setTableData(tData);
            }
            setLoading(false);
        },
        [shouldIncludeTableRow, sortByVarianceScore]
    );

    useEffect(() => {
        if (isError) {
            console.error(error);
            setLoading(false);
            return;
        }

        if (xaiResponse && !dataLoading) {
            setLoading(true);
            try {
                restructurizeColExplanation(xaiResponse as XAIResponse);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        } else if (!dataLoading && !xaiResponse) {
            setLoading(false);
        }
    }, [xaiResponse, isError, error, dataLoading, restructurizeColExplanation]);

    /**
     * Get updated table data filtered for a specific table name
     */
    const getUpdatedTableData = useCallback(
        (name: string) => {
            const tData = tableData.filter((info) =>
                shouldIncludeTableRow(
                    info,
                    name,
                    xaiData?.navigation?.navigation_tree as {
                        [key: string]: string[];
                    }
                )
            );
            return tData;
        },
        [shouldIncludeTableRow, tableData, xaiData?.navigation]
    );

    return useMemo(
        () => ({
            xaiData,
            tableData,
            filteredTableData,
            initialSelected,
            loading,
            getUpdatedTableData,
            sortByVarianceScore
        }),
        [
            xaiData,
            tableData,
            filteredTableData,
            initialSelected,
            loading,
            getUpdatedTableData,
            sortByVarianceScore
        ]
    );
};
