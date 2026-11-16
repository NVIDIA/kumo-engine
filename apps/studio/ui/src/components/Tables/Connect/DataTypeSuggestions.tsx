// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from "react";

import { Column } from "gen/v1/types/column";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";

import { ctaLevels } from "types/constants";

import { capitalizeFirstLetter } from "utils/Helpers/stringHelpers";

/**
 * Suggesting Data type changes recommendations
 */
export interface DataTypeSuggestionsProps {
    columns: Column[];
}
const DataTypeSuggestions = ({ columns }: DataTypeSuggestionsProps) => {
    const getDataTypeSuggestions = useCallback((selectedColumns: Column[]) => {
        const categoricalColsWithFloatDtype: Column[] = [];
        selectedColumns?.forEach((column) => {
            if (
                column?.dtype?.includes("float") &&
                column?.stype?.includes("categorical")
            ) {
                categoricalColsWithFloatDtype.push(column);
            }
        });
        return { categoricalColsWithFloatDtype };
    }, []);

    const { categoricalColsWithFloatDtype } = getDataTypeSuggestions(columns);

    if (categoricalColsWithFloatDtype.length < 1) {
        return null;
    }

    const byDtype = categoricalColsWithFloatDtype.reduce(
        (acc, col) => {
            const dtype = col.dtype ?? "float";
            if (!acc[dtype]) acc[dtype] = [];
            acc[dtype].push(col);
            return acc;
        },
        {} as Record<string, Column[]>
    );

    const suggestions = Object.entries(byDtype).map(([dtype, cols]) => {
        const columnList = cols.map((c) => `"${c.name}"`).join(", ");
        const isPlural = cols.length > 1;
        const stype = cols[0]?.stype;
        return {
            title: "Data type suggestion",
            description:
                ` Use of data type <${dtype}> for ${isPlural ? "columns" : "column"} ` +
                `${columnList} with Type "${capitalizeFirstLetter(stype)}" ` +
                `may result in data mismatch because of precision issue. Consider ` +
                `converting the data type to integers in your data source.`,
            is_error: false,
            is_dismissible: false,
            ctaLevel: ctaLevels.Warning
        };
    });

    return (
        <WarningErrorCTA
            id={"error-non-int-dtype"}
            title="Warning"
            ctaLevel={ctaLevels.Warning}
            data={suggestions}
            handleClose={() => {}}
        />
    );
};

export default WithErrorBoundary(DataTypeSuggestions);
