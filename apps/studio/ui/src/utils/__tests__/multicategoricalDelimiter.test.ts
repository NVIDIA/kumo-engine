// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SourceColumn } from "gen/v0/types";
import { Column } from "gen/v1/types/column";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { describe, expect, it } from "vitest";

import { ColumnArgs, FlowType } from "types/tables";

import {
    assignSampleRows,
    parseColsAndSmplRows,
    toApiColumn,
    updateWithRecommendeStype
} from "utils/tables";

const sourceColumn = (overrides: Partial<SourceColumn> = {}): SourceColumn =>
    ({
        name: "tags",
        stype: "multicategorical",
        dtype: "string",
        is_primary: false,
        multicategorical_delimiter: ",",
        ...overrides
    }) as SourceColumn;

const tableArgs = (column: Column): KumoapiTableTableDefinition =>
    ({
        cols: [column],
        source_table: {
            connector_id: "connector_00000000000000000000000000000000",
            file_type: "PARQUET",
            source: { mode: "relative", path: "items" }
        }
    }) as KumoapiTableTableDefinition;

describe("multicategorical delimiter preprocessing", () => {
    it("preserves an inferred source delimiter in create flow", () => {
        const result = parseColsAndSmplRows(
            [sourceColumn()],
            [],
            FlowType.CREATE,
            "",
            ""
        );

        expect(result.sourceColumns[0].multicategorical_delimiter).toBe(",");
        expect(result.cols[0].multicategorical_delimiter).toBe(",");
    });

    it("preserves the persisted delimiter in edit and selection rebuilds", () => {
        const existing = {
            name: "tags",
            stype: "multicategorical",
            dtype: "string",
            multicategorical_delimiter: "|"
        } as Column;
        const result = parseColsAndSmplRows(
            [sourceColumn({ stype: "categorical" })],
            [existing],
            FlowType.EDIT,
            "",
            ""
        );

        expect(result.sourceColumns[0].multicategorical_delimiter).toBe("|");
        expect(result.cols[0].multicategorical_delimiter).toBe("|");
        expect(
            toApiColumn(result.sourceColumns[0]).multicategorical_delimiter
        ).toBe("|");
    });

    it("carries sample-inferred proof into the recommended API column", () => {
        const current: ColumnArgs = {
            name: "tags",
            stype: "categorical",
            dtype: "string",
            sTypeLabel: "Categorical",
            userUpdatedType: false
        };
        const [withSamples, recommendations] = assignSampleRows(
            [sourceColumn()],
            [current],
            [["red,blue"]]
        );
        const args = tableArgs(toApiColumn(current));

        const updated = updateWithRecommendeStype(
            [current],
            [{ value: "categorical", label: "Categorical" }],
            args,
            recommendations,
            true
        );

        expect(withSamples[0].multicategorical_delimiter).toBe(",");
        expect(recommendations[0].multicategoricalDelimiter).toBe(",");
        expect(updated.updatedColumn[0].multicategorical_delimiter).toBe(",");
        expect(updated.newArgs.cols[0].multicategorical_delimiter).toBe(",");
    });

    it("clears stale proof when a recommendation changes semantic type", () => {
        const current: ColumnArgs = {
            name: "tags",
            stype: "multicategorical",
            dtype: "string",
            multicategorical_delimiter: ";"
        };
        const args = tableArgs(toApiColumn(current));

        const updated = updateWithRecommendeStype(
            [current],
            [{ value: "multicategorical", label: "Multicategorical" }],
            args,
            [{ name: "tags", newStype: "categorical" }],
            false
        );

        expect(updated.updatedColumn[0].multicategorical_delimiter).toBeNull();
        expect(updated.newArgs.cols[0].multicategorical_delimiter).toBeNull();
    });
});
