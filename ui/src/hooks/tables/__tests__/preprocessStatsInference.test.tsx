// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";

import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { Column } from "gen/v1/types/column";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import {
    holdInferMetadataResponses,
    setInferMetadataResponseCols,
    setInferMetadataResponseError
} from "components/Tables/Connect/__tests__/MockFunction/InferMetadata.mock";
import { setSampleRowsStypeOverrides } from "components/Tables/Connect/__tests__/MockFunction/SampleRowsResponse.test.mock";

import { ColumnArgs, FlowType } from "types/tables";

import {
    canInferStatsBasedMetadata,
    reconcileStypeRecommendations
} from "utils/tables";

import { useTablePreprocessing } from "../preprocess";

import "tests/mockAxios";

vi.mock("utils/navigator", () => ({
    useCustomNavigation: vi.fn(() => vi.fn())
}));

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => "project-tests"
}));

const SOURCE_TABLE_NAME = "user_fact_table";
const CONNECTOR_ID = "connector-086";

// The wizard's `source_table`, stamped by `applySelectedSourceTable` before
// the column fetch (same shape the create call submits).
const BIGQUERY_SOURCE_TABLE = {
    bigquery_connector_id: CONNECTOR_ID,
    table_name: SOURCE_TABLE_NAME,
    project_id: "example-project",
    dataset_id: "synthetic",
    data_source_type: "BIGQUERY" as const
};

const CUSTOM_SQL_SOURCE_TABLE = {
    connector_id: CONNECTOR_ID,
    sql: "SELECT 1 AS one",
    table: "custom_sql_probe",
    data_source_type: "CUSTOM_SQL" as const
};

const OTHER_SOURCE_TABLE_NAME = "other_fact_table";

const OTHER_BIGQUERY_SOURCE_TABLE = {
    bigquery_connector_id: CONNECTOR_ID,
    table_name: OTHER_SOURCE_TABLE_NAME,
    project_id: "example-project",
    dataset_id: "synthetic",
    data_source_type: "BIGQUERY" as const
};

// What the SDK's stats-based `infer_metadata` returns for the same source
// the stats-free preview fixtures (`MockResponse/ColumnResponse.ts`,
// `MockResponse/SampleRowsResponse.ts`) describe. Mirrors the divergence
// classes of finding 086 / kumo-ai/kumo#30131: a high-cardinality string
// becomes `text` (preview: categorical) and a low-cardinality int becomes
// `categorical` (preview: numerical); two columns agree; the remaining
// preview columns are not covered by the response.
const STATS_BASED_COLS: Column[] = [
    { name: "InvoiceNo", stype: "categorical", dtype: "string" },
    { name: "Description", stype: "text", dtype: "string" },
    { name: "Quantity", stype: "categorical", dtype: "int64" },
    { name: "UnitPrice", stype: "numerical", dtype: "float64" }
] as Column[];

// The stats-free preview stypes of the shared fixtures, keyed by column.
const PREVIEW_STYPES: Record<string, string> = {
    InvoiceNo: "categorical",
    StockCode: "ID",
    Description: "categorical",
    Quantity: "numerical",
    InvoiceDate: "multicategorical",
    UnitPrice: "numerical",
    CustomerID: "numerical",
    Country: "categorical"
};

const previewColumnArgs = (): KumoapiTableTableDefinition["cols"] =>
    Object.entries(PREVIEW_STYPES).map(([name, stype]) => ({
        name,
        stype,
        dtype:
            name === "Quantity"
                ? "int64"
                : ["UnitPrice", "CustomerID"].includes(name)
                  ? "float64"
                  : "string"
    })) as KumoapiTableTableDefinition["cols"];

const inferMetadataCalls = () =>
    mockAxiosInstance.post.mock.calls.filter(
        ([url]) => url === "v1/tables/infer_metadata"
    );

let queryClient: QueryClient;
const setLoading = vi.fn();
const setUnnamedConSourceTable = vi.fn();
const handleError = vi.fn();

const renderPreprocessing = (
    flowType: FlowType,
    sourceTable: KumoapiTableTableDefinition["source_table"],
    initialCols: KumoapiTableTableDefinition["cols"] = []
) => {
    const initialArgs = {
        cols: initialCols,
        source_table: sourceTable,
        pkey: "",
        time_col: "",
        end_time_col: ""
    } as unknown as KumoapiTableTableDefinition;

    return renderHook(
        ({ selectedSourceTableName }: { selectedSourceTableName: string }) => {
            const [tableArgs, setTableArgs] = useState(initialArgs);
            const preprocessing = useTablePreprocessing({
                connectorId: CONNECTOR_ID,
                selectedSourceTableName,
                tableArgs,
                flowType,
                connectorNotExists: false,
                selectFromTableArgs: false,
                cancelFetchSampleCal: false,
                setTableArgs,
                setUnnamedConSourceTable,
                setLoading,
                handleError
            });
            return { tableArgs, setTableArgs, preprocessing };
        },
        {
            wrapper: createWrapper(queryClient),
            initialProps: { selectedSourceTableName: SOURCE_TABLE_NAME }
        }
    );
};

const stypesByName = (cols: Array<{ name?: string; stype?: string }>) =>
    Object.fromEntries(cols.map((col) => [col.name, col.stype]));

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setInferMetadataResponseCols(null);
    setSampleRowsStypeOverrides(null);
});

afterEach(() => {
    queryClient.clear();
});

describe("useTablePreprocessing stats-based stype inference (kumo#30131)", () => {
    it("Should request stats-based inference with the SDK request shape", async () => {
        setInferMetadataResponseCols(STATS_BASED_COLS);

        const { result } = renderPreprocessing(
            FlowType.CREATE,
            BIGQUERY_SOURCE_TABLE
        );

        await waitFor(() => {
            expect(inferMetadataCalls()).toHaveLength(1);
        });

        const [, payload, options] = inferMetadataCalls()[0];
        // `cols: []` mirrors an SDK `infer_metadata` call on the raw source,
        // so both authoring paths start inference from the same input.
        expect(payload).toEqual({
            cols: [],
            source_table: BIGQUERY_SOURCE_TABLE
        });
        expect(options?.params).toEqual({ project_id: "project-tests" });
        expect(result.current.preprocessing).toBeDefined();
    });

    it("Should apply the stats-based stypes as create-flow defaults, matching the SDK path", async () => {
        setInferMetadataResponseCols(STATS_BASED_COLS);

        const { result } = renderPreprocessing(
            FlowType.CREATE,
            BIGQUERY_SOURCE_TABLE
        );

        await waitFor(() => {
            const argStypes = stypesByName(result.current.tableArgs.cols);
            expect(argStypes["Description"]).toBe("text");
            expect(argStypes["Quantity"]).toBe("categorical");
        });

        // Parity: every column the stats-based (SDK) inference covers ends up
        // in the create payload with exactly the SDK stype.
        const argStypes = stypesByName(result.current.tableArgs.cols);
        for (const col of STATS_BASED_COLS) {
            expect(argStypes[col.name]).toBe(col.stype);
        }

        // Columns the stats-based response does not cover keep the
        // stats-free preview stypes.
        expect(argStypes["StockCode"]).toBe(PREVIEW_STYPES["StockCode"]);
        expect(argStypes["Country"]).toBe(PREVIEW_STYPES["Country"]);

        // The rendered column table and stype dropdowns follow suit.
        const columnStypes = stypesByName(result.current.preprocessing.columns);
        expect(columnStypes["Description"]).toBe("text");
        expect(columnStypes["Quantity"]).toBe("categorical");
        const descriptionIndex = result.current.preprocessing.columns.findIndex(
            (col: ColumnArgs) => col.name === "Description"
        );
        expect(
            result.current.preprocessing.columnSTypes[descriptionIndex]?.value
        ).toBe("text");
    });

    it("Should offer (not auto-apply) the stats-based stypes in the edit flow", async () => {
        setInferMetadataResponseCols(STATS_BASED_COLS);

        const { result } = renderPreprocessing(
            FlowType.EDIT,
            BIGQUERY_SOURCE_TABLE,
            previewColumnArgs()
        );

        await waitFor(() => {
            expect(result.current.preprocessing.newStypes).toEqual(
                expect.arrayContaining([
                    expect.objectContaining({
                        name: "Description",
                        newStype: "text"
                    }),
                    expect.objectContaining({
                        name: "Quantity",
                        newStype: "categorical"
                    })
                ])
            );
        });

        // Saved stypes are not silently rewritten.
        const argStypes = stypesByName(result.current.tableArgs.cols);
        expect(argStypes["Description"]).toBe("categorical");
        expect(argStypes["Quantity"]).toBe("numerical");

        // Accepting the offer converges the table on the SDK stypes.
        act(() => {
            result.current.preprocessing.applyRecommendedTypes(false);
        });
        await waitFor(() => {
            const applied = stypesByName(result.current.tableArgs.cols);
            expect(applied["Description"]).toBe("text");
            expect(applied["Quantity"]).toBe("categorical");
        });
    });

    it("Should keep the stats-free preview stypes when inference fails", async () => {
        setInferMetadataResponseError();

        const { result } = renderPreprocessing(
            FlowType.CREATE,
            BIGQUERY_SOURCE_TABLE
        );

        await waitFor(() => {
            expect(inferMetadataCalls()).toHaveLength(1);
        });
        await waitFor(() => {
            expect(result.current.preprocessing.columns.length).toBeGreaterThan(
                0
            );
        });

        const argStypes = stypesByName(result.current.tableArgs.cols);
        for (const [name, stype] of Object.entries(PREVIEW_STYPES)) {
            expect(argStypes[name]).toBe(stype);
        }
        expect(result.current.preprocessing.newStypes).toEqual([]);
        // The failure is a silent fallback, not a wizard error.
        expect(handleError).not.toHaveBeenCalled();
    });

    it("Should override auto-applied sample recommendations, but not user retypes, when the stats-based result arrives after the sample rows", async () => {
        // The sample fetch recommends Description -> ID, diverging from
        // both the preview (categorical) and the stats-based result (text).
        setSampleRowsStypeOverrides({ Description: "ID" });
        const heldResponses = holdInferMetadataResponses();

        const { result } = renderPreprocessing(
            FlowType.CREATE,
            BIGQUERY_SOURCE_TABLE
        );

        // The sample-based recommendation is auto-applied first while the
        // stats-based inference is still in flight.
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Description"]
            ).toBe("ID");
        });
        expect(heldResponses).toHaveLength(1);

        // A genuine user retype before the stats-based result lands.
        const quantityIndex = result.current.preprocessing.columns.findIndex(
            (col: ColumnArgs) => col.name === "Quantity"
        );
        act(() => {
            result.current.preprocessing.onSetColumnSType(
                { value: "ID", label: "ID" },
                quantityIndex
            );
        });
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Quantity"]
            ).toBe("ID");
        });

        await act(async () => {
            heldResponses[0].release(STATS_BASED_COLS);
        });

        // The stats-based stype wins over the auto-applied sample
        // recommendation, so the outcome matches the stats-first ordering.
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Description"]
            ).toBe("text");
        });
        // The user's explicit retype is never overridden (stats-based
        // inference proposed `categorical` for Quantity).
        expect(stypesByName(result.current.tableArgs.cols)["Quantity"]).toBe(
            "ID"
        );
    });

    it("Should discard a stale stats-based response after switching source tables", async () => {
        const heldResponses = holdInferMetadataResponses();
        const sharedSourceTable = { ...BIGQUERY_SOURCE_TABLE };

        const { result, rerender } = renderPreprocessing(
            FlowType.CREATE,
            sharedSourceTable
        );

        await waitFor(() => {
            expect(heldResponses).toHaveLength(1);
        });
        await waitFor(() => {
            expect(result.current.preprocessing.columns.length).toBeGreaterThan(
                0
            );
        });

        // The user switches to another source table (with same-named
        // columns) while the first inference request is still in flight.
        // Production shallow-copies tableArgs and mutates the nested
        // source_table IN PLACE (`applySelectedSourceTable`), so the new
        // selection reuses the same object reference; the selected name is
        // derived from tableArgs, so both change in the same commit.
        act(() => {
            result.current.setTableArgs((args) => {
                sharedSourceTable.table_name = OTHER_SOURCE_TABLE_NAME;
                return {
                    ...args,
                    cols: [],
                    pkey: "",
                    time_col: "",
                    end_time_col: ""
                };
            });
            rerender({ selectedSourceTableName: OTHER_SOURCE_TABLE_NAME });
        });

        await waitFor(() => {
            expect(heldResponses).toHaveLength(2);
        });
        // Both in-flight requests carry the same mutated object, so the
        // request payload cannot identify which selection a response
        // answers.
        expect(heldResponses[1].payload?.source_table).toBe(
            heldResponses[0].payload?.source_table
        );
        expect(heldResponses[1].payload?.source_table).toEqual(
            OTHER_BIGQUERY_SOURCE_TABLE
        );

        // The stale response for the first table answers after the switch.
        await act(async () => {
            heldResponses[0].release([
                { name: "Description", stype: "text", dtype: "string" }
            ] as Column[]);
        });
        // The current table's response answers last and is applied.
        await act(async () => {
            heldResponses[1].release([
                { name: "Quantity", stype: "categorical", dtype: "int64" }
            ] as Column[]);
        });

        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Quantity"]
            ).toBe("categorical");
        });
        // The stale table's stype was not stamped onto the same-named
        // column of the newly selected table.
        expect(stypesByName(result.current.tableArgs.cols)["Description"]).toBe(
            PREVIEW_STYPES["Description"]
        );
    });

    it("Should apply the next table's stats after switching from a table whose stats already applied, even when they arrive after its sample rows", async () => {
        const heldResponses = holdInferMetadataResponses();
        const sharedSourceTable = { ...BIGQUERY_SOURCE_TABLE };

        const { result, rerender } = renderPreprocessing(
            FlowType.CREATE,
            sharedSourceTable
        );

        await waitFor(() => {
            expect(heldResponses).toHaveLength(1);
        });
        await waitFor(() => {
            expect(result.current.preprocessing.columns.length).toBeGreaterThan(
                0
            );
        });

        // The first table's stats land and are applied as defaults.
        await act(async () => {
            heldResponses[0].release(STATS_BASED_COLS);
        });
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Description"]
            ).toBe("text");
        });

        // Switch to another source table the way production does: the
        // shared source_table object is mutated in place, and the selected
        // name (derived from tableArgs) changes in the same commit.
        act(() => {
            result.current.setTableArgs((args) => {
                sharedSourceTable.table_name = OTHER_SOURCE_TABLE_NAME;
                return {
                    ...args,
                    cols: [],
                    pkey: "",
                    time_col: "",
                    end_time_col: ""
                };
            });
            rerender({ selectedSourceTableName: OTHER_SOURCE_TABLE_NAME });
        });

        await waitFor(() => {
            expect(heldResponses).toHaveLength(2);
        });
        // The new table's sample rows resolve first (the common production
        // ordering), restoring the stats-free preview stypes.
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Description"]
            ).toBe(PREVIEW_STYPES["Description"]);
        });

        // Its stats response arrives last and must still be applied.
        await act(async () => {
            heldResponses[1].release([
                { name: "Description", stype: "text", dtype: "string" }
            ] as Column[]);
        });
        await waitFor(() => {
            expect(
                stypesByName(result.current.tableArgs.cols)["Description"]
            ).toBe("text");
        });
    });

    it("Should not request stats-based inference for a Custom SQL source", async () => {
        const { result } = renderPreprocessing(
            FlowType.CREATE,
            CUSTOM_SQL_SOURCE_TABLE
        );

        await waitFor(() => {
            expect(result.current.preprocessing.columns.length).toBeGreaterThan(
                0
            );
        });

        expect(inferMetadataCalls()).toHaveLength(0);
    });
});

describe("canInferStatsBasedMetadata", () => {
    it("Should accept physical warehouse sources and reject the rest", () => {
        expect(canInferStatsBasedMetadata(BIGQUERY_SOURCE_TABLE)).toBe(true);
        expect(canInferStatsBasedMetadata(CUSTOM_SQL_SOURCE_TABLE)).toBe(false);
        expect(
            canInferStatsBasedMetadata({
                table: "gone",
                data_source_type: "UNAVAILABLE"
            })
        ).toBe(false);
        expect(canInferStatsBasedMetadata(undefined)).toBe(false);
        // A source without a resolvable connector cannot be inferred.
        expect(
            canInferStatsBasedMetadata({
                ...BIGQUERY_SOURCE_TABLE,
                bigquery_connector_id: ""
            })
        ).toBe(false);
    });
});

describe("reconcileStypeRecommendations", () => {
    const columnArgs = (
        name: string,
        stype: string,
        dtype: string
    ): ColumnArgs => ({ name, stype, dtype }) as ColumnArgs;

    it("Should prefer the stats-based stype and fall back to sample recommendations", () => {
        const recommendations = reconcileStypeRecommendations(
            [{ name: "a", stype: "text", dtype: "string" } as Column],
            [
                { name: "a", newStype: "ID" },
                { name: "b", newStype: "categorical" }
            ],
            [
                columnArgs("a", "categorical", "string"),
                columnArgs("b", "numerical", "int64")
            ]
        );
        expect(recommendations).toEqual([
            { name: "a", newStype: "text", multicategoricalDelimiter: null },
            {
                name: "b",
                newStype: "categorical",
                multicategoricalDelimiter: null
            }
        ]);
    });

    it("Should emit nothing when the target matches the current stype", () => {
        expect(
            reconcileStypeRecommendations(
                [
                    {
                        name: "a",
                        stype: "categorical",
                        dtype: "string"
                    } as Column
                ],
                [],
                [columnArgs("a", "categorical", "string")]
            )
        ).toEqual([]);
    });

    it("Should skip targets invalid for the column dtype and unsupported columns", () => {
        expect(
            reconcileStypeRecommendations(
                [
                    // `text` is not a valid stype for an int64 column.
                    { name: "a", stype: "text", dtype: "string" } as Column,
                    { name: "b", stype: "text", dtype: "string" } as Column
                ],
                [],
                [
                    columnArgs("a", "numerical", "int64"),
                    columnArgs("b", "unsupported", "string")
                ]
            )
        ).toEqual([]);
    });

    it("Should carry the stats-based multicategorical delimiter", () => {
        expect(
            reconcileStypeRecommendations(
                [
                    {
                        name: "a",
                        stype: "multicategorical",
                        dtype: "string",
                        multicategorical_delimiter: "|"
                    } as Column
                ],
                [],
                [columnArgs("a", "categorical", "string")]
            )
        ).toEqual([
            {
                name: "a",
                newStype: "multicategorical",
                multicategoricalDelimiter: "|"
            }
        ]);
    });
});
