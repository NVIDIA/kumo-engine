// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Row } from "@kumo-ai/kumo-ui-catalog";
import { describe, expect, it } from "vitest";

import { PredictionOutputTableColumns } from "../PredictionOutputTable";

type ColumnArgs = {
    artifact_type?: string;
    output_path?: string;
    table_name?: string;
    max_categories?: string;
    num_cols?: number;
    num_rows?: number;
    size_bytes?: number;
    component?: React.ReactNode;
};

type ColumnDefinition = ReturnType<typeof PredictionOutputTableColumns>[number];

const createRow = (data: ColumnArgs): Row<ColumnArgs> => {
    return { original: data } as Row<ColumnArgs>;
};

const renderCell = (column: ColumnDefinition, row: Row<ColumnArgs>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({ row } as never);
};

describe("PredictionOutputTableColumns", () => {
    describe("Column Configuration", () => {
        it.each([
            {
                storageType: undefined,
                expectedLength: 5,
                expectedIds: [
                    "artifact_type",
                    "table_name",
                    "max_categories",
                    "configs",
                    "size"
                ],
                description: "without storage type"
            },
            {
                storageType: "BIGQUERY" as const,
                expectedLength: 5,
                expectedIds: [
                    "artifact_type",
                    "table_name",
                    "max_categories",
                    "configs",
                    "size"
                ],
                description: "with BIGQUERY storage"
            },
            {
                storageType: "SNOWFLAKE" as const,
                expectedLength: 5,
                expectedIds: [
                    "artifact_type",
                    "table_name",
                    "max_categories",
                    "configs",
                    "size"
                ],
                description: "with SNOWFLAKE storage"
            },
            {
                storageType: "DATABRICKS" as const,
                expectedLength: 5,
                expectedIds: [
                    "artifact_type",
                    "table_name",
                    "max_categories",
                    "configs",
                    "size"
                ],
                description: "with DATABRICKS storage"
            },
            {
                storageType: "S3" as const,
                expectedLength: 6,
                expectedIds: [
                    "artifact_type",
                    "output_path",
                    "max_categories",
                    "file_type",
                    "configs",
                    "size"
                ],
                description: "with S3 storage"
            },
            ...(["GCS", "ADLS"] as const).map((storageType) => ({
                storageType,
                expectedLength: 6,
                expectedIds: [
                    "artifact_type",
                    "output_path",
                    "max_categories",
                    "file_type",
                    "configs",
                    "size"
                ],
                description: `with ${storageType} storage`
            }))
        ])(
            "Should return correct columns $description",
            ({ storageType, expectedLength, expectedIds }) => {
                const columns = PredictionOutputTableColumns(storageType);

                expect(columns).toHaveLength(expectedLength);
                expectedIds.forEach((id, index) => {
                    expect(columns[index].id).toBe(id);
                });
            }
        );
    });

    describe("Cell Rendering", () => {
        describe("Artifact Type Column", () => {
            const columns = PredictionOutputTableColumns();
            const artifactTypeColumn = columns[0];

            it("Should render capitalized artifact type", () => {
                const row = createRow({ artifact_type: "PREDICTIONS" });
                const result = renderCell(artifactTypeColumn, row);

                expect(result.props.children).toBe("Predictions");
            });

            it("Should handle lowercase artifact type", () => {
                const row = createRow({ artifact_type: "scores" });
                const result = renderCell(artifactTypeColumn, row);

                expect(result.props.children).toBe("Scores");
            });

            it("Should render emdash symbol when artifact type is missing", () => {
                const row = createRow({});
                const result = renderCell(artifactTypeColumn, row);

                expect(result.props.type).toBe("emdash");
            });

            it("Should render emdash symbol when artifact type is empty", () => {
                const row = createRow({ artifact_type: "" });
                const result = renderCell(artifactTypeColumn, row);

                expect(result.props.type).toBe("emdash");
            });
        });

        describe("Table Name Column (Non-S3)", () => {
            const columns = PredictionOutputTableColumns("BIGQUERY");
            const tableNameColumn = columns[1];

            it("Should render table name with CopyCell component", () => {
                const row = createRow({ table_name: "project.dataset.table" });
                const result = renderCell(tableNameColumn, row);

                // CopyCell is a function component
                expect(result.type.name).toBe("CopyCell");
                expect(result.props.value).toBe("project.dataset.table");
            });

            it("Should render emdash when table name is N/A or missing", () => {
                const testCases = [
                    { table_name: "N/A" },
                    { table_name: "n/a" },
                    {}
                ];

                testCases.forEach((testCase) => {
                    const row = createRow(testCase);
                    const result = renderCell(tableNameColumn, row);
                    expect(result.props.type).toBe("emdash");
                });
            });
        });

        describe("Output Path Column (object stores)", () => {
            const columns = PredictionOutputTableColumns("GCS");
            const outputPathColumn = columns[1];

            it("Should render output path with CopyCell component", () => {
                const row = createRow({
                    output_path: "gs://bucket/path/to/predictions.csv"
                });
                const result = renderCell(outputPathColumn, row);

                // CopyCell is a function component
                expect(result.type.name).toBe("CopyCell");
                expect(result.props.value).toBe(
                    "gs://bucket/path/to/predictions.csv"
                );
            });

            it("Should render emdash when output path is N/A or missing", () => {
                const testCases = [{ output_path: "N/A" }, {}];

                testCases.forEach((testCase) => {
                    const row = createRow(testCase);
                    const result = renderCell(outputPathColumn, row);
                    expect(result.props.type).toBe("emdash");
                });
            });
        });

        describe("Max Categories Column", () => {
            const columns = PredictionOutputTableColumns();
            const maxCategoriesColumn = columns[2];

            it("Should render max categories value", () => {
                const row = createRow({ max_categories: "100" });
                const result = renderCell(maxCategoriesColumn, row);

                expect(result.props.children).toBe("100");
            });

            it("Should render emdash when max categories is missing or empty", () => {
                const testCases = [{}, { max_categories: "" }];

                testCases.forEach((testCase) => {
                    const row = createRow(testCase);
                    const result = renderCell(maxCategoriesColumn, row);
                    expect(result.props.type).toBe("emdash");
                });
            });
        });

        describe("File Type Column (object stores)", () => {
            const columns = PredictionOutputTableColumns("ADLS");
            const fileTypeColumn = columns.find(
                (col) => col.id === "file_type"
            );

            it("Should extract and render file type from output path", () => {
                const row = createRow({
                    output_path:
                        "abfss://fs@acct.dfs.core.windows.net/predictions.csv"
                });
                const result = renderCell(fileTypeColumn!, row);

                expect(result.props.children).toBe("Csv");
            });

            it("Should handle parquet file type", () => {
                const row = createRow({
                    output_path: "s3://bucket/data.parquet"
                });
                const result = renderCell(fileTypeColumn!, row);

                expect(result.props.children).toBe("Parquet");
            });

            it("Should handle file paths with trailing slash", () => {
                const row = createRow({
                    output_path: "s3://bucket/data.json/"
                });
                const result = renderCell(fileTypeColumn!, row);

                expect(result.props.children).toBe("Json");
            });

            it("Should render emdash when output path has no extension", () => {
                const row = createRow({
                    output_path: "s3://bucket/predictions"
                });
                const result = renderCell(fileTypeColumn!, row);

                expect(result.props.type).toBe("emdash");
            });

            it("Should render emdash when output path is missing", () => {
                const row = createRow({});
                const result = renderCell(fileTypeColumn!, row);

                expect(result.props.type).toBe("emdash");
            });
        });

        describe("Configs Column", () => {
            const columns = PredictionOutputTableColumns();
            const configsColumn = columns.find((col) => col.id === "configs");

            it("Should render rows and columns configuration", () => {
                const row = createRow({ num_rows: 1000000, num_cols: 50 });
                const result = renderCell(configsColumn!, row);

                const labelContent = result.props.children.props.children;
                expect(labelContent).toContain("1.0M rows");
                expect(labelContent).toContain("50 columns");
            });

            it("Should format large numbers with million system", () => {
                const row = createRow({ num_rows: 2500000, num_cols: 100 });
                const result = renderCell(configsColumn!, row);

                const labelContent = result.props.children.props.children;
                expect(labelContent).toContain("2.5M rows");
                expect(labelContent).toContain("100 columns");
            });

            it("Should handle small numbers", () => {
                const row = createRow({ num_rows: 500, num_cols: 10 });
                const result = renderCell(configsColumn!, row);

                const labelContent = result.props.children.props.children;
                expect(labelContent).toContain("500 rows");
                expect(labelContent).toContain("10 columns");
            });

            it("Should render emdash when num_cols is missing", () => {
                const row = createRow({ num_rows: 1000 });
                const result = renderCell(configsColumn!, row);

                expect(result.props.children.props.type).toBe("emdash");
            });

            it("Should render emdash when both values are missing", () => {
                const row = createRow({});
                const result = renderCell(configsColumn!, row);

                expect(result.props.children.props.type).toBe("emdash");
            });
        });

        describe("Size Column", () => {
            const columns = PredictionOutputTableColumns();
            const sizeColumn = columns.find((col) => col.id === "size");

            it("Should render size in bytes", () => {
                const row = createRow({ size_bytes: 1024 });
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.children).toBe("1.0 KB");
            });

            it("Should render size in megabytes", () => {
                const row = createRow({ size_bytes: 1048576 });
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.children).toBe("1.0 MB");
            });

            it("Should render size in gigabytes", () => {
                const row = createRow({ size_bytes: 1073741824 });
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.children).toBe("1.1 GB");
            });

            it("Should render emdash when size_bytes is missing", () => {
                const row = createRow({});
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.type).toBe("emdash");
            });

            it("Should render emdash when size_bytes is zero", () => {
                const row = createRow({ size_bytes: 0 });
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.type).toBe("emdash");
            });

            it("Should handle large file sizes", () => {
                const row = createRow({ size_bytes: 5368709120 });
                const result = renderCell(sizeColumn!, row);

                expect(result.props.children.props.children).toBe("5.4 GB");
            });
        });
    });

    describe("Column Headers", () => {
        it("Should have correct headers for non-S3 storage", () => {
            const columns = PredictionOutputTableColumns("BIGQUERY");

            const headers = columns.map((col) => {
                if (typeof col.header === "function") {
                    return col.header({} as never);
                }
                return col.header;
            });

            const headerTexts = headers.map((h) => {
                if (h && typeof h === "object" && "props" in h) {
                    return h.props.children;
                }
                return h;
            });

            expect(headerTexts).toContain("Output Type");
            expect(headerTexts).toContain("Table Name");
            expect(headerTexts).toContain("Max Categories");
            expect(headerTexts).toContain("Configs");
            expect(headerTexts).toContain("Size");
        });

        it("Should have correct headers for S3 storage", () => {
            const columns = PredictionOutputTableColumns("S3");

            const headers = columns.map((col) => {
                if (typeof col.header === "function") {
                    return col.header({} as never);
                }
                return col.header;
            });

            const headerTexts = headers.map((h) => {
                if (h && typeof h === "object" && "props" in h) {
                    return h.props.children;
                }
                return h;
            });

            expect(headerTexts).toContain("Output Type");
            expect(headerTexts).toContain("Output Path");
            expect(headerTexts).toContain("Max Categories");
            expect(headerTexts).toContain("File Type");
            expect(headerTexts).toContain("Configs");
            expect(headerTexts).toContain("Size");
        });
    });

    describe("CopyCell Component", () => {
        it("Should render CopyCell for both long and short values", () => {
            const s3Columns = PredictionOutputTableColumns("S3");
            const outputPathColumn = s3Columns[1];

            const longPath =
                "s3://very-long-bucket-name/very/deep/path/structure/predictions.csv";
            const shortPath = "s3://bucket/file.csv";

            const longRow = createRow({ output_path: longPath });
            const shortRow = createRow({ output_path: shortPath });

            const longResult = renderCell(outputPathColumn, longRow);
            const shortResult = renderCell(outputPathColumn, shortRow);

            // Both should render CopyCell component
            expect(longResult.type.name).toBe("CopyCell");
            expect(shortResult.type.name).toBe("CopyCell");
            expect(longResult.props.value).toBe(longPath);
            expect(shortResult.props.value).toBe(shortPath);
        });
    });

    describe("Special Cases", () => {
        it("Should handle mixed case artifact types", () => {
            const columns = PredictionOutputTableColumns();
            const artifactTypeColumn = columns[0];

            const row = createRow({ artifact_type: "PrEdIcTiOnS" });
            const result = renderCell(artifactTypeColumn, row);

            expect(result.props.children).toBe("Predictions");
        });

        it("Should differentiate between S3 and non-S3 column sets", () => {
            const s3Columns = PredictionOutputTableColumns("S3");
            const bqColumns = PredictionOutputTableColumns("BIGQUERY");

            const s3ColumnIds = s3Columns.map((col) => col.id);
            const bqColumnIds = bqColumns.map((col) => col.id);

            expect(s3ColumnIds).toContain("output_path");
            expect(s3ColumnIds).toContain("file_type");
            expect(s3ColumnIds).not.toContain("table_name");

            expect(bqColumnIds).toContain("table_name");
            expect(bqColumnIds).not.toContain("output_path");
            expect(bqColumnIds).not.toContain("file_type");
        });

        it("Should handle zero num_rows with valid num_cols", () => {
            const columns = PredictionOutputTableColumns();
            const configsColumn = columns.find((col) => col.id === "configs");

            const row = createRow({ num_rows: 0, num_cols: 10 });
            const result = renderCell(configsColumn!, row);

            const labelContent = result.props.children.props.children;
            expect(labelContent).toContain("0 rows");
            expect(labelContent).toContain("10 columns");
        });
    });
});
