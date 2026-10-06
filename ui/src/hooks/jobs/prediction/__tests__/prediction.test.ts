// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { renderHook, waitFor } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { jobStatus, outputStorage } from "utils/Constants";

import {
    useGenPredErrorsById,
    usePredictionOutputs,
    usePredictionWarnings
} from "../prediction";

const makeSingleOutput = (overrides: Record<string, unknown> = {}) =>
    ({
        artifact_type: "PREDICTIONS",
        directory: "s3://bucket/dir",
        table_schema: "myschema",
        table_name: "predictions",
        extension: ".csv",
        errorInfo: {},
        ...overrides
    }) as never;

const makeOutputs = (
    key = "output-1",
    overrides: Record<string, unknown> = {}
) => ({
    [key]: makeSingleOutput(overrides)
});

const makeConnector = (
    value = "connector-123",
    mode: "rooted" | "any_uri" = "rooted"
) => ({
    value,
    label: "My Connector",
    config: {
        type: "s3" as const,
        target:
            mode === "rooted"
                ? {
                      mode: "rooted" as const,
                      bucket: "bucket",
                      prefix: ""
                  }
                : { mode: "any_uri" as const }
    }
});

const makeCloudConnector = (type: "gcs" | "adls") => ({
    value: `${type}-connector`,
    label: `${type.toUpperCase()} Connector`,
    config:
        type === "gcs"
            ? {
                  type: "gcs" as const,
                  target: {
                      mode: "rooted" as const,
                      bucket: "bucket",
                      prefix: "root"
                  }
              }
            : {
                  type: "adls" as const,
                  target: {
                      mode: "rooted" as const,
                      storage_account: "account",
                      filesystem: "filesystem",
                      prefix: "root",
                      endpoint_suffix: "dfs.core.windows.net"
                  }
              }
});

const makeAnyUriConnector = (type: "s3" | "gcs" | "adls") => ({
    value: `any-uri-${type}`,
    label: `Any URI ${type.toUpperCase()}`,
    config: {
        type,
        target: { mode: "any_uri" as const }
    }
});

const getItem = (items: unknown[]) =>
    items[0] as unknown as Record<string, unknown>;

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

describe("usePredictionOutputs", () => {
    it.each([
        ["outputs is empty", outputStorage.S3, {}],
        ["destination is unknown", "unknown", makeOutputs()]
    ])("Should return empty array when %s", (_, destination, outputs) => {
        const { result } = renderWithQueryClient(() =>
            usePredictionOutputs(destination, outputs)
        );

        expect(result.current).toEqual([]);
    });

    it.each([
        [
            "no trailing slash",
            "s3://bucket/dir",
            "preds",
            ".csv",
            "s3://bucket/dir/preds.csv"
        ],
        [
            "trailing slash",
            "s3://bucket/dir/",
            "preds",
            ".csv",
            "s3://bucket/dir/preds.csv"
        ],
        ["no table_name", "s3://bucket/dir", "", ".csv", "s3://bucket/dir/"]
    ])(
        "S3: should build a rooted relative file_path when directory has %s",
        (_, directory, table_name, extension, expectedPath) => {
            const outputs = makeOutputs("key", {
                directory,
                table_name,
                extension
            });
            const { result } = renderWithQueryClient(() =>
                usePredictionOutputs(outputStorage.S3, outputs, makeConnector())
            );

            const item = getItem(result.current);
            expect(item.file_path).toBe(
                expectedPath.replace("s3://bucket/", "")
            );
            expect(item.storage_type).toBe("S3");
            expect(item.extra_fields).toEqual([]);
        }
    );

    it.each([
        [outputStorage.S3, "s3" as const, "s3://output-bucket/results"],
        [outputStorage.GCS, "gcs" as const, "gs://output-bucket/results"],
        [
            outputStorage.ADLS,
            "adls" as const,
            "abfss://output@account.dfs.core.windows.net/results"
        ]
    ])(
        "%s: omits source-only any-URI connectors from prediction outputs",
        (destination, type, directory) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionOutputs(
                    destination,
                    makeOutputs("key", { directory }),
                    makeAnyUriConnector(type)
                )
            );

            expect(result.current).toEqual([]);
        }
    );

    it.each([
        [outputStorage.GCS, "gcs" as const, "gs://bucket/root/results"],
        [
            outputStorage.ADLS,
            "adls" as const,
            "abfss://filesystem@account.dfs.core.windows.net/root/results"
        ]
    ])(
        "%s: builds the rooted relative file_path contract",
        (destination, type, directory) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionOutputs(
                    destination,
                    makeOutputs("key", { directory }),
                    makeCloudConnector(type)
                )
            );

            expect(getItem(result.current)).toMatchObject({
                storage_type: destination,
                connector_id: `${type}-connector`,
                file_path: "results/predictions.csv",
                extra_fields: []
            });
        }
    );

    it.each([
        [
            "a relative path for rootless S3",
            makeConnector("rootless", "any_uri"),
            "relative/dir"
        ],
        [
            "a URI outside a rooted S3 connector",
            makeConnector("rooted", "rooted"),
            "s3://other-bucket/dir"
        ]
    ])(
        "omits an invalid cross-mode output for %s",
        (_, connector, directory) => {
            const { result } = renderWithQueryClient(() =>
                usePredictionOutputs(
                    outputStorage.S3,
                    makeOutputs("key", { directory }),
                    connector
                )
            );

            expect(result.current).toEqual([]);
        }
    );

    it.each([
        [outputStorage.Snowflake, "SNOWFLAKE", "preds"],
        [outputStorage.Bigquery, "BIGQUERY", "preds"],
        [outputStorage.Databricks, "DATABRICKS", "preds"]
    ])(
        "%s: should set correct storage_type, connector_id, and table_name",
        (destination, expectedStorageType, expectedTableName) => {
            const connector = makeConnector("my-connector-id");
            const outputs = makeOutputs("key", { table_name: "preds" });
            const { result } = renderWithQueryClient(() =>
                usePredictionOutputs(destination, outputs, connector)
            );

            const item = getItem(result.current);
            expect(item.storage_type).toBe(expectedStorageType);
            expect(item.connector_id).toBe("my-connector-id");
            expect(item.table_name).toBe(expectedTableName);
        }
    );

    it("Should create one output item per output entry", () => {
        const outputs = {
            "out-1": makeSingleOutput(),
            "out-2": makeSingleOutput({ table_name: "preds2" })
        };
        const { result } = renderWithQueryClient(() =>
            usePredictionOutputs(outputStorage.S3, outputs, makeConnector())
        );

        expect(result.current).toHaveLength(2);
    });
});

describe("useGenPredErrorsById", () => {
    it.each([
        ["id is empty", ""],
        ["valid id is provided", "gen-job-123"]
    ])("Should return empty errors and warnings when %s", async (_, id) => {
        const { result } = renderWithQueryClient(() =>
            useGenPredErrorsById(id, undefined)
        );

        await waitFor(() => expect(result.current.fetchingErrors).toBe(false));
        expect(result.current.errors).toEqual([]);
        expect(result.current.warnings).toEqual([]);
    });
});

describe("usePredictionWarnings", () => {
    it.each([
        ["id is empty", "", jobStatus.Running],
        ["status is Cancelled", "bp-job-123", jobStatus.Cancelled],
        ["valid id and Running status", "bp-job-123", jobStatus.Running]
    ])("Should return empty arrays when %s", async (_, id, status) => {
        const { result } = renderWithQueryClient(() =>
            usePredictionWarnings(id, status as never)
        );

        await waitFor(() => expect(result.current.fetchingErrors).toBe(false));
        expect(result.current.errors).toEqual([]);
        expect(result.current.warnings).toEqual([]);
    });
});
