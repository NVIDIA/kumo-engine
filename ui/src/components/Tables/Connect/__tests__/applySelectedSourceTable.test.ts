// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ADLSSourceTable, GCSSourceTable, S3SourceTable } from "gen/v0/types";
import {
    KumoapiSourceTableFileType,
    KumoapiTableTableDefinitionSourceTable
} from "gen/v1/types";

import { TableDropdownOption } from "types/inputs";
import { DataSourceType } from "types/tables";

import { applySelectedSourceTable } from "../ConnectTables";

const s3SourceTable = (): KumoapiTableTableDefinitionSourceTable =>
    ({
        data_source_type: DataSourceType.S3,
        connector_id: "connector-project-upload",
        source: { mode: "relative", path: "" },
        file_type: KumoapiSourceTableFileType.CSV
    }) as KumoapiTableTableDefinitionSourceTable;

describe("applySelectedSourceTable", () => {
    it("sets the selected source name and file type", () => {
        const sourceTable = s3SourceTable();
        applySelectedSourceTable(sourceTable, {
            value: "customer",
            label: "customer",
            fileType: KumoapiSourceTableFileType.CSV
        } as TableDropdownOption);

        const s3 = sourceTable as S3SourceTable;
        expect(s3.source).toEqual({ mode: "relative", path: "customer" });
        expect(s3.file_type).toBe("CSV");
    });

    it("preserves a canonical any-URI source returned by exact resolution", () => {
        const sourceTable = s3SourceTable();
        applySelectedSourceTable(sourceTable, {
            value: "s3://remote-bucket/deep/customer.parquet",
            label: "s3://remote-bucket/deep/customer.parquet",
            fileType: KumoapiSourceTableFileType.PARQUET,
            objectStoreSource: {
                mode: "uri",
                uri: "s3://remote-bucket/deep/customer.parquet"
            }
        });

        expect((sourceTable as S3SourceTable).source).toEqual({
            mode: "uri",
            uri: "s3://remote-bucket/deep/customer.parquet"
        });
    });

    it.each([
        [DataSourceType.GCS, "gcs-connector"],
        [DataSourceType.ADLS, "adls-connector"]
    ])("sets a %s source table without converting it to S3", (type, id) => {
        const sourceTable = {
            data_source_type: type,
            connector_id: id,
            source: { mode: "relative", path: "" },
            file_type: KumoapiSourceTableFileType.PARQUET
        } as KumoapiTableTableDefinitionSourceTable;

        applySelectedSourceTable(sourceTable, {
            value: "2026/07/deep/table.csv",
            label: "2026/07/deep/table.csv",
            fileType: KumoapiSourceTableFileType.CSV
        });

        const fileSource = sourceTable as GCSSourceTable | ADLSSourceTable;
        expect(fileSource.data_source_type).toBe(type);
        expect(fileSource.connector_id).toBe(id);
        expect(fileSource.source).toEqual({
            mode: "relative",
            path: "2026/07/deep/table.csv"
        });
        expect(fileSource.file_type).toBe(KumoapiSourceTableFileType.CSV);
    });

    it.each([
        [DataSourceType.GCS, "gs://other-bucket/deep/table.parquet"],
        [
            DataSourceType.ADLS,
            "abfss://other@account.dfs.core.windows.net/deep/table.parquet"
        ]
    ])("preserves an exact full-URI %s source", (type, uri) => {
        const sourceTable = {
            data_source_type: type,
            connector_id: `${type.toLowerCase()}-connector`,
            source: { mode: "uri", uri: "" },
            file_type: KumoapiSourceTableFileType.PARQUET
        } as KumoapiTableTableDefinitionSourceTable;

        applySelectedSourceTable(sourceTable, {
            value: uri,
            label: uri,
            fileType: KumoapiSourceTableFileType.PARQUET,
            objectStoreSource: { mode: "uri", uri }
        });

        expect(
            (sourceTable as GCSSourceTable | ADLSSourceTable).source
        ).toEqual({ mode: "uri", uri });
    });

    it("does not silently default the type of a file-based table", () => {
        expect(() =>
            applySelectedSourceTable(s3SourceTable(), {
                value: "deep/directory-dataset",
                label: "deep/directory-dataset"
            })
        ).toThrow("File type is required for S3 table");
    });

    it("applies the project upload connector and its file type", () => {
        const sourceTable = s3SourceTable();
        // Uploaded-file dropdown options carry no path.
        applySelectedSourceTable(sourceTable, {
            value: "uploaded_file",
            label: "uploaded_file",
            connectorId: "connector-project-upload",
            fileType: KumoapiSourceTableFileType.PARQUET
        } as TableDropdownOption);

        const s3 = sourceTable as S3SourceTable;
        expect(s3.source).toEqual({
            mode: "relative",
            path: "uploaded_file"
        });
        expect(s3.file_type).toBe("PARQUET");
        // The connector the table came from is applied for column fetches.
        expect(s3.connector_id).toBe("connector-project-upload");
    });
});
