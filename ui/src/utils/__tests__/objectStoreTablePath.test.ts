// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
    adlsTargetForBaseLocation,
    displayPathForObjectStoreConnector,
    filePathForRootedObjectStoreConnector,
    gcsTargetForBaseLocation,
    isAnyUriObjectStoreConnector,
    isAnyUriS3Connector,
    isObjectStoreConnectorConfig,
    normalizeObjectStoreTablePath,
    ObjectStoreConnectorConfig,
    s3SourceForConnector,
    s3TargetForBaseLocation
} from "utils/objectStoreTablePath";

const rootedS3 = {
    type: "s3",
    target: {
        mode: "rooted",
        bucket: "company-data",
        prefix: "events/root",
        region: null
    }
} satisfies ObjectStoreConnectorConfig;

const rootlessS3 = {
    type: "s3",
    target: { mode: "any_uri" }
} satisfies ObjectStoreConnectorConfig;

const gcs = {
    type: "gcs",
    target: {
        mode: "rooted",
        bucket: "training-data",
        prefix: "raw"
    }
} satisfies ObjectStoreConnectorConfig;

const adls = {
    type: "adls",
    target: {
        mode: "rooted",
        storage_account: "companylake",
        filesystem: "curated",
        prefix: "orders",
        endpoint_suffix: "dfs.core.windows.net"
    }
} satisfies ObjectStoreConnectorConfig;

const anyUriGcs = {
    type: "gcs",
    target: { mode: "any_uri" }
} satisfies ObjectStoreConnectorConfig;

const anyUriAdls = {
    type: "adls",
    target: { mode: "any_uri" }
} satisfies ObjectStoreConnectorConfig;

describe("normalizeObjectStoreTablePath", () => {
    it("recognizes rooted and rootless typed S3 configs", () => {
        expect(isObjectStoreConnectorConfig(rootedS3)).toBe(true);
        expect(isObjectStoreConnectorConfig(rootlessS3)).toBe(true);
        expect(isAnyUriS3Connector(rootedS3)).toBe(false);
        expect(isAnyUriS3Connector(rootlessS3)).toBe(true);
    });

    it("preserves arbitrary-depth rooted relative paths", () => {
        expect(
            normalizeObjectStoreTablePath(
                "2026/07/06/events/facts.parquet",
                rootedS3
            )
        ).toBe("2026/07/06/events/facts.parquet");
        expect(
            normalizeObjectStoreTablePath("daily/exports/events.csv", gcs)
        ).toBe("daily/exports/events.csv");
    });

    it.each([
        [
            rootedS3,
            "s3://company-data/events/root/2026/07/events.parquet",
            "2026/07/events.parquet"
        ],
        [
            gcs,
            "gs://training-data/raw/2026/07/examples.csv",
            "2026/07/examples.csv"
        ],
        [
            adls,
            "abfss://curated@companylake.dfs.core.windows.net/orders/2026/07/facts.parquet",
            "2026/07/facts.parquet"
        ]
    ])("normalizes a same-root full URI", (config, uri, expected) => {
        expect(normalizeObjectStoreTablePath(uri, config)).toBe(expected);
    });

    it("requires and canonicalizes a full URI for rootless S3", () => {
        expect(
            normalizeObjectStoreTablePath(
                "s3://other-bucket/deep/table%20name%3Fpart.parquet",
                rootlessS3
            )
        ).toBe("s3://other-bucket/deep/table%20name%3Fpart.parquet");
        expect(() =>
            normalizeObjectStoreTablePath("deep/table.parquet", rootlessS3)
        ).toThrow("full S3 URI");
    });

    it("creates mode-specific S3 sources and rooted output paths", () => {
        expect(s3SourceForConnector("deep/table.parquet", rootedS3)).toEqual({
            mode: "relative",
            path: "deep/table.parquet"
        });
        expect(
            s3SourceForConnector(
                "s3://other-bucket/deep/table.parquet",
                rootlessS3
            )
        ).toEqual({
            mode: "uri",
            uri: "s3://other-bucket/deep/table.parquet"
        });
        expect(
            filePathForRootedObjectStoreConnector(
                "deep/exports/predictions.csv",
                rootedS3
            )
        ).toBe("deep/exports/predictions.csv");
        expect(() =>
            filePathForRootedObjectStoreConnector(
                "s3://output-bucket/deep/exports/",
                rootlessS3
            )
        ).toThrow("source-only");
    });

    it("maps an optional S3 base location to the internal target modes", () => {
        expect(s3TargetForBaseLocation("")).toEqual({ mode: "any_uri" });
        expect(
            s3TargetForBaseLocation(
                "s3://company-data/events/2026/daily/",
                "us-west-2"
            )
        ).toEqual({
            mode: "rooted",
            bucket: "company-data",
            prefix: "events/2026/daily",
            region: "us-west-2"
        });
        expect(s3TargetForBaseLocation("s3://company-data")).toEqual({
            mode: "rooted",
            bucket: "company-data",
            prefix: "",
            region: null
        });
        // A bare bucket/prefix (no scheme) is normalized to s3://.
        expect(s3TargetForBaseLocation("company-data/events/2026")).toEqual({
            mode: "rooted",
            bucket: "company-data",
            prefix: "events/2026",
            region: null
        });
    });

    it("maps GCS and ADLS base locations to rooted or any-URI targets", () => {
        expect(gcsTargetForBaseLocation("")).toEqual({ mode: "any_uri" });
        expect(gcsTargetForBaseLocation("gs://training-data/raw")).toEqual({
            mode: "rooted",
            bucket: "training-data",
            prefix: "raw"
        });
        // A bare bucket/prefix (no scheme) is normalized to gs://.
        expect(gcsTargetForBaseLocation("training-data/raw")).toEqual({
            mode: "rooted",
            bucket: "training-data",
            prefix: "raw"
        });
        // Surrounding whitespace is still rejected (no silent trim).
        expect(() => gcsTargetForBaseLocation(" training-data")).toThrow();
        expect(adlsTargetForBaseLocation("")).toEqual({ mode: "any_uri" });
        expect(
            adlsTargetForBaseLocation(
                "abfss://curated@companylake.dfs.core.windows.net/orders"
            )
        ).toEqual(adls.target);
    });

    it("leaves provider identifier semantics to canonical API validation", () => {
        expect(gcsTargetForBaseLocation("gs://Google_Bucket/raw")).toEqual({
            mode: "rooted",
            bucket: "Google_Bucket",
            prefix: "raw"
        });
        expect(
            adlsTargetForBaseLocation(
                "abfss://bad--filesystem@account-with-hyphen.example.invalid/raw"
            )
        ).toEqual({
            mode: "rooted",
            storage_account: "account-with-hyphen",
            filesystem: "bad--filesystem",
            prefix: "raw",
            endpoint_suffix: "example.invalid"
        });
    });

    it("requires canonical full GCS and ADLS URIs for any-URI connectors", () => {
        expect(isAnyUriObjectStoreConnector(anyUriGcs)).toBe(true);
        expect(
            normalizeObjectStoreTablePath(
                "gs://other-bucket/deep/table.parquet",
                anyUriGcs
            )
        ).toBe("gs://other-bucket/deep/table.parquet");
        expect(
            normalizeObjectStoreTablePath(
                "abfss://other@account.dfs.core.windows.net/deep/table.parquet",
                anyUriAdls
            )
        ).toContain("abfss://other@account.dfs.core.windows.net/");
        expect(() =>
            normalizeObjectStoreTablePath("deep/table.parquet", anyUriGcs)
        ).toThrow("canonical GCS");
        expect(displayPathForObjectStoreConnector(anyUriGcs)).toBe(
            "Any gs:// URI"
        );
        expect(displayPathForObjectStoreConnector(anyUriAdls)).toBe(
            "Any abfss:// URI"
        );
    });

    it.each([
        "https://company-data/events",
        " s3://company-data/events",
        "s3://company-data//events",
        "s3://Bad_Bucket/events",
        "s3://company-data/events/../private",
        "s3://company-data/events%2Fprivate",
        "s3://company-data/events?version=1"
    ])("rejects invalid S3 base location %j", (baseLocation) => {
        expect(() => s3TargetForBaseLocation(baseLocation)).toThrow();
    });

    it.each([
        "",
        " table.csv",
        "/absolute/table.csv",
        "one//table.csv",
        "one/../table.csv",
        "one\\table.csv",
        "one/table.csv?version=1"
    ])("rejects unsafe rooted relative path %j", (path) => {
        expect(() => normalizeObjectStoreTablePath(path, rootedS3)).toThrow();
    });

    it.each([
        "s3://company-data/events/root2/table.csv",
        "s3://other-bucket/events/root/table.csv",
        "gs://company-data/events/root/table.csv",
        "s3://company-data/events/root/../table.csv",
        "s3://company-data/events/root/%2Ftable.csv",
        "s3://company-data/events/root",
        "s3://company-data/events/root/table.csv?version=1"
    ])("rejects a URI outside the exact rooted connector %j", (uri) => {
        expect(() => normalizeObjectStoreTablePath(uri, rootedS3)).toThrow();
    });

    it.each([
        "s3://Bad_Bucket/path/table.csv",
        "s3://127.0.0.1/path/table.csv",
        "s3://bucket//table.csv",
        "s3://bucket/path/%2Ftable.csv",
        "https://bucket/path/table.csv"
    ])("rejects a non-canonical rootless URI %j", (uri) => {
        expect(() => normalizeObjectStoreTablePath(uri, rootlessS3)).toThrow();
    });

    it.each([
        [rootedS3, "deep/%2F/table.parquet"],
        [rootedS3, "deep/%252F/table.parquet"],
        [rootlessS3, "s3://valid-bucket/deep/%2F/table.parquet"],
        [rootlessS3, "s3://valid-bucket/deep/%252F/table.parquet"]
    ])("rejects representative encoded danger in %j", (config, value) => {
        expect(() => s3SourceForConnector(value, config)).toThrow();
        if (config.target.mode === "rooted") {
            expect(() =>
                filePathForRootedObjectStoreConnector(value, config)
            ).toThrow();
        }
    });

    it.each([
        [gcs, "gs://training-data/raw/deep%2Ftable.csv"],
        [
            adls,
            "abfss://curated@companylake.dfs.core.windows.net/orders/deep%5Ctable.csv"
        ]
    ])(
        "rejects encoded separators in rooted object-store URI",
        (config, uri) => {
            expect(() => normalizeObjectStoreTablePath(uri, config)).toThrow(
                "encoded separator"
            );
        }
    );

    it("rejects a cross-scheme value from either bare-path base location", () => {
        expect(() => gcsTargetForBaseLocation("s3://x")).toThrow();
        expect(() => s3TargetForBaseLocation("gs://x")).toThrow();
    });

    it.each(["training-data ", "training-data\t"])(
        "rejects a GCS base location with trailing whitespace %j",
        (baseLocation) => {
            expect(() => gcsTargetForBaseLocation(baseLocation)).toThrow();
        }
    );

    it.each(["company data/events", ":8080/x"])(
        "rejects a bare GCS base location whose derived bucket is invalid %j",
        (baseLocation) => {
            expect(() => gcsTargetForBaseLocation(baseLocation)).toThrow();
        }
    );

    it("accepts an uppercase GS:// scheme for parity with S3", () => {
        expect(gcsTargetForBaseLocation("GS://training-data/raw")).toEqual({
            mode: "rooted",
            bucket: "training-data",
            prefix: "raw"
        });
    });

    it("accepts a valid long dotted GCS bucket name (>63 chars)", () => {
        // GCS allows dotted bucket names up to 222 chars; the client sanity
        // check must not reject them the way S3's 63-char cap would.
        const longBucket =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa." +
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.cc";
        expect(gcsTargetForBaseLocation(`${longBucket}/raw`)).toEqual({
            mode: "rooted",
            bucket: longBucket,
            prefix: "raw"
        });
    });
});
