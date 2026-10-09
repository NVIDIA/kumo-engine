// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ADLSConnectorResourceConfig } from "gen/v0/types/aDLSConnectorResourceConfig";
import type { GCSConnectorResourceConfig } from "gen/v0/types/gCSConnectorResourceConfig";
import type { S3ConnectorResourceConfig } from "gen/v0/types/s3ConnectorResourceConfig";
import type {
    ADLSSourceTableSource,
    GCSSourceTableSource,
    S3SourceTableSource
} from "gen/v1/types";

import type { ConnectorConfig } from "types/connectors";

import {
    decodeS3Path,
    encodeS3Path,
    FULL_URI,
    normalizeDecodedS3RelativePath,
    normalizePath,
    normalizeS3Uri,
    objectStoreBaseLocation,
    URI_SCHEME
} from "./objectStoreBaseLocation";

export {
    adlsTargetForBaseLocation,
    gcsTargetForBaseLocation,
    s3TargetForBaseLocation
} from "./objectStoreBaseLocation";

export type S3ConnectorConfig = S3ConnectorResourceConfig & { type: "s3" };
export type ObjectStoreConnectorConfig =
    | S3ConnectorConfig
    | (GCSConnectorResourceConfig & { type: "gcs" })
    | (ADLSConnectorResourceConfig & { type: "adls" });

export const isObjectStoreOutputStorage = (
    storageType: string | undefined
): boolean => ["S3", "GCS", "ADLS"].includes(storageType ?? "");

type ObjectStoreRoot = {
    scheme: "s3" | "gs" | "abfss";
    authority: string;
    prefix: string;
};

export const isS3ConnectorConfig = (
    config: ConnectorConfig | undefined
): config is S3ConnectorConfig => config?.type === "s3";

export const isGCSConnectorConfig = (
    config: ConnectorConfig | undefined
): config is GCSConnectorResourceConfig & { type: "gcs" } =>
    config?.type === "gcs";

export const isADLSConnectorConfig = (
    config: ConnectorConfig | undefined
): config is ADLSConnectorResourceConfig & { type: "adls" } =>
    config?.type === "adls";

export const isAnyUriS3Connector = (
    config: ConnectorConfig | undefined
): boolean => isS3ConnectorConfig(config) && config.target.mode === "any_uri";

export const isAnyUriObjectStoreConnector = (
    config: ConnectorConfig | undefined
): boolean =>
    isObjectStoreConnectorConfig(config) && config.target.mode === "any_uri";

export const isRootedObjectStoreConnector = (
    config: ConnectorConfig | undefined
): config is ObjectStoreConnectorConfig =>
    isObjectStoreConnectorConfig(config) && config.target.mode !== "any_uri";

export const isObjectStoreConnectorConfig = (
    config: ConnectorConfig | undefined
): config is ObjectStoreConnectorConfig =>
    isS3ConnectorConfig(config) ||
    isGCSConnectorConfig(config) ||
    isADLSConnectorConfig(config);

const rootForConfig = (config: ObjectStoreConnectorConfig): ObjectStoreRoot => {
    switch (config.type) {
        case "s3": {
            if (config.target.mode === "any_uri") {
                throw new Error(
                    "This S3 connector requires a full S3 URI for every table."
                );
            }
            const prefix = normalizePath(config.target.prefix ?? "", {
                allowEmpty: true
            });
            return {
                scheme: "s3",
                authority: config.target.bucket,
                prefix
            };
        }
        case "gcs":
            if (config.target.mode === "any_uri") {
                throw new Error(
                    "This GCS connector requires a full GCS URI for every table."
                );
            }
            return {
                scheme: "gs",
                authority: config.target.bucket,
                prefix: normalizePath(config.target.prefix ?? "", {
                    allowEmpty: true
                })
            };
        case "adls":
            if (config.target.mode === "any_uri") {
                throw new Error(
                    "This ADLS connector requires a full ADLS URI for every table."
                );
            }
            return {
                scheme: "abfss",
                authority: `${config.target.filesystem}@${config.target.storage_account}.${config.target.endpoint_suffix ?? "dfs.core.windows.net"}`,
                prefix: normalizePath(config.target.prefix ?? "", {
                    allowEmpty: true
                })
            };
    }
};

const relativePathFromUri = (
    value: string,
    config: ObjectStoreConnectorConfig,
    preserveTrailingSlash = false
): string => {
    if (value.includes("?") || value.includes("#")) {
        throw new Error("Table URI cannot contain a query or fragment.");
    }
    const match = FULL_URI.exec(value);
    if (!match) throw new Error("Enter a valid object-store URI.");

    const [, scheme, authority, encodedPath = ""] = match;
    const root = rootForConfig(config);
    if (
        scheme.toLowerCase() !== root.scheme ||
        authority.toLowerCase() !== root.authority.toLowerCase()
    ) {
        throw new Error("Table URI must use the selected connector's root.");
    }
    const decodedPath =
        config.type === "s3"
            ? decodeS3Path(encodedPath.replace(/^\//, ""))
            : (() => {
                  if (/%(?:2f|5c)/i.test(encodedPath)) {
                      throw new Error(
                          "Object-store URI cannot contain an encoded separator."
                      );
                  }
                  return decodeURIComponent(encodedPath).replace(/^\//, "");
              })();
    const fullPath = normalizePath(decodedPath, { preserveTrailingSlash });
    const fullComponents = fullPath.replace(/\/$/, "").split("/");
    const rootComponents = root.prefix ? root.prefix.split("/") : [];
    if (
        fullComponents.length <= rootComponents.length ||
        rootComponents.some(
            (component, index) => component !== fullComponents[index]
        )
    ) {
        throw new Error("Table URI must be below the selected connector root.");
    }
    const relative = fullComponents.slice(rootComponents.length).join("/");
    return preserveTrailingSlash && fullPath.endsWith("/")
        ? `${relative}/`
        : relative;
};

/** Normalize one object-store coordinate according to connector target mode. */
export const normalizeObjectStoreTablePath = (
    value: string,
    config: ObjectStoreConnectorConfig,
    preserveTrailingSlash = false
): string => {
    if (isAnyUriObjectStoreConnector(config)) {
        if (config.type === "s3") return normalizeS3Uri(value, false);
        const expectedScheme = config.type === "gcs" ? "gs" : "abfss";
        const { authority, prefix } = objectStoreBaseLocation(
            value,
            expectedScheme
        );
        if (!prefix) {
            throw new Error("The full table URI must include a table path.");
        }
        return `${expectedScheme}://${authority}/${encodeS3Path(prefix)}`;
    }
    if (!value || value !== value.trim()) {
        throw new Error(
            "Enter a non-empty table path without surrounding whitespace."
        );
    }
    if (URI_SCHEME.test(value)) {
        return relativePathFromUri(value, config, preserveTrailingSlash);
    }
    if (value.includes("?") || value.includes("#")) {
        throw new Error("Table path cannot contain a query or fragment.");
    }
    return config.type === "s3"
        ? normalizeDecodedS3RelativePath(value, preserveTrailingSlash)
        : normalizePath(value, { preserveTrailingSlash });
};

export const s3SourceForConnector = (
    value: string,
    config: S3ConnectorConfig
): S3SourceTableSource => {
    const normalized = normalizeObjectStoreTablePath(value, config);
    return isAnyUriS3Connector(config)
        ? { mode: "uri", uri: normalized }
        : { mode: "relative", path: normalized };
};

/** Build the relative file_path contract for a rooted object-store output. */
export const filePathForRootedObjectStoreConnector = (
    value: string,
    config: ObjectStoreConnectorConfig
): string => {
    if (config.target.mode === "any_uri") {
        throw new Error("Any-URI connectors are source-only.");
    }
    return normalizeObjectStoreTablePath(value, config, true);
};

export const sourcePathFromS3Source = (source: S3SourceTableSource): string =>
    "uri" in source ? source.uri : source.path;

export const sourcePathFromObjectStoreSource = (
    source: S3SourceTableSource | GCSSourceTableSource | ADLSSourceTableSource
): string => ("uri" in source ? source.uri : source.path);

export const displayPathForS3Connector = (
    config: S3ConnectorConfig
): string => {
    if (config.target.mode === "any_uri") return "Any s3:// URI";
    const prefix = config.target.prefix?.replace(/^\/+|\/+$/g, "");
    return `s3://${config.target.bucket}${prefix ? `/${prefix}` : ""}`;
};

export const displayPathForObjectStoreConnector = (
    config: ObjectStoreConnectorConfig
): string => {
    if (config.type === "s3") return displayPathForS3Connector(config);
    if (config.target.mode === "any_uri") {
        return config.type === "gcs" ? "Any gs:// URI" : "Any abfss:// URI";
    }
    const root = rootForConfig(config);
    return `${root.scheme}://${root.authority}${root.prefix ? `/${root.prefix}` : ""}`;
};
