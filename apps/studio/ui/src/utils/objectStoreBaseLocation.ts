// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ADLSConnectorResourceConfig } from "gen/v0/types/aDLSConnectorResourceConfig";
import type { GCSConnectorResourceConfig } from "gen/v0/types/gCSConnectorResourceConfig";
import type { S3ConnectorResourceConfig } from "gen/v0/types/s3ConnectorResourceConfig";

export const URI_SCHEME = /^[A-Za-z][A-Za-z0-9+.-]*:/;
export const FULL_URI = /^([A-Za-z][A-Za-z0-9+.-]*):\/\/([^/?#]+)(\/[^?#]*)?$/;
const S3_BUCKET = /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/;
const S3_RESERVED_PREFIXES = ["xn--", "sthree-", "amzn-s3-demo-"];
const S3_RESERVED_SUFFIXES = [
    "-s3alias",
    "--ol-s3",
    ".mrap",
    "--x-s3",
    "--table-s3"
];
// A permissive client-side sanity check for a GCS bucket component: reject a
// derived bucket that carries whitespace or other clearly-invalid characters
// (a stray ``:port``, spaces) while leaving exact GCS naming semantics (case,
// the ``goog`` prefix, per-component dotted rules) to canonical API validation.
// Unlike ``S3_BUCKET`` (max 63) this tolerates the underscores/uppercase GCS
// accepts and the up-to-222-char length of dotted bucket names.
const GCS_BUCKET = /^[A-Za-z0-9][A-Za-z0-9._-]{1,220}[A-Za-z0-9]$/;

export const normalizePath = (
    path: string,
    {
        allowEmpty = false,
        preserveTrailingSlash = false
    }: { allowEmpty?: boolean; preserveTrailingSlash?: boolean } = {}
): string => {
    const hadTrailingSlash = path.endsWith("/");
    const withoutTrailingSlash = path.replace(/\/+$/, "");
    if (!withoutTrailingSlash) {
        if (allowEmpty) return "";
        throw new Error("Enter a path below the bucket or connector root.");
    }
    if (
        withoutTrailingSlash.startsWith("/") ||
        withoutTrailingSlash.includes("\\") ||
        [...withoutTrailingSlash].some(
            (character) =>
                character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127
        )
    ) {
        throw new Error("Path must use safe bucket-relative components.");
    }
    const components = withoutTrailingSlash.split("/");
    if (
        components.some(
            (component) =>
                component === "" || component === "." || component === ".."
        )
    ) {
        throw new Error(
            "Path cannot contain empty, current, or parent components."
        );
    }
    const normalized = components.join("/");
    return preserveTrailingSlash && hadTrailingSlash
        ? `${normalized}/`
        : normalized;
};

const validateS3Bucket = (bucket: string): string => {
    if (
        !S3_BUCKET.test(bucket) ||
        bucket.includes("..") ||
        S3_RESERVED_PREFIXES.some((prefix) => bucket.startsWith(prefix)) ||
        S3_RESERVED_SUFFIXES.some((suffix) => bucket.endsWith(suffix)) ||
        /^\d{1,3}(?:\.\d{1,3}){3}$/.test(bucket)
    ) {
        throw new Error("Enter a valid canonical S3 bucket name.");
    }
    return bucket;
};

const validateGcsBucket = (bucket: string): string => {
    if (!GCS_BUCKET.test(bucket)) {
        throw new Error(
            "Enter a valid GCS base location such as gs://bucket/prefix."
        );
    }
    return bucket;
};

export const decodeS3Path = (path: string): string => {
    if (/%(?![0-9A-Fa-f]{2})/.test(path)) {
        throw new Error("S3 path contains an invalid percent escape.");
    }
    if (/%(?:2f|5c)/i.test(path)) {
        throw new Error("S3 path cannot contain an encoded slash.");
    }
    let decoded: string;
    try {
        decoded = decodeURIComponent(path);
    } catch {
        throw new Error("S3 path contains invalid UTF-8.");
    }
    if (/%(?:2f|5c|2e|00)/i.test(decoded)) {
        throw new Error(
            "S3 path cannot contain a nested encoded separator, traversal, or null byte."
        );
    }
    return decoded;
};

export const normalizeDecodedS3RelativePath = (
    value: string,
    preserveTrailingSlash = false
): string => {
    if (!value || value !== value.trim()) {
        throw new Error(
            "Enter a non-empty S3 path without surrounding whitespace."
        );
    }
    if (URI_SCHEME.test(value) || value.includes("?") || value.includes("#")) {
        throw new Error(
            "Relative S3 path cannot be a URI, query, or fragment."
        );
    }
    if (/%[0-9A-Fa-f]{2}/.test(value)) {
        throw new Error(
            "Relative S3 paths must be decoded, not percent-encoded."
        );
    }
    return normalizePath(value, { preserveTrailingSlash });
};

export const encodeS3Path = (path: string): string =>
    path
        .split("/")
        .map((component) =>
            encodeURIComponent(component).replace(
                /%(?:24|26|2B|2C|3B|3D|3A|40)/gi,
                (escape) =>
                    String.fromCharCode(Number.parseInt(escape.slice(1), 16))
            )
        )
        .join("/");

export const normalizeS3Uri = (
    value: string,
    preserveTrailingSlash: boolean
): string => {
    if (!value || value !== value.trim()) {
        throw new Error(
            "Enter a non-empty S3 URI without surrounding whitespace."
        );
    }
    if (value.includes("?") || value.includes("#")) {
        throw new Error("S3 URI cannot contain a query or fragment.");
    }
    const match = FULL_URI.exec(value);
    if (!match || match[1].toLowerCase() !== "s3") {
        throw new Error("Enter a full S3 URI such as s3://bucket/path.");
    }
    const [, , authority, encodedPath = ""] = match;
    if (authority.includes("@") || authority.includes(":")) {
        throw new Error("S3 URI cannot contain credentials or a port.");
    }
    const bucket = validateS3Bucket(authority);
    if (!encodedPath.startsWith("/") || encodedPath.startsWith("//")) {
        throw new Error("S3 URI must contain one bucket/path separator.");
    }
    const decodedPath = decodeS3Path(encodedPath.slice(1));
    const path = normalizePath(decodedPath, { preserveTrailingSlash });
    return `s3://${bucket}/${encodeS3Path(path)}`;
};

/** Map the optional user-facing S3 base location to the tagged API target. */
export const s3TargetForBaseLocation = (
    baseLocation: string,
    region = ""
): S3ConnectorResourceConfig["target"] => {
    if (!baseLocation.trim()) return { mode: "any_uri" };
    if (
        baseLocation !== baseLocation.trim() ||
        baseLocation.includes("?") ||
        baseLocation.includes("#")
    ) {
        throw new Error(
            "Enter an S3 base location without whitespace, a query, or a fragment."
        );
    }

    // Accept a bare ``bucket/prefix``; the S3 connector type already fixes the
    // scheme. An explicit (possibly mistyped) scheme is preserved so the
    // canonical check below still rejects a cross-scheme value.
    const canonical = URI_SCHEME.test(baseLocation)
        ? baseLocation
        : `s3://${baseLocation}`;
    const match = FULL_URI.exec(canonical);
    if (!match || match[1].toLowerCase() !== "s3") {
        throw new Error(
            "Enter a full S3 base location such as s3://bucket/prefix."
        );
    }
    const [, , authority, encodedPath = ""] = match;
    if (authority.includes("@") || authority.includes(":")) {
        throw new Error(
            "S3 base location cannot contain credentials or a port."
        );
    }

    const bucket = validateS3Bucket(authority);
    if (encodedPath.startsWith("//")) {
        throw new Error("S3 base location must not contain an empty path.");
    }
    const decodedPath = decodeS3Path(encodedPath.replace(/^\//, ""));
    const prefix = normalizePath(decodedPath, { allowEmpty: true });

    return {
        mode: "rooted",
        bucket,
        prefix,
        region: region.trim() || null
    };
};

export const objectStoreBaseLocation = (
    baseLocation: string,
    scheme: "gs" | "abfss"
): { authority: string; prefix: string } => {
    const provider = scheme === "gs" ? "GCS" : "ADLS";
    if (
        baseLocation !== baseLocation.trim() ||
        baseLocation.includes("?") ||
        baseLocation.includes("#")
    ) {
        throw new Error(
            `Enter a ${provider} base location without whitespace, a query, or a fragment.`
        );
    }
    const match = FULL_URI.exec(baseLocation);
    // URI schemes are case-insensitive (RFC 3986), so accept an uppercase
    // ``GS://`` / ``ABFSS://`` for parity with the S3 path.
    if (!match || match[1].toLowerCase() !== scheme) {
        throw new Error(
            scheme === "gs"
                ? "Enter a canonical GCS base location such as gs://bucket/prefix."
                : "Enter a canonical ADLS base location such as abfss://filesystem@account.dfs.core.windows.net/prefix."
        );
    }
    const [, , authority, encodedPath = ""] = match;
    if (encodedPath.startsWith("//")) {
        throw new Error(`Enter a canonical ${provider} base location.`);
    }
    if (/%(?:2f|5c)/i.test(encodedPath)) {
        throw new Error(
            `${provider} path cannot contain an encoded separator.`
        );
    }
    let decodedPath: string;
    try {
        decodedPath = decodeURIComponent(encodedPath.replace(/^\//, ""));
    } catch {
        throw new Error(`${provider} path contains invalid UTF-8.`);
    }
    return {
        authority,
        prefix: normalizePath(decodedPath, { allowEmpty: true })
    };
};

export const gcsTargetForBaseLocation = (
    baseLocation: string
): GCSConnectorResourceConfig["target"] => {
    if (!baseLocation.trim()) return { mode: "any_uri" };
    // The GCS connector type already fixes the scheme, so accept a bare
    // ``bucket/prefix`` and normalize it to a canonical ``gs://`` URI. Leave an
    // explicit scheme (e.g. a mistyped ``s3://``) or a value with surrounding
    // whitespace untouched so ``objectStoreBaseLocation`` still rejects a
    // cross-scheme value and reports the canonical whitespace error.
    const canonical =
        URI_SCHEME.test(baseLocation) || baseLocation !== baseLocation.trim()
            ? baseLocation
            : `gs://${baseLocation}`;
    const { authority, prefix } = objectStoreBaseLocation(canonical, "gs");
    // ``objectStoreBaseLocation`` does no bucket validation, so a bare input
    // such as ``company data/events`` or ``:8080/x`` would otherwise yield a
    // malformed bucket. Sanity-check the derived bucket, matching the S3 path.
    const bucket = validateGcsBucket(authority);
    return { mode: "rooted", bucket, prefix };
};

export const adlsTargetForBaseLocation = (
    baseLocation: string
): ADLSConnectorResourceConfig["target"] => {
    if (!baseLocation.trim()) return { mode: "any_uri" };
    const { authority, prefix } = objectStoreBaseLocation(
        baseLocation,
        "abfss"
    );
    const at = authority.indexOf("@");
    if (at <= 0 || at !== authority.lastIndexOf("@")) {
        throw new Error(
            "Enter an ADLS filesystem, storage account, and endpoint."
        );
    }
    const filesystem = authority.slice(0, at);
    const host = authority.slice(at + 1);
    const dot = host.indexOf(".");
    if (dot <= 0 || dot === host.length - 1) {
        throw new Error(
            "Enter an ADLS filesystem, storage account, and endpoint."
        );
    }
    const storage_account = host.slice(0, dot);
    const endpoint_suffix = host.slice(dot + 1);
    return {
        mode: "rooted",
        storage_account,
        filesystem,
        prefix,
        endpoint_suffix
    };
};
