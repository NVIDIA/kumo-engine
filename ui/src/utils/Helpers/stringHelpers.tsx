// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { Symbol } from "@kumo-ai/kumo-ui-catalog";

import { Nullable } from "./typeGuards";

/**
 * Replace the splitter for databricks table schema and table name
 * with something more readable.
 */
export function replaceSchemaTableNameSplitter(data: string): string {
    try {
        return String(data)?.replace("__kumo__", "_");
    } catch (error) {
        console.error("Unable to replaceSchemaTableNameSplitter", error);
        return data;
    }
}

/**
 * Get a short hash from an ID string.
 */
export function getIdHash(id: string | null | undefined): string {
    return id?.split("-").reverse()[0]?.slice(0, 5) ?? "-";
}

/**
 * Capitalize the first letter of the given value.
 */
export function capitalizeFirstLetter(value: Nullable<string>): string {
    if (!value) {
        return ""; // Return an empty string if value is undefined or empty
    }
    return value?.charAt(0)?.toUpperCase() + value?.slice(1);
}

/**
 * Convert a string to title case.
 */
export function toTitleCase(value?: string): string {
    if (!value) {
        return "";
    }
    return value.replace(
        /\w\S*/g,
        (text) =>
            text?.charAt(0)?.toUpperCase() + text?.substring(1)?.toLowerCase()
    );
}

/**
 * Capitalize the first letter of each word, with spaces.
 * Converts underscores to spaces.
 */
export function toPascalCaseWithSpace(
    value: string | null | undefined
): string | null | undefined {
    if (!value) return value;
    return value
        ?.toLowerCase()
        ?.split("_")
        ?.map((word) => word?.charAt(0)?.toUpperCase() + word?.slice(1))
        ?.join(" ");
}

/**
 * Check if the given value is a dash ("-") and return EmDash symbol.
 */
export function checkDash(value: string): React.ReactNode {
    if (
        value === null ||
        value === undefined ||
        value === "-" ||
        value === ""
    ) {
        return <Symbol type="emdash" />;
    }
    return <>{value}</>;
}

/**
 * Format a query string for tooltip display by adding newlines before keywords.
 */
export const formatQueryForTooltip = (query: string): string => {
    if (!query) {
        return "";
    }
    try {
        return query
            ?.toString()
            ?.replace(/(?<!\n)(FOR EACH|WHERE|ASSUMING)/g, "\n$1")
            ?.trim();
    } catch (e) {
        console.log("error in formatQueryForTooltip", query, "error", e);
    }
    return query?.toString()?.trim();
};

/**
 * Format an object-store path for display by removing its authority.
 */
export const getFormattedObjectStorePath = (path?: string): string => {
    const parts = path
        ?.replace(/^[A-Za-z][A-Za-z0-9+.-]*:\/\//, "")
        ?.split("/");
    parts?.shift();
    return `../${parts?.join("/")}`;
};
