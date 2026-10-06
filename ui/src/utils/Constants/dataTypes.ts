// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableDataTypes } from "types/constants";

export const tableSemanticTypes: string[] = [
    "numerical",
    "categorical",
    "multicategorical",
    "ID",
    "text",
    "timestamp",
    "sequence",
    "image"
];

export const timestampColumn: string[] = ["timestamp"];

export const dataTypeDescription: {
    [key: string]: string;
} = {
    "numerical":
        "Numbers and fractions whose values relative to each other are important (eg amounts).",
    "categorical":
        "Column with a limited set of possible values (up to several thousand).",
    "ID": "Numbers or text that uniquely identify an entity in your graph (Eg PK or FKs).",
    "multicategorical": "Comma-separated list of categorical values.",
    "text": "Free-form text with semantic meaning we can process with a language model (eg descriptions).",
    "timestamp":
        "Numbers or strings that represent a date or time (eg date_created).",
    "sequence":
        "Lists of floats, all of equal length. Typically the output of another AI model.",
    "image":
        "URL strings that link to publicly accessible images used for visualization purposes only. Processing of images within Kumo models will be available in future releases."
};

export const warehouseSource = {
    SnowFlake: "snowflake",
    S3: "s3",
    GCS: "gcs",
    ADLS: "adls",
    BigQuery: "bigquery",
    Databricks: "databricks",
    Local: "local"
} as const;

export type warehouseSource =
    (typeof warehouseSource)[keyof typeof warehouseSource];

export const tableDataTypes = [
    "bool",
    "int",
    "byte",
    "int16",
    "int32",
    "int64",
    "float",
    "float32",
    "float64",
    "string",
    "binary",
    "date",
    "time",
    "timedelta",
    "floatlist",
    "intlist",
    "stringlist"
];

export const tableDataTypesToSTypesMapping: TableDataTypes = {
    bool: ["categorical"],
    int: ["numerical", "categorical", "ID"],
    byte: ["numerical", "categorical", "ID"],
    int16: ["numerical", "categorical", "ID"],
    int32: ["numerical", "categorical", "ID"],
    int64: ["numerical", "categorical", "ID"],
    float: ["numerical", "categorical", "ID"],
    float32: ["numerical", "categorical", "ID"],
    float64: ["numerical", "categorical", "ID"],
    string: [
        "categorical",
        "multicategorical",
        "ID",
        "text",
        "sequence",
        "image",
        "timestamp"
    ],
    binary: [
        "categorical",
        "multicategorical",
        "ID",
        "text",
        "sequence",
        "image"
    ],
    date: ["timestamp"],
    time: ["timestamp"],
    timedelta: ["numerical", "categorical"],
    floatlist: ["sequence"],
    intlist: ["sequence"]
};

export const ctaInfoItems = {
    split_setup_validation: "split_adjustment_for_empty_validation_set",
    possible_data_missing: "possible_data_missing"
};
