// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const digitsRegex: RegExp = /^[0-9\b]+$/;

export const floatRegex: RegExp = /^[+-]?\d*(?:[.,]\d*)?$/;

export const keywords: string[] = [
    "assuming",
    "rank",
    "for-each",
    "predict",
    "where",
    "forecast",
    "timeframes"
];

export const operators: string[] = [
    "page",
    "page-1",
    "page-2",
    "page-3",
    "page-4",
    "page-5",
    "and",
    "not_contains",
    "contains",
    "ends_with",
    "starts_with",
    "is_in",
    "is_not_null",
    "is_null",
    "like",
    "not-like",
    "not",
    "or"
];

export const type_name: string[] = [
    "avg",
    "count_distinct",
    "count",
    "first",
    "last",
    "list_distinct",
    "list",
    "max",
    "min",
    "sum"
];
