// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export enum testResponseType {
    ALL = "ALL",
    DEFAULT = "DEFAULT",
    ERROR = "ERROR",
    EMPTY = "EMPTY",
    INVALID = "INVALID",
    VALID_NAME = "VALID-NAME", // Used for entity name validation
    EXISTS = "EXISTS", // Used for entity name validation
    FAILED = "FAILED",
    QUEUED = "QUEUED",
    CANCELLED = "CANCELLED",
    RUNNING = "RUNNING",
    FILTERED_OUT = "FILTERED_OUT",
    VALID = "VALID",
    INVALID_RESPONSE = "ERROR_RESPONSE",
    PARTIAL = "PARTIAL",
    ARCHIVED = "ARCHIVED",
    WITH_WARNINGS = "WITH_WARNINGS",
    UNREACHABLE_TABLES = "UNREACHABLE_TABLES"
}

export type test = "abc";
