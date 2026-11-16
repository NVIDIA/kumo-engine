// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ADLSSourceTableSource,
    GCSSourceTableSource,
    KumoapiSourceTableFileType,
    S3SourceTableSource
} from "gen/v1/types";

import { ConnectorConfig } from "types/connectors";

export type SelectOption = {
    value: string;
    label: string;
    labelDescription?: string;
    disabled?: boolean;
};

export type DropdownOption = {
    value?: string | number;
    label?: string;
    type?: string;
    origValue?: string;
    isDisabled?: boolean;
    subLabel?: string;
};
export type ConnectorDropdownOption = {
    config?: ConnectorConfig;
} & DropdownOption;

export type ColumnStatsdownOption = DropdownOption & {
    pkey?: string | null;
    timeCol?: string | null;
    endTimeCol?: string | null;
};

export type TableDropdownOption = {
    value: string;
    label: string;
    valid?: boolean;
    reason?: string;
    fileType?: KumoapiSourceTableFileType;
    /** Canonical typed source returned by exact object-store resolution. */
    objectStoreSource?:
        | S3SourceTableSource
        | GCSSourceTableSource
        | ADLSSourceTableSource;
    // Connector the table was fetched from. Set in the local file-upload flow
    // where csv + parquet connector tables are merged into one list.
    connectorId?: string;
};

export enum InputState {
    None = "none",
    Validating = "validating",
    Valid = "valid",
    InValid = "invalid",
    Warning = "warning",
    ValidationError = "validationerror"
}
