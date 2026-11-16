// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrWarnDetails } from "gen/v0/types";

export type CTALevel = {
    color: string;
    value: string;
};

export type ErrorInfo = {
    color: string;
    info: string;
};

export type CTAInfo = {
    errors?: null | Array<ErrWarnDetails>;
    warnings?: null | Array<ErrWarnDetails>;
    info_items?: null | Array<ErrWarnDetails>;
    showCTAErrors?: boolean;
    showCTAWarnings?: boolean;
    warnings_fetched?: boolean;
};

export enum JobErrorType {
    ERROR = 0,
    INFO = 1
}
