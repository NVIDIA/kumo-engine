// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GetTrainingJobsWarningsById200 } from "gen/v0/types";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";
import { CTAInfo } from "types/warnings";

import { CtaSourceName } from "./Constants";

export function getCta(
    sourceName: CtaSourceName,
    id: string,
    handleError?: HandleErrorFunc,
    response?: GetTrainingJobsWarningsById200
): [boolean, CTAInfo] {
    let success = false;
    const cta: CTAInfo = {
        errors: [],
        warnings: [],
        info_items: [],
        showCTAErrors: false,
        showCTAWarnings: false
    };

    try {
        const jobWarnings = response?.[id];
        if (jobWarnings) {
            success = true;
            cta.errors = jobWarnings.errors ?? [];
            cta.warnings = jobWarnings.warnings ?? [];
            cta.info_items = jobWarnings.info_items ?? [];
            cta.showCTAErrors = cta.errors.length > 0;
            cta.showCTAWarnings = cta.warnings.length > 0;
        }
    } catch (error) {
        if (handleError) {
            const title = `Unable to fetch ${sourceName} warnings`;
            handleError(error as CustomError, title);
        }
    }
    return [success, cta];
}
