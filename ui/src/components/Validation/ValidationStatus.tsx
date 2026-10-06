// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    ValidationCard,
    ValidationState as CatalogValidationState,
    ValidationStatus as CatalogValidationStatus
} from "@kumo-ai/kumo-ui-catalog";
import type { ErrorDetail } from "gen/v1/types";

import { InputState } from "types/inputs";

export interface ValidationStatusProps {
    state: InputState;
    errorInfo?: ErrorDetail[];
    emptyStateText?: string;
}

const STATE_MAP: Record<InputState, CatalogValidationState> = {
    [InputState.None]: "none",
    [InputState.Valid]: "valid",
    [InputState.Validating]: "validating",
    [InputState.Warning]: "warning",
    [InputState.InValid]: "invalid",
    [InputState.ValidationError]: "none"
};

const ValidationStatus = ({
    state,
    errorInfo,
    emptyStateText = "Empty Query"
}: ValidationStatusProps) => {
    const errorData = errorInfo?.filter((d) => d?.type === 0) ?? [];
    const warningData = errorInfo?.filter((d) => d?.type === 1) ?? [];
    const hasIssues = errorInfo && errorInfo.length > 0;

    return (
        <CatalogValidationStatus
            state={STATE_MAP[state]}
            emptyText={emptyStateText}
            hoverContent={
                hasIssues ? (
                    <>
                        {warningData.length > 0 && (
                            <ValidationCard
                                data={warningData}
                                variant="warning"
                            />
                        )}
                        {errorData.length > 0 && (
                            <ValidationCard data={errorData} variant="error" />
                        )}
                    </>
                ) : undefined
            }
        />
    );
};

export default ValidationStatus;
