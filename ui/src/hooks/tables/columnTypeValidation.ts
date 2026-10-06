// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from "react";

import { useWithDebounce } from "@kumo-ai/kumo-ui-catalog";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";

import { CustomError } from "types/common";
import { HandleErrorFunc } from "types/validation";

import { errorStatusCode, timestampColumn } from "utils/Constants";

type ColErr = { errTitle: string; errMessage: string };

/** Pkey column must have semantic type ID. */
function validatePkey(tableArgs: KumoapiTableTableDefinition): ColErr | null {
    const pIndex = tableArgs?.cols?.findIndex(
        (element) => element?.name === tableArgs?.pkey
    );
    if (pIndex > -1 && tableArgs?.cols[pIndex]?.stype !== "ID") {
        return {
            errTitle: "Primary Key (pkey) must be an ID.",
            errMessage:
                "Please change the semantic type (stype) of your column to kumo.ID and try again!"
        };
    }
    return null;
}

/** Create-date column must be a Timestamp and differ from the end-date column. */
function validateTimeCol(
    tableArgs: KumoapiTableTableDefinition
): ColErr | null {
    const cIndex = tableArgs?.cols?.findIndex(
        (e) => e?.name === tableArgs?.time_col
    );
    if (cIndex <= -1) return null;
    if (!timestampColumn.includes(tableArgs?.cols[cIndex]?.stype as string)) {
        return {
            errTitle: "Create Date Type is not valid!",
            errMessage: "Please select Timestamp columns only"
        };
    }
    if (tableArgs?.end_time_col === tableArgs?.cols[cIndex]?.name) {
        return {
            errTitle: "Date columns should not be same as end date!",
            errMessage: "Create date and end date shouldn't be same"
        };
    }
    return null;
}

/** End-date column must be a Timestamp and differ from the create-date column. */
function validateEndTimeCol(
    tableArgs: KumoapiTableTableDefinition
): ColErr | null {
    const eIndex = tableArgs?.cols?.findIndex(
        (e) => e?.name === tableArgs?.end_time_col
    );
    if (eIndex <= -1) return null;
    if (!timestampColumn.includes(tableArgs?.cols[eIndex]?.stype as string)) {
        return {
            errTitle: "End Date Type is not valid!",
            errMessage: "Please select Timestamp columns only"
        };
    }
    if (tableArgs?.time_col === tableArgs?.cols[eIndex]?.name) {
        return {
            errTitle: "Date columns should not be same as create date!",
            errMessage: "Create date and end date shouldn't be same"
        };
    }
    return null;
}

/** First validation error (pkey, then create-date, then end-date), else empty. */
function validateTableColumns(tableArgs: KumoapiTableTableDefinition): ColErr {
    return (
        validatePkey(tableArgs) ??
        validateTimeCol(tableArgs) ??
        validateEndTimeCol(tableArgs) ?? { errTitle: "", errMessage: "" }
    );
}

export function useKeyColumnTypeValidation({
    tableArgs,
    clearErrors,
    handleError
}: {
    tableArgs: KumoapiTableTableDefinition;
    clearErrors: () => void;
    handleError: HandleErrorFunc;
}) {
    // Tracks whether the currently shown alert was raised by this hook, so we
    // only clear our own column-type error. Clearing unconditionally would wipe
    // unrelated alerts (e.g. a "Table Validation failed!" error from submit).
    const hasColumnErrorRef = useRef(false);
    const validateTableArgs = useCallback(() => {
        const { errTitle, errMessage } = validateTableColumns(tableArgs);
        if (errTitle) {
            const err: CustomError = {
                response: {
                    status: errorStatusCode.UnProcessableEntityError,
                    data: errMessage
                },
                name: "",
                message: errMessage
            };
            handleError(err, errTitle);
            hasColumnErrorRef.current = true;
        } else if (hasColumnErrorRef.current) {
            clearErrors();
            hasColumnErrorRef.current = false;
        }
    }, [tableArgs, clearErrors, handleError]);

    const restartTableArgValidation = useWithDebounce(validateTableArgs, 1);

    useEffect(restartTableArgValidation, [restartTableArgValidation]);
}
