// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from "@testing-library/react";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { vi } from "vitest";

import { useKeyColumnTypeValidation } from "../columnTypeValidation";

const clearErrors = vi.fn();
const handleError = vi.fn();

beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
});

afterEach(() => {
    vi.useRealTimers();
});

const createTableArgs = ({
    pkeyStype = "ID",
    timeColStype = "timestamp",
    endTimeColStype = "timestamp",
    timeCol = "created_at",
    endTimeCol = "ended_at"
} = {}) =>
    ({
        cols: [
            { name: "id", stype: pkeyStype, dtype: "string" },
            { name: "created_at", stype: timeColStype, dtype: "time" },
            { name: "ended_at", stype: endTimeColStype, dtype: "time" },
            { name: "value", stype: "numerical", dtype: "float64" }
        ],
        pkey: "id",
        time_col: timeCol,
        end_time_col: endTimeCol,
        source_table: {
            connector_id: "test",
            source: { mode: "relative", path: "test_table" },
            data_source_type: "S3",
            file_type: "PARQUET"
        }
    }) as KumoapiTableTableDefinition;

const renderValidation = (
    tableArgs: KumoapiTableTableDefinition = createTableArgs()
) => {
    const { rerender } = renderHook(
        (args: KumoapiTableTableDefinition) =>
            useKeyColumnTypeValidation({
                tableArgs: args,
                clearErrors,
                handleError
            }),
        { initialProps: tableArgs }
    );
    act(() => {
        vi.advanceTimersByTime(1000);
    });
    return {
        rerender: (next: KumoapiTableTableDefinition) => {
            rerender(next);
            act(() => {
                vi.advanceTimersByTime(1000);
            });
        }
    };
};

describe("useKeyColumnTypeValidation", () => {
    it("Should not clear errors when columns are valid and no column error was shown", () => {
        // Nothing was raised by this hook, so it must not touch the shared
        // alert channel (which could hold an unrelated submit error).
        renderValidation();

        expect(clearErrors).not.toHaveBeenCalled();
        expect(handleError).not.toHaveBeenCalled();
    });

    it("Should clear its own error once the columns become valid again", () => {
        const { rerender } = renderValidation(
            createTableArgs({ pkeyStype: "numerical" })
        );
        expect(handleError).toHaveBeenCalledTimes(1);
        expect(clearErrors).not.toHaveBeenCalled();

        // Fix the column type -> the previously raised error is cleared.
        rerender(createTableArgs({ pkeyStype: "ID" }));
        expect(clearErrors).toHaveBeenCalledTimes(1);
    });

    it.each([
        {
            description: "pkey column stype is not ID",
            args: { pkeyStype: "numerical" },
            expectedTitle: "Primary Key (pkey) must be an ID."
        },
        {
            description: "time_col stype is not timestamp",
            args: { timeColStype: "numerical" },
            expectedTitle: "Create Date Type is not valid!"
        },
        {
            description: "end_time_col stype is not timestamp",
            args: { endTimeColStype: "numerical" },
            expectedTitle: "End Date Type is not valid!"
        },
        {
            description: "time_col and end_time_col are the same",
            args: { endTimeCol: "created_at" },
            expectedTitle: "Date columns should not be same as end date!"
        },
        {
            description: "time_col matches end_time_col from end date side",
            args: { timeCol: "ended_at" },
            expectedTitle: "Date columns should not be same as end date!"
        }
    ])(
        "Should call handleError when $description",
        ({ args, expectedTitle }) => {
            renderValidation(createTableArgs(args));

            expect(handleError).toHaveBeenCalledWith(
                expect.any(Object),
                expectedTitle
            );
            expect(clearErrors).not.toHaveBeenCalled();
        }
    );

    it("Should not raise or clear errors when time_col and end_time_col are not set", () => {
        renderValidation(createTableArgs({ timeCol: "", endTimeCol: "" }));

        expect(handleError).not.toHaveBeenCalled();
        expect(clearErrors).not.toHaveBeenCalled();
    });

    it("Should only report pkey error when both pkey and time_col are invalid", () => {
        renderValidation(
            createTableArgs({
                pkeyStype: "numerical",
                timeColStype: "numerical"
            })
        );

        expect(handleError).toHaveBeenCalledTimes(1);
        expect(handleError).toHaveBeenCalledWith(
            expect.any(Object),
            "Primary Key (pkey) must be an ID."
        );
    });
});
