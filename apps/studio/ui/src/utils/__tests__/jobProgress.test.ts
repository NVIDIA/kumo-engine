// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { estimateJobProgressPercent } from "utils/jobProgress";

// A 30-minute estimate window.
const START = "2026-07-17T01:19:31.237505Z";
const END = "2026-07-17T01:49:31.237505Z";

describe("estimateJobProgressPercent", () => {
    it("computes percent from the ISO-duration elapsed_time over the window", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT15M"
            })
        ).toBeCloseTo(50);
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT7M30S"
            })
        ).toBeCloseTo(25);
    });

    it("parses the production wire format (kumo#29935 capture)", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT27M45.504461S"
            })
        ).toBeCloseTo((1665.504461 / 1800) * 100);
    });

    it("caps at 98 once elapsed reaches the estimate window", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT30M"
            })
        ).toBe(98);
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT2H"
            })
        ).toBe(98);
    });

    it("returns 0 for a zero or negative elapsed_time", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "PT0S"
            })
        ).toBe(0);
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "-PT30S"
            })
        ).toBe(0);
    });

    it("returns 0 when elapsed_time is missing or unparseable", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END
            })
        ).toBe(0);
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: END,
                elapsed_time: "not a duration"
            })
        ).toBe(0);
    });

    it("returns 0 when estimated_end_time is null or missing", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: null,
                elapsed_time: "PT15M"
            })
        ).toBe(0);
        expect(
            estimateJobProgressPercent({
                start_time: START,
                elapsed_time: "PT15M"
            })
        ).toBe(0);
        expect(estimateJobProgressPercent(null)).toBe(0);
        expect(estimateJobProgressPercent(undefined)).toBe(0);
    });

    it("returns 0 for an empty or unparseable window", () => {
        expect(
            estimateJobProgressPercent({
                start_time: START,
                estimated_end_time: START,
                elapsed_time: "PT15M"
            })
        ).toBe(0);
        expect(
            estimateJobProgressPercent({
                start_time: END,
                estimated_end_time: START,
                elapsed_time: "PT15M"
            })
        ).toBe(0);
        expect(
            estimateJobProgressPercent({
                start_time: "not-a-date",
                estimated_end_time: END,
                elapsed_time: "PT15M"
            })
        ).toBe(0);
    });
});
