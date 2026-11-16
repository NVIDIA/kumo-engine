// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
    formatToLongLocalDateString,
    formatToNumericLocalDateString,
    formatToShortLocalDateString,
    parseDurationToSeconds,
    secondsToHm
} from "utils/timeConverter";

// Regression tests for kumo-ai/kumo#29906: timestamps render in the
// viewer's local timezone with an explicit zone label. The timeZone option
// exists for exactly these tests; product code omits it so the viewer's
// own timezone is used.

// 4:15 PM PDT / 8:15 AM+1d JST / 11:15 PM UTC.
const JULY_INSTANT = "2026-07-10T23:15:14.098040Z";
// January exercises the DST boundary: America/Los_Angeles is PST (UTC-8).
const JANUARY_INSTANT = "2026-01-15T14:30:00.000000+00:00";

const LA = { timeZone: "America/Los_Angeles" };
const TOKYO = { timeZone: "Asia/Tokyo" };
const UTC = { timeZone: "UTC" };

describe("formatToShortLocalDateString", () => {
    it("Should render the viewer's local wall time with a zone label", () => {
        expect(formatToShortLocalDateString(JULY_INSTANT, LA)).toBe(
            "Jul 10, 2026, 4:15 PM PDT"
        );
        expect(formatToShortLocalDateString(JULY_INSTANT, UTC)).toBe(
            "Jul 10, 2026, 11:15 PM UTC"
        );
    });

    it("Should render different wall times for different viewer timezones", () => {
        // Same instant, three zones: the wall time, the calendar date, and
        // the label all follow the viewer.
        expect(formatToShortLocalDateString(JULY_INSTANT, TOKYO)).toBe(
            "Jul 11, 2026, 8:15 AM GMT+9"
        );
        expect(formatToShortLocalDateString(JULY_INSTANT, TOKYO)).not.toBe(
            formatToShortLocalDateString(JULY_INSTANT, LA)
        );
    });

    it("Should label DST correctly for the date being formatted", () => {
        expect(formatToShortLocalDateString(JULY_INSTANT, LA)).toContain("PDT");
        expect(formatToShortLocalDateString(JANUARY_INSTANT, LA)).toBe(
            "Jan 15, 2026, 6:30 AM PST"
        );
    });

    it("Should treat a naive (offset-less) string as a UTC instant", () => {
        expect(formatToShortLocalDateString("2026-07-10T23:15:14", LA)).toBe(
            "Jul 10, 2026, 4:15 PM PDT"
        );
    });

    it("Should parse an explicit negative offset to the correct instant", () => {
        // Same instant as JULY_INSTANT, expressed in a -07:00 offset.
        expect(
            formatToShortLocalDateString("2026-07-10T16:15:14-07:00", UTC)
        ).toBe("Jul 10, 2026, 11:15 PM UTC");
        expect(
            formatToShortLocalDateString("2026-07-10T16:15:14-07:00", LA)
        ).toBe("Jul 10, 2026, 4:15 PM PDT");
        expect(
            formatToShortLocalDateString("2026-07-10T16:15:14-0700", UTC)
        ).toBe("Jul 10, 2026, 11:15 PM UTC");
    });

    it("Should format a Date object", () => {
        expect(formatToShortLocalDateString(new Date(JULY_INSTANT), LA)).toBe(
            "Jul 10, 2026, 4:15 PM PDT"
        );
    });

    it("Should follow the environment timezone when no override is given", () => {
        const envZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        expect(formatToShortLocalDateString(JULY_INSTANT)).toBe(
            formatToShortLocalDateString(JULY_INSTANT, { timeZone: envZone })
        );
    });

    it("Should return '-' for empty or invalid input", () => {
        expect(formatToShortLocalDateString(undefined)).toBe("-");
        expect(formatToShortLocalDateString("")).toBe("-");
        expect(formatToShortLocalDateString("not a date")).toBe("-");
    });
});

describe("formatToLongLocalDateString", () => {
    it("Should render weekday, seconds, and a zone label in local time", () => {
        expect(formatToLongLocalDateString(JULY_INSTANT, LA)).toBe(
            "Fri, Jul 10, 2026, 16:15:14 PDT"
        );
        expect(formatToLongLocalDateString(JULY_INSTANT, UTC)).toBe(
            "Fri, Jul 10, 2026, 23:15:14 UTC"
        );
        expect(formatToLongLocalDateString(JULY_INSTANT, TOKYO)).toBe(
            "Sat, Jul 11, 2026, 08:15:14 GMT+9"
        );
    });

    it("Should label DST correctly for the date being formatted", () => {
        expect(formatToLongLocalDateString(JANUARY_INSTANT, LA)).toBe(
            "Thu, Jan 15, 2026, 06:30:00 PST"
        );
    });

    it("Should return '-' for empty or invalid input", () => {
        expect(formatToLongLocalDateString(undefined)).toBe("-");
        expect(formatToLongLocalDateString("")).toBe("-");
        expect(formatToLongLocalDateString("not a date")).toBe("-");
    });
});

describe("formatToNumericLocalDateString", () => {
    it("Should render a numeric local date with a zone label", () => {
        expect(formatToNumericLocalDateString(JULY_INSTANT, LA)).toBe(
            "07/10/2026, 4:15 PM PDT"
        );
        expect(formatToNumericLocalDateString(JULY_INSTANT, TOKYO)).toBe(
            "07/11/2026, 8:15 AM GMT+9"
        );
        expect(formatToNumericLocalDateString(JANUARY_INSTANT, LA)).toBe(
            "01/15/2026, 6:30 AM PST"
        );
    });

    it("Should format a Date object", () => {
        expect(
            formatToNumericLocalDateString(new Date(JULY_INSTANT), UTC)
        ).toBe("07/10/2026, 11:15 PM UTC");
    });

    it("Should return '-' for empty or invalid input", () => {
        expect(formatToNumericLocalDateString(undefined)).toBe("-");
        expect(formatToNumericLocalDateString("")).toBe("-");
        expect(formatToNumericLocalDateString("not a date")).toBe("-");
    });
});

describe("secondsToHm", () => {
    it("Should render 1-59 seconds as <1m instead of 0m", () => {
        expect(secondsToHm(1)).toBe("<1m");
        expect(secondsToHm(8)).toBe("<1m");
        expect(secondsToHm(59)).toBe("<1m");
    });

    it("Should keep 0m for zero (trainer time unavailable)", () => {
        expect(secondsToHm(0)).toBe("0m");
    });

    it("Should render whole minutes and hours unchanged", () => {
        expect(secondsToHm(60)).toBe("1m");
        expect(secondsToHm(300)).toBe("5m");
        expect(secondsToHm(3600)).toBe("1h 0m");
        expect(secondsToHm(3661)).toBe("1h 1m");
    });

    it("Should pass through undefined and null", () => {
        expect(secondsToHm(undefined)).toBeUndefined();
        expect(secondsToHm(null)).toBeNull();
    });
});

// Regression tests for kumo-ai/kumo#29935: the training/prediction progress
// bar stayed empty because `elapsed_time` is serialized as an ISO 8601
// duration string (e.g. "PT2M30S"), which `Number(...)` turns into NaN.
describe("parseDurationToSeconds", () => {
    it("Should parse ISO 8601 duration strings the API emits", () => {
        expect(parseDurationToSeconds("PT0S")).toBe(0);
        expect(parseDurationToSeconds("PT30S")).toBe(30);
        expect(parseDurationToSeconds("PT2M30S")).toBe(150);
        expect(parseDurationToSeconds("PT2M30.5S")).toBe(150.5);
        expect(parseDurationToSeconds("PT1H2M3S")).toBe(3723);
        expect(parseDurationToSeconds("P1DT30S")).toBe(86430);
        expect(parseDurationToSeconds("P2D")).toBe(172800);
        expect(parseDurationToSeconds("P1W")).toBe(604800);
        expect(parseDurationToSeconds("-PT30S")).toBe(-30);
    });

    it("Should tolerate plain numeric seconds for backward compatibility", () => {
        expect(parseDurationToSeconds(150)).toBe(150);
        expect(parseDurationToSeconds("150")).toBe(150);
        expect(parseDurationToSeconds("150.5")).toBe(150.5);
    });

    it("Should return null for missing or unparseable input", () => {
        expect(parseDurationToSeconds(undefined)).toBeNull();
        expect(parseDurationToSeconds(null)).toBeNull();
        expect(parseDurationToSeconds("")).toBeNull();
        expect(parseDurationToSeconds("not a duration")).toBeNull();
        expect(parseDurationToSeconds("P")).toBeNull();
        expect(parseDurationToSeconds("PT")).toBeNull();
        expect(parseDurationToSeconds(NaN)).toBeNull();
    });
});
