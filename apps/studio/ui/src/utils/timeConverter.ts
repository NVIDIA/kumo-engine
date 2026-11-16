// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseNumToMinIntDigits } from "./Helpers/numberHelpers";
import { isValid, Nullish } from "./Helpers/typeGuards";
import { timeZone } from "./Constants";

/**
 * Parses a timestamp into a Date, treating naive (offset-less) strings as
 * UTC, which is what the Kumo API serves. Returns null for missing or
 * unparseable input.
 */
function toValidDate(input: string | Date | undefined): Date | null {
    if (!input) {
        return null;
    }
    if (input instanceof Date) {
        return isNaN(input.getTime()) ? null : input;
    }
    if (typeof input !== "string") {
        return null;
    }
    const hasTimeZoneOffset = /(Z|[+-]\d{2}:?\d{2})$/.test(input);
    const date = hasTimeZoneOffset ? new Date(input) : new Date(input + "Z");
    return isNaN(date.getTime()) ? null : date;
}

/**
 * Options for the local formatters. `timeZone` overrides the viewer's
 * timezone; it exists for deterministic tests and must not be set by
 * product code (the whole point is to render the viewer's clock).
 */
type LocalFormatOptions = {
    timeZone?: string;
};

const localeStringOptions = (
    formatOptions: Intl.DateTimeFormatOptions,
    options?: LocalFormatOptions
): Intl.DateTimeFormatOptions => ({
    ...formatOptions,
    timeZoneName: "short",
    ...(options?.timeZone ? { timeZone: options.timeZone } : {})
});

/**
 * Renders the wire (UTC) wall time of an ISO datetime string, without
 * converting to the viewer's timezone. Use this ONLY for data-domain
 * timestamps (values that describe the dataset's clock, e.g. train/test
 * split boundaries or anchor times), never for resource metadata like
 * created/updated times — those render in the viewer's local timezone via
 * the formatters below.
 *
 * Input format: ISO 8601 datetime string (e.g., "2025-11-08T10:26:51.078904+00:00")
 * Output format:
 *   - With time: "2025-11-08 10:26:51 UTC" (when includeUTC=true)
 *   - Date only: "2025-11-08 UTC" (when time is 00:00:00 and alwaysShowTime=false)
 *   - Without UTC: "2025-11-08 10:26:51" (when includeUTC=false)
 *   - Always show time: "2025-11-08 00:00:00" (when alwaysShowTime=true)
 */
export function parseDateTime(
    timeStringISO: string | Date,
    includeUTC = true,
    alwaysShowTime = false
): string {
    try {
        if (timeStringISO instanceof Date) {
            timeStringISO = timeStringISO?.toISOString();
        }
        if (!timeStringISO || !timeStringISO?.split) {
            // Invalid time string
            return "-";
        }
        // Split date and time
        const dateTimeText = timeStringISO?.split("T");
        const date = dateTimeText[0]; // Date is the first part
        // Split time text
        const hms = dateTimeText[1]?.split(":");
        const secs = hms?.length > 1 ? hms[2]?.split(/[.+-]/)[0] : "00";

        if (
            !alwaysShowTime &&
            hms[0] === "00" &&
            hms[1] === "00" &&
            secs === "00"
        ) {
            // All of hr, min, sec are zero;return date only.
            return date + (includeUTC ? " " + timeZone : "");
        } else {
            return (
                date +
                " " +
                hms[0] +
                ":" +
                hms[1] +
                ":" +
                secs +
                (includeUTC ? " " + timeZone : "")
            );
        }
    } catch (error) {
        console.error("Unable to parseDateTime", error);
        return "-";
    }
}

/**
 * Extracts only the date portion of an ISO datetime string, i.e. the wire
 * (UTC) calendar date. Like parseDateTime, this is for data-domain values
 * (fact time ranges, snapshot data dates) that describe the dataset's
 * clock rather than the viewer's.
 *
 * Input format: ISO 8601 datetime string (e.g., "2025-11-08T10:26:51.078904+00:00")
 * Output format: "2025-11-08"
 */
export function parseDate(timeStringISO: string | Date): string {
    try {
        if (timeStringISO instanceof Date) {
            timeStringISO = timeStringISO.toISOString();
        }
        if (!timeStringISO || !timeStringISO.includes("T")) {
            return "-";
        }
        const date = timeStringISO.split("T")[0];
        return date;
    } catch (error) {
        console.error("Unable to extractDateOnly", error);
        return "-";
    }
}

/**
 * Formats a timestamp as a numeric date string in the viewer's local
 * timezone, with the timezone labeled.
 *
 * Input format: ISO 8601 datetime string (e.g., "2026-07-10T23:15:14.098040Z")
 * Output format: "07/10/2026, 4:15 PM PDT" (for a viewer in America/Los_Angeles)
 */
export const formatToNumericLocalDateString = (
    dateString: string | Date | undefined,
    options?: LocalFormatOptions
): string => {
    try {
        const date = toValidDate(dateString);
        if (!date) {
            return "-";
        }
        return date.toLocaleString(
            "en-US",
            localeStringOptions(
                {
                    month: "2-digit",
                    day: "2-digit",
                    year: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                    hour12: true
                },
                options
            )
        );
    } catch (error) {
        console.error("Unable to formatToNumericLocalDateString", error);
        return "-";
    }
};

/**
 * Formats a timestamp as a long date string, with day of week and seconds,
 * in the viewer's local timezone, with the timezone labeled.
 *
 * Input format: ISO 8601 datetime string (e.g., "2026-07-10T23:15:14.098040Z")
 * Output format: "Fri, Jul 10, 2026, 16:15:14 PDT" (for a viewer in America/Los_Angeles)
 */
export function formatToLongLocalDateString(
    dateTimeString: string | Date | undefined,
    options?: LocalFormatOptions
): string {
    try {
        const date = toValidDate(dateTimeString);
        if (!date) {
            return "-";
        }
        return date.toLocaleString(
            "en-US",
            localeStringOptions(
                {
                    weekday: "short",
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                    hourCycle: "h23"
                },
                options
            )
        );
    } catch (error) {
        console.error("Unable to formatToLongLocalDateString", error);
        return "-";
    }
}

/**
 * Formats a timestamp as a short date string in the viewer's local
 * timezone, with the timezone labeled. This is the app's default format
 * for resource metadata timestamps (created/updated columns and fields).
 *
 * Input format: ISO 8601 datetime string (e.g., "2026-07-10T23:15:14.098040Z")
 * Output format: "Jul 10, 2026, 4:15 PM PDT" (for a viewer in America/Los_Angeles)
 */
export function formatToShortLocalDateString(
    dateTimeString: string | Date | undefined,
    options?: LocalFormatOptions
): string {
    try {
        const date = toValidDate(dateTimeString);
        if (!date) {
            return "-";
        }
        return date.toLocaleString(
            "en-US",
            localeStringOptions(
                {
                    month: "short",
                    day: "2-digit",
                    year: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                    hour12: true
                },
                options
            )
        );
    } catch (error) {
        console.error("Unable to formatToShortLocalDateString", error);
        return "-";
    }
}

export function timeDifference(dateString: string | Date | undefined): string {
    if (!dateString) {
        return "-";
    }
    try {
        if (dateString instanceof Date) {
            dateString = dateString?.toISOString();
        }
        const currentDate = new Date();
        const inputDate = new Date(dateString);
        const differenceInSeconds = Math.floor(
            (currentDate?.getTime() - inputDate?.getTime()) / 1000
        );
        const minutes = Math.floor(differenceInSeconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        const months = Math.floor(days / 30);
        const years = Math.floor(months / 12);

        if (years > 0) {
            return years === 1 ? "1 year ago" : `${years} years ago`;
        } else if (months > 0) {
            return months === 1 ? "1 month ago" : `${months} months ago`;
        } else if (days > 0) {
            return days === 1 ? "1 day ago" : `${days} days ago`;
        } else if (hours > 0) {
            return hours === 1 ? "1 hour ago" : `${hours} hours ago`;
        } else if (minutes > 0) {
            return minutes === 1 ? "1 minute ago" : `${minutes} minutes ago`;
        } else {
            return "Just now";
        }
    } catch (error) {
        console.error("Unable to timeDifference", error);
        return "-";
    }
}

// Returns time difference in milliseconds
function getTimeDiff(
    startTime: string | Date | Nullish,
    endTime: string | Date | Nullish
): number {
    try {
        if (startTime instanceof Date) {
            startTime = startTime?.toISOString();
        }
        if (endTime instanceof Date) {
            endTime = endTime?.toISOString();
        }
        let start: Date = new Date();
        let end: Date = new Date();

        if (isValid(startTime)) {
            start = new Date(startTime as string);
        }
        if (isValid(endTime)) {
            end = new Date(endTime as string);
        }

        const diff: number = end?.getTime() - start?.getTime(); // Difference in milliseconds
        return diff;
    } catch (error) {
        console.error("Unable to jobRunTimeFromJobResponse", error);
        return 0;
    }
}

/**
 * Formats a duration in seconds as "Xh Ym" (or "Ym" under an hour). Real
 * sub-minute durations (1-59s) render as "<1m" instead of rounding down
 * to "0m"; exactly 0 still renders "0m" because callers use 0 to mean
 * "no duration recorded" rather than a sub-minute one. undefined/null
 * pass through unchanged.
 */
export function secondsToHm(
    seconds: number | undefined | null
): string | number | undefined | null {
    if (seconds === undefined || seconds === null) return seconds;

    const d = Number(seconds);

    if (d > 0 && d < 60) {
        return "<1m";
    }

    const h = Math.floor(d / 3600);
    const m = Math.floor((d % 3600) / 60);

    const hDisplay = h > 0 ? `${h}h ` : "";
    const mDisplay = `${m}m`;

    return hDisplay + mDisplay;
}

/**
 * Parses a value into a number of seconds. The Kumo API serializes
 * `timedelta` fields (e.g. `elapsed_time`) as ISO 8601 duration strings such
 * as `"PT0S"`, `"PT2M30S"` or `"P1DT30.5S"`, which `Number(...)` turns into
 * `NaN`. This helper understands those strings, and also tolerates a plain
 * numeric value already expressed in seconds (for backward compatibility).
 * Returns `null` when the input cannot be parsed.
 */
export function parseDurationToSeconds(
    value: string | number | undefined | null
): number | null {
    if (value === undefined || value === null) return null;

    // Already numeric (or a plain numeric string): interpret as seconds.
    if (typeof value === "number") {
        return Number.isFinite(value) ? value : null;
    }
    const asNumber = Number(value);
    if (value.trim() !== "" && Number.isFinite(asNumber)) {
        return asNumber;
    }

    // ISO 8601 duration: P[nD]T[nH][nM][nS] (weeks handled via P[nW]).
    const match =
        /^(-)?P(?:([\d.]+)W)?(?:([\d.]+)D)?(?:T(?:([\d.]+)H)?(?:([\d.]+)M)?(?:([\d.]+)S)?)?$/.exec(
            value.trim()
        );
    if (!match) return null;

    const [, sign, weeks, days, hours, minutes, seconds] = match;
    // Reject an empty duration like "P" or "PT" that matched nothing.
    if (!weeks && !days && !hours && !minutes && !seconds) return null;

    const total =
        (Number(weeks) || 0) * 604800 +
        (Number(days) || 0) * 86400 +
        (Number(hours) || 0) * 3600 +
        (Number(minutes) || 0) * 60 +
        (Number(seconds) || 0);

    return sign === "-" ? -total : total;
}

export function monthsDifference(
    date1: string = "",
    date2: string = ""
): number {
    //The function calculates the difference in months between two given dates,
    try {
        const d1 = new Date(date1);
        const d2 = new Date(date2);

        if (isNaN(d1?.getTime()) || isNaN(d2?.getTime())) {
            console.error("One or both dates are invalid.");
            return NaN;
        }

        const laterDate = d1 > d2 ? d1 : d2;
        const earlierDate = d1 > d2 ? d2 : d1;

        let yearLater = laterDate?.getFullYear();
        let monthLater = laterDate?.getMonth();
        const yearEarlier = earlierDate?.getFullYear();
        const monthEarlier = earlierDate?.getMonth();

        if (monthEarlier > monthLater) {
            yearLater -= 1;
            monthLater += 12;
        }

        const monthsDiff =
            (yearLater - yearEarlier) * 12 + (monthLater - monthEarlier);

        return monthsDiff;
    } catch (error) {
        console.error(error);
        return NaN;
    }
}

export const isValidISODate = (dateStr: string): boolean => {
    // This function checks if a given string is a valid date-time string.
    try {
        const isoDateRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/;
        if (isoDateRegex?.test(dateStr)) {
            const parsedDate = new Date(dateStr);
            return !isNaN(parsedDate?.getTime());
        }
        return false;
    } catch (error) {
        console.error(error);
        return false;
    }
};

/**
 * Method to calculate job run time from job start and end time.
 * Returns job run time.
 *
 * @param startTime job start time
 * @param endTime job end time
 */
export function jobRunTimeFromJobResponse(
    startTime: string | Date | Nullish,
    endTime: string | Date | Nullish
): string {
    try {
        let runTime: string = "";

        const diff: number = getTimeDiff(startTime, endTime);
        let diffMin: number = Math.ceil(diff / 60000); // Difference in minutes
        if (diffMin > 59) {
            const diffHour: number = Math.floor(diffMin / 60);
            diffMin = Math.ceil(diffMin % 60);
            runTime = `${parseNumToMinIntDigits(diffHour)}h`;
            if (diffMin > 0) {
                runTime += ` ${parseNumToMinIntDigits(diffMin)}m`;
            }
        } else if (diffMin > 0) {
            runTime = `${parseNumToMinIntDigits(diffMin)}m`;
        }

        return runTime;
    } catch (error) {
        console.error("Unable to jobRunTimeFromJobResponse", error);
        return "";
    }
}

export function isValidTimestamp<T>(data: T | undefined | null): boolean {
    let retValue = false;
    if (
        data !== undefined &&
        data !== null &&
        data?.toString()?.trim() !== "" &&
        data?.toString()?.trim() !== "-"
    ) {
        retValue = true;
    }
    return retValue;
}
