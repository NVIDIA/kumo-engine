// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Convert bytes to a human-readable decimal size. */
export function bytesToSize(bytes: number, decimals: number = 1): string {
    try {
        if (bytes === -1) return "-";
        if (bytes === 0) return "0 Bytes";

        const dm = decimals < 0 ? 0 : decimals;
        const units = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];
        let value = bytes;
        let unit = 0;
        while (value >= 1000 && unit < units.length - 1) {
            value /= 1000;
            unit++;
        }
        return `${value.toFixed(dm)} ${units[unit]}`;
    } catch (error) {
        console.error("Unable to bytesToSize", error);
        return "-";
    }
}

interface ParseOptions {
    millionSystem?: boolean;
    decimals?: number;
    disableMiniFraction?: boolean;
}

/**
 * Method to parse numbers. This method takes two arguments value and options.
 * Value is the number to be parsed and options is the object with values
 * millionSystem and number of decimal points.
 *
 * @param {number|string} value number to be formated
 * @param {object} options object with values for millionSystem and
 * decimal points.
 *
 * @example let value = parseNumbers(1234566, {millionSystem: true})
 */
export function parseNumbers(
    value: number | string | null | undefined,
    options: ParseOptions = {}
): string | number | null {
    if (value === -1) {
        return "-";
    }
    if (value === null || value === undefined || isNaN(Number(value))) {
        return value === undefined ? null : value;
    }
    const number = Number(value);

    // Handling scientific notation for very small numbers
    if (Math.abs(number) < 0.0001 && number !== 0) {
        return number?.toLocaleString("en-US", {
            notation: "scientific",
            maximumFractionDigits: 3,
            minimumFractionDigits: 3
        } as Intl.NumberFormatOptions);
    }

    // Handling compact notation if `millionSystem` option is set
    if (options?.millionSystem) {
        const FractionDigits = Math.abs(number) < 1000 ? 0 : 1;
        return number.toLocaleString("en-US", {
            minimumFractionDigits: FractionDigits,
            maximumFractionDigits: FractionDigits,
            notation: "compact",
            compactDisplay: "short"
        } as Intl.NumberFormatOptions);
    }

    // Handling decimal places for general numbers
    let decimal = options?.decimals || 3;
    if (value?.toString()?.startsWith("0.") && decimal < 4) {
        const t = value?.toString()?.split(".")[1];
        let exponent = 0;
        if (t?.startsWith("00")) {
            exponent = 0;
        } else if (t?.startsWith("0")) {
            exponent = 1;
        }
        let num = number;
        while (num < 1 && num !== 0) {
            num *= 10;
            exponent++;
        }
        decimal = Math.max(exponent, 2);
    }
    const FractionDigits = Number.isInteger(number) ? 0 : decimal;
    return number.toLocaleString("en-US", {
        minimumFractionDigits: options.disableMiniFraction ? 0 : FractionDigits,
        maximumFractionDigits: FractionDigits,
        useGrouping: true
    } as Intl.NumberFormatOptions);
}

/**
 * Method to parse given numbers to specified minimum integer digits.
 * This method takes two arguments value and intDigts.
 * Value is the number to be parsed and intDigits is the minimum integer digits
 * in the returned value.
 *
 * @param {number} value number to be formated
 * @param {number} intDigits Required integer digits. Default value is 2.
 */
export function parseNumToMinIntDigits(
    value: number,
    intDigits: number = 2
): string {
    return value?.toLocaleString("en-US", {
        minimumIntegerDigits: intDigits,
        useGrouping: false
    });
}
