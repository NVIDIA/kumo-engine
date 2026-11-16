// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

interface ParseOptions {
    millionSystem?: boolean;
    decimals?: number;
}
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
        minimumFractionDigits: FractionDigits,
        maximumFractionDigits: FractionDigits,
        useGrouping: true
    } as Intl.NumberFormatOptions);
}

// Helper function to create diagonal stripe pattern
export const createDiagonalPattern = (
    backgroundColor: string,
    lineColor: string,
    spacing: number = 16
) => {
    const canvas = document.createElement("canvas");
    canvas.width = spacing;
    canvas.height = spacing;
    const ctx = canvas.getContext("2d");

    if (!ctx) return backgroundColor;

    // Fill background
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, spacing, spacing);

    // Draw diagonal lines (right-left pattern)
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    ctx.beginPath();

    // Main diagonal line
    ctx.moveTo(0, spacing);
    ctx.lineTo(spacing, 0);

    // Additional lines to ensure pattern continuity
    ctx.moveTo(-spacing / 2, spacing / 2);
    ctx.lineTo(spacing / 2, -spacing / 2);

    ctx.moveTo(spacing / 2, spacing * 1.5);
    ctx.lineTo(spacing * 1.5, spacing / 2);

    ctx.stroke();

    return canvas;
};
