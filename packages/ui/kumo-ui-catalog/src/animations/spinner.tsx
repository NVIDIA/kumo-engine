// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

type SizeType = "xs" | "sm" | "md" | "lg" | "xl" | number;

interface SpinnerProps {
    size?: SizeType;
    className?: string;
}

/**
 * Progress indication component using Tailwind CSS v4
 *
 * @param size - Size of the spinner ('xs', 'sm', 'md', 'lg', 'xl' or number in pixels)
 * @param className - Additional CSS classes to apply
 * @returns Spinner component
 */
export const Spinner: React.FC<SpinnerProps> = ({
    size = "sm",
    className = ""
}) => {
    // Convert size to tailwind classes
    const containerSizeClass = getSizeClass(size as SizeType);
    const innerSizeClass = getInnerSizeClass(size as SizeType);

    return (
        <div
            className={`flex items-center justify-center ${containerSizeClass} ${className}`}>
            <div
                className={`${innerSizeClass} border-brand-green-subtle border-t-brand-green animate-spin rounded-full border-4`}
                role="status"
                aria-label="Loading"
            />
        </div>
    );
};

/**
 * Helper function to convert SizeType to the appropriate Tailwind size class
 */
function getSizeClass(size: SizeType): string {
    if (typeof size === "number") {
        // Convert pixel values to the closest Tailwind size class
        if (size <= 4) return "h-1 w-1";
        if (size <= 8) return "h-2 w-2";
        if (size <= 12) return "h-3 w-3";
        if (size <= 16) return "h-4 w-4";
        if (size <= 20) return "h-5 w-5";
        if (size <= 24) return "h-6 w-6";
        if (size <= 28) return "h-7 w-7";
        if (size <= 32) return "h-8 w-8";
        if (size <= 36) return "h-9 w-9";
        if (size <= 40) return "h-10 w-10";
        if (size <= 48) return "h-12 w-12";
        if (size <= 56) return "h-14 w-14";
        if (size <= 64) return "h-16 w-16";
        if (size <= 80) return "h-20 w-20";
        return "h-24 w-24"; // Default for larger sizes
    }

    // Map string sizes to Tailwind classes
    switch (size) {
        case "xs":
            return "h-4 w-4";
        case "sm":
            return "h-6 w-6";
        case "md":
            return "h-8 w-8";
        case "lg":
            return "h-10 w-10";
        case "xl":
            return "h-12 w-12";
        default:
            return "h-8 w-8";
    }
}

/**
 * Helper function to get the inner spinner size class
 * Maintains the proportional relationship between container and spinner
 */
function getInnerSizeClass(size: SizeType): string {
    if (typeof size === "number") {
        return getSizeClass(size - 4);
    }

    // For named sizes, use a slightly smaller size for the inner element
    switch (size) {
        case "xs":
            return "h-3 w-3";
        case "sm":
            return "h-5 w-5";
        case "md":
            return "h-7 w-7";
        case "lg":
            return "h-9 w-9";
        case "xl":
            return "h-10 w-10";
        default:
            return "h-7 w-7";
    }
}

export default Spinner;
