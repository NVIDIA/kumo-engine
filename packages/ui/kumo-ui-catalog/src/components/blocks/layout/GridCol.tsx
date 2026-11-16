// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import "@/grid.css";
import { cn } from "@/lib/utils";

/**
 * GridCol component - must be a direct child of GridRow
 *
 * ## Basic Usage
 * - If no span props are provided, defaults to col-span-1
 * - Use "auto" to size column to its content
 * - Spans are relative to the parent GridRow's column count at each breakpoint
 *
 * ## Offsets
 * Offsets use CSS Grid's `col-start` property to skip columns. When using offsets:
 * - **IMPORTANT**: offset + span must not exceed parent's column count
 * - Offsets are absolute positions, so adjust them at each responsive breakpoint
 * - Parent GridRow must have explicit column count (xs, sm, md, etc.)
 *
 * @example Basic spans
 * <GridRow xs={12}>
 *   <GridCol span={6}>50% width</GridCol>
 *   <GridCol span={6}>50% width</GridCol>
 * </GridRow>
 *
 * @example Responsive spans
 * <GridRow xs={12} md={6}>
 *   <GridCol xs={12} md={3}>Full on mobile, half on tablet</GridCol>
 *   <GridCol xs={12} md={3}>Full on mobile, half on tablet</GridCol>
 * </GridRow>
 *
 * @example Using offsets correctly with responsive grids
 * // ✓ CORRECT: Adjust offsets at each breakpoint
 * <GridRow xs={12} md={6}>
 *   <GridCol
 *     xs={6} offset={3}      // 12 cols: skip 3, use 6 = 9 total ✓
 *     md={3} offsetMd={1}    // 6 cols: skip 1, use 3 = 4 total ✓
 *   >
 *     Centered content
 *   </GridCol>
 * </GridRow>
 *
 * @example WRONG: Static offset with responsive grid
 * // ✗ WRONG: Will overflow at md breakpoint
 * <GridRow xs={12} md={6}>
 *   <GridCol span={6} offset={3}>
 *     // Mobile (12 cols): offset=3 + span=6 = 9 ✓ Works
 *     // Tablet (6 cols): offset=3 + span=6 = 9 ✗ Exceeds 6 columns!
 *   </GridCol>
 * </GridRow>
 */
export interface GridColProps extends React.HTMLAttributes<HTMLDivElement> {
    /**
     * Column span (1-12) - sets width as fraction of parent's grid columns
     * @default 1 (if no responsive spans are set)
     */
    span?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for extra small screens (xs: <640px, default/mobile)
     * Same as 'span' prop, but more explicit
     */
    xs?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for small screens (sm: 640px+)
     */
    sm?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for medium screens (md: 980px+)
     */
    md?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for large screens (lg: 1300px+)
     */
    lg?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for extra large screens (xl: 1600px+)
     */
    xl?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column span for 2xl screens (2xl: 1920px+)
     */
    "2xl"?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";
    /**
     * Column offset (base/mobile) - skips N columns before placing this column.
     * Uses CSS Grid `col-start`, so offset + span must not exceed parent's column count.
     * When using responsive grids, adjust offset at each breakpoint where columns change.
     */
    offset?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for extra small screens (same as 'offset')
     */
    offsetXs?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for small screens (sm: 640px+)
     */
    offsetSm?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for medium screens (md: 980px+)
     */
    offsetMd?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for large screens (lg: 1300px+)
     */
    offsetLg?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for extra large screens (xl: 1600px+)
     */
    offsetXl?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Column offset for 2xl screens (2xl: 1920px+)
     */
    offset2xl?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
    /**
     * Order of the column
     */
    order?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "first" | "last";
}

const colSpanClasses: Record<number | "auto", string> = {
    auto: "col-auto",
    1: "col-span-1",
    2: "col-span-2",
    3: "col-span-3",
    4: "col-span-4",
    5: "col-span-5",
    6: "col-span-6",
    7: "col-span-7",
    8: "col-span-8",
    9: "col-span-9",
    10: "col-span-10",
    11: "col-span-11",
    12: "col-span-12"
};

const colSpanClassesSm: Record<number | "auto", string> = {
    auto: "sm:col-auto",
    1: "sm:col-span-1",
    2: "sm:col-span-2",
    3: "sm:col-span-3",
    4: "sm:col-span-4",
    5: "sm:col-span-5",
    6: "sm:col-span-6",
    7: "sm:col-span-7",
    8: "sm:col-span-8",
    9: "sm:col-span-9",
    10: "sm:col-span-10",
    11: "sm:col-span-11",
    12: "sm:col-span-12"
};

const colSpanClassesMd: Record<number | "auto", string> = {
    auto: "md:col-auto",
    1: "md:col-span-1",
    2: "md:col-span-2",
    3: "md:col-span-3",
    4: "md:col-span-4",
    5: "md:col-span-5",
    6: "md:col-span-6",
    7: "md:col-span-7",
    8: "md:col-span-8",
    9: "md:col-span-9",
    10: "md:col-span-10",
    11: "md:col-span-11",
    12: "md:col-span-12"
};

const colSpanClassesLg: Record<number | "auto", string> = {
    auto: "lg:col-auto",
    1: "lg:col-span-1",
    2: "lg:col-span-2",
    3: "lg:col-span-3",
    4: "lg:col-span-4",
    5: "lg:col-span-5",
    6: "lg:col-span-6",
    7: "lg:col-span-7",
    8: "lg:col-span-8",
    9: "lg:col-span-9",
    10: "lg:col-span-10",
    11: "lg:col-span-11",
    12: "lg:col-span-12"
};

const colSpanClassesXl: Record<number | "auto", string> = {
    auto: "xl:col-auto",
    1: "xl:col-span-1",
    2: "xl:col-span-2",
    3: "xl:col-span-3",
    4: "xl:col-span-4",
    5: "xl:col-span-5",
    6: "xl:col-span-6",
    7: "xl:col-span-7",
    8: "xl:col-span-8",
    9: "xl:col-span-9",
    10: "xl:col-span-10",
    11: "xl:col-span-11",
    12: "xl:col-span-12"
};

const colSpanClasses2xl: Record<number | "auto", string> = {
    auto: "2xl:col-auto",
    1: "2xl:col-span-1",
    2: "2xl:col-span-2",
    3: "2xl:col-span-3",
    4: "2xl:col-span-4",
    5: "2xl:col-span-5",
    6: "2xl:col-span-6",
    7: "2xl:col-span-7",
    8: "2xl:col-span-8",
    9: "2xl:col-span-9",
    10: "2xl:col-span-10",
    11: "2xl:col-span-11",
    12: "2xl:col-span-12"
};

/**
 * Offset classes using CSS Grid's col-start property
 *
 * How offsets work:
 * - offset={1} means "skip 1 column", so start at column 2 (col-start-2)
 * - offset={3} means "skip 3 columns", so start at column 4 (col-start-4)
 *
 * IMPORTANT: When using offsets with responsive grids:
 * - Always check that offset + span doesn't exceed parent's column count
 * - Use responsive offset props (offsetSm, offsetMd, etc.) when GridRow columns change
 *
 * Example:
 *   GridRow xs={12} md={6}
 *   - Mobile: offset={3} + span={6} = 9 columns needed ✓ (fits in 12)
 *   - Tablet: offset={3} + span={6} = 9 columns needed ✗ (exceeds 6!)
 *   - Fix: Add offsetMd={1} to adjust for smaller grid
 */
const offsetClasses: Record<number, string> = {
    1: "col-start-2",
    2: "col-start-3",
    3: "col-start-4",
    4: "col-start-5",
    5: "col-start-6",
    6: "col-start-7",
    7: "col-start-8",
    8: "col-start-9",
    9: "col-start-10",
    10: "col-start-11",
    11: "col-start-12",
    12: "col-start-13"
};

const offsetClassesSm: Record<number, string> = {
    1: "sm:col-start-2",
    2: "sm:col-start-3",
    3: "sm:col-start-4",
    4: "sm:col-start-5",
    5: "sm:col-start-6",
    6: "sm:col-start-7",
    7: "sm:col-start-8",
    8: "sm:col-start-9",
    9: "sm:col-start-10",
    10: "sm:col-start-11",
    11: "sm:col-start-12",
    12: "sm:col-start-13"
};

const offsetClassesMd: Record<number, string> = {
    1: "md:col-start-2",
    2: "md:col-start-3",
    3: "md:col-start-4",
    4: "md:col-start-5",
    5: "md:col-start-6",
    6: "md:col-start-7",
    7: "md:col-start-8",
    8: "md:col-start-9",
    9: "md:col-start-10",
    10: "md:col-start-11",
    11: "md:col-start-12",
    12: "md:col-start-13"
};

const offsetClassesLg: Record<number, string> = {
    1: "lg:col-start-2",
    2: "lg:col-start-3",
    3: "lg:col-start-4",
    4: "lg:col-start-5",
    5: "lg:col-start-6",
    6: "lg:col-start-7",
    7: "lg:col-start-8",
    8: "lg:col-start-9",
    9: "lg:col-start-10",
    10: "lg:col-start-11",
    11: "lg:col-start-12",
    12: "lg:col-start-13"
};

const offsetClassesXl: Record<number, string> = {
    1: "xl:col-start-2",
    2: "xl:col-start-3",
    3: "xl:col-start-4",
    4: "xl:col-start-5",
    5: "xl:col-start-6",
    6: "xl:col-start-7",
    7: "xl:col-start-8",
    8: "xl:col-start-9",
    9: "xl:col-start-10",
    10: "xl:col-start-11",
    11: "xl:col-start-12",
    12: "xl:col-start-13"
};

const offsetClasses2xl: Record<number, string> = {
    1: "2xl:col-start-2",
    2: "2xl:col-start-3",
    3: "2xl:col-start-4",
    4: "2xl:col-start-5",
    5: "2xl:col-start-6",
    6: "2xl:col-start-7",
    7: "2xl:col-start-8",
    8: "2xl:col-start-9",
    9: "2xl:col-start-10",
    10: "2xl:col-start-11",
    11: "2xl:col-start-12",
    12: "2xl:col-start-13"
};

const orderClasses: Record<number | "first" | "last", string> = {
    first: "order-first",
    last: "order-last",
    1: "order-1",
    2: "order-2",
    3: "order-3",
    4: "order-4",
    5: "order-5",
    6: "order-6",
    7: "order-7",
    8: "order-8",
    9: "order-9",
    10: "order-10",
    11: "order-11",
    12: "order-12"
};

export const GridCol = React.forwardRef<HTMLDivElement, GridColProps>(
    (
        {
            children,
            className,
            span,
            xs,
            sm,
            md,
            lg,
            xl,
            "2xl": xxl,
            offset,
            offsetXs,
            offsetSm,
            offsetMd,
            offsetLg,
            offsetXl,
            offset2xl,
            order,
            ...props
        },
        ref
    ) => {
        // xs and span both refer to the base (mobile-first) breakpoint
        const baseSpan = xs || span;
        const baseOffset = offsetXs || offset;

        // Development-mode validation: warn about potential offset issues (only once per component instance)
        const hasWarnedRef = React.useRef(false);
        React.useEffect(() => {
            if (
                process.env.NODE_ENV === "development" &&
                !hasWarnedRef.current
            ) {
                const hasResponsiveSpans = !!(sm || md || lg || xl || xxl);
                const hasResponsiveOffsets = !!(
                    offsetSm ||
                    offsetMd ||
                    offsetLg ||
                    offsetXl ||
                    offset2xl
                );
                const hasBaseOffset = !!(offset || offsetXs);

                // Warn if using offsets with responsive spans but no responsive offsets
                if (
                    hasBaseOffset &&
                    hasResponsiveSpans &&
                    !hasResponsiveOffsets
                ) {
                    console.warn(
                        `GridCol: You're using responsive spans (sm/md/lg/xl/2xl) with a static offset. ` +
                            `This may cause layout issues when the parent GridRow's column count changes. ` +
                            `Consider adding responsive offsets (offsetSm, offsetMd, etc.) to match your responsive spans. ` +
                            `See GridCol documentation for examples.`
                    );
                    hasWarnedRef.current = true;
                }
            }
        }, [
            sm,
            md,
            lg,
            xl,
            xxl,
            offset,
            offsetXs,
            offsetSm,
            offsetMd,
            offsetLg,
            offsetXl,
            offset2xl
        ]);

        return (
            <div
                ref={ref}
                className={cn(
                    // Default to col-span-1 if no span is specified at any breakpoint
                    !baseSpan &&
                        !sm &&
                        !md &&
                        !lg &&
                        !xl &&
                        !xxl &&
                        "col-span-1",
                    // Apply span classes for each breakpoint
                    baseSpan && colSpanClasses[baseSpan],
                    sm && colSpanClassesSm[sm],
                    md && colSpanClassesMd[md],
                    lg && colSpanClassesLg[lg],
                    xl && colSpanClassesXl[xl],
                    xxl && colSpanClasses2xl[xxl],
                    // Apply offset classes (uses col-start-* which requires parent to have explicit grid-cols-*)
                    // WARNING: offset + span must not exceed parent GridRow's column count at each breakpoint!
                    // Use responsive offsets (offsetSm, offsetMd, etc.) when parent columns change responsively
                    baseOffset && offsetClasses[baseOffset],
                    offsetSm && offsetClassesSm[offsetSm],
                    offsetMd && offsetClassesMd[offsetMd],
                    offsetLg && offsetClassesLg[offsetLg],
                    offsetXl && offsetClassesXl[offsetXl],
                    offset2xl && offsetClasses2xl[offset2xl],
                    // Apply order if specified
                    order && orderClasses[order],
                    className
                )}
                {...props}>
                {children}
            </div>
        );
    }
);

GridCol.displayName = "GridCol";
