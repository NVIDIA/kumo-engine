// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";
import "@/grid.css";
import { cn } from "@/lib/utils";

type GutterSize = "none" | "sm" | "normal" | "lg" | "xl";
type AlignItems = "start" | "center" | "end" | "stretch" | "baseline";
type JustifyContent =
    | "start"
    | "center"
    | "end"
    | "between"
    | "around"
    | "evenly";
/**
 * Grid column configuration
 * - Numbers 1-12: Creates a grid with that many columns
 * - "auto": Creates a horizontal auto-flow layout where columns size to content
 *   (useful for toolbars, button groups, etc.)
 */
type GridCols = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | "auto";

/**
 * GridRow component - creates a CSS Grid container
 *
 * ## Usage
 * - Defaults to a 12-column grid on mobile (xs={12})
 * - Use responsive props (sm, md, lg, xl, 2xl) to change column count at breakpoints
 * - Use "auto" for horizontal layouts where columns size to content
 * - Must contain GridCol components as direct children for proper layout
 *
 * ## Working with Offsets
 * When GridCol children use offsets, the column count at each breakpoint matters:
 * - GridCol offsets use absolute column positions (col-start-*)
 * - If you change column count responsively, GridCol offsets should also adjust
 * - See GridCol documentation for detailed offset examples
 *
 * @example Basic grid
 * <GridRow xs={12} md={6}>
 *   <GridCol span={6}>Content</GridCol>
 *   <GridCol span={6}>Content</GridCol>
 * </GridRow>
 *
 * @example Grid with centered content using offsets
 * <GridRow xs={12}>
 *   <GridCol span={6} offset={3}>Centered</GridCol>
 * </GridRow>
 */
export interface GridRowProps extends React.HTMLAttributes<HTMLDivElement> {
    /**
     * Gutter size between columns
     * @default "normal"
     */
    gutter?: GutterSize;
    /**
     * Align items vertically
     */
    align?: AlignItems;
    /**
     * Justify content horizontally
     */
    justify?: JustifyContent;
    /**
     * Number of columns in the grid (base/mobile)
     * @default 12
     */
    xs?: GridCols;
    /**
     * Number of columns at sm breakpoint (640px+)
     */
    sm?: GridCols;
    /**
     * Number of columns at md breakpoint (980px+)
     */
    md?: GridCols;
    /**
     * Number of columns at lg breakpoint (1300px+)
     */
    lg?: GridCols;
    /**
     * Number of columns at xl breakpoint (1600px+)
     */
    xl?: GridCols;
    /**
     * Number of columns at 2xl breakpoint (1920px+)
     */
    "2xl"?: GridCols;
}

// Gap classes for gutter (using CSS Grid gap instead of negative margins)
const gapClasses = {
    none: "",
    sm: "gap-2",
    normal: "gap-6",
    lg: "gap-8",
    xl: "gap-12"
};

const alignClasses = {
    start: "items-start",
    center: "items-center",
    end: "items-end",
    stretch: "items-stretch",
    baseline: "items-baseline"
};

const justifyClasses = {
    start: "justify-start",
    center: "justify-center",
    end: "justify-end",
    between: "justify-between",
    around: "justify-around",
    evenly: "justify-evenly"
};

// Static grid column classes - Tailwind can see and include these
const gridColsClasses: Record<Exclude<GridCols, "auto">, string> = {
    1: "grid-cols-1",
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-4",
    5: "grid-cols-5",
    6: "grid-cols-6",
    7: "grid-cols-7",
    8: "grid-cols-8",
    9: "grid-cols-9",
    10: "grid-cols-10",
    11: "grid-cols-11",
    12: "grid-cols-12"
};

const gridColsClassesSm: Record<Exclude<GridCols, "auto">, string> = {
    1: "sm:grid-cols-1",
    2: "sm:grid-cols-2",
    3: "sm:grid-cols-3",
    4: "sm:grid-cols-4",
    5: "sm:grid-cols-5",
    6: "sm:grid-cols-6",
    7: "sm:grid-cols-7",
    8: "sm:grid-cols-8",
    9: "sm:grid-cols-9",
    10: "sm:grid-cols-10",
    11: "sm:grid-cols-11",
    12: "sm:grid-cols-12"
};

const gridColsClassesMd: Record<Exclude<GridCols, "auto">, string> = {
    1: "md:grid-cols-1",
    2: "md:grid-cols-2",
    3: "md:grid-cols-3",
    4: "md:grid-cols-4",
    5: "md:grid-cols-5",
    6: "md:grid-cols-6",
    7: "md:grid-cols-7",
    8: "md:grid-cols-8",
    9: "md:grid-cols-9",
    10: "md:grid-cols-10",
    11: "md:grid-cols-11",
    12: "md:grid-cols-12"
};

const gridColsClassesLg: Record<Exclude<GridCols, "auto">, string> = {
    1: "lg:grid-cols-1",
    2: "lg:grid-cols-2",
    3: "lg:grid-cols-3",
    4: "lg:grid-cols-4",
    5: "lg:grid-cols-5",
    6: "lg:grid-cols-6",
    7: "lg:grid-cols-7",
    8: "lg:grid-cols-8",
    9: "lg:grid-cols-9",
    10: "lg:grid-cols-10",
    11: "lg:grid-cols-11",
    12: "lg:grid-cols-12"
};

const gridColsClassesXl: Record<Exclude<GridCols, "auto">, string> = {
    1: "xl:grid-cols-1",
    2: "xl:grid-cols-2",
    3: "xl:grid-cols-3",
    4: "xl:grid-cols-4",
    5: "xl:grid-cols-5",
    6: "xl:grid-cols-6",
    7: "xl:grid-cols-7",
    8: "xl:grid-cols-8",
    9: "xl:grid-cols-9",
    10: "xl:grid-cols-10",
    11: "xl:grid-cols-11",
    12: "xl:grid-cols-12"
};

const gridColsClasses2xl: Record<Exclude<GridCols, "auto">, string> = {
    1: "2xl:grid-cols-1",
    2: "2xl:grid-cols-2",
    3: "2xl:grid-cols-3",
    4: "2xl:grid-cols-4",
    5: "2xl:grid-cols-5",
    6: "2xl:grid-cols-6",
    7: "2xl:grid-cols-7",
    8: "2xl:grid-cols-8",
    9: "2xl:grid-cols-9",
    10: "2xl:grid-cols-10",
    11: "2xl:grid-cols-11",
    12: "2xl:grid-cols-12"
};

export const GridRow = React.forwardRef<HTMLDivElement, GridRowProps>(
    (
        {
            children,
            className,
            gutter = "normal",
            align,
            justify,
            xs = 12, // Default to 12-column grid
            sm,
            md,
            lg,
            xl,
            "2xl": xxl,
            ...props
        },
        ref
    ) => {
        return (
            <div
                ref={ref}
                className={cn(
                    "grid",
                    gapClasses[gutter],
                    align && alignClasses[align],
                    justify && justifyClasses[justify],
                    // Grid columns - static classes that Tailwind can analyze
                    xs && xs !== "auto" && gridColsClasses[xs],
                    xs === "auto" && "auto-cols-auto grid-flow-col",
                    sm && sm !== "auto" && gridColsClassesSm[sm],
                    sm === "auto" && "sm:auto-cols-auto sm:grid-flow-col",
                    md && md !== "auto" && gridColsClassesMd[md],
                    md === "auto" && "md:auto-cols-auto md:grid-flow-col",
                    lg && lg !== "auto" && gridColsClassesLg[lg],
                    lg === "auto" && "lg:auto-cols-auto lg:grid-flow-col",
                    xl && xl !== "auto" && gridColsClassesXl[xl],
                    xl === "auto" && "xl:auto-cols-auto xl:grid-flow-col",
                    xxl && xxl !== "auto" && gridColsClasses2xl[xxl],
                    xxl === "auto" && "2xl:auto-cols-auto 2xl:grid-flow-col",
                    className
                )}
                {...props}>
                {children}
            </div>
        );
    }
);

GridRow.displayName = "GridRow";
