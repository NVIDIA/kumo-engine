// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@kumo-ai/kumo-ui-catalog";

/**
 * Tailwind classes for a wrapper around a catalog `DataTable` that make the
 * table fill its flex parent's height when the current page is full, so the
 * body rows absorb the sub-row remainder left by the page-size math (a couple
 * px each — imperceptible; the header keeps its fixed height) and the table
 * reaches the bottom with pagination pinned.
 *
 * Pass `isPageFull = data.length >= pageSize`. When the page is not full (a
 * short/last page with fewer rows than fit) this returns `undefined`, leaving
 * the table content-sized with natural row height and plain space below.
 *
 * The catalog `DataTable` exposes no fill/height prop, so the stretch is done
 * via its rendered structure (positional, since the bordered container and the
 * pagination nav share the same border class):
 *   div                          <- DataTable root        ([&>div])
 *     div.border-… (1st child)    <- bordered table box     (:first-child)
 *       div                       <- inner wrapper
 *         div.overflow-auto       <- scroll area
 *           table                 <- stretched to h-full
 *     div (2nd child)             <- pagination footer (stays at the bottom)
 *
 * The wrapper must be a flex child of a flex column with a definite height for
 * `flex-1` to resolve.
 */
export const getDataTableFillClasses = (
    isPageFull: boolean
): string | undefined =>
    isPageFull
        ? cn(
              "flex min-h-0 flex-1 flex-col",
              "[&>div]:flex [&>div]:min-h-0 [&>div]:flex-1 [&>div]:flex-col",
              "[&>div>div:first-child]:flex [&>div>div:first-child]:min-h-0 [&>div>div:first-child]:flex-1 [&>div>div:first-child]:flex-col",
              "[&>div>div:first-child>div]:flex [&>div>div:first-child>div]:min-h-0 [&>div>div:first-child>div]:flex-1 [&>div>div:first-child>div]:flex-col",
              "[&>div>div:first-child>div>div]:min-h-0 [&>div>div:first-child>div>div]:flex-1",
              "[&_table]:h-full"
          )
        : undefined;
