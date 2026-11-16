// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GridRow } from "@/components/blocks/layout/GridRow";
import { GridCol } from "@/components/blocks/layout/GridCol";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Title } from "@/components/ui/title";
import { Description } from "@/components/ui/description";
import { Switch } from "@/components/ui/switch";

import type { Meta, StoryObj } from "@storybook/react-vite";
import { AddCircle } from "@/icons";

// Helper component for visualizing grid columns
const GridPlaceholder = ({
    children,
    className = ""
}: {
    children?: React.ReactNode;
    className?: string;
}) => (
    <div
        className={`rounded border border-slate-300 bg-slate-100 p-4 text-center text-sm text-slate-600 ${className}`}>
        {children}
    </div>
);

// =============================================================================
// GridRow Stories
// =============================================================================

const gridRowMeta: Meta<typeof GridRow> = {
    title: "Blocks/Grid",
    component: GridRow,
    tags: ["autodocs"],
    parameters: {
        layout: "padded"
    },
    argTypes: {
        xs: {
            control: { type: "select" },
            options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "auto"],
            description: "Number of columns (base/mobile)"
        },
        sm: {
            control: { type: "select" },
            options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "auto"],
            description: "Number of columns at sm breakpoint (640px+)"
        },
        md: {
            control: { type: "select" },
            options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "auto"],
            description: "Number of columns at md breakpoint (980px+)"
        },
        lg: {
            control: { type: "select" },
            options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "auto"],
            description: "Number of columns at lg breakpoint (1300px+)"
        },
        gutter: {
            control: { type: "select" },
            options: ["none", "sm", "normal", "lg", "xl"],
            description: "Gap between columns"
        },
        align: {
            control: { type: "select" },
            options: ["start", "center", "end", "stretch", "baseline"],
            description: "Vertical alignment"
        },
        justify: {
            control: { type: "select" },
            options: ["start", "center", "end", "between", "around", "evenly"],
            description: "Horizontal justification"
        }
    }
};

export default gridRowMeta;
type GridRowStory = StoryObj<typeof GridRow>;

export const GridRowOnly: GridRowStory = {
    name: "GridRow Only (No GridCol)",
    render: () => (
        <GridRow xs={3} gutter="normal">
            <GridPlaceholder>Item 1</GridPlaceholder>
            <GridPlaceholder>Item 2</GridPlaceholder>
            <GridPlaceholder>Item 3</GridPlaceholder>
        </GridRow>
    )
};

export const GridRowAutoLayout: GridRowStory = {
    name: "GridRow Auto Layout",
    render: () => (
        <GridRow xs="auto" gutter="sm">
            <Button variant="light-outline" size="sm">
                Cancel
            </Button>
            <Button variant="primary" size="sm">
                Save
            </Button>
        </GridRow>
    )
};

export const Default: GridRowStory = {
    render: () => (
        <GridRow xs={12}>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const TwoColumns: GridRowStory = {
    render: () => (
        <GridRow xs={12}>
            <GridCol span={6}>
                <GridPlaceholder>Col 6 (50%)</GridPlaceholder>
            </GridCol>
            <GridCol span={6}>
                <GridPlaceholder>Col 6 (50%)</GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const ThreeColumns: GridRowStory = {
    render: () => (
        <GridRow xs={12}>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const FourColumns: GridRowStory = {
    render: () => (
        <GridRow xs={12}>
            <GridCol span={3}>
                <GridPlaceholder>Col 3</GridPlaceholder>
            </GridCol>
            <GridCol span={3}>
                <GridPlaceholder>Col 3</GridPlaceholder>
            </GridCol>
            <GridCol span={3}>
                <GridPlaceholder>Col 3</GridPlaceholder>
            </GridCol>
            <GridCol span={3}>
                <GridPlaceholder>Col 3</GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const UnevenColumns: GridRowStory = {
    render: () => (
        <GridRow xs={12}>
            <GridCol span={8}>
                <GridPlaceholder>Col 8 (wider)</GridPlaceholder>
            </GridCol>
            <GridCol span={4}>
                <GridPlaceholder>Col 4</GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const GutterSizes: GridRowStory = {
    render: () => (
        <div className="space-y-8">
            <div>
                <Label className="mb-2">Gutter: none</Label>
                <GridRow xs={12} gutter="none">
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Gutter: sm</Label>
                <GridRow xs={12} gutter="sm">
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Gutter: normal (default)</Label>
                <GridRow xs={12} gutter="normal">
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Gutter: lg</Label>
                <GridRow xs={12} gutter="lg">
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Gutter: xl</Label>
                <GridRow xs={12} gutter="xl">
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Col</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
        </div>
    )
};

export const VerticalAlignment: GridRowStory = {
    render: () => (
        <div className="space-y-8">
            <div>
                <Label className="mb-2">Align: start</Label>
                <GridRow xs={12} align="start" className="min-h-24 bg-slate-50">
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder className="py-8">
                            Taller content
                        </GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Align: center</Label>
                <GridRow
                    xs={12}
                    align="center"
                    className="min-h-24 bg-slate-50">
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder className="py-8">
                            Taller content
                        </GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
            <div>
                <Label className="mb-2">Align: end</Label>
                <GridRow xs={12} align="end" className="min-h-24 bg-slate-50">
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder className="py-8">
                            Taller content
                        </GridPlaceholder>
                    </GridCol>
                    <GridCol span={4}>
                        <GridPlaceholder>Short</GridPlaceholder>
                    </GridCol>
                </GridRow>
            </div>
        </div>
    )
};

// =============================================================================
// GridCol Stories
// =============================================================================

export const GridColSpans: StoryObj<typeof GridCol> = {
    render: () => (
        <div className="space-y-4">
            <GridRow xs={12}>
                <GridCol span={12}>
                    <GridPlaceholder>span=12 (full width)</GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={6}>
                    <GridPlaceholder>span=6</GridPlaceholder>
                </GridCol>
                <GridCol span={6}>
                    <GridPlaceholder>span=6</GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={4}>
                    <GridPlaceholder>span=4</GridPlaceholder>
                </GridCol>
                <GridCol span={4}>
                    <GridPlaceholder>span=4</GridPlaceholder>
                </GridCol>
                <GridCol span={4}>
                    <GridPlaceholder>span=4</GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={3}>
                    <GridPlaceholder>span=3</GridPlaceholder>
                </GridCol>
                <GridCol span={3}>
                    <GridPlaceholder>span=3</GridPlaceholder>
                </GridCol>
                <GridCol span={3}>
                    <GridPlaceholder>span=3</GridPlaceholder>
                </GridCol>
                <GridCol span={3}>
                    <GridPlaceholder>span=3</GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
                <GridCol span={2}>
                    <GridPlaceholder>2</GridPlaceholder>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const GridColOffsets: StoryObj<typeof GridCol> = {
    render: () => (
        <div className="space-y-4">
            <GridRow xs={12}>
                <GridCol span={6} offset={3}>
                    <GridPlaceholder>
                        span=6, offset=3 (centered)
                    </GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={4} offset={4}>
                    <GridPlaceholder>span=4, offset=4</GridPlaceholder>
                </GridCol>
            </GridRow>
            <GridRow xs={12}>
                <GridCol span={3}>
                    <GridPlaceholder>span=3</GridPlaceholder>
                </GridCol>
                <GridCol span={3} offset={3}>
                    <GridPlaceholder>span=3, offset=3</GridPlaceholder>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const ResponsiveSpans: StoryObj<typeof GridCol> = {
    render: () => (
        <GridRow xs={12}>
            <GridCol xs={12} md={6} lg={4}>
                <GridPlaceholder>
                    Full on mobile, half on tablet, third on desktop
                </GridPlaceholder>
            </GridCol>
            <GridCol xs={12} md={6} lg={4}>
                <GridPlaceholder>
                    Full on mobile, half on tablet, third on desktop
                </GridPlaceholder>
            </GridCol>
            <GridCol xs={12} md={12} lg={4}>
                <GridPlaceholder>
                    Full on mobile/tablet, third on desktop
                </GridPlaceholder>
            </GridCol>
        </GridRow>
    )
};

export const AllGutterSizesComparison: StoryObj<typeof GridRow> = {
    name: "Gutter Sizes Comparison",
    render: () => (
        <div className="space-y-8">
            {(["none", "sm", "normal", "lg", "xl"] as const).map((gutter) => (
                <div key={gutter}>
                    <Label className="mb-2 block">
                        gutter=&quot;{gutter}&quot;
                    </Label>
                    <GridRow xs={4} gutter={gutter}>
                        <GridCol>
                            <GridPlaceholder>1</GridPlaceholder>
                        </GridCol>
                        <GridCol>
                            <GridPlaceholder>2</GridPlaceholder>
                        </GridCol>
                        <GridCol>
                            <GridPlaceholder>3</GridPlaceholder>
                        </GridCol>
                        <GridCol>
                            <GridPlaceholder>4</GridPlaceholder>
                        </GridCol>
                    </GridRow>
                </div>
            ))}
        </div>
    )
};

// =============================================================================
// Combined Usage Stories
// =============================================================================

export const TableListHeader: StoryObj<typeof GridRow> = {
    name: "Example: Table List Header",
    render: () => (
        <div className="space-y-4">
            {/* Row 1: Title with count */}
            <GridRow xs={12}>
                <GridCol span={12}>
                    <div className="flex items-baseline gap-2">
                        <Title variant="primary-heading">Tables</Title>
                        <span className="text-xl text-slate-400">30</span>
                    </div>
                </GridCol>
            </GridRow>

            {/* Row 2: Search, Sort, and Add button */}
            <GridRow xs={12} align="center">
                <GridCol xs={12} md={4}>
                    <div className="flex w-full items-center gap-2 rounded-md border px-3 py-2">
                        <span className="text-slate-400">Q</span>
                        <span className="text-sm text-slate-400">
                            Search tables
                        </span>
                    </div>
                </GridCol>
                <GridCol xs={12} md={4}>
                    <select
                        aria-label="Sort order"
                        className="w-full rounded-md border px-3 py-2 text-sm">
                        <option>Sort by: Most recent</option>
                    </select>
                </GridCol>
                <GridCol xs={12} md={4}>
                    <div className="flex justify-end">
                        <Button variant="primary" size="sm">
                            <AddCircle />
                            Add table
                        </Button>
                    </div>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const SettingsPanel: StoryObj<typeof GridRow> = {
    name: "Example: Settings Panel",
    render: () => (
        <div className="max-w-2xl space-y-6">
            <GridRow xs={12} align="center">
                <GridCol span={10}>
                    <div>
                        <Label
                            htmlFor="baseline-comparison"
                            className="text-base font-semibold">
                            Baseline Comparison
                        </Label>
                        <Description variant="secondary" size="sm">
                            Compare your predictive model&apos;s performance
                            against baseline models to evaluate improvements.
                        </Description>
                    </div>
                </GridCol>
                <GridCol span={2}>
                    <div className="flex justify-end">
                        <Switch id="baseline-comparison" />
                    </div>
                </GridCol>
            </GridRow>

            <GridRow xs={12} align="center">
                <GridCol span={8}>
                    <div>
                        <Label className="text-base font-semibold">
                            Run Mode
                        </Label>
                        <Description variant="secondary" size="sm">
                            The Run Mode field is used to determine the default
                            values for your model&apos;s Model Plan based on
                            common usage patterns.
                        </Description>
                    </div>
                </GridCol>
                <GridCol span={4}>
                    <div className="flex justify-end">
                        <select
                            aria-label="Run mode"
                            className="min-w-32 rounded-md border px-4 py-2 text-sm">
                            <option>FAST</option>
                            <option>BALANCED</option>
                            <option>QUALITY</option>
                        </select>
                    </div>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const FormLayout: StoryObj<typeof GridRow> = {
    name: "Example: Form Layout",
    render: () => (
        <div className="max-w-2xl space-y-6">
            <GridRow xs={12}>
                <GridCol xs={12} md={6}>
                    <div>
                        <Label htmlFor="grid-first-name">First Name</Label>
                        <input
                            id="grid-first-name"
                            type="text"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="John"
                        />
                    </div>
                </GridCol>
                <GridCol xs={12} md={6}>
                    <div>
                        <Label htmlFor="grid-last-name">Last Name</Label>
                        <input
                            id="grid-last-name"
                            type="text"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="Doe"
                        />
                    </div>
                </GridCol>
            </GridRow>

            <GridRow xs={12}>
                <GridCol span={12}>
                    <div>
                        <Label htmlFor="grid-email">Email Address</Label>
                        <input
                            id="grid-email"
                            type="email"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="john.doe@example.com"
                        />
                    </div>
                </GridCol>
            </GridRow>

            <GridRow xs={12}>
                <GridCol xs={12} md={4}>
                    <div>
                        <Label htmlFor="grid-city">City</Label>
                        <input
                            id="grid-city"
                            type="text"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="San Francisco"
                        />
                    </div>
                </GridCol>
                <GridCol xs={12} md={4}>
                    <div>
                        <Label htmlFor="grid-state">State</Label>
                        <input
                            id="grid-state"
                            type="text"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="CA"
                        />
                    </div>
                </GridCol>
                <GridCol xs={12} md={4}>
                    <div>
                        <Label htmlFor="grid-zip">Zip Code</Label>
                        <input
                            id="grid-zip"
                            type="text"
                            className="w-full rounded-md border px-3 py-2"
                            placeholder="94102"
                        />
                    </div>
                </GridCol>
            </GridRow>

            <GridRow xs={12} justify="end">
                <GridCol span="auto">
                    <GridRow xs="auto" gutter="sm">
                        <GridCol>
                            <Button variant="light-outline" size="sm">
                                Cancel
                            </Button>
                        </GridCol>
                        <GridCol>
                            <Button variant="primary" size="sm">
                                Save
                            </Button>
                        </GridCol>
                    </GridRow>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const DashboardLayout: StoryObj<typeof GridRow> = {
    name: "Example: Dashboard Layout",
    render: () => (
        <div className="space-y-6">
            <GridRow xs={12}>
                <GridCol xs={12} lg={8}>
                    <GridPlaceholder className="min-h-64">
                        Main Chart Area
                    </GridPlaceholder>
                </GridCol>
                <GridCol xs={12} lg={4}>
                    <div className="space-y-4">
                        <GridPlaceholder className="min-h-28">
                            Stat Card 1
                        </GridPlaceholder>
                        <GridPlaceholder className="min-h-28">
                            Stat Card 2
                        </GridPlaceholder>
                    </div>
                </GridCol>
            </GridRow>

            <GridRow xs={12}>
                <GridCol xs={12} md={6} lg={3}>
                    <GridPlaceholder className="min-h-24">
                        Metric 1
                    </GridPlaceholder>
                </GridCol>
                <GridCol xs={12} md={6} lg={3}>
                    <GridPlaceholder className="min-h-24">
                        Metric 2
                    </GridPlaceholder>
                </GridCol>
                <GridCol xs={12} md={6} lg={3}>
                    <GridPlaceholder className="min-h-24">
                        Metric 3
                    </GridPlaceholder>
                </GridCol>
                <GridCol xs={12} md={6} lg={3}>
                    <GridPlaceholder className="min-h-24">
                        Metric 4
                    </GridPlaceholder>
                </GridCol>
            </GridRow>
        </div>
    )
};

export const NestedGrids: StoryObj<typeof GridRow> = {
    name: "Example: Nested Grids",
    render: () => (
        <GridRow xs={12}>
            <GridCol span={6}>
                <div className="rounded-lg border p-4">
                    <Label className="mb-4">Left Panel</Label>
                    <GridRow xs={12} gutter="sm">
                        <GridCol span={6}>
                            <GridPlaceholder>Nested 1</GridPlaceholder>
                        </GridCol>
                        <GridCol span={6}>
                            <GridPlaceholder>Nested 2</GridPlaceholder>
                        </GridCol>
                    </GridRow>
                </div>
            </GridCol>
            <GridCol span={6}>
                <div className="rounded-lg border p-4">
                    <Label className="mb-4">Right Panel</Label>
                    <GridRow xs={12} gutter="sm">
                        <GridCol span={4}>
                            <GridPlaceholder>A</GridPlaceholder>
                        </GridCol>
                        <GridCol span={4}>
                            <GridPlaceholder>B</GridPlaceholder>
                        </GridCol>
                        <GridCol span={4}>
                            <GridPlaceholder>C</GridPlaceholder>
                        </GridCol>
                    </GridRow>
                </div>
            </GridCol>
        </GridRow>
    )
};

export const TwoColumnWithSidebar: StoryObj<typeof GridRow> = {
    name: "Example: Two Column with Sidebar",
    render: () => (
        <GridRow xs={12}>
            <GridCol xs={12} lg={3}>
                <div className="space-y-4 rounded-lg border p-4">
                    <Title variant="section-header">Navigation</Title>
                    <div className="space-y-2">
                        <div className="rounded bg-slate-100 px-3 py-2 text-sm">
                            Overview
                        </div>
                        <div className="px-3 py-2 text-sm text-slate-600">
                            Training
                        </div>
                        <div className="px-3 py-2 text-sm text-slate-600">
                            Evaluation
                        </div>
                        <div className="px-3 py-2 text-sm text-slate-600">
                            Predictions
                        </div>
                    </div>
                </div>
            </GridCol>
            <GridCol xs={12} lg={9}>
                <div className="min-h-64 rounded-lg border p-4">
                    <Title variant="secondary-heading">Main Content Area</Title>
                    <Description variant="secondary" size="sm" className="mt-2">
                        This is where the main content would be displayed. The
                        sidebar takes 3/12 columns and the main content takes
                        9/12 columns on large screens. On smaller screens, both
                        stack vertically.
                    </Description>
                </div>
            </GridCol>
        </GridRow>
    )
};
