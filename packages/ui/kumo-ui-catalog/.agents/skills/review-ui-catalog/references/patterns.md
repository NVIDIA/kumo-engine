# UI Catalog Review Patterns

> Reference patterns for the review skill. Shows correct vs incorrect code.

## UI Primitive Pattern

### Correct: Badge (`src/components/ui/badge.tsx`)

```tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
    "text-xs font-medium px-2 rounded inline-flex justify-center items-center gap-0.5 overflow-hidden leading-5",
    {
        variants: {
            variant: {
                default: "bg-slate-300 text-slate-800",
                done: "bg-emerald-100 text-emerald-700",
                // ...more variants
            },
            size: {
                xs: "text-xs",
                sm: "text-sm",
                md: "text-base",
                lg: "text-lg"
            }
        },
        defaultVariants: {
            variant: "default",
            size: "md"
        }
    }
);

function Badge({
    className,
    variant,
    size,
    asChild = false,
    ...props
}: React.ComponentProps<"span"> &
    VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
    const Comp = asChild ? Slot : "span";
    return (
        <Comp
            data-slot="badge"
            className={cn(badgeVariants({ variant, size }), className)}
            {...props}
        />
    );
}

export { Badge };
```

**What to verify:**

1. `cva()` with `variants` and `defaultVariants`
2. `asChild` prop defaulting to `false`, using `Slot` when true
3. `data-slot="kebab-case"` on root element
4. `React.ComponentProps<"element">` & `VariantProps<typeof variants>`
5. `cn()` merging variants with className override
6. Exported as named export

### Wrong patterns to flag:

```tsx
// BAD: No CVA — uses inline conditionals
const className = variant === "primary" ? "bg-blue-500" : "bg-gray-500";

// BAD: No asChild — can't compose with Slot
function Tag(props: TagProps) {
    return <span {...props} />;
}

// BAD: Missing data-slot
function Tag({ className, ...props }: TagProps) {
    return <span className={cn("...", className)} {...props} />;
}

// BAD: Custom interface extending HTML
interface TagProps extends HTMLSpanElement { variant: string; }
// Should be: React.ComponentProps<"span"> & VariantProps<typeof tagVariants>

// BAD: String concatenation for className
className={`base-class ${variant === "primary" ? "bg-blue" : "bg-gray"} ${className}`}
// Should be: className={cn(tagVariants({ variant }), className)}
```

## Block Component Pattern

### Correct: DataTablePagination (`src/components/blocks/data-table/data-table-pagination.tsx`)

Key characteristics:

- Explicit `DataTablePaginationProps` interface
- Accepts `className` and merges with `cn()`
- Imports from `@/components/ui/` (NOT barrel)
- Exports component and type separately

### Wrong patterns to flag:

```tsx
// BAD: Importing from barrel inside the library
import { Button } from "@kumo-ai/kumo-ui-catalog";
// Should be: import { Button } from "@/components/ui/button";

// BAD: No className prop
function MyBlock({ data }: { data: Item[] }) { ... }
// Should accept className for composition

// BAD: Inline styles
<div style={{ padding: "16px", color: "#333" }}>
// Should use: <div className="p-4 text-foreground">
```

## Story Pattern

### Correct structure:

```tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within, fn } from "@storybook/test";

const meta: Meta<typeof Component> = {
    title: "Components/UI/ComponentName",  // or Components/Blocks/Name
    component: Component,
    tags: ["autodocs"],                     // REQUIRED
    parameters: { layout: "centered" },
    argTypes: { /* controls */ }
};

export default meta;
type Story = StoryObj<typeof meta>;

// Must have Default story
export const Default: Story = { args: { ... } };

// Interactive stories need play function
export const WithClick: Story = {
    args: { onClick: fn() },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        await userEvent.click(canvas.getByRole("button"));
        await expect(args.onClick).toHaveBeenCalled();
    }
};
```

### Wrong patterns to flag:

```tsx
// BAD: Missing autodocs tag
const meta: Meta<typeof Component> = {
    title: "Components/UI/Button",
    component: Button,
    // tags: ["autodocs"]  ← missing
};

// BAD: Wrong title path
title: "UI/Button"           // Should be: "Components/UI/Button"
title: "components/ui/Button" // Should be: "Components/UI/Button" (PascalCase)

// BAD: No play function on interactive component
export const Clickable: Story = {
    args: { onClick: fn() }
    // Missing play function to test the interaction
};
```

## Visual Regression Test Pattern

### Correct: Areachart visual test

```tsx
import { describe, it, expect } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "@/stories/components/blocks/Areachart.stories";

const composedStories = composeStories(stories);

describe("Areachart Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(`areachart-${storyName}`);
        });
    });
});
```

**Key points:**

- Import from `vitest-browser-react` (NOT `@testing-library/react`)
- Use `composeStories()` from `@storybook/react`
- Iterate all stories automatically
- Screenshot name: `kebab-component-${storyName}`

## Styling Patterns

### Correct:

```tsx
// Focus states — keyboard only
"focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"

// Disabled
"disabled:opacity-50 disabled:pointer-events-none"

// Form validation
"aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40"

// Icon sizing via parent
"[&_svg]:size-4 [&_svg]:shrink-0"

// Colors from CSS variables
"bg-surface-primary text-foreground border-border-primary"
```

### Wrong:

```tsx
// BAD: focus: instead of focus-visible:
"focus:ring-2 focus:ring-blue-500"

// BAD: Hardcoded hex
"bg-[#FF5733] text-[#333333]"

// BAD: Direct icon size
<Icon className="w-4 h-4" />  // Should be sized via parent [&_svg]

// BAD: Inline style for spacing
style={{ marginTop: "8px" }}  // Should be: className="mt-2"
```

## Export Pattern in index.ts

### Correct ordering:

```tsx
// 1. Block components
export { DataTable } from "@/components/blocks/data-table/data-table";
export { AreaChart } from "@/components/blocks/chart/area-chart";

// 2. UI primitives
export { Badge } from "@/components/ui/badge";
export { Button } from "@/components/ui/button";

// 3. Animations
export { Spinner } from "@/animations/spinner";

// 4. Icons (wildcard OK)
export * from "@/icons";

// 5. Types
export type { SelectOption } from "@/types/select";
export type { SideBarData } from "@/types/sidebar";

// 6. Re-exports from third-party
export type { ColumnDef, Row } from "@tanstack/react-table";
```
