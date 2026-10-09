---
name: code-ui-catalog
description: Scaffold and implement UI catalog components following conventions. Use when creating new primitives, block components, stories, or tests in kumo-ui-catalog.
---

# Code UI Catalog

You are a UI catalog coding agent. You create production-quality, accessible React components following the CVA + Radix + Tailwind pattern.

## Invocation

```
/code-ui-catalog <task-description>
/code-ui-catalog add a new Tag component with dismissible variant
/code-ui-catalog add visual tests for Badge component
/code-ui-catalog add a Stepper block component
```

## Step 0: Load Context

From the monorepo root, read `AGENTS.md` and `packages/kumo-ui-catalog/CLAUDE.md`.
Then work from `packages/kumo-ui-catalog/`; the `src/` paths and `npm` commands
below are relative to that package directory.

## Step 1: Understand the Task

Determine the artifact type:

- **ui-primitive** — Radix-based component in `src/components/ui/`
- **block** — Composite component in `src/components/blocks/`
- **story** — Storybook story for an existing component
- **visual-test** — Visual regression test
- **icon** — New icon export
- **type** — New type definition

Search for similar existing components to match local conventions before writing code.

## Exemplar Files

Study these real files before writing new code:

| Type              | Exemplar File                                                                        | Why                                                                    |
| ----------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| UI Primitive      | `src/components/ui/badge.tsx`                                                        | Clean CVA + asChild + data-slot + ComponentProps pattern               |
| Block Component   | `src/components/blocks/data-table/data-table-pagination.tsx`                         | Explicit props interface, composes UI primitives, className support    |
| Story (with play) | `src/stories/components/blocks/DataTable.stories.tsx`                                | autodocs tag, argTypes, play function with userEvent + assertions      |
| Visual Test       | `src/stories/components/blocks/__tests__/VisualRegression/areachart.visual.test.tsx` | composeStories pattern, iterates all stories, screenshot matching      |
| Barrel Export     | `src/index.ts` lines 4-9 (blocks), 35 (UI), 179-221 (types)                          | Export grouping: blocks → UI → animations → icons → types → re-exports |

## Step 2: Check Prerequisites

1. **Check existing components**: Does something similar already exist? Search `src/components/` and `src/index.ts`.
2. **Check Radix**: Is there a Radix primitive that provides the headless behavior? Prefer Radix over custom implementations.
3. **Check color tokens**: If the component needs colors, check `src/colors.css` for existing tokens.

## Step 3: Implement

### UI Primitive

Create `src/components/ui/my-component.tsx`:

```tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const myComponentVariants = cva(
    // Base styles
    "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors",
    {
        variants: {
            variant: {
                default:
                    "bg-primary text-primary-foreground hover:bg-primary/90",
                secondary:
                    "bg-secondary text-secondary-foreground hover:bg-secondary/80",
                outline:
                    "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
                destructive:
                    "bg-destructive text-destructive-foreground hover:bg-destructive/90"
            },
            size: {
                default: "h-10 px-4 py-2",
                sm: "h-9 px-3 text-xs",
                lg: "h-11 px-8"
            }
        },
        defaultVariants: {
            variant: "default",
            size: "default"
        }
    }
);

function MyComponent({
    className,
    variant,
    size,
    asChild = false,
    ...props
}: React.ComponentProps<"div"> &
    VariantProps<typeof myComponentVariants> & {
        asChild?: boolean;
    }) {
    const Comp = asChild ? Slot : "div";
    return (
        <Comp
            data-slot="my-component"
            className={cn(
                myComponentVariants({ variant, size, className })
            )}
            {...props}
        />
    );
}

export { MyComponent, myComponentVariants };
```

**Checklist:**

- [ ] CVA with variants + defaultVariants
- [ ] `asChild` prop with Radix `Slot`
- [ ] `data-slot="kebab-case-name"` on root
- [ ] `cn()` for className merging
- [ ] `React.ComponentProps<"element">` for HTML props
- [ ] Export both component and variants
- [ ] `focus-visible:ring-[3px]` for interactive elements
- [ ] `disabled:opacity-50 disabled:pointer-events-none` for disableable elements
- [ ] `aria-invalid:ring-destructive/20` for form elements

### Block Component

Create `src/components/blocks/my-block/my-block.tsx`:

```tsx
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface MyBlockProps {
    title: string;
    data: DataItem[];
    onAction?: (item: DataItem) => void;
    className?: string;
}

function MyBlock({ title, data, onAction, className }: MyBlockProps) {
    return (
        <Card className={cn("w-full", className)}>
            <CardHeader>{title}</CardHeader>
            <CardContent>
                {data.map((item) => (
                    <Button
                        key={item.id}
                        variant="outline"
                        onClick={() => onAction?.(item)}>
                        {item.label}
                    </Button>
                ))}
            </CardContent>
        </Card>
    );
}

export { MyBlock };
export type { MyBlockProps };
```

**Checklist:**

- [ ] Explicit props interface
- [ ] `className` prop with `cn()` merge
- [ ] Import from `@/components/ui/` (not barrel)
- [ ] Export both component and props type

### Story

Create `src/stories/components/ui/MyComponent.stories.tsx`:

```tsx
import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within, fn } from "@storybook/test";
import { MyComponent } from "@/components/ui/my-component";

const meta: Meta<typeof MyComponent> = {
    title: "Components/UI/MyComponent",
    component: MyComponent,
    tags: ["autodocs"],
    parameters: { layout: "centered" },
    argTypes: {
        variant: {
            control: "select",
            options: ["default", "secondary", "outline", "destructive"]
        },
        size: {
            control: "select",
            options: ["default", "sm", "lg"]
        }
    }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        children: "My Component"
    }
};

export const Secondary: Story = {
    args: {
        children: "Secondary",
        variant: "secondary"
    }
};

export const Small: Story = {
    args: {
        children: "Small",
        size: "sm"
    }
};

export const Disabled: Story = {
    args: {
        children: "Disabled",
        disabled: true
    }
};

export const WithInteraction: Story = {
    args: {
        children: "Click me",
        onClick: fn()
    },
    play: async ({ canvasElement, args }) => {
        const canvas = within(canvasElement);
        const element = canvas.getByText("Click me");
        await userEvent.click(element);
        await expect(args.onClick).toHaveBeenCalledOnce();
    }
};
```

**Checklist:**

- [ ] `tags: ["autodocs"]`
- [ ] Title: `Components/UI/Name` or `Components/Blocks/Name`
- [ ] Default story
- [ ] Key variant stories
- [ ] Disabled state story (if applicable)
- [ ] At least one story with `play` function (for interactive components)
- [ ] Use `fn()` for callback props

### Visual Regression Test

Create `src/stories/components/__tests__/VisualRegression/my-component.visual.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render } from "vitest-browser-react";
import { composeStories } from "@storybook/react";
import * as stories from "../../ui/MyComponent.stories";

const composedStories = composeStories(stories);

describe("MyComponent Visual Regression", () => {
    Object.entries(composedStories).forEach(([storyName, Story]) => {
        it(`${storyName} story matches screenshot`, async () => {
            const { container } = await render(<Story />);
            await expect(container).toMatchScreenshot(
                `my-component-${storyName}`
            );
        });
    });
});
```

## Step 4: Wire Up

After implementing:

- [ ] Export from `src/index.ts` (component + variants for primitives, component + type for blocks)
- [ ] Export new types with `export type { ... }` in `src/index.ts`
- [ ] Place exports in the correct section (blocks first, then UI, then types)

## Step 5: Validate

```bash
# Type check
npm run type-check

# Lint + format
npm run lint && npm run format

# Unit tests
npm run test

# Visual tests (if added)
npm run test:visual

# Storybook (manual check)
npm run storybook
```

Fix any errors before considering the task complete.

## Styling Reference

### Color Tokens (from `src/colors.css`)

Use Tailwind classes that reference CSS variables:

```
Brand:    --pink, --second-pink, --primary-dark, --secondary-light
Grey:     --kumo-grey-50 through --kumo-grey-1000
Pink:     --kumo-pink-50 through --kumo-pink-950
Surface:  --surface-primary, --surface-secondary, --surface-brand-navigation-primary
Border:   --border-primary, --border-secondary
Icon:     --icon-primary, --icon-secondary
```

### Common Tailwind Patterns

```tsx
// Interactive focus
"focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"

// Disabled
"disabled:opacity-50 disabled:pointer-events-none"

// Form validation
"aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40"

// Icon sizing (in parent)
"[&_svg]:size-4 [&_svg]:shrink-0"

// Transitions
"transition-colors"  // color changes
"transition-[color,box-shadow]"  // color + shadow
```

### Formatting

- **Print width:** 80 characters
- **Tab width:** 4 spaces
- **Trailing commas:** none
- **Semicolons:** always
- **Arrow parens:** always
- Tailwind classes are auto-sorted by the Prettier plugin
