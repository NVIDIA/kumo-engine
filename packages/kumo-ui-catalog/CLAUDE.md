# UI Catalog — Agent Coding Guide

> `@kumo-ai/kumo-ui-catalog` — Kumo's shared component library and design system.
> Built on Radix UI + Tailwind CSS + CVA. Consumed in-tree by the Kumo UI.

## Agent Skills

| Skill                | Purpose                                                                   |
| -------------------- | ------------------------------------------------------------------------- |
| `/code-ui-catalog`   | Scaffold new components, stories, and tests following catalog conventions |
| `/review-ui-catalog` | Review PRs against catalog standards (CVA, data-slot, exports, a11y)      |

## Project Structure

| Directory                | Purpose                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| `src/components/ui/`     | ~50 primitive components (Button, Dialog, Select, etc.)           |
| `src/components/blocks/` | Composite components (DataTable, Charts, Graphs, Sidebar, Layout) |
| `src/icons/`             | 104 Material-style icons (re-exported from Lucide)                |
| `src/animations/`        | Spinner, ProgressBar, Bouncingball, AnimatedDone                  |
| `src/hooks/`             | Custom React hooks                                                |
| `src/lib/utils.ts`       | `cn()` utility (clsx + tailwind-merge)                            |
| `src/types/`             | TypeScript types (charts, graphs, sidebar, forms)                 |
| `src/stories/`           | Storybook stories + visual regression tests                       |
| `src/index.ts`           | **Barrel export** — all public components (~243 exports)          |
| `src/colors.css`         | Color design tokens (CSS variables)                               |
| `src/grid.css`           | Grid system styles                                                |

## Tech Stack

- **React 19** + **TypeScript 5.6** (strict mode)
- **Vite 6** (library mode, outputs UMD + ESM)
- **Tailwind CSS 4** with Vite plugin (no PostCSS)
- **Radix UI** primitives (headless, accessible)
- **CVA** (Class Variance Authority) for component variants
- **Storybook 10** for docs + visual testing
- **Vitest** for unit tests, **Playwright** for visual regression
- **ECharts** for charts, **@xyflow/react** for graph visualization

## Key Commands

| Command                      | Purpose                              |
| ---------------------------- | ------------------------------------ |
| `npm run dev`                | Vite dev server                      |
| `npm run build`              | Build to `dist/` (ESM + UMD + types) |
| `npm run storybook`          | Run Storybook (port 6006)            |
| `npm run test`               | Vitest unit tests                    |
| `npm run test:visual`        | Visual regression tests (browser)    |
| `npm run test:visual:update` | Update visual snapshots              |
| `npm run lint`               | ESLint fix                           |
| `npm run format`             | Prettier format                      |
| `npm run type-check`         | TypeScript check (no emit)           |
| `npm run coverage`           | Coverage report                      |

## Component Pattern (UI Primitives)

All UI primitives follow this pattern:

```tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const myComponentVariants = cva(
  "base-tailwind-classes", // base styles
  {
    variants: {
      variant: {
        default: "variant-classes",
        secondary: "variant-classes",
        destructive: "variant-classes",
      },
      size: {
        default: "size-classes",
        sm: "size-classes",
        lg: "size-classes",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

function MyComponent({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof myComponentVariants> & {
  asChild?: boolean;
}) {
  const Comp = asChild ? Slot : "div";
  return (
    <Comp
      data-slot="my-component"
      className={cn(myComponentVariants({ variant, size, className }))}
      {...props}
    />
  );
}

export { MyComponent, myComponentVariants };
```

**Key conventions:**

- `asChild` prop for polymorphic rendering via Radix `Slot`
- `data-slot="component-name"` for testing/querying
- `cn()` to merge CVA variants with className overrides
- Export both the component and its variants
- Use `React.ComponentProps<"element">` for props (not custom interfaces when extending HTML elements)
- Disabled: `disabled:opacity-50 disabled:pointer-events-none`
- Focus: `focus-visible:ring-[3px]`
- Validation: `aria-invalid:ring-destructive/20`

## Block Component Pattern (Composite)

Block components compose UI primitives for specific use cases:

```tsx
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface MyBlockProps {
  data: MyDataType;
  onAction?: () => void;
  className?: string;
}

function MyBlock({ data, onAction, className }: MyBlockProps) {
  return (
    <Card className={cn("default-styles", className)}>
      <CardContent>
        {/* Block-specific logic */}
        <Button onClick={onAction}>Action</Button>
      </CardContent>
    </Card>
  );
}

export { MyBlock };
```

**Key conventions:**

- Import from relative `@/components/ui/` paths (not the barrel)
- Accept `className` for composition
- Use `cn()` for merging defaults with overrides
- Define explicit props interfaces (not extending HTML elements)

## Story Pattern

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
    variant: { control: "select", options: ["default", "secondary"] },
    size: { control: "select", options: ["default", "sm", "lg"] },
    onClick: { action: "clicked" },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { children: "Click me", variant: "default" },
};

export const WithInteraction: Story = {
  args: { children: "Click me", onClick: fn() },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);
    await userEvent.click(canvas.getByRole("button"));
    await expect(args.onClick).toHaveBeenCalled();
  },
};
```

**Story conventions:**

- Title path: `Components/UI/Name` or `Components/Blocks/Name`
- Always include `tags: ["autodocs"]`
- Use `play` functions for interaction testing
- Use `fn()` for action callbacks
- Stories in `src/stories/components/{ui,blocks}/`

## Visual Regression Tests

```tsx
import { composeStories } from "@storybook/react-vite";
import { render } from "@testing-library/react";
import * as stories from "../stories/components/ui/MyComponent.stories";

const composedStories = composeStories(stories);

describe("MyComponent Visual Regression", () => {
  Object.entries(composedStories).forEach(([name, Story]) => {
    it(`${name} matches screenshot`, async () => {
      const { container } = await render(<Story />);
      await expect(container).toMatchScreenshot(`my-component-${name}`);
    });
  });
});
```

## Adding a New Component

### UI Primitive

1. Create `src/components/ui/my-component.tsx` following the CVA pattern above
2. Export from `src/index.ts`:
    ```tsx
    export { MyComponent, myComponentVariants } from "@/components/ui/my-component";
    ```
3. Create story: `src/stories/components/ui/MyComponent.stories.tsx`
4. Create visual test: `src/stories/components/__tests__/my-component.visual.test.tsx`
5. Run `npm run type-check && npm run test`

### Block Component

1. Create `src/components/blocks/my-block/my-block.tsx`
2. Export from `src/index.ts`
3. Create story: `src/stories/components/blocks/MyBlock.stories.tsx`
4. Export any new types from `src/types/` and `src/index.ts`

## Styling Conventions

- **Tailwind CSS 4** — use utility classes, not custom CSS
- **CSS variables** for colors — defined in `src/colors.css`
- **Dark mode** via `dark:` modifier — CSS variables adapt automatically
- **No inline styles** — use Tailwind or `cn()` with CVA
- **Prettier plugin** auto-sorts Tailwind classes
- **Icons** sized with `[&_svg]:size-N` pattern inside components

## Integration with Kumo UI

The main consumer is `kumo/ui`. For local development:

```bash
# In kumo/ui — run with local catalog hot-reload
npm run local:catalog

# Or via Docker from kumo root
UI_CATALOG_DEV=1 make devserver
```

For the current in-tree dependency and application commands, follow the
[repository UI guide](../../docs/development/ui.md#commands).

## Formatting & Linting

- **Prettier:** 80 char width, 4-space indent, trailing comma off, tailwindcss plugin
- **ESLint:** flat config, react-hooks enforced, react-refresh warn
- **Pre-commit:** Husky + lint-staged (format + lint on staged files)
- **Node:** 22.x required
