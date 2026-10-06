# Kumo Design Tokens

> Source of truth: `packages/kumo-ui-catalog/src/kumo-ui-catalog.css`
> Figma access: If a Figma URL is provided, extract the file ID from the URL. The Figma MCP server (if configured in `.mcp.json`) handles authentication automatically.

## Color Tokens — Use These Instead of Tailwind Generics

| Purpose | CSS Variable | Value | Tailwind Class | DON'T use |
|---------|-------------|-------|----------------|-----------|
| Primary text / headings | `--title-primary` | `#000000` | `text-title-primary` | `text-slate-800`, `text-gray-900` |
| Secondary text / labels | `--title-secondary` | `#636363` | `text-title-secondary` | `text-slate-500`, `text-gray-500` |
| Primary border (buttons) | `--border-primary` | `#636363` | `border-border-primary` | `border-slate-500` |
| Secondary border (panels) | `--border-secondary` | `#cccccc` | `border-border-secondary` | `border-slate-200`, `border-slate-300`, `border-gray-300` |
| Surface background | `--surface-grey` | `#f7f7f7` | `bg-surface-grey` | `bg-slate-100`, `bg-gray-100` |
| Tooltip dark bg | `--tooltip-dark` | `#3a3a41` | `style={{ backgroundColor: "var(--tooltip-dark)" }}` | `bg-gray-800`, `bg-gray-900` |
| Icon primary | `--icon-primary` | = title-primary | `text-icon-primary` | `text-gray-700` |
| Icon secondary | `--icon-secondary` | = title-secondary | `text-icon-secondary` | `text-gray-400`, `text-gray-500` |
| Brand green (primary action) | `--brand-green` | `#76b900` | via catalog `Button` | hardcoded brand hex |
| Link in dark tooltip | — | sky-300 | `text-sky-300` | `text-blue-300` |

## Legacy Figma Color → Token Mapping

These RGB values describe the older palette; use the semantic tokens below.
Their current values are listed in the table above.

| Figma RGB | Figma Role | Design Token |
|-----------|-----------|-------------|
| `rgb(41,45,54)` | Dark text, titles | `text-title-primary` |
| `rgb(95,113,140)` | Secondary text, labels, icons | `text-title-secondary` |
| `rgb(210,220,228)` | Borders, separators | `border-border-secondary` |
| `rgb(244,246,248)` | Surface backgrounds | `bg-surface-grey` |
| `rgb(58,58,65)` | Tooltip/popover background | `var(--tooltip-dark)` |
| `rgb(229,0,118)` / `rgb(252,19,115)` | Pink accent, active states | Primary action (via catalog `Button`) |
| `rgb(255,255,255)` | White backgrounds | `bg-white` |

## Typography Tokens

| Figma | Size | Weight | Use |
|-------|------|--------|-----|
| 26px / 700 | `text-3xl` | `font-bold` | Page titles |
| 16px / 700 | `text-base` | `font-bold` | Section titles (Configurations, Graph Preview) |
| 16px / 500 | `text-base` | `font-medium` | Tab labels, field labels (primary) |
| 14px / 500 | `text-sm` | `font-medium` | Field labels (secondary), descriptions, popover text |

## Tooltip / Popover Pattern

All info-icon popovers in the app must use:

```tsx
<HoverCard>
    <HoverCardTrigger className="cursor-pointer flex items-center">
        <Info className="size-3.5 text-icon-secondary" />
    </HoverCardTrigger>
    <HoverCardContent
        arrow={false}
        className="w-72 text-white text-sm font-medium p-4 rounded space-y-2"
        style={{ backgroundColor: "var(--tooltip-dark, #3a3a41)" }}>
        <p>Description text here.</p>
        <TextLink ... className="text-sky-300" />
    </HoverCardContent>
</HoverCard>
```

- Icon: `size-3.5` (14px), `text-icon-secondary`
- Popover bg: `var(--tooltip-dark)` — NOT `bg-gray-800`
- Text: `text-white text-sm font-medium`
- Padding: `p-4` (16px)
- Gap between items: `space-y-2` (8px)
- Links: `text-sky-300`
- Border radius: `rounded` (4px)

## Tailwind v4 Gotchas

- **Arbitrary values like `text-[26px]` may not compile.** Use standard classes: `text-3xl` = 26px.
- **`[&_selector]` overrides may not work.** Use scoped `<style>` tags with `#id` selectors instead.
- **Check if a class exists before using it.** Some Tailwind classes aren't in the JIT bundle until first used. If a class doesn't apply, check compiled CSS or use inline styles.
