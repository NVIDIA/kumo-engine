# Agent Configuration

This directory follows the [agentskills.io](https://agentskills.io) specification for skills.

## Structure

```
.agents/
└── skills/
    ├── code-ui-catalog/
    │   └── SKILL.md
    └── review-ui-catalog/
        ├── SKILL.md
        ├── references/
        │   └── patterns.md
        └── scripts/
            └── diff-scan.sh
```

## Skills

| Skill               | Purpose                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| `code-ui-catalog`   | Scaffold new components, stories, and visual tests following catalog conventions |
| `review-ui-catalog` | Review PRs against catalog standards (CVA, Radix, exports, a11y, stories)        |
