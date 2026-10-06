# Agent Hooks

These hooks are registered in the repository Claude Code settings.
Their sources live here; `.claude/hooks/` links to them.

## Python Linting Hook

**File:** `python_lint_check.py`

**Trigger:** Runs automatically after `Write` or `Edit` operations on Python files

**What it does:**
1. **isort**: Automatically sorts Python imports according to PEP 8 standards
2. **flake8**: Checks for code style violations and common errors

### How it Works

When Claude Code writes or edits a Python file, this hook:
- Runs `isort` to fix import ordering automatically
- Runs `flake8` to detect violations
- Provides feedback about any issues found
- Does NOT block operations (just provides feedback)

### Example Output

```
✓ isort: imports sorted
⚠ flake8 violations:
kumo/data/session_utils.py:16:1: F401 'kumo.data.backend_utils' imported but unused
```

### Requirements

```bash
pip install isort flake8
```

---

## Documentation Validator

**File:** `validate_agent_docs.py`

**Trigger:** Runs at session end or manually

**What it does:**
1. Auto-syncs an existing “Available Skills” table in AGENTS.md/CLAUDE.md from skill definitions
2. Warns about undocumented MCP servers

The hook reads nested `*/SKILL.md` files and legacy flat skill files. It does
not search the user’s home directory for a catalog. The current root guide has no “Available
Skills” table, so its text is left unchanged; the hook still reports discovered
skills and checks configured MCP permissions against documented servers.

### Running Manually

```bash
python3 .agents/hooks/validate_agent_docs.py
```

---

## Hooks and manual scripts

| Type | Location | Execution |
| --- | --- | --- |
| Hooks | `.agents/hooks/` | Registered Claude events or manual invocation |
| Optional MCP sync | `.agents/scripts/sync-mcp.py` | Manual invocation only |

The [MCP source](../mcp-servers.json) contains only the optional Codex entry.
Running the sync script writes MCP permissions to `.claude/settings.json` and
creates `.codex/config.toml`; review those local changes before using them.

Use the [read-only repository checks](../../docs/maintenance/README.md#validation)
for links, instruction imports, and declared skill adapters. These complement
the retained session-end hook without generating documentation.

---

## See Also

- [Main Setup Guide](../README.md)
- [Claude Code Hooks Documentation](https://code.claude.com/docs/en/hooks)
- [isort Documentation](https://pycqa.github.io/isort/)
- [flake8 Documentation](https://flake8.pycqa.org/)
