#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Validate and sync agent configuration with AGENTS.md documentation.

This script:
1. Auto-syncs an existing skills table in AGENTS.md from .agents/skills/ files
2. Warns about undocumented MCP servers (user adds manually)

Registered by the repository Claude Code SessionEnd configuration.
Supports both AGENTS.md and CLAUDE.md (for backward compatibility).

Run automatically via SessionEnd hook or manually.
"""

import json
import os
import re
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    if "CLAUDE_PROJECT_DIR" in os.environ:
        return Path(os.environ["CLAUDE_PROJECT_DIR"])
    return Path(__file__).parent.parent.parent


def get_skill_info(skill_file: Path) -> dict:
    """Extract name and description from a skill markdown file."""
    content = skill_file.read_text()

    # Default values
    name = skill_file.stem
    description = ""

    # Try to parse YAML frontmatter (Claude format)
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]

            # Extract name
            name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
            if name_match:
                name = name_match.group(1).strip()

            # Extract description
            desc_match = re.search(
                r"^description:\s*(.+)$", frontmatter, re.MULTILINE
            )
            if desc_match:
                description = desc_match.group(1).strip()

    # Try to parse SOP format (Universal format)
    # Look for **Trigger:** line for description
    trigger_match = re.search(r"\*\*Trigger:\*\*\s*(.+)$", content, re.MULTILINE)
    if trigger_match and not description:
        description = trigger_match.group(1).strip()

    # Look for "# Skill: Name" pattern
    skill_name_match = re.search(r"^#\s*Skill:\s*(.+)$", content, re.MULTILINE)
    if skill_name_match:
        name = skill_name_match.group(1).strip().lower().replace(" ", "-")

    return {"name": name, "description": description}


def get_defined_skills(project_root: Path) -> dict:
    """Get skill info from .agents/skills/ and .claude/commands/ directories."""
    skills = {}

    # Check .agents/skills/ first (new universal location)
    agents_skills_dir = project_root / ".agents" / "skills"
    if agents_skills_dir.exists():
        for f in (*agents_skills_dir.glob("*.md"),
                  *agents_skills_dir.glob("*/SKILL.md")):
            info = get_skill_info(f)
            skills[info["name"]] = info["description"]

    # Also check .claude/commands/ for backward compatibility
    commands_dir = project_root / ".claude" / "commands"
    if commands_dir.exists():
        for f in commands_dir.glob("*.md"):
            # Skip symlinks to avoid duplicates
            if f.is_symlink():
                continue
            info = get_skill_info(f)
            if info["name"] not in skills:
                skills[info["name"]] = info["description"]

    return skills


def get_configured_mcp_tools(project_root: Path) -> set:
    """Get MCP tool prefixes from settings.json permissions."""
    settings_file = project_root / ".claude" / "settings.json"
    if not settings_file.exists():
        return set()

    try:
        settings = json.loads(settings_file.read_text())
    except json.JSONDecodeError:
        return set()

    mcp_prefixes = set()
    permissions = settings.get("permissions", {}).get("allow", [])

    for perm in permissions:
        if perm.startswith("mcp__"):
            parts = perm.split("__")
            if len(parts) >= 2:
                # Normalize aws-knowledge-mcp-server to "aws"
                server = parts[1]
                if "aws" in server.lower():
                    mcp_prefixes.add("aws")
                else:
                    mcp_prefixes.add(server.lower())

    return mcp_prefixes


def get_documented_mcp_servers(content: str) -> set:
    """Get MCP server names documented in AGENTS.md/CLAUDE.md content."""
    documented = set()

    # Look for "### ServerName MCP" style headers
    header_pattern = r"###\s+(\w+)\s+MCP"
    matches = re.findall(header_pattern, content, re.IGNORECASE)
    documented.update(m.lower() for m in matches)

    # Also look for mcp__servername__ patterns
    tool_pattern = r"mcp__([a-z0-9-]+)__"
    matches = re.findall(tool_pattern, content, re.IGNORECASE)
    documented.update(m.lower() for m in matches)

    return documented


def generate_skills_table(skills: dict) -> str:
    """Generate markdown table for skills."""
    lines = [
        "| Skill | Purpose |",
        "|-------|---------|",
    ]

    for name in sorted(skills.keys()):
        desc = skills[name] or "No description"
        # Truncate long descriptions
        if len(desc) > 60:
            desc = desc[:57] + "..."
        lines.append(f"| `/{name}` | {desc} |")

    return "\n".join(lines)


def update_skills_table(doc_file: Path, skills: dict) -> bool:
    """Update the skills table in AGENTS.md or CLAUDE.md. Returns True if updated."""
    if not doc_file.exists():
        return False

    content = doc_file.read_text()

    # Find the "## Available Skills" or "### Available Skills" section and its table
    pattern = r"(##?#?\s*Available Skills\s*\n\s*\n)(\|[^\n]+\|\n\|[-|\s]+\|\n(?:\|[^\n]+\|\n)*)"

    new_table = generate_skills_table(skills)

    def replace_table(match):
        return match.group(1) + new_table + "\n"

    new_content, count = re.subn(pattern, replace_table, content)

    if count > 0 and new_content != content:
        doc_file.write_text(new_content)
        return True

    return False


def main():
    project_root = get_project_root()

    # Check for AGENTS.md first, then CLAUDE.md
    agents_md = project_root / "AGENTS.md"
    claude_md = project_root / "CLAUDE.md"

    doc_file = None
    if agents_md.exists():
        doc_file = agents_md
    elif claude_md.exists():
        doc_file = claude_md

    if not doc_file:
        print("⚠️  Neither AGENTS.md nor CLAUDE.md found - skipping validation")
        return 0

    print("━" * 50)
    print("  Agent Docs Consistency Check")
    print("━" * 50)
    print(f"  Using: {doc_file.name}")

    # ===== Skills: Auto-sync =====
    print("\n📋 Skills (auto-sync):")
    defined_skills = get_defined_skills(project_root)

    if not defined_skills:
        print("   No skills defined in .agents/skills/ or .claude/commands/")
    else:
        updated = update_skills_table(doc_file, defined_skills)
        if updated:
            print(f"   ✅ Skills table updated in {doc_file.name}")
            for name in sorted(defined_skills.keys()):
                print(f"      - /{name}")
        else:
            print("   Skills table unchanged (already current or no matching section)")
            for name in sorted(defined_skills.keys()):
                print(f"      - /{name}")

    # ===== MCP: Warn only =====
    print("\n🔌 MCP Servers (manual sync):")
    configured_mcp = get_configured_mcp_tools(project_root)
    content = doc_file.read_text()
    documented_mcp = get_documented_mcp_servers(content)

    has_mcp_issues = False

    if not configured_mcp:
        print("   No MCP tools configured in settings.json")
    else:
        for mcp in sorted(configured_mcp):
            if mcp in documented_mcp or any(mcp in doc for doc in documented_mcp):
                print(f"   ✅ {mcp}")
            else:
                print(f"   ⚠️  {mcp} - NOT documented in {doc_file.name}")
                has_mcp_issues = True

    print("\n" + "━" * 50)

    if has_mcp_issues:
        print("💡 To document MCP servers, add a section to {doc_file.name}:")
        print("   ### <ServerName> MCP - Use For:")
        print("   - **Feature**: `tool_name` - Description")
        print("━" * 50)
        return 1
    else:
        print("✅ All documentation is consistent!")
        print("━" * 50)
        return 0


if __name__ == "__main__":
    sys.exit(main())
