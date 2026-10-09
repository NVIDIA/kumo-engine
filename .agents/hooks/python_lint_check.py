#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code PostToolUse hook for Python linting.

The repository's `.claude/settings.json` registers this script for Claude Code
Write/Edit calls. It reads the hook event from stdin and expects the edited
path in `tool_input.file_path`. Events without an existing `.py` file are
ignored. For Python files, isort may rewrite imports and flake8 reports
violations through Claude's `hookSpecificOutput.additionalContext`. Lint
failures provide feedback but do not block the edit.

Codex does not load `.claude/settings.json`. Its `apply_patch` PostToolUse
event supplies patch text in `tool_input.command`, not `tool_input.file_path`,
so pointing a Codex hook at this script unchanged would skip those edits.
Equivalent Codex linting needs its own registration and an input adapter.
"""
import json
import os
import subprocess
import sys


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # Only process Python files
    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    # Verify file exists
    if not os.path.exists(file_path):
        sys.exit(0)

    feedback_messages = []

    # Step 1: Run isort to auto-fix import ordering
    try:
        result = subprocess.run(
            ["isort", file_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            feedback_messages.append("✓ isort: imports sorted")
        else:
            feedback_messages.append(f"⚠ isort failed: {result.stderr}")

    except FileNotFoundError:
        feedback_messages.append(
            "⚠ isort not installed. Run: pip install isort"
        )
    except subprocess.TimeoutExpired:
        feedback_messages.append("⚠ isort timed out")
    except Exception as e:
        feedback_messages.append(f"⚠ isort error: {e}")

    # Step 2: Run flake8 to check for violations
    try:
        result = subprocess.run(
            ["flake8", file_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            feedback_messages.append("✓ flake8: no violations")
        else:
            violations = result.stdout.strip()
            feedback_messages.append(f"⚠ flake8 violations:\n{violations}")

    except FileNotFoundError:
        feedback_messages.append(
            "⚠ flake8 not installed. Run: pip install flake8"
        )
    except subprocess.TimeoutExpired:
        feedback_messages.append("⚠ flake8 timed out")
    except Exception as e:
        feedback_messages.append(f"⚠ flake8 error: {e}")

    # Provide feedback to Claude
    if feedback_messages:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n".join(feedback_messages)
            }
        }
        print(json.dumps(output))

    # Exit with success (we don't block operations, just provide feedback)
    sys.exit(0)


if __name__ == "__main__":
    main()
