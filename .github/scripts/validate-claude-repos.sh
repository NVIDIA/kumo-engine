#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate claude-repos.json configuration file

set -e

CONFIG_FILE=".github/claude-repos.json"

echo "Validating Claude multi-repo configuration..."

# Check if file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: $CONFIG_FILE not found"
    exit 1
fi

# Use Python for JSON validation (more portable than jq)
python3 << 'PYTHON_SCRIPT'
import json
import sys

config_file = ".github/claude-repos.json"

try:
    with open(config_file) as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print(f"ERROR: {config_file} is not valid JSON: {e}")
    sys.exit(1)

# Check required fields
if 'repositories' not in config:
    print("ERROR: Missing 'repositories' field")
    sys.exit(1)

if 'multi_repo_keywords' not in config:
    print("ERROR: Missing 'multi_repo_keywords' field")
    sys.exit(1)

# Validate each repository has required fields
repos = config['repositories']
print(f"Found {len(repos)} repositories")

for i, repo in enumerate(repos):
    name = repo.get('name')
    owner = repo.get('owner')
    branch = repo.get('default_branch')

    if not name:
        print(f"ERROR: Repository at index {i} missing 'name' field")
        sys.exit(1)

    if not owner:
        print(f"ERROR: Repository '{name}' missing 'owner' field")
        sys.exit(1)

    if not branch:
        print(f"ERROR: Repository '{name}' missing 'default_branch' field")
        sys.exit(1)

    print(f"  ✓ {owner}/{name} (default branch: {branch})")

# Validate keywords
keywords = config['multi_repo_keywords']
print(f"Found {len(keywords)} multi-repo keywords")

if len(keywords) < 1:
    print("ERROR: At least one multi-repo keyword is required")
    sys.exit(1)

for keyword in keywords:
    print(f"  ✓ '{keyword}'")

print()
print("✅ Configuration is valid!")
print()
print("To enable multi-repo support, ensure:")
print("1. The .github/workflows/claude.yml has been updated with multi-repo jobs")
print("2. KUMO_GITHUB_BOT_TOKEN secret has access to all configured repositories")
PYTHON_SCRIPT
