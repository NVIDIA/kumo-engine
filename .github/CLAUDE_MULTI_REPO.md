# Claude Multi-Repo Support

This document describes how to use Claude across multiple Kumo repositories.

## Overview

The Claude GitHub workflow supports making coordinated changes across repositories in the Kumo ecosystem. The Kumo API and SDK now live together in this repository, so their contract changes can be made atomically.

## Current configuration

`.github/claude-repos.json` currently lists only **kumo**. The API, SDK,
parser, core, catalog, and service code are in this repository. Use a normal
single-repository request and one Kumo PR for changes across those paths.
The keyword-triggered dispatcher below is retained for configurations that
include other repositories; with today's list, it dispatches only to Kumo.

## Usage

### Single Repository (Default)

```
@claude fix the authentication bug in server.py
```

### Multi-Repository Mode

Use one of these keywords to trigger multi-repo mode:
- `multi-repo`
- `all repos`
- `across repos`

```
@claude multi-repo: Update the API version to 2.0 in all repos
```

This will:
1. Dispatch requests to each configured repository
2. Create separate branches and PRs in each configured repository
3. Post a summary comment with links to each repo

## Examples

### Update Version Across Repos
```
@claude multi-repo: Update the API_VERSION constant from "1.9" to "2.0"
```

### Update Dependencies
```
@claude across repos: Update pydantic to version 2.5.0 in requirements.txt
```

### Add Configuration
```
@claude all repos: Add ENABLE_FEATURE_X config option with default False
```

### Rename Function
```
@claude multi-repo: Rename get_db_connection to get_database_connection
```

## Best Practices

1. **Be specific** - Include file names, exact values, and clear instructions
2. **Keep contract changes atomic** - Update the in-tree API and SDK together when their contract changes
3. **Review each PR** - Changes in each repo require independent review

## Limitations

- Each repository gets independent branches/PRs (not linked)
- Changes are not atomic - one repo might succeed while another fails
- Requires `KUMO_GITHUB_BOT_TOKEN` with write access to all repos

## Troubleshooting

**Multi-repo not triggering?**
- Check keyword is present in comment
- Verify `.github/claude-repos.json` exists
- Ensure `KUMO_GITHUB_BOT_TOKEN` has permissions

**Changes missing in some repos?**
- Check workflow runs in each repository
- Some repos might not have the referenced files

## Configuration

To add/remove repositories, edit `.github/claude-repos.json`:

```json
{
  "repositories": [
    {"name": "kumo", "owner": "kumo-ai", "default_branch": "master"}
  ],
  "multi_repo_keywords": ["multi-repo", "all repos", "across repos"]
}
```

Validate with: `bash .github/scripts/validate-claude-repos.sh`
