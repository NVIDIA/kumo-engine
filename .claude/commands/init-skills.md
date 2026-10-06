---
description: Indexing `kumo-skills-catalog` for additional skills installation
---

Clone (or pull) the kumo-skills-catalog and make the sync script available.

## Steps

1. Check if `.agents/skills-catalog/` already exists:
   - If yes: `git -C .agents/skills-catalog pull`
   - If no: `git clone https://github.com/kumo-ai/kumo-skills-catalog.git .agents/skills-catalog`

2. Verify the sync script is available at `.agents/skills-catalog/sync-skills-catalog.py`

3. Run `python3 .agents/skills-catalog/sync-skills-catalog.py --list` to show available skills

4. Tell the user they can now use:
   - `python3 .agents/skills-catalog/sync-skills-catalog.py --list` — browse available skills
   - `python3 .agents/skills-catalog/sync-skills-catalog.py --add <name> ...` — install skills
   - `python3 .agents/skills-catalog/sync-skills-catalog.py --remove <name> ...` — uninstall skills
   - `python3 .agents/skills-catalog/sync-skills-catalog.py --reset` — remove all catalog skills
