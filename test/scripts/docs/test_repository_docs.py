# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral fixtures for local documentation and agent-interface validation."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CHECKER = Path(__file__).resolve().parents[3] / 'scripts/docs/check_repository_docs.py'


class RepositoryDocsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.write('AGENTS.md', '# Instructions\n[Guide](docs/guide.md#usage)\n')
        self.write('CLAUDE.md', '@AGENTS.md\n')
        self.write('docs/guide.md', '# Guide\n## Usage\n')
        self.write('.agents/skills/sample/SKILL.md',
                   '---\nname: sample\ndescription: Use for sample work.\n---\n# Sample\n')
        (self.root / '.claude').mkdir()
        (self.root / '.claude/skills').symlink_to('../.agents/skills')
        self.write('docs/maintenance/agent-contract.json', json.dumps({
            'instruction_scopes': ['.'], 'skill_adapter': '.claude/skills',
            'skill_source': '.agents/skills'}))
        self.write('docs/maintenance/documentation-policy.json', json.dumps({
            'maintained': ['AGENTS.md', 'docs/**/*.md', '.agents/**/*.md']}))

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def check(self, *args):
        return subprocess.run([sys.executable, str(CHECKER), '--root', str(self.root),
                               '--check', *args], capture_output=True, text=True)

    def test_valid_tree_is_read_only(self):
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob('*') if p.is_file()}
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_link_and_anchor_are_reported(self):
        self.write('docs/guide.md', '# Guide\n[Missing](gone.md)\n[Bad](#absent)\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('gone.md', result.stdout)
        self.assertIn('absent', result.stdout)

    def test_fenced_examples_are_not_links(self):
        self.write('docs/guide.md', '# Guide\n## Usage\n```md\n[Example](absent.md)\n```\n')
        self.assertEqual(self.check().returncode, 0)

    def test_reference_and_encoded_space_links(self):
        self.write('docs/guide.md', '# Guide\n## Usage\n[Details][d]\n[d]: <other%20guide.md#some-title>\n')
        self.write('docs/other guide.md', '# Some `title`\n')
        self.assertEqual(self.check().returncode, 0)
        (self.root / 'docs/other guide.md').unlink()
        self.assertIn('other%20guide.md', self.check().stdout)

    def test_adapter_cannot_diverge(self):
        self.write('CLAUDE.md', '@AGENTS.md\nConflicting instructions\n')
        self.assertIn('CLAUDE.md', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def test_missing_scope_source_is_reported(self):
        (self.root / 'AGENTS.md').unlink()
        self.assertIn('AGENTS.md', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def test_broken_or_external_skill_adapter_is_rejected(self):
        (self.root / '.claude/skills').unlink()
        (self.root / '.claude/skills').symlink_to('/tmp/not-repository-skills')
        self.assertIn('skill adapter', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def use_command_adapters(self):
        contract = self.root / 'docs/maintenance/agent-contract.json'
        value = json.loads(contract.read_text())
        value['skill_adapter'] = '.claude/commands'
        value['skill_adapter_kind'] = 'commands'
        value['skill_commands'] = ['sample']
        contract.write_text(json.dumps(value))
        (self.root / '.claude/skills').unlink()
        command = self.root / '.claude/commands/sample.md'
        command.parent.mkdir()
        command.symlink_to('../../.agents/skills/sample/SKILL.md')
        return command

    def test_existing_claude_commands_reference_canonical_skills(self):
        command = self.use_command_adapters()
        self.assertEqual(self.check().returncode, 0)
        command.unlink()
        command.write_text(
            'Read and follow `.agents/skills/sample/SKILL.md` from the repository root.\n'
            'Apply that procedure to this request: $ARGUMENTS\n')
        self.assertEqual(self.check().returncode, 0)
        command.write_text('A divergent second copy.\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('command adapter', result.stdout)
        command.unlink()
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('command adapter', result.stdout)

    def test_standalone_project_commands_are_allowed(self):
        self.use_command_adapters()
        self.write('.claude/commands/project-status.md', 'Summarize local changes.\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_skills_do_not_require_command_adapters(self):
        self.use_command_adapters()
        self.write('.agents/skills/extra/SKILL.md',
                   '---\nname: extra\ndescription: Another public procedure.\n---\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_declared_command_cannot_outlive_its_skill(self):
        command = self.use_command_adapters()
        command.unlink()
        command.write_text(
            'Read and follow `.agents/skills/sample/SKILL.md` from the repository root.\n'
            'Apply that procedure to this request: $ARGUMENTS\n')
        (self.root / '.agents/skills/sample/SKILL.md').unlink()
        self.write('.agents/skills/extra/SKILL.md',
                   '---\nname: extra\ndescription: Another public procedure.\n---\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sample.md: canonical skill is missing', result.stdout)

    def test_duplicate_skill_names_are_rejected(self):
        self.write('.agents/skills/other/SKILL.md',
                   '---\nname: sample\ndescription: Duplicate\n---\n')
        self.assertIn('duplicate skill', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def test_imported_instruction_scope_keeps_upstream_files_unchanged(self):
        contract = self.root / 'docs/maintenance/agent-contract.json'
        value = json.loads(contract.read_text())
        value['imported_instruction_scopes'] = ['vendor/component']
        contract.write_text(json.dumps(value))
        self.write('vendor/component/AGENTS.md', '# Upstream instructions\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write('vendor/unlisted/AGENTS.md', '# Unexpected scope\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('undeclared instruction', result.stdout)

    def test_undeclared_instruction_scope_is_reported(self):
        self.write('component/AGENTS.md', '# Hidden instruction scope\n')
        self.assertIn('undeclared instruction', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def test_new_markdown_is_checked_without_registration(self):
        self.write('new-package/docs/guide.md', '[Broken](missing.md)\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('new-package/docs/guide.md', result.stdout)
        self.assertIn('missing.md', result.stdout)

    def test_check_does_not_require_document_inventory(self):
        (self.root / 'docs/maintenance/documentation-policy.json').unlink()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ignored_local_markdown_is_not_checked(self):
        self.write('.gitignore', 'scratch/\n')
        self.write('scratch/notes.md', '[Local note](missing.md)\n')
        self.assertEqual(self.check().returncode, 0)

    def test_tracked_ignored_markdown_is_still_checked(self):
        self.write('.gitignore', 'archive/\n')
        self.write('archive/guide.md', '[Broken](missing.md)\n')
        subprocess.run(['git', '-C', str(self.root), 'add', '-f',
                        'archive/guide.md'], check=True)
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('archive/guide.md', result.stdout)

    def test_markdown_symlinks_use_authored_file_location(self):
        self.write('docs/linked.md', '[Guide](guide.md#usage)\n')
        (self.root / 'alias.md').symlink_to('docs/linked.md')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write('docs/linked.md', '[Missing](absent.md)\n')
        self.assertIn('absent.md', self.check().stdout)

    def test_upstream_mirror_links_are_reported_without_blocking(self):
        self.write('packages/diskgraph/docs/guide.md', '[Upstream](missing.md)\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Upstream mirror:', result.stdout)
        self.assertIn('missing.md', result.stdout)
        self.write('packages/owned/docs/guide.md', '[Owned](missing.md)\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('packages/owned/docs/guide.md', result.stdout)

    def test_link_cannot_escape_repository(self):
        self.write('docs/guide.md', '# Guide\n## Usage\n[Outside](../../outside.md)\n')
        self.assertIn('outside repository', self.check().stdout)
        self.assertNotEqual(self.check().returncode, 0)

    def test_instruction_adapter_cannot_read_an_external_symlink(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'CLAUDE.md'
            target.write_text('@AGENTS.md\n')
            (self.root / 'CLAUDE.md').unlink()
            (self.root / 'CLAUDE.md').symlink_to(target)
            result = self.check()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('outside repository', result.stdout)

    def test_skill_source_cannot_escape_repository(self):
        import shutil
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / 'sample').mkdir()
            (target / 'sample/SKILL.md').write_text(
                '---\nname: sample\ndescription: Example\n---\n')
            shutil.rmtree(self.root / '.agents/skills')
            (self.root / '.agents/skills').symlink_to(target)
            result = self.check()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('outside repository', result.stdout)

    def test_malformed_link_reports_diagnostic_instead_of_traceback(self):
        self.write('docs/guide.md', '# Guide\n## Usage\n[Broken](https://[bad)\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('malformed link', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_nested_worktrees_do_not_add_instruction_scopes(self):
        self.write('.gitignore', '.worktrees/\nworktrees/\n')
        for folder in ('.worktrees', 'worktrees'):
            self.write(f'{folder}/feature/AGENTS.md', '# Other checkout\n')
            self.write(f'{folder}/feature/CLAUDE.md', '@AGENTS.md\n')
            self.write(f'{folder}/feature/.git', 'gitdir: /unused/metadata\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_other_nested_repository_does_not_add_instruction_scopes(self):
        self.write('scratch/checkout/.git', 'gitdir: /unused/metadata\n')
        self.write('scratch/checkout/AGENTS.md', '# Other checkout\n')
        self.write('scratch/checkout/CLAUDE.md', '@AGENTS.md\n')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_declared_instruction_alias_links_exactly_to_adjacent_source(self):
        contract = self.root / 'docs/maintenance/agent-contract.json'
        value = json.loads(contract.read_text())
        value['instruction_aliases'] = ['.']
        contract.write_text(json.dumps(value))
        adapter = self.root / 'CLAUDE.md'
        adapter.unlink()
        adapter.symlink_to('AGENTS.md')
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        adapter.unlink()
        adapter.symlink_to('docs/guide.md')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('declared instruction alias', result.stdout)


class AgentDocsHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = CHECKER.parents[2] / '.agents/hooks/validate_agent_docs.py'
        spec = importlib.util.spec_from_file_location('agent_docs_hook', source)
        self.hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.hook)
        skills = self.root / '.agents/skills'
        (skills / 'example').mkdir(parents=True)
        (skills / 'example/SKILL.md').write_text(
            '---\nname: example\ndescription: Current nested skill\n---\n')
        (skills / 'legacy.md').write_text(
            '---\nname: legacy\ndescription: Existing flat skill\n---\n')
        commands = self.root / '.claude/commands'
        commands.mkdir(parents=True)
        (commands / 'example.md').symlink_to(skills / 'example/SKILL.md')

    def test_nested_and_flat_skills_update_existing_table(self):
        skills = self.hook.get_defined_skills(self.root)
        self.assertEqual(skills, {'example': 'Current nested skill',
                                  'legacy': 'Existing flat skill'})
        guide = self.root / 'AGENTS.md'
        guide.write_text('## Available Skills\n\n| Skill | Purpose |\n'
                         '|-------|---------|\n| /old | Obsolete |\n')
        self.assertTrue(self.hook.update_skills_table(guide, skills))
        self.assertIn('`/example`', guide.read_text())
        self.assertIn('`/legacy`', guide.read_text())
        self.assertNotIn('/old', guide.read_text())

    def test_hook_preserves_guide_without_table_and_never_searches_home(self):
        guide = self.root / 'AGENTS.md'
        original = '# Agent navigation\nRead .agents/README.md for skills.\n'
        guide.write_text(original)
        with patch.dict(os.environ, {'CLAUDE_PROJECT_DIR': str(self.root)}), \
                patch.object(Path, 'home', side_effect=AssertionError(
                    'Repository hook must not search the home directory')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.hook.main(), 0)
        self.assertEqual(guide.read_text(), original)


if __name__ == '__main__':
    unittest.main()
