#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only local references and agent-interface checks; never fetch URLs.

Discover repository Markdown through Git, including untracked, unignored files.
Imported DiskGraph documentation link failures remain warnings.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


def prose(text):
    """Remove fenced examples and HTML comments while preserving line numbers."""
    text = re.sub(r'<!--.*?-->', lambda m: '\n' * m[0].count('\n'), text,
                  flags=re.DOTALL)
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append('')
        else:
            lines.append('' if fence else line)
    return '\n'.join(lines)


def anchors(text):
    result = set()
    used = Counter()
    for line in prose(text).splitlines():
        heading = re.match(r'^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?$', line)
        if heading:
            label = re.sub(r'<[^>]+>', '', heading[1])
            label = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', label)
            slug = re.sub(r'[^\w\-\s]', '', label.lower()).replace(' ', '-')
            count = used[slug]
            used[slug] += 1
            result.add(slug + (f'-{count}' if count else ''))
        result.update(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', line))
    return result


def links(text):
    """Yield inline/image and reference-definition destinations, outside code."""
    for number, line in enumerate(prose(text).splitlines(), 1):
        line = re.sub(r'`+[^`]*`+', '', line)
        # One nested pair covers ordinary Markdown paths containing parentheses.
        for match in re.finditer(
                r'!?\[[^]\n]*\]\(\s*(<[^>]+>|(?:[^\s()]|\([^()]*\))+)', line):
            yield number, match[1].strip('<>')
        definition = re.match(r'^\s*\[[^]]+\]:\s*(<[^>]+>|\S+)', line)
        if definition:
            yield number, definition[1].strip('<>')
        for match in re.finditer(r'<(?:img|a)\b[^>]*\b(?:src|href)=["\']([^"\']+)', line):
            yield number, match[1]


def markdown_files(root):
    """Include tracked and new Markdown, excluding ignored local files."""
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '--cached', '--others',
         '--exclude-standard', '-z'], check=True, capture_output=True)
    return {root / os.fsdecode(name) for name in result.stdout.split(b'\0')
            if name.lower().endswith(b'.md')
            and ((root / os.fsdecode(name)).is_file()
                 or (root / os.fsdecode(name)).is_symlink())}


def check(root):
    errors = []
    maintenance = root / 'docs/maintenance'
    try:
        contract = json.loads((maintenance / 'agent-contract.json').read_text())
        files = markdown_files(root)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return [f'Cannot read repository Markdown/agent contract: {exc}'], 0

    for path in sorted(files):
        display = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            errors.append(f'{display}: Markdown file outside repository')
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeError) as exc:
            errors.append(f'{display}: cannot read: {exc}')
            continue
        file_errors = []
        for line, target in links(text):
            try:
                parsed = urlsplit(target)
            except ValueError:
                file_errors.append(f'{display}:{line}: {target}: malformed link')
                continue
            if parsed.scheme or parsed.netloc:
                continue
            relative = unquote(parsed.path)
            dest = ((root / relative.lstrip('/')) if relative.startswith('/')
                    else (path.resolve().parent / relative) if relative else path).resolve()
            label = f'{display}:{line}: {target}'
            if not dest.is_relative_to(root):
                file_errors.append(f'{label}: outside repository')
            elif not dest.exists():
                file_errors.append(f'{label}: missing local target')
            elif parsed.fragment and dest.is_file() and dest.suffix.lower() == '.md':
                if unquote(parsed.fragment) not in anchors(dest.read_text()):
                    file_errors.append(f'{label}: missing heading/anchor')

        if path.relative_to(root).parts[:2] == ('packages', 'diskgraph'):
            for error in file_errors:
                print(f'DiskGraph documentation: {error}')
        else:
            errors.extend(file_errors)

    expected_scopes = set(contract['instruction_scopes'])
    imported_scopes = set(contract.get('imported_instruction_scopes', []))
    aliases = set(contract.get('instruction_aliases', []))
    if not aliases <= expected_scopes:
        errors.append('instruction aliases must name declared instruction scopes')
    for directory, dirs, filenames in os.walk(root, followlinks=False):
        here = Path(directory)
        if here != root and ('.git' in dirs or '.git' in filenames):
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs
                   if name not in {'.git', 'node_modules', '.venv', 'venv',
                                   '__pycache__', '.worktrees', 'worktrees'}]
        for instruction in ('AGENTS.md', 'CLAUDE.md'):
            if instruction in filenames:
                scope = here.relative_to(root).as_posix()
                if scope not in expected_scopes | imported_scopes:
                    errors.append(f'{scope}/{instruction}: undeclared instruction scope')
    for scope in sorted(expected_scopes):
        source = root / scope / 'AGENTS.md'
        adapter = root / scope / 'CLAUDE.md'
        if any(not p.resolve().is_relative_to(root) for p in (source, adapter)):
            errors.append(f'{scope}: instruction source/adapter outside repository')
            continue
        if not source.is_file():
            errors.append(f'{scope}/AGENTS.md: missing instruction source')
        if scope in aliases:
            if not adapter.is_symlink() or adapter.resolve() != source.resolve():
                errors.append(f'{scope}/CLAUDE.md: declared instruction alias must link to adjacent AGENTS.md')
        elif not adapter.is_file() or adapter.read_text() != '@AGENTS.md\n':
            errors.append(f'{scope}/CLAUDE.md: must import only adjacent AGENTS.md')
    collections = [(contract['skill_source'], contract['skill_adapter'],
                    contract.get('skill_adapter_kind', 'directory'))]
    names = {}
    entries = []
    for source_path, adapter_path, kind in collections:
        source = root / source_path
        adapter = root / adapter_path
        if kind == 'commands':
            for skill in contract.get('skill_commands', []):
                name = skill + '.md'
                target = source / skill / 'SKILL.md'
                if not target.is_file():
                    errors.append(f'{adapter_path}/{name}: canonical skill is missing')
                    continue
                command = adapter / name
                wrapper = (f'Read and follow `{target.relative_to(root)}` from the repository root.\n'
                           'Apply that procedure to this request: $ARGUMENTS\n')
                linked = command.is_symlink() and command.resolve() == target.resolve()
                imported = (not command.is_symlink() and command.is_file()
                            and command.read_text() == wrapper)
                if not linked and not imported:
                    errors.append(f'{adapter_path}/{name}: command adapter must reference its canonical skill')
        elif not source.is_dir() or not adapter.is_symlink() or adapter.resolve() != source.resolve():
            errors.append(f'{adapter_path}: skill adapter must link to its canonical skill directory')
        if not source.resolve().is_relative_to(root):
            errors.append(f'{source_path}: canonical skill source outside repository')
        else:
            entries.extend(sorted(source.glob('*/SKILL.md')))
    for entry in entries:
        if not entry.resolve().is_relative_to(root):
            errors.append(f'{entry.relative_to(root)}: skill outside repository')
            continue
        text = entry.read_text()
        header = re.match(r'^---\n(.*?)\n---(?:\n|$)', text, re.DOTALL)
        name = re.search(r'^name:\s*([a-z0-9-]+)\s*$', header[1], re.MULTILINE) if header else None
        description = re.search(r'^description:\s*\S', header[1], re.MULTILINE) if header else None
        if not name or not description:
            errors.append(f'{entry.relative_to(root)}: missing valid skill name/description')
            continue
        if name[1] in names:
            errors.append(f'duplicate skill name {name[1]}: {entry.relative_to(root)}')
        names[name[1]] = entry
        if name[1] != entry.parent.name:
            errors.append(f'{entry.relative_to(root)}: skill name differs from directory')
    if not names:
        errors.append('no canonical public skills found')
    commands = root / '.claude/commands'
    if contract.get('skill_adapter_kind') != 'commands' and commands.exists() and any(commands.iterdir()):
        errors.append('legacy Claude commands may duplicate canonical skills')
    return errors, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--check', action='store_true', help='Explicit read-only check (the default).')
    args = parser.parse_args()
    errors, count = check(args.root.resolve())
    for error in errors:
        print(error)
    if errors:
        print(f'{len(errors)} documentation/agent errors')
        return 1
    print(f'Checked {count} repository Markdown files and the agent interface (read-only).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
