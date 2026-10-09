# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The skills cite Kumo modules and SDK names. Check each one still exists in this repo.

Module paths are written as the agent sees them in a notebook (`kumo_core.train.trainer`), not as
monorepo paths, and are resolved here against the package sources. `kumoml.*` lives in a separate
repository and is not checked.
"""
import ast
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / 'agent-skills' / 'skills'
PACKAGES = {
    'kumo.engine': REPO / 'packages' / 'kumo-engine' / 'kumo' / 'engine',
    'kumo_core': REPO / 'packages' / 'kumo-core' / 'src' / 'kumo_core',
    'kumoapi': REPO / 'packages' / 'kumo-api' / 'kumoapi',
    'kumo_snowflake': REPO / 'packages' / 'kumo-snowflake' / 'src' / 'kumo_snowflake',
}
ROOT = r'(?:kumo\.engine|kumo_core|kumoapi|kumo_snowflake)'
DOTTED = re.compile(rf'`({ROOT}(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`')
FENCE = re.compile(r'^[ \t]*```[^\n]*\n(.*?)^[ \t]*```', re.DOTALL | re.MULTILINE)
FROM_IMPORT = re.compile(rf'^[ \t]*from[ \t]+({ROOT}(?:\.\w+)*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)', re.MULTILINE)
PLAIN_IMPORT = re.compile(r'^[ \t]*import[ \t]+([^\n#]*)', re.MULTILINE)
ALIAS = re.compile(r'\bkumo\.([A-Za-z_][A-Za-z0-9_]*)')

# Names the skills say do not exist (superseded or hosted-only) and must never be emitted.
NOT_EXPORTED = {
    'ComputeType', 'Dtype', 'EdgeType', 'FilesystemArtifactLocation', 'LocalBackend', 'NodeType',
    'Session', 'Storage', 'Stype', 'create_identity', 'get_backend', 'init', 'set_backend',
    'test_identity_credentials',
}


def skill_text() -> str:
    return '\n'.join(p.read_text() for p in sorted(SKILLS.rglob('*.md')))


@cache
def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def child(path: Path, name: str) -> Path | None:
    """The submodule `name` of the package directory `path`, matched on exact case.

    `Path.is_dir()` is case-insensitive on macOS, so `Trainer` would wrongly match `trainer/`.
    """
    entries = {e.name for e in path.iterdir()}
    if name in entries and (path / name).is_dir():
        return path / name
    if f'{name}.py' in entries:
        return path / f'{name}.py'
    return None


@dataclass(frozen=True)
class Module:
    file: Path
    name: str  # dotted, as the agent writes it

    @property
    def body(self) -> list[ast.stmt]:
        return parse(self.file).body

    def absolute(self, node: ast.ImportFrom) -> str:
        """The dotted module a `from ... import` statement names, relative imports resolved."""
        if not node.level:
            return node.module or ''
        package = self.name.split('.')
        if self.file.name != '__init__.py':
            package.pop()
        package = package[:len(package) - (node.level - 1)]
        return '.'.join(package + ([node.module] if node.module else []))


def bindings(body: list[ast.stmt], attr: str) -> list[tuple[ast.stmt, str | None]]:
    """Statements that bind `attr` in a module or class body, with the imported name if any."""
    found: list[tuple[ast.stmt, str | None]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == attr:
                found.append((node, None))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == attr for t in targets):
                found.append((node, None))
        elif isinstance(node, ast.ImportFrom):
            found += [(node, a.name) for a in node.names if (a.asname or a.name) == attr]
        elif isinstance(node, ast.Import):
            found += [(node, None) for a in node.names if (a.asname or a.name).split('.')[0] == attr]
        elif isinstance(node, ast.If):
            found += bindings(node.body + node.orelse, attr)
        elif isinstance(node, ast.Try):
            found += bindings(node.body + node.orelse + node.finalbody, attr)
            for handler in node.handlers:
                found += bindings(handler.body, attr)
    return found


def walk(body: list[ast.stmt], mod: Module, attrs: list[str], owner: str,
         bases: list[ast.expr], seen: frozenset[str]) -> str | None:
    """Follow `attrs` through a module or class body, crossing re-exports and base classes.

    Walking stops, and the name is accepted, at anything the monorepo cannot answer for: a
    function, a plain assignment, a plain `import`, a name imported from outside the cited
    packages, or a base class defined elsewhere (stdlib, third party).
    """
    if not attrs:
        return None
    attr, rest = attrs[0], attrs[1:]
    found = bindings(body, attr)
    if not found:
        for base in bases:
            base = base.value if isinstance(base, ast.Subscript) else base
            if not isinstance(base, ast.Name):
                return None  # an attribute base; not followed
            if base.id in ('Generic', 'Protocol'):
                continue
            if not bindings(mod.body, base.id):
                return None  # a builtin
            if walk(mod.body, mod, [base.id] + attrs, mod.name, [], seen) is None:
                return None
        return f'{attr!r} is not defined in {owner}'
    # Prefer a real definition over an import of the same name.
    node, imported = next((f for f in found if f[1] is None), found[0])
    if isinstance(node, ast.ClassDef):
        return walk(node.body, mod, rest, f'{owner}.{attr}', node.bases, seen)
    if isinstance(node, ast.ImportFrom) and imported is not None:
        target = mod.absolute(node)
        if any(target == p or target.startswith(p + '.') for p in PACKAGES):
            return resolve('.'.join([target, imported] + rest), seen)
    return None


def resolve(dotted: str, seen: frozenset[str] = frozenset()) -> str | None:
    """Return why `dotted` does not resolve, or None if it does."""
    if dotted in seen:
        return None  # a re-export cycle; nothing further to learn
    seen = seen | {dotted}
    prefix = next(p for p in PACKAGES if dotted == p or dotted.startswith(p + '.'))
    parts = dotted[len(prefix):].lstrip('.').split('.') if dotted != prefix else []
    path, name = PACKAGES[prefix], prefix
    while parts and path.is_dir():
        nxt = child(path, parts[0])
        if nxt is None:
            break
        path, name = nxt, f'{name}.{parts.pop(0)}'
    module = path / '__init__.py' if path.is_dir() else path
    if not module.is_file():
        return f'no module at {module.relative_to(REPO)}'
    mod = Module(module, name)
    return walk(mod.body, mod, parts, name, [], seen)


def exported() -> set[str]:
    for node in parse(PACKAGES['kumo.engine'] / '__init__.py').body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError('kumo.engine.__init__ has no __all__')


def code_imports(markdown: str) -> set[str]:
    """Dotted names imported by Python statements inside fenced code blocks."""
    found = set()
    for block in FENCE.findall(markdown):
        for module, names in FROM_IMPORT.findall(block):
            for line in names.strip('()').split('\n'):
                for name in line.split('#')[0].split(','):
                    name = name.replace('(', '').replace(')', '').split(' as ')[0].strip()
                    if name.isidentifier():  # skips `*`
                        found.add(f'{module}.{name}')
        for statement in PLAIN_IMPORT.findall(block):
            for name in statement.split(','):
                name = name.split(' as ')[0].strip()
                if re.fullmatch(ROOT + r'(?:\.\w+)*', name):
                    found.add(name)
    return found


CITED = sorted(set(DOTTED.findall(skill_text())) | code_imports(skill_text()))


def test_citations_are_found() -> None:
    assert len(CITED) >= 30, CITED  # guards against the pattern silently matching nothing


@pytest.mark.parametrize('dotted', CITED)
def test_cited_module_exists(dotted: str) -> None:
    assert resolve(dotted) is None, resolve(dotted)


def test_kumo_names_match_the_public_api() -> None:
    used = set(ALIAS.findall(skill_text())) - {'engine'}
    public = exported()
    assert used - public - NOT_EXPORTED == set(), 'kumo.<name> cited but not in kumo.engine.__all__'
    assert NOT_EXPORTED & public == set(), 'a name the skills call nonexistent is now exported'
    assert NOT_EXPORTED - used == set(), 'NOT_EXPORTED lists a name the skills no longer mention'


def test_resolve_reports_missing_names() -> None:
    assert resolve('kumo_core.train.trainer.Trainer.explain') is None
    assert resolve('kumo_core.train.trainer.Trainer.no_such_method') is not None
    assert resolve('kumo_core.no_such_module') is not None
    assert resolve('kumo.engine.Trainer.fit') is None
    assert resolve('kumo.engine.Trainer.no_such_method') is not None
    assert resolve('kumo_core.Train.trainer') is not None  # wrong case must not match `train/`
    assert resolve('kumo.engine.Trainer') is None
    assert resolve('kumo_snowflake.spcs.no_such_name') is not None


def test_code_imports_are_parsed() -> None:
    sample = '''Prose: from kumoapi.nope import Ignored

```python
import kumo.engine as kumo
import os, kumo_core.train
from kumoapi.typing import Dtype, Stype  # trailing comment
from kumoapi.model_plan import (
    GNNModelPlan,  # why
    RunMode as Mode,
)
from kumoml.thing import NotChecked
from kumo_snowflake.spcs import SPCS_TOKEN_PATH
```
'''
    assert code_imports(sample) == {
        'kumo.engine', 'kumo_core.train', 'kumoapi.typing.Dtype', 'kumoapi.typing.Stype',
        'kumoapi.model_plan.GNNModelPlan', 'kumoapi.model_plan.RunMode',
        'kumo_snowflake.spcs.SPCS_TOKEN_PATH',
    }
