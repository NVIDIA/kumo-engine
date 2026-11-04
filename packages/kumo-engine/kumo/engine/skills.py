# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Install the `kumo-data-science` agent skill that ships with this wheel.

The skill teaches a coding agent to build, train and run Kumo models with
*this* SDK, so it ships inside the wheel and installs from it, matching the
installed Kumo. (Each ``kumo-engine-v*`` GitHub release also carries the same
packages as ``kumo-data-science-<host>.zip``, for installs without the wheel.)::

    import kumo.engine as kumo
    kumo.install_skills()                    # every local agent found, or Genie
    kumo.install_skills('claude')            # ~/.claude/skills/
    kumo.install_skills('codex', scope='project')   # ./.agents/skills/

or from a shell, ``python -m kumo.engine.skills [agent] [--scope ...]``. A
shell has no Snowpark session, so the ``coco`` account scopes there sign in
with a ``connections.toml`` entry, as the ``snow`` CLI does::

    python -m kumo.engine.skills coco --scope workspace --workspace MY_WS \
        --connection my_conn

One host package per agent, built from ``agent-skills/`` in the repository
(``agent-skills/scripts/package_skill.py``):

=========  =========  ==============================================================
agent      package    ``scope`` → where the skill directory goes
=========  =========  ==============================================================
claude     generic    ``user`` ``~/.claude/skills`` · ``project`` ``./.claude/skills``
codex      generic    ``user`` ``~/.agents/skills`` · ``project`` ``./.agents/skills``
coco       coco       ``user`` ``~/.snowflake/cortex/skills`` (CLI, Desktop) ·
                      ``project`` ``./.cortex/skills`` · ``workspace`` a Snowsight
                      workspace · ``extension`` a Cortex Extension in the account
genie      genie      ``/Users/<you>/.assistant/skills`` in the Databricks workspace
genie-cli  generic    as ``codex``: the Genie Code CLI reads ``.agents/skills``
=========  =========  ==============================================================

``path=`` installs a package into any skills directory instead, for an agent
not listed here. Each install replaces the ``kumo-data-science/`` directory it
writes, and only that directory: a package carries exactly one host's files,
and a second host left behind would be read too.

Versions: each package is stamped ``SKILL-BUILD: <date>-<sha>``. An install
prints the build it installed and the one it replaced. Installing again
always replaces a local or Genie copy; a Cortex Extension gets a new version
only when the package changed. ``import kumo.engine`` warns when a local
install is from another build, and ``python -m kumo.engine.skills --check``
lists the local installs and exits 1 if any is stale.

This module uses the standard library and ``filelock``, plus
``databricks-sdk`` for Genie and a Snowpark session for the ``coco`` account
scopes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from filelock import FileLock

SKILL = 'kumo-data-science'

# Agent → (host package, {scope: skills directory}).
AGENTS: dict[str, tuple[str, dict[str, str]]] = {
    'claude': ('generic', {
        'user': '~/.claude/skills',
        'project': '.claude/skills'
    }),
    'codex': ('generic', {
        'user': '~/.agents/skills',
        'project': '.agents/skills'
    }),
    'coco': ('coco', {
        'user': '~/.snowflake/cortex/skills',
        'project': '.cortex/skills'
    }),
    'genie': ('genie', {}),
    # Genie Code CLI runs locally and reaches Databricks through its shell,
    # like any coding agent, and reads the same directories as Codex. A copy
    # in ~/.genie/skills as well would load as a second, competing skill.
    'genie-cli': ('generic', {
        'user': '~/.agents/skills',
        'project': '.agents/skills'
    }),
}
COCO_ACCOUNT_SCOPES = ('workspace', 'extension')
# A home directory that marks an agent as present, for `install_skills()`.
HOME_MARKERS = [
    ('claude', '.claude'),
    ('codex', '.codex'),
    ('codex', '.agents'),
    ('coco', '.snowflake/cortex'),
    ('genie-cli', '.genie'),
]
GENIE_SKILLS = '/Users/{user}/.assistant/skills'
_MARKER = re.compile(r'SKILL-BUILD: ([^`*\s]+)')
# How long a Cortex Extension install waits for Snowflake's scan of the new
# version, which Cortex Code needs to have passed before it installs it.
SCAN_TIMEOUT = 300.0
SCAN_POLL = 5.0

_BUNDLED = Path(__file__).resolve().parent / '_skills'
# In a source checkout (an editable install) the skill is built fresh from
# agent-skills/, so an edit there is what gets installed.
_PACKAGER = (Path(__file__).resolve().parents[4] / 'agent-skills' / 'scripts' /
             'package_skill.py')


@contextmanager
def skill_package(host: str) -> Iterator[Path]:
    r"""Yield the ``kumo-data-science/`` directory of one host's package.

    Args:
        host: ``generic``, ``coco`` or ``genie``.
    """
    if _PACKAGER.is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location('package_skill',
                                                      _PACKAGER)
        assert spec and spec.loader
        package_skill = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package_skill)
        with tempfile.TemporaryDirectory() as tmp:
            pkg = package_skill.build(host, Path(tmp))
            # The check a wheel build makes, before any destination changes.
            errors = package_skill.unresolved(pkg)
            if errors:
                raise ValueError(
                    f'The {host!r} skill package has {len(errors)} '
                    f'unresolved reference(s), e.g. {errors[0]}')
            yield pkg
        return
    pkg = _BUNDLED / host / SKILL
    if not (pkg / 'SKILL.md').is_file():
        raise FileNotFoundError(
            f'This kumo-engine build carries no {host!r} skill package '
            f'(looked in {pkg}).')
    yield pkg


def _files(pkg: Path) -> list[Path]:
    return sorted(p for p in pkg.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts)


def build_of(skill_md: str) -> Optional[str]:
    r"""The ``SKILL-BUILD`` stamp of a ``SKILL.md``, or ``None``."""
    m = _MARKER.search(skill_md)
    return m.group(1) if m else None


def _read_build(skill_dir: Path) -> Optional[str]:
    try:
        return build_of((skill_dir / 'SKILL.md').read_text(encoding='utf-8'))
    except OSError:
        return None


def _report(pkg: Path, target: str, previous: Optional[str] = None,
            note: str = '') -> None:
    build = _read_build(pkg)
    if note:
        status = note
    elif previous is None:
        status = 'new'
    elif previous == build:
        status = 'reinstalled'
    else:
        status = f'replaced {previous}'
    print(f'{SKILL} {build} -> {target} ({status})')


def _install_local(pkg: Path, skills_dir: Path) -> str:
    previous = _read_build(skills_dir.expanduser() / SKILL)
    target = str(_copy(pkg, skills_dir))
    _report(pkg, target, previous)
    return target


def bundled_build() -> Optional[str]:
    r"""The build this wheel carries, or ``None`` in a source checkout."""
    if _PACKAGER.is_file():
        return None
    return _read_build(_BUNDLED / 'generic' / SKILL)


def installed_skills() -> dict[str, Optional[str]]:
    r"""The local installs of the skill, user and current project scope, and
    the build of each."""
    dirs = {
        str(Path(d).expanduser().resolve() / SKILL)
        for _, scopes in AGENTS.values()
        for d in scopes.values()
    }
    return {d: _read_build(Path(d)) for d in sorted(dirs) if Path(d).is_dir()}


def update_call(skill_dir: str) -> str:
    r"""The ``install_skills`` call that updates the install at ``skill_dir``,
    one of :func:`installed_skills`."""
    for agent, (_, scopes) in AGENTS.items():
        for scope, d in scopes.items():
            if str(Path(d).expanduser().resolve() / SKILL) == skill_dir:
                return (f"install_skills('{agent}')" if scope == 'user' else
                        f"install_skills('{agent}', scope='{scope}')")
    return f'install_skills(path={str(Path(skill_dir).parent)!r})'


def stale_skills() -> dict[str, Optional[str]]:
    r"""The local installs whose build is not this wheel's."""
    build = bundled_build()
    if build is None:
        return {}
    return {d: b for d, b in installed_skills().items() if b != build}


class SkillStaleWarning(UserWarning):
    r"""A local install of the skill is from another build than this wheel's.
    """


def warn_if_stale() -> None:
    r"""Warns, once per import of ``kumo.engine``, about local installs of the
    skill from another build than this wheel's.

    The warning is a :class:`SkillStaleWarning`. Where warnings are errors, it
    is dropped rather than raised, so it never fails the import.
    """
    try:
        stale = stale_skills()
    except Exception:  # never let a check of skills break the import
        return
    if stale:
        listed = '; '.join(f'{d} ({b}): kumo.engine.{update_call(d)}'
                           for d, b in stale.items())
        user_dirs = {
            str(Path(scopes['user']).expanduser().resolve() / SKILL)
            for _, scopes in AGENTS.values() if 'user' in scopes
        }
        shared = (' A user-scope install is shared by every Python '
                  'environment, so environments with different kumo-engine '
                  "builds replace each other's; give each project its own "
                  "with scope='project'." if user_dirs & stale.keys() else '')
        try:
            warnings.warn(
                f'The {SKILL} agent skill is not the one this kumo-engine '
                f'({bundled_build()}) ships. To update it, run: {listed}.'
                f'{shared}', SkillStaleWarning, stacklevel=2)
        except SkillStaleWarning:  # promoted to an error, e.g. by -W error
            pass


def _copy(pkg: Path, skills_dir: Path) -> Path:
    r"""Replaces ``skills_dir/kumo-data-science`` with ``pkg``.

    The package is copied in full beside the target first and then renamed
    into place, under a lock on the destination, so concurrent installs never
    mix files and a failed copy leaves the previous install intact.
    """
    root = skills_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / SKILL
    with FileLock(_lock_path(root)):
        staging = Path(tempfile.mkdtemp(prefix=f'.{SKILL}.', dir=root))
        try:
            new, old = staging / SKILL, staging / 'previous'
            shutil.copytree(pkg, new,
                            ignore=shutil.ignore_patterns('__pycache__'))
            if target.exists():
                target.rename(old)
            try:
                new.rename(target)
            except OSError:
                if old.exists():
                    old.rename(target)
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return target


def _lock_path(root: Path) -> Path:
    r"""The lock file for installs into ``root``.

    It is kept in the temporary directory, so none is left among the skills
    (or in a project's version control).
    """
    key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f'{SKILL}-{key}.lock'


def _detect() -> list[str]:
    r"""The agents to install for when none is named."""
    if os.environ.get('DATABRICKS_RUNTIME_VERSION'):
        return ['genie']
    if Path('/snowflake/session/token').exists():
        raise ValueError(
            'In a Snowflake notebook, name the destination: '
            "install_skills('coco', scope='workspace', workspace=...) for a "
            "Snowsight workspace, or install_skills('coco', "
            "scope='extension') for a Cortex Extension in this account.")
    found = [
        agent
        for agent, marker in HOME_MARKERS if (Path.home() / marker).is_dir()
    ] or ['claude', 'codex']
    # One install per directory: Codex and Genie Code CLI share theirs.
    by_dir: dict[str, str] = {}
    for agent in found:
        by_dir.setdefault(AGENTS[agent][1]['user'], agent)
    return list(by_dir.values())


def install_skills(
    agent: Optional[str] = None,
    *,
    scope: str = 'user',
    path: Optional[str | Path] = None,
    session: Any = None,
    workspace: Optional[str] = None,
    extension: str = 'KUMO_DATA_SCIENCE',
    stage: str = 'KUMO_SKILL_PKG',
    client: Any = None,
) -> list[str]:
    r"""Installs the ``kumo-data-science`` skill for a coding agent.

    The skill is the one built from the same commit as this wheel.

    Args:
        agent: ``claude``, ``codex``, ``coco``, ``genie`` (Genie Code in the
            Databricks workspace) or ``genie-cli`` (the Genie Code CLI).
            ``None`` picks Genie on Databricks; elsewhere, every agent whose
            home directory exists (``~/.claude``, ``~/.codex`` or
            ``~/.agents``, ``~/.snowflake/cortex``, ``~/.genie``), or Claude
            Code and Codex if none does. With Claude Code present, CoCo's
            ``user`` install is skipped, since Cortex Code would load the
            Claude Code copy instead.
        scope: ``user`` (the agent's home skills directory) or ``project``
            (under the current directory). For ``coco`` also ``workspace``
            and ``extension``. ``genie`` takes only ``user``: its skills
            folder is per user.
        path: a skills directory to install into instead; ``agent`` then
            picks only the package (default ``generic``).
        session: ``coco`` ``workspace``/``extension``: the Snowpark session.
            Defaults to the active one.
        workspace: ``coco`` ``workspace``: the Snowsight workspace, as
            its name as Snowsight shows it (in your personal database) or a
            full ``USER$<user>.PUBLIC."<name>"``. One created with SQL gets
            its live version added.
        extension: ``coco`` ``extension``: the Cortex Extension, created if
            missing, in the session's current database and schema unless
            qualified.
        stage: ``coco`` ``extension``: the stage the version is added from,
            created with server-side encryption if missing.
        client: ``genie``: a ``databricks.sdk.WorkspaceClient``. Defaults to
            one with default authentication (the notebook's own, inside
            Databricks).

    Each install prints the build it installed and the one it replaced.

    Returns:
        Where each install went.
    """
    if path is not None:
        host = AGENTS.get(agent or '', ('generic', ))[0]
        with skill_package(host) as pkg:
            return [_install_local(pkg, Path(path))]
    if agent is None:
        # Only coco has account scopes, so they name it; detection would try
        # every local agent first, or refuse inside a Snowflake notebook.
        names = ['coco'] if scope in COCO_ACCOUNT_SCOPES else _detect()
        # Cortex Code would load the Claude Code copy instead.
        shadowed = scope == 'user' and {'claude', 'coco'} <= set(names)
        targets = [
            target for name in names
            if not (shadowed and name == 'coco') for target in install_skills(
                name, scope=scope, session=session, workspace=workspace,
                extension=extension, stage=stage, client=client)
        ]
        if shadowed:
            print(f'{SKILL} -> {AGENTS["coco"][1]["user"]} (skipped: Cortex '
                  f'Code reads {AGENTS["claude"][1]["user"]} first; use '
                  "install_skills('coco', scope='project') or "
                  "scope='extension')")
        return targets
    if agent not in AGENTS:
        raise ValueError(
            f'Unknown agent {agent!r}; expected one of {sorted(AGENTS)}, '
            'or pass path= for any other agent.')
    host, scopes = AGENTS[agent]
    account = agent == 'coco' and scope in COCO_ACCOUNT_SCOPES
    # The arguments are checked before the package is built or a session
    # is looked for, so a missing one is the error reported.
    if agent != 'genie' and not account and scope not in scopes:
        raise ValueError(
            f'{agent}: scope must be one of '
            f'{sorted(scopes) + (list(COCO_ACCOUNT_SCOPES) if agent == "coco" else [])}'
        )
    if account and scope == 'workspace' and not workspace:
        raise ValueError("scope='workspace' needs workspace=")
    if agent == 'genie' and scope != 'user':
        raise ValueError('genie installs to your workspace skills folder and '
                         f'takes no scope (got scope={scope!r}).')
    with skill_package(host) as pkg:
        if agent == 'genie':
            return [_install_genie(pkg, client)]
        if account:
            session = session or _active_session()
            if scope == 'workspace':
                return [_install_coco_workspace(pkg, session, workspace)]
            return [_install_coco_extension(pkg, session, extension, stage)]
        target = _install_local(pkg, Path(scopes[scope]))
        if scope == 'user' and agent in ('claude', 'coco'):
            _warn_coco_shadowed()
        return [target]


def _warn_coco_shadowed() -> None:
    r"""Warns when CoCo would load the Claude Code copy.

    Cortex Code also reads ``~/.claude/skills``, and a skill there shadows the
    same name in ``~/.snowflake/cortex/skills``, so CoCo would run the
    ``generic`` package instead of its own.
    """
    claude = Path(AGENTS['claude'][1]['user']).expanduser() / SKILL
    coco = Path(AGENTS['coco'][1]['user']).expanduser() / SKILL
    if claude.is_dir() and coco.is_dir():
        warnings.warn(
            f'Cortex Code reads {claude.parent} as well as {coco.parent}, '
            f'and loads the Claude Code copy of {SKILL}, which lacks the '
            'Snowflake host guidance. For CoCo, install with '
            "scope='project' or scope='extension' instead.", stacklevel=3)


def _install_genie(pkg: Path, client: Any = None) -> str:
    r"""Uploads ``pkg`` to the user's Genie Code skills folder and reads every
    file back. A new Genie Code chat loads it.

    The Workspace API cannot rename, so the package is written over the
    previous install and only then are the files it no longer ships deleted.
    A failure part way leaves old and new files together, never an empty
    folder, and running the install again completes it.

    There is no lock, so the folder is checked again after the prune: the
    install that changes it last checks the final state against its own
    build. If every concurrent install succeeds, the folder holds exactly one
    complete build, the last one's.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ImportFormat, ObjectType

    w = client or WorkspaceClient()
    target = (f'{GENIE_SKILLS.format(user=w.current_user.me().user_name)}'
              f'/{SKILL}')
    remote = {
        f'{target}/{p.relative_to(pkg).as_posix()}': p
        for p in _files(pkg)
    }
    previous = _genie_build(w, target)
    for path, p in remote.items():
        w.workspace.mkdirs(path.rsplit('/', 1)[0])
        w.workspace.upload(path, io.BytesIO(p.read_bytes()),
                           format=ImportFormat.AUTO, overwrite=True)
    bad = _genie_unequal(w, remote)
    if bad:
        raise RuntimeError(f'{len(bad)} file(s) did not read back intact, '
                           f'e.g. {bad[0]}')
    # Take the whole inventory before deleting anything.
    stale = _genie_walk(w, target, remote, ObjectType.DIRECTORY)[1]
    for path in stale:
        w.workspace.delete(path, recursive=True)
    found, extra = _genie_walk(w, target, remote, ObjectType.DIRECTORY)
    changed = sorted(set(remote) - found) + extra or _genie_unequal(w, remote)
    if changed:
        raise RuntimeError(f'Another install changed {target} while this one '
                           f'ran (e.g. {changed[0]}); run install_skills '
                           'again.')
    _report(pkg, target, previous)
    return target


def _genie_build(w: Any, target: str) -> Optional[str]:
    r"""The build of the Genie install at ``target``, if there is one."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.workspace import ExportFormat
    try:
        md = w.workspace.download(f'{target}/SKILL.md',
                                  format=ExportFormat.AUTO).read()
    except NotFound:
        return None
    return build_of(md.decode('utf-8', 'replace'))


def _genie_unequal(w: Any, remote: dict[str, Path]) -> list[str]:
    r"""The files of ``remote`` whose workspace copy differs from the local
    one.
    """
    from databricks.sdk.service.workspace import ExportFormat
    return [
        path for path, p in remote.items() if w.workspace.download(
            path, format=ExportFormat.AUTO).read() != p.read_bytes()
    ]


def _genie_walk(w: Any, target: str, remote: dict[str, Path],
                directory: Any) -> tuple[set[str], list[str]]:
    r"""Walks the folders under ``target`` that ``remote`` ships. Returns the
    shipped files found and every entry found that is not shipped: the
    highest unshipped file or folder, empty ones too.
    """
    shipped = {target}
    for path in remote:
        while (path := path.rsplit('/', 1)[0]) not in shipped:
            shipped.add(path)
    found: set[str] = set()
    unshipped: list[str] = []
    queue = [target]
    while queue:
        for obj in w.workspace.list(queue.pop()):
            if obj.path in remote:
                found.add(obj.path)
            elif obj.path not in shipped:
                unshipped.append(obj.path)
            elif obj.object_type == directory:
                queue.append(obj.path)
    return found, unshipped


def _active_session() -> Any:
    from snowflake.snowpark.context import get_active_session
    return get_active_session()


def _put(session: Any, pkg: Path, root: str) -> None:
    r"""PUTs every file of ``pkg`` under ``root``, a stage or workspace path
    ending in ``.../kumo-data-science``.
    """
    for p in _files(pkg):
        rel = p.relative_to(pkg).parent.as_posix()
        dest = root if rel == '.' else f'{root}/{rel}'
        session.sql(
            f"PUT '{_lit('file://' + p.as_posix())}' "
            f"'{_lit(dest)}/' AUTO_COMPRESS=FALSE OVERWRITE=TRUE").collect()


def _lit(text: str) -> str:
    r"""Escapes ``text`` for a single-quoted SQL string literal."""
    return text.replace('\\', '\\\\').replace("'", "\\'")


_IDENT = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)'


def _workspace_name(session: Any, workspace: str) -> str:
    r"""The SQL name of a Snowsight workspace.

    ``DB.SCHEMA.NAME`` is used as written. Anything else is the name as
    Snowsight shows it, in the user's personal database, quoted so that case,
    dots and spaces survive.
    """
    if re.fullmatch(rf'{_IDENT}\.{_IDENT}\.{_IDENT}', workspace):
        return workspace
    user = session.sql('SELECT CURRENT_USER()').collect()[0][0]
    name = workspace.replace('"', '""')
    user = user.replace('"', '""')
    return f'"USER${user}".PUBLIC."{name}"'


def _rows(session: Any, text: str) -> list[dict[str, Any]]:
    return [r.as_dict() for r in session.sql(text).collect()]


def _install_coco_workspace(pkg: Path, session: Any, workspace: str) -> str:
    r"""Uploads ``pkg`` to a Snowsight workspace's ``.snowflake/cortex/skills``,
    the copy Cortex Code reads in that workspace.

    A workspace created with SQL has no live version and takes no files until
    one is added, so it is added here. The package is written over the
    previous install, and only then are the files it no longer ships removed.

    There is no lock, so the folder is listed again after the prune: the
    install that changes it last checks the final state against its own
    build. Workspace files cannot be read back, so each is checked by path
    and size, a file of ``s`` bytes listing as ``(s // 16 + 1) * 16``. A
    concurrent build whose files all list at the same sizes as this one's
    cannot be told apart; the last install then wins, file by file.
    """
    ws = _workspace_name(session, workspace)
    versions = _rows(session, f'SHOW VERSIONS IN WORKSPACE {ws}')
    if not any(str(v.get('is_live')).lower() == 'true' for v in versions):
        session.sql(
            f'ALTER WORKSPACE {ws} ADD LIVE VERSION FROM LAST').collect()
    root = f'snow://workspace/{ws}/versions/live/.snowflake/cortex/skills/{SKILL}'
    _put(session, pkg, root)

    # Empty `.folder` files mark folders made in Snowsight. Remove, for each
    # file not shipped, the highest folder under the root that holds nothing
    # shipped.
    sizes = {
        p.relative_to(pkg).as_posix(): (p.stat().st_size // 16 + 1) * 16
        for p in _files(pkg)
    }
    shipped = {''}
    for path in sizes:
        while (path := path.rpartition('/')[0]) not in shipped:
            shipped.add(path)
    removed: list[str] = []
    kept: set[str] = set()
    for rel in _workspace_files(session, root):
        folder = rel.rpartition('/')[0]
        if rel in sizes or _is_marker(rel) and folder in shipped:
            continue
        top = rel
        while (parent := top.rpartition('/')[0]) not in shipped:
            top = parent
        # REMOVE matches by prefix: a folder is named with its slash, and a
        # file is kept if it is a prefix of a shipped one.
        target = top + '/' if top != rel else top
        if any(target.startswith(r) for r in removed):
            continue
        if any(f != target and f.startswith(target) for f in sizes):
            kept.add(rel)
            continue
        session.sql(f"REMOVE '{_lit(root)}/{_lit(target)}'").collect()
        removed.append(target)

    found = {
        rel: size
        for rel, size in _workspace_files(session, root).items()
        if not _is_marker(rel) and rel not in kept
    }
    changed = sorted(rel for rel in sizes.keys() | found.keys()
                     if found.get(rel) != sizes.get(rel))
    if changed:
        raise RuntimeError(f'Another install changed {root} while this one '
                           f'ran (e.g. {changed[0]}); run install_skills '
                           'again.')
    _report(pkg, root, note='uploaded')
    return root


def _workspace_files(session: Any, root: str) -> dict[str, int]:
    r"""The files under ``root``, a workspace path, by their path below it,
    with the size LIST gives each.
    """
    rel_root = root.partition('/versions/live/')[2]
    files = {}
    for row in _rows(session, f"LIST '{_lit(root)}/'"):
        # LIST names each file from /versions/live/ on.
        rel = row['name'].split('/versions/live/', 1)[1]
        if rel.startswith(rel_root + '/'):
            files[rel[len(rel_root) + 1:]] = int(row['size'])
    return files


def _is_marker(rel: str) -> bool:
    return rel == '.folder' or rel.endswith('/.folder')


def _digest(pkg: Path) -> str:
    r"""A digest of the package's paths and contents."""
    h = hashlib.sha256()
    for p in _files(pkg):
        h.update(p.relative_to(pkg).as_posix().encode() + b'\0')
        h.update(p.read_bytes() + b'\0')
    return h.hexdigest()[:16]


def _extension_versions(session: Any, extension: str) -> list[dict[str, Any]]:
    try:
        return _rows(session, f'SHOW VERSIONS IN CORTEX EXTENSION {extension}')
    except Exception as e:
        if 'does not exist' in str(e):
            return []
        raise


def _install_coco_extension(pkg: Path, session: Any, extension: str,
                            stage: str) -> str:
    r"""Publishes ``pkg`` as a new version of a Cortex Extension.

    The version root holds only ``skills/kumo-data-science/``, as a Cortex
    Extension requires. Installing it for a user still happens in Cortex Code
    (``cortex skill add <extension> --catalog``); ``cortex skill check`` then
    reports later versions, and ``cortex skill update`` installs them.

    Each version is added from a stage folder named after a digest of the
    package, which Snowflake records as the version's source. When the
    default version already holds this package, nothing is published, so
    reinstalls do not pile up identical versions. A version is usable only
    once Snowflake has scanned it, so the install waits for the scan, of an
    unchanged version too.
    """
    digest = _digest(pkg)
    for v in _extension_versions(session, extension):
        source = v.get('source_location_uri') or ''
        m = re.search(rf'/(kumo-ds-{digest}-[0-9a-f]+)/', source)
        if str(v.get('is_default')).lower() == 'true' and m:
            version = _await_scan(session, extension, m.group(1))
            _report(pkg, extension, note=f'unchanged, {version}')
            return extension
    session.sql(f'CREATE STAGE IF NOT EXISTS {stage} '
                "ENCRYPTION=(TYPE='SNOWFLAKE_SSE') "
                'DIRECTORY=(ENABLE=TRUE)').collect()
    # A folder of its own per install, so concurrent installs sharing the
    # stage never publish each other's files. A version holds its own copy
    # of the files, so the folder is removed once it is added.
    folder = f'kumo-ds-{digest}-{uuid.uuid4().hex[:8]}'
    root = f'@{stage}/{folder}'
    try:
        _put(session, pkg, f'{root}/skills/{SKILL}')
        # Created from the package, so its first version is not empty.
        session.sql(f'CREATE CORTEX EXTENSION IF NOT EXISTS {extension} '
                    f'FROM {root}/').collect()
        if not _our_version(session, extension, folder):
            session.sql(f'ALTER CORTEX EXTENSION {extension} '
                        f'ADD VERSION FROM {root}/').collect()
    finally:
        session.sql(f'REMOVE {root}/').collect()
    version = _await_scan(session, extension, folder)
    _report(pkg, extension, note=version)
    return extension


def _connect(connection: str) -> Any:
    r"""A Snowpark session for a ``connections.toml`` entry, as the ``snow``
    CLI uses.
    """
    from snowflake.snowpark import Session
    return Session.builder.config('connection_name', connection).create()


def _our_version(session: Any, extension: str,
                 folder: str) -> Optional[dict[str, Any]]:
    for v in _extension_versions(session, extension):
        if f'/{folder}/' in (v.get('source_location_uri') or ''):
            return v
    return None


def _await_scan(session: Any, extension: str, folder: str) -> str:
    r"""Waits for Snowflake's scan of the version added from ``folder``.
    Returns its name, and a note if the scan is still running."""
    deadline = time.monotonic() + SCAN_TIMEOUT
    while True:
        v = _our_version(session, extension, folder)
        if v is None:
            raise RuntimeError(f'{extension} has no version from {folder}.')
        status = str(v.get('effective_status') or v.get('scan_status') or '')
        if 'FAIL' in status or 'BLOCK' in status:
            raise RuntimeError(
                f'Snowflake rejected {extension} {v["name"]}: {status}, '
                f'{v.get("scan_findings")}')
        if status != 'SCANNING':
            return str(v['name'])
        if time.monotonic() >= deadline:
            warnings.warn(
                f'{extension} {v["name"]} is still being scanned; Cortex Code '
                'can install it once SHOW VERSIONS IN CORTEX EXTENSION '
                f'{extension} shows it passed.', stacklevel=4)
            return f'{v["name"]}, scanning'
        time.sleep(SCAN_POLL)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m kumo.engine.skills',
        description='Install the kumo-data-science agent skill. The coco '
        'account scopes, workspace and extension, sign in with --connection.')
    ap.add_argument('agent', nargs='?', choices=sorted(AGENTS))
    ap.add_argument('--scope', default='user',
                    choices=['user', 'project', *COCO_ACCOUNT_SCOPES])
    ap.add_argument('--path', type=Path,
                    help='install into this skills directory instead')
    ap.add_argument(
        '--connection', metavar='NAME',
        help='coco --scope workspace|extension: the Snowflake connection '
        '(a connections.toml entry, as the snow CLI uses) to install with')
    ap.add_argument('--workspace', help='coco --scope workspace: its name')
    ap.add_argument('--extension', default='KUMO_DATA_SCIENCE',
                    help='coco --scope extension: the Cortex Extension')
    ap.add_argument('--stage', default='KUMO_SKILL_PKG',
                    help='coco --scope extension: the stage it is added from')
    ap.add_argument(
        '--profile', help='genie: the Databricks CLI profile to authenticate '
        'with (default: default authentication)')
    ap.add_argument(
        '--check', action='store_true',
        help='list the local installs and exit 1 if any is not '
        "this wheel's build")
    args = ap.parse_args(argv)
    if args.check:
        stale = stale_skills()
        for d, b in installed_skills().items():
            print(f'{d}: {b}' +
                  (f' (stale; run {update_call(d)})' if d in stale else ''))
        print(f'this kumo-engine: {bundled_build() or "source checkout"}')
        ap.exit(1 if stale else 0)
    account = (args.path is None and args.agent in (None, 'coco')
               and args.scope in COCO_ACCOUNT_SCOPES)
    # Checked before connecting, so a missing argument is the error reported.
    if account and not args.connection:
        ap.error(f'--scope {args.scope} needs --connection NAME: a shell has '
                 'no active Snowflake session')
    if args.connection and not account:
        ap.error('--connection is only for coco --scope workspace or '
                 'extension')
    if account and args.scope == 'workspace' and not args.workspace:
        ap.error('--scope workspace needs --workspace')
    client = None
    if args.profile:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient(profile=args.profile)
    try:
        session = _connect(args.connection) if account else None
    except Exception as e:  # e.g. an unknown connection name or bad login
        ap.exit(1, f'{ap.prog}: error: --connection {args.connection}: {e}\n')
    try:
        install_skills(args.agent, scope=args.scope, path=args.path,
                       session=session, workspace=args.workspace,
                       extension=args.extension, stage=args.stage,
                       client=client)
    # OSError: e.g. a --path through a file, or a directory not writable.
    except (ValueError, OSError, RuntimeError) as e:
        ap.exit(1, f'{ap.prog}: error: {e}\n')
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    main()
