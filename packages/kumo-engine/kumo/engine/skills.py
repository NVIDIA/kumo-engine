# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Install the `kumo-data-science` agent skill that ships with this wheel.

The skill teaches a coding agent to build, train and run Kumo models with
*this* SDK, so it ships inside the wheel and installs from it; it never comes
from a separate download that can drift from the installed Kumo::

    import kumo.engine as kumo
    kumo.install_skills()                    # every local agent found, or Genie
    kumo.install_skills('claude')            # ~/.claude/skills/
    kumo.install_skills('codex', scope='project')   # ./.agents/skills/

or from a shell, ``python -m kumo.engine.skills [agent] [--scope ...]``.

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
=========  =========  ==============================================================

``path=`` installs a package into any skills directory instead, for an agent
not listed here. Each install replaces the ``kumo-data-science/`` directory it
writes, and only that directory: a package carries exactly one host's files,
and a second host left behind would be read too.

This module uses only the standard library, plus ``databricks-sdk`` for
Genie and a Snowpark session for the ``coco`` account scopes.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

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
}
COCO_ACCOUNT_SCOPES = ('workspace', 'extension')
# A home directory that marks an agent as present, for `install_skills()`.
HOME_MARKERS = [
    ('claude', '.claude'),
    ('codex', '.codex'),
    ('codex', '.agents'),
    ('coco', '.snowflake/cortex'),
]
GENIE_SKILLS = '/Users/{user}/.assistant/skills'

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


def _copy(pkg: Path, skills_dir: Path) -> Path:
    r"""Replaces ``skills_dir/kumo-data-science`` with ``pkg``.

    The package is copied in full beside the target first and then renamed
    into place, under a lock on the destination, so concurrent installs never
    mix files and a failed copy leaves the previous install intact.
    """
    root = skills_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / SKILL
    with open(root / f'.{SKILL}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
        agent for agent, marker in HOME_MARKERS
        if (Path.home() / marker).is_dir()
    ]
    return list(dict.fromkeys(found)) or ['claude', 'codex']


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
        agent: ``claude``, ``codex``, ``coco`` or ``genie``. ``None`` picks
            Genie on Databricks; elsewhere, every agent whose home directory
            exists (``~/.claude``, ``~/.codex`` or ``~/.agents``,
            ``~/.snowflake/cortex``), or Claude Code and Codex if none does.
        scope: ``user`` (the agent's home skills directory) or ``project``
            (under the current directory). For ``coco`` also ``workspace``
            and ``extension``. Ignored for ``genie``, whose skills folder is
            per user.
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

    Returns:
        Where each install went.
    """
    if path is not None:
        host = AGENTS[agent][0] if agent else 'generic'
        with skill_package(host) as pkg:
            return [str(_copy(pkg, Path(path)))]
    if agent is None:
        return [
            target for name in _detect()
            for target in install_skills(
                name, scope=scope, session=session, workspace=workspace,
                extension=extension, stage=stage, client=client)
        ]
    if agent not in AGENTS:
        raise ValueError(
            f'Unknown agent {agent!r}; expected one of {sorted(AGENTS)}, '
            'or pass path= for any other agent.')
    host, scopes = AGENTS[agent]
    with skill_package(host) as pkg:
        if agent == 'genie':
            return [_install_genie(pkg, client)]
        if agent == 'coco' and scope in COCO_ACCOUNT_SCOPES:
            session = session or _active_session()
            if scope == 'workspace':
                if not workspace:
                    raise ValueError("scope='workspace' needs workspace=")
                return [_install_coco_workspace(pkg, session, workspace)]
            return [_install_coco_extension(pkg, session, extension, stage)]
        if scope not in scopes:
            raise ValueError(
                f'{agent}: scope must be one of '
                f'{sorted(scopes) + (list(COCO_ACCOUNT_SCOPES) if agent == "coco" else [])}'
            )
        return [str(_copy(pkg, Path(scopes[scope])))]


def _install_genie(pkg: Path, client: Any = None) -> str:
    r"""Uploads ``pkg`` to the user's Genie Code skills folder and reads every
    file back. A new Genie Code chat loads it.

    The Workspace API cannot rename, so the package is written over the
    previous install and only then are the files it no longer ships deleted.
    A failure part way leaves old and new files together, never an empty
    folder, and running the install again completes it.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.workspace import ExportFormat, ImportFormat

    w = client or WorkspaceClient()
    target = (f'{GENIE_SKILLS.format(user=w.current_user.me().user_name)}'
              f'/{SKILL}')
    remote = {
        f'{target}/{p.relative_to(pkg).as_posix()}': p
        for p in _files(pkg)
    }
    for path, p in remote.items():
        w.workspace.mkdirs(path.rsplit('/', 1)[0])
        w.workspace.upload(path, io.BytesIO(p.read_bytes()),
                           format=ImportFormat.AUTO, overwrite=True)
    bad = [
        path for path, p in remote.items() if w.workspace.download(
            path, format=ExportFormat.AUTO).read() != p.read_bytes()
    ]
    if bad:
        raise RuntimeError(f'{len(bad)} file(s) did not read back intact, '
                           f'e.g. {bad[0]}')
    # Delete, for each file not shipped, the highest folder under `target`
    # that holds nothing shipped; the listing yields files, not folders.
    shipped = {target}
    for path in remote:
        while (path := path.rsplit('/', 1)[0]) not in shipped:
            shipped.add(path)
    deleted: list[str] = []
    for obj in w.workspace.list(target, recursive=True):
        if obj.path in remote:
            continue
        top = obj.path
        while (parent := top.rsplit('/', 1)[0]) not in shipped:
            top = parent
        if not any(top == d or top.startswith(d + '/') for d in deleted):
            w.workspace.delete(top, recursive=True)
            deleted.append(top)
    return target


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
        session.sql(f"PUT '{_lit('file://' + p.as_posix())}' "
                    f"'{_lit(dest)}/' AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
                    ).collect()


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
    return f'USER${user}.PUBLIC."{name}"'


def _rows(session: Any, text: str) -> list[dict[str, Any]]:
    return [r.as_dict() for r in session.sql(text).collect()]


def _install_coco_workspace(pkg: Path, session: Any, workspace: str) -> str:
    r"""Uploads ``pkg`` to a Snowsight workspace's ``.snowflake/cortex/skills``,
    the copy Cortex Code reads in that workspace.

    A workspace created with SQL has no live version and takes no files until
    one is added, so it is added here. The package is written over the
    previous install, and only then are the files it no longer ships removed.
    """
    ws = _workspace_name(session, workspace)
    versions = _rows(session, f'SHOW VERSIONS IN WORKSPACE {ws}')
    if not any(str(v.get('is_live')).lower() == 'true' for v in versions):
        session.sql(f'ALTER WORKSPACE {ws} ADD LIVE VERSION FROM LAST'
                    ).collect()
    live = f'snow://workspace/{ws}/versions/live/'
    rel_root = f'.snowflake/cortex/skills/{SKILL}'
    root = live + rel_root
    _put(session, pkg, root)

    # LIST names each file from /versions/live/ on; empty `.folder` files
    # mark folders made in Snowsight. Remove, for each file not shipped, the
    # highest folder under the root that holds nothing shipped.
    shipped_files = {p.relative_to(pkg).as_posix() for p in _files(pkg)}
    shipped = {''}
    for path in shipped_files:
        while (path := path.rpartition('/')[0]) not in shipped:
            shipped.add(path)
    removed: list[str] = []
    for row in _rows(session, f"LIST '{_lit(root)}/'"):
        rel = row['name'].split('/versions/live/', 1)[1]
        if not rel.startswith(rel_root + '/'):
            continue
        rel = rel[len(rel_root) + 1:]
        folder = rel.rpartition('/')[0]
        if rel in shipped_files or (rel.endswith('/.folder')
                                    or rel == '.folder') and folder in shipped:
            continue
        top = rel
        while (parent := top.rpartition('/')[0]) not in shipped:
            top = parent
        # REMOVE matches by prefix: a folder is named with its slash, and a
        # file is skipped if it is a prefix of a shipped one.
        target = top + '/' if top != rel else top
        if any(target.startswith(r) for r in removed) or any(
                f != target and f.startswith(target) for f in shipped_files):
            continue
        session.sql(f"REMOVE '{_lit(root)}/{_lit(target)}'").collect()
        removed.append(target)
    return root


def _install_coco_extension(pkg: Path, session: Any, extension: str,
                            stage: str) -> str:
    r"""Publishes ``pkg`` as a new version of a Cortex Extension.

    The version root holds only ``skills/kumo-data-science/``, as a Cortex
    Extension requires. Installing it for a user still happens in Cortex Code
    (Shared skills); that copy does not follow later versions.
    """
    session.sql(f'CREATE STAGE IF NOT EXISTS {stage} '
                "ENCRYPTION=(TYPE='SNOWFLAKE_SSE') "
                'DIRECTORY=(ENABLE=TRUE)').collect()
    # A folder of its own per install, so concurrent installs sharing the
    # stage never publish each other's files. A version holds its own copy
    # of the files, so the folder is removed once it is added.
    root = f'@{stage}/kumo-ds-{uuid.uuid4().hex}'
    try:
        _put(session, pkg, f'{root}/skills/{SKILL}')
        session.sql(
            f'CREATE CORTEX EXTENSION IF NOT EXISTS {extension}').collect()
        session.sql(f'ALTER CORTEX EXTENSION {extension} '
                    f'ADD VERSION FROM {root}/').collect()
    finally:
        session.sql(f'REMOVE {root}/').collect()
    return extension


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m kumo.engine.skills',
        description='Install the kumo-data-science agent skill.')
    ap.add_argument('agent', nargs='?', choices=sorted(AGENTS))
    ap.add_argument('--scope', default='user',
                    choices=['user', 'project', *COCO_ACCOUNT_SCOPES])
    ap.add_argument('--path', type=Path,
                    help='install into this skills directory instead')
    ap.add_argument('--workspace', help="coco --scope workspace: its name")
    args = ap.parse_args(argv)
    for target in install_skills(args.agent, scope=args.scope, path=args.path,
                                 workspace=args.workspace):
        print(f'{SKILL} -> {target}')


if __name__ == '__main__':
    main()
