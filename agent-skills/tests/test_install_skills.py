# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`kumo.engine.install_skills()`: the wheel carries every host package and installs the right one.

`kumo/engine/skills.py` uses only the standard library, so it is loaded here by path, without
importing `kumo.engine` and the training stack behind it.
"""
import importlib.util
import io
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package_skill = load('package_skill', ROOT / 'agent-skills' / 'scripts' / 'package_skill.py')
skills = load('kumo_engine_skills', ROOT / 'packages' / 'kumo-engine' / 'kumo' / 'engine' / 'skills.py')


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('DATABRICKS_RUNTIME_VERSION', raising=False)
    monkeypatch.chdir(tmp_path)
    return home


def hosts(target: Path) -> list[str]:
    return sorted(p.name for p in (target / 'hosts').iterdir())


def test_the_packager_is_found_from_a_checkout() -> None:
    assert skills._PACKAGER == ROOT / 'agent-skills' / 'scripts' / 'package_skill.py'
    assert {host for host, _ in skills.AGENTS.values()} == set(package_skill.HOSTS)


def test_bundle_holds_one_package_per_host(tmp_path: Path) -> None:
    package_skill.bundle(tmp_path / '_skills')
    assert sorted(p.name for p in (tmp_path / '_skills').iterdir()) == sorted(package_skill.HOSTS)
    for host, expected in [('generic', ['generic']), ('coco', ['snowflake-coco']),
                           ('genie', ['databricks-genie'])]:
        pkg = tmp_path / '_skills' / host / 'kumo-data-science'
        assert (pkg / 'SKILL.md').is_file()
        assert hosts(pkg) == expected
        assert package_skill.unresolved(pkg) == []


@pytest.mark.parametrize('agent,scope,where,host', [
    ('claude', 'user', 'home/.claude/skills', 'generic'),
    ('claude', 'project', '.claude/skills', 'generic'),
    ('codex', 'user', 'home/.agents/skills', 'generic'),
    ('codex', 'project', '.agents/skills', 'generic'),
    ('coco', 'user', 'home/.snowflake/cortex/skills', 'snowflake-coco'),
    ('coco', 'project', '.cortex/skills', 'snowflake-coco'),
])
def test_local_agents(home: Path, agent: str, scope: str, where: str, host: str) -> None:
    [target] = skills.install_skills(agent, scope=scope)
    assert Path(target) == (home.parent / where / 'kumo-data-science').resolve()
    assert hosts(Path(target)) == [host]
    assert 'SKILL-BUILD: source' not in (Path(target) / 'SKILL.md').read_text()


def test_reinstall_replaces_only_its_own_directory(home: Path) -> None:
    root = home / '.claude' / 'skills'
    (root / 'kumo-data-science' / 'hosts' / 'snowflake-coco').mkdir(parents=True)
    (root / 'other-skill').mkdir()
    (root / 'other-skill' / 'SKILL.md').write_text('mine')
    skills.install_skills('claude')
    assert hosts(root / 'kumo-data-science') == ['generic']
    assert (root / 'other-skill' / 'SKILL.md').read_text() == 'mine'


def test_path_installs_anywhere(home: Path, tmp_path: Path) -> None:
    [target] = skills.install_skills(path=tmp_path / 'somewhere')
    assert hosts(Path(target)) == ['generic']
    [target] = skills.install_skills('coco', path=tmp_path / 'else')
    assert hosts(Path(target)) == ['snowflake-coco']


def test_no_agent_installs_for_the_agents_present(home: Path) -> None:
    assert len(skills.install_skills()) == 2   # none present: Claude Code and Codex
    for d in ['.codex', '.snowflake/cortex']:
        (home / d).mkdir(parents=True, exist_ok=True)
    (home / '.claude').rename(home / 'moved')
    targets = skills.install_skills()
    assert [Path(t).parts[-3] for t in targets] == ['.agents', 'cortex']


def test_bad_arguments(home: Path) -> None:
    with pytest.raises(ValueError, match='Unknown agent'):
        skills.install_skills('cursor')
    with pytest.raises(ValueError, match='scope must be one of'):
        skills.install_skills('claude', scope='workspace')


def test_the_bundled_copy_is_used_outside_a_checkout(
        home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_skill.bundle(tmp_path / '_skills')
    marker = tmp_path / '_skills' / 'generic' / 'kumo-data-science' / 'BUNDLED'
    marker.write_text('')
    monkeypatch.setattr(skills, '_PACKAGER', tmp_path / 'missing.py')
    monkeypatch.setattr(skills, '_BUNDLED', tmp_path / '_skills')
    [target] = skills.install_skills('claude')
    assert (Path(target) / 'BUNDLED').is_file()
    monkeypatch.setattr(skills, '_BUNDLED', tmp_path / 'empty')
    with pytest.raises(FileNotFoundError, match='carries no'):
        skills.install_skills('claude')


class FakeDatabricks:
    """A `WorkspaceClient` holding files in a dict, plus the `databricks.sdk` modules it needs."""

    def __init__(self, files: dict[str, bytes], fail_after: int = -1,
                 folders: tuple[str, ...] = ()) -> None:
        self.files = files
        self.folders = set(folders)   # folders beyond those holding a file, e.g. empty ones
        self.fail_after = fail_after   # uploads that succeed before one raises; -1 never
        self.deleted: list[str] = []
        self.current_user = types.SimpleNamespace(
            me=lambda: types.SimpleNamespace(user_name='someone@example.com'))
        self.workspace = self

    def dirs(self) -> set[str]:
        out = set(self.folders)
        for p in self.files:
            while (p := p.rsplit('/', 1)[0]):
                out.add(p)
        return out

    def delete(self, path: str, recursive: bool = False) -> None:
        assert recursive
        if path not in self.files and path not in self.dirs():
            raise LookupError(f'Path ({path}) doesn\'t exist.')
        for p in [p for p in self.files if p == path or p.startswith(path + '/')]:
            del self.files[p]
        self.folders = {d for d in self.folders if d != path and not d.startswith(path + '/')}
        self.deleted.append(path)

    def list(self, path: str, recursive: bool = False):
        # Like the SDK: a recursive listing is a lazy breadth-first walk that
        # lists each folder only when it reaches it, and yields files only.
        queue = [path]
        while queue:
            path = queue.pop(0)
            dirs = self.dirs()
            if path not in dirs:
                raise LookupError(f'Path ({path}) doesn\'t exist.')
            children = sorted({p for p in [*self.files, *dirs]
                               if p.rsplit('/', 1)[0] == path})
            for p in children:
                if p in dirs:
                    if recursive:
                        queue.append(p)
                        continue
                    yield types.SimpleNamespace(path=p, object_type='DIRECTORY')
                else:
                    yield types.SimpleNamespace(path=p, object_type='FILE')

    def mkdirs(self, path: str) -> None:
        pass

    def upload(self, path: str, content, format, overwrite: bool) -> None:
        assert format == 'AUTO' and overwrite
        if self.fail_after == 0:
            raise ConnectionError('upload failed')
        self.fail_after -= 1
        self.files[path] = content.read()

    def download(self, path: str, format):
        return io.BytesIO(self.files[path])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mods = {
            'databricks': types.ModuleType('databricks'),
            'databricks.sdk': types.SimpleNamespace(WorkspaceClient=lambda: self),
            'databricks.sdk.service': types.ModuleType('databricks.sdk.service'),
            'databricks.sdk.service.workspace': types.SimpleNamespace(
                ImportFormat=types.SimpleNamespace(AUTO='AUTO'),
                ExportFormat=types.SimpleNamespace(AUTO='AUTO'),
                ObjectType=types.SimpleNamespace(DIRECTORY='DIRECTORY')),
        }
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)


GENIE = '/Users/someone@example.com/.assistant/skills'


def test_genie_on_databricks(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = {f'{GENIE}/kumo-data-science/hosts/generic/adapter.md': b'stale',
           f'{GENIE}/kumo-data-science/hosts/generic/runtime.md': b'stale',
           f'{GENIE}/kumo-data-science/old-notes.md': b'stale',
           f'{GENIE}/kumo-data-science/SKILL.md': b'old router',
           f'{GENIE}/tabular/SKILL.md': b'another skill'}
    ws = FakeDatabricks(dict(old))
    ws.install(monkeypatch)
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '17.3')
    assert skills.install_skills() == [f'{GENIE}/kumo-data-science']
    shipped = {p.split('/kumo-data-science/')[1] for p in ws.files if '/kumo-data-science/' in p}
    assert 'SKILL.md' in shipped and 'hosts/databricks-genie/adapter.md' in shipped
    assert ws.files[f'{GENIE}/kumo-data-science/SKILL.md'] != b'old router'
    assert not any(p.startswith('hosts/generic') or p == 'old-notes.md' for p in shipped)
    # The stale host goes as one folder; nothing outside kumo-data-science/ is touched.
    assert sorted(ws.deleted) == [f'{GENIE}/kumo-data-science/hosts/generic',
                                  f'{GENIE}/kumo-data-science/old-notes.md']
    assert ws.files[f'{GENIE}/tabular/SKILL.md'] == b'another skill'


def test_genie_removes_nested_and_empty_stale_folders(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = f'{GENIE}/kumo-data-science'
    ws = FakeDatabricks({f'{base}/hosts/old/a.md': b'stale',
                         f'{base}/hosts/old/sub/b.md': b'stale',
                         f'{GENIE}/tabular/SKILL.md': b'another skill'},
                        folders=(f'{base}/hosts/empty', f'{base}/hosts/old/sub/deeper'))
    ws.install(monkeypatch)
    skills.install_skills('genie')
    assert sorted(ws.deleted) == [f'{base}/hosts/empty', f'{base}/hosts/old']
    assert {p.rsplit('/', 1)[1] for p in ws.dirs() if p.startswith(f'{base}/hosts/')} == {
        'databricks-genie'}
    assert ws.files[f'{GENIE}/tabular/SKILL.md'] == b'another skill'


def test_a_failed_genie_reinstall_keeps_the_previous_skill(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = {f'{GENIE}/kumo-data-science/SKILL.md': b'old router',
           f'{GENIE}/kumo-data-science/hosts/databricks-genie/zz-last.md': b'old'}
    ws = FakeDatabricks(dict(old), fail_after=3)
    ws.install(monkeypatch)
    with pytest.raises(ConnectionError):
        skills.install_skills('genie')
    assert ws.deleted == []
    assert ws.files[f'{GENIE}/kumo-data-science/hosts/databricks-genie/zz-last.md'] == b'old'
    ws.fail_after = -1
    skills.install_skills('genie')   # a rerun completes it
    assert f'{GENIE}/kumo-data-science/hosts/databricks-genie/zz-last.md' not in ws.files


def test_genie_fails_when_a_file_does_not_read_back(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeDatabricks({f'{GENIE}/kumo-data-science/old.md': b'old'})
    ws.install(monkeypatch)
    monkeypatch.setattr(ws, 'download', lambda path, format: io.BytesIO(b'x'))
    with pytest.raises(RuntimeError, match='did not read back'):
        skills.install_skills('genie')
    assert ws.deleted == []   # nothing is pruned after a failed check


class FakeRow(tuple):
    def __new__(cls, **fields: object) -> 'FakeRow':
        row = super().__new__(cls, fields.values())
        row.fields = fields
        return row

    def as_dict(self) -> dict:
        return self.fields


class FakeSession:
    r"""Answers CURRENT_USER (``user``), SHOW VERSIONS (live unless ``live=False``) and
    LIST (``listed``, names relative to the workspace's live version)."""

    def __init__(self, live: bool = True, listed: tuple[str, ...] = (),
                 user: str = 'ME') -> None:
        self.user = user
        self.sql_log: list[str] = []
        self.live = live
        self.listed = listed

    def sql(self, text: str):
        self.sql_log.append(text)
        if text.startswith('SHOW VERSIONS'):
            rows = [FakeRow(name='VERSION$1', is_live='false')]
            rows += [FakeRow(name=None, is_live='true')] if self.live else []
        elif text.startswith('LIST'):
            rows = [FakeRow(name=f'/versions/live/{n}') for n in self.listed]
        else:
            rows = [FakeRow(user=self.user)]
        return types.SimpleNamespace(collect=lambda: rows)


def test_coco_extension(home: Path) -> None:
    session = FakeSession()
    assert skills.install_skills('coco', scope='extension', session=session) == [
        'KUMO_DATA_SCIENCE']
    log = session.sql_log
    assert "ENCRYPTION=(TYPE='SNOWFLAKE_SSE')" in log[0]
    puts = [s for s in log if s.startswith('PUT')]
    root = puts[0].split("'")[3].split('/skills/')[0]
    assert re.fullmatch(r'@KUMO_SKILL_PKG/kumo-ds-[0-9a-f]{32}', root)
    assert all(f"'{root}/skills/kumo-data-science" in s for s in puts)
    assert any('/hosts/snowflake-coco/' in s for s in puts)
    assert log[-2:] == [f'ALTER CORTEX EXTENSION KUMO_DATA_SCIENCE ADD VERSION FROM {root}/',
                        f'REMOVE {root}/']


def test_concurrent_extension_installs_use_their_own_folders(home: Path) -> None:
    first, second = FakeSession(), FakeSession()
    skills.install_skills('coco', scope='extension', session=first)
    skills.install_skills('coco', scope='extension', session=second, extension='OTHER')
    assert first.sql_log[-1] != second.sql_log[-1]
    assert not any(s.startswith('REMOVE') for s in first.sql_log[:-1])


def test_a_failed_extension_install_removes_its_folder(home: Path) -> None:
    class Failing(FakeSession):
        def sql(self, text: str):
            if text.startswith('ALTER CORTEX'):
                raise RuntimeError('add version failed')
            return super().sql(text)

    session = Failing()
    with pytest.raises(RuntimeError, match='add version failed'):
        skills.install_skills('coco', scope='extension', session=session)
    assert session.sql_log[-1].startswith('REMOVE @KUMO_SKILL_PKG/kumo-ds-')


def test_no_agent_forwards_the_snowflake_destination(home: Path) -> None:
    (home / '.snowflake' / 'cortex').mkdir(parents=True)
    session = FakeSession()
    assert skills.install_skills(scope='extension', session=session, extension='MY_EXT',
                                 stage='MY_STAGE') == ['MY_EXT']
    assert session.sql_log[-2].startswith('ALTER CORTEX EXTENSION MY_EXT ADD VERSION FROM @MY_STAGE/')
    [root] = skills.install_skills(scope='workspace', session=FakeSession(), workspace='WS')
    assert root.startswith('snow://workspace/"USER$ME".PUBLIC."WS"/')


def test_coco_workspace(home: Path) -> None:
    session = FakeSession()
    [root] = skills.install_skills('coco', scope='workspace', session=session, workspace='MY_WS')
    assert root == ('snow://workspace/"USER$ME".PUBLIC."MY_WS"/versions/live/'
                    '.snowflake/cortex/skills/kumo-data-science')
    assert all(f"'{root}" in s for s in session.sql_log if s.startswith('PUT'))
    assert not any(s.startswith(('ALTER', 'REMOVE')) for s in session.sql_log)
    with pytest.raises(ValueError, match='needs workspace'):
        skills.install_skills('coco', scope='workspace', session=session)


def test_coco_workspace_without_a_live_version_gets_one(home: Path) -> None:
    session = FakeSession(live=False)
    skills.install_skills('coco', scope='workspace', session=session, workspace='NEW')
    log = session.sql_log
    add = log.index('ALTER WORKSPACE "USER$ME".PUBLIC."NEW" ADD LIVE VERSION FROM LAST')
    assert log[add - 1] == 'SHOW VERSIONS IN WORKSPACE "USER$ME".PUBLIC."NEW"'
    assert add < min(i for i, s in enumerate(log) if s.startswith('PUT'))


@pytest.mark.parametrize('given, sql_name', [
    ('UAT_testing', '"USER$ME".PUBLIC."UAT_testing"'),
    ('my.ws', '"USER$ME".PUBLIC."my.ws"'),
    ('say "hi"', '"USER$ME".PUBLIC."say ""hi"""'),
    ('DB.SCH.WS', 'DB.SCH.WS'),
    ('USER$ME.PUBLIC."My Ws"', 'USER$ME.PUBLIC."My Ws"'),
    ('"USER$ME".PUBLIC."My Ws"', '"USER$ME".PUBLIC."My Ws"'),
])
def test_coco_workspace_names(home: Path, given: str, sql_name: str) -> None:
    session = FakeSession()
    [root] = skills.install_skills('coco', scope='workspace', session=session, workspace=given)
    assert root.startswith(f'snow://workspace/{sql_name}/versions/live/')
    assert f'SHOW VERSIONS IN WORKSPACE {sql_name}' in session.sql_log


def test_coco_workspace_quotes_an_email_user(home: Path) -> None:
    session = FakeSession(user='A.B@NVIDIA.COM')
    [root] = skills.install_skills('coco', scope='workspace', session=session, workspace='WS')
    assert root.startswith('snow://workspace/"USER$A.B@NVIDIA.COM".PUBLIC."WS"/versions/live/')
    assert 'SHOW VERSIONS IN WORKSPACE "USER$A.B@NVIDIA.COM".PUBLIC."WS"' in session.sql_log


def test_sql_literals_are_escaped(home: Path) -> None:
    session = FakeSession()
    skills.install_skills('coco', scope='workspace', session=session, workspace="it's")
    put = next(s for s in session.sql_log if s.startswith('PUT'))
    assert """'snow://workspace/"USER$ME".PUBLIC."it\\'s"/versions/live/""" in put
    assert skills._lit('a\\b') == 'a\\\\b'


def test_coco_workspace_removes_what_the_build_no_longer_ships(home: Path) -> None:
    base = '.snowflake/cortex/skills/kumo-data-science'
    session = FakeSession(listed=(
        f'{base}/SKILL.md',
        f'{base}/.folder',
        f'{base}/hosts/.folder',
        f'{base}/hosts/snowflake-coco/adapter.md',
        f'{base}/hosts/obsolete/.folder',
        f'{base}/hosts/obsolete/a.md',
        f'{base}/hosts/obsolete/deep/b.md',
        f'{base}/old-notes.md',
        f'{base}/SKILL',  # a prefix of SKILL.md: REMOVE would take both
        '.snowflake/cortex/skills/other-skill/SKILL.md',
    ))
    [root] = skills.install_skills('coco', scope='workspace', session=session, workspace='WS')
    removes = [s for s in session.sql_log if s.startswith('REMOVE')]
    assert removes == [f"REMOVE '{root}/hosts/obsolete/'", f"REMOVE '{root}/old-notes.md'"]


def test_snowflake_notebook_asks_for_a_destination(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path.exists
    monkeypatch.setattr(Path, 'exists',
                        lambda self: str(self) == '/snowflake/session/token' or real(self))
    with pytest.raises(ValueError, match='name the destination'):
        skills.install_skills()


def test_a_failed_copy_keeps_the_previous_install(home: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    [target] = skills.install_skills('claude')
    before = (Path(target) / 'SKILL.md').read_text()

    def broken(src, dst, **kwargs):
        Path(dst).mkdir(parents=True)
        (Path(dst) / 'SKILL.md').write_text('half copied')
        raise OSError('disk full')

    monkeypatch.setattr(skills.shutil, 'copytree', broken)
    with pytest.raises(OSError, match='disk full'):
        skills.install_skills('claude')
    assert (Path(target) / 'SKILL.md').read_text() == before
    root = Path(target).parent
    names = sorted(p.name for p in root.iterdir())
    assert names == ['.kumo-data-science.lock', 'kumo-data-science']


def test_a_checkout_install_checks_references_first(home: Path, tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    [target] = skills.install_skills('claude')
    broken = tmp_path / 'package_skill.py'
    broken.write_text(
        'from pathlib import Path\n'
        'def build(host, out):\n'
        '    pkg = out / host / "kumo-data-science"\n'
        '    pkg.mkdir(parents=True)\n'
        '    (pkg / "SKILL.md").write_text("[gone](accept/missing.md)")\n'
        '    return pkg\n'
        'def unresolved(pkg):\n'
        '    return ["SKILL.md: link (accept/missing.md) does not resolve"]\n')
    monkeypatch.setattr(skills, '_PACKAGER', broken)
    with pytest.raises(ValueError, match='unresolved reference'):
        skills.install_skills('claude')
    assert 'missing.md' not in (Path(target) / 'SKILL.md').read_text()


def test_packaging_without_git_ships_what_git_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    tracked = package_skill.git_tracked(package_skill.SKILLS)

    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(package_skill.subprocess, 'run', no_git)
    assert package_skill.git_tracked(package_skill.SKILLS) == sorted(tracked)
