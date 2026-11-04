# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`kumo.engine.install_skills()`: the wheel carries every host package and installs the right one.

`kumo/engine/skills.py` uses only the standard library and `filelock`, so it is loaded here by
path, without importing `kumo.engine` and the training stack behind it.
"""
import ast
import importlib.util
import io
import re
import sys
import threading
import types
import warnings
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
ENGINE = ROOT / 'packages' / 'kumo-engine' / 'kumo' / 'engine'
skills = load('kumo_engine_skills', ENGINE / 'skills.py')


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


def test_path_installs_for_an_agent_not_listed(home: Path, tmp_path: Path) -> None:
    [target] = skills.install_skills('cursor', path=tmp_path / 'cursor')
    assert hosts(Path(target)) == ['generic']


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
    with pytest.raises(ValueError, match='takes no scope'):
        skills.install_skills('genie', scope='project')


def test_an_install_leaves_no_lock_file(home: Path) -> None:
    skills.install_skills('claude', scope='project')
    assert [p.name for p in Path('.claude/skills').iterdir()] == ['kumo-data-science']


def test_coco_beside_claude_code_warns(home: Path) -> None:
    # Cortex Code reads ~/.claude/skills too, and that copy wins.
    skills.install_skills('claude')
    with pytest.warns(UserWarning, match='Cortex Code reads'):
        skills.install_skills('coco')
    with pytest.warns(UserWarning, match='Cortex Code reads'):
        skills.install_skills('claude')


def test_cli_reports_errors_without_a_traceback(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        skills.main(['genie', '--scope', 'project'])
    assert exc.value.code == 1
    assert 'error: genie installs' in capsys.readouterr().err


def test_cli_reports_a_filesystem_error_without_a_traceback(
        home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / 'a-file').write_text('')
    with pytest.raises(SystemExit) as exc:
        skills.main(['--path', str(tmp_path / 'a-file' / 'skills')])
    assert exc.value.code == 1
    assert 'error: ' in capsys.readouterr().err


def test_a_missing_workspace_is_reported_before_the_session(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_session():
        raise RuntimeError('no active session')

    monkeypatch.setattr(skills, '_active_session', no_session)
    with pytest.raises(ValueError, match='needs workspace='):
        skills.install_skills('coco', scope='workspace')
    with pytest.raises(ValueError, match='needs workspace='):
        skills.install_skills(scope='workspace')


def test_skills_imports_no_posix_only_module() -> None:
    # `kumo.engine` imports it, so it must import on Windows.
    tree = ast.parse((ENGINE / 'skills.py').read_text())
    names = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names}
    names |= {n.module.split('.')[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {'fcntl', 'termios', 'pwd', 'grp', 'resource'}


def test_an_install_waits_for_the_lock(home: Path, tmp_path: Path) -> None:
    from filelock import FileLock
    root = tmp_path / 'locked'
    root.mkdir()
    done = threading.Event()
    with FileLock(skills._lock_path(root.resolve())):
        thread = threading.Thread(
            target=lambda: (skills.install_skills(path=root), done.set()))
        thread.start()
        assert not done.wait(1)
        assert not (root / 'kumo-data-science').exists()
    thread.join(60)
    assert done.is_set() and hosts(root / 'kumo-data-science') == ['generic']


def test_kumo_engine_imports_install_skills_on_first_use(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # `python -m kumo.engine.skills` warns if importing `kumo.engine` already
    # imported it. The package's own imports need the training stack, so its
    # top level is read, and its `__getattr__` run against stand-in modules.
    tree = ast.parse((ENGINE / '__init__.py').read_text())
    assert not [n for n in tree.body if isinstance(n, ast.ImportFrom)
                and (n.module or '').startswith('kumo.engine.skills')]
    [getattr_] = [n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == '__getattr__']
    namespace: dict = {'__name__': 'kumo.engine'}
    exec(compile(ast.Module([getattr_], []), 'kumo/engine/__init__.py', 'exec'), namespace)
    for name in ['kumo', 'kumo.engine']:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'kumo.engine.skills', skills)
    assert namespace['__getattr__']('install_skills') is skills.install_skills
    with pytest.raises(AttributeError):
        namespace['__getattr__']('nothing')


@pytest.mark.parametrize('argv, checked', [
    (['python', '-m', 'kumo.engine.skills', '--check'], False),
    (['python', '-W', 'error', '-m', 'kumo.engine.skills'], False),
    (['python', 'train.py'], True),
    (['python', '-m', 'kumo.engine.other', 'kumo.engine.skills'], True),
])
def test_the_import_check_skips_the_skills_command(
        monkeypatch: pytest.MonkeyPatch, argv: list, checked: bool) -> None:
    # Under `python -m kumo.engine.skills`, importing `kumo.engine` must not import the module a
    # second time to check. Its `_warn_if_skill_stale` is run against a stand-in `sys` and module.
    tree = ast.parse((ENGINE / '__init__.py').read_text())
    [guard] = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == '_warn_if_skill_stale']
    namespace: dict = {'sys': types.SimpleNamespace(orig_argv=argv)}
    exec(compile(ast.Module([guard], []), 'kumo/engine/__init__.py', 'exec'), namespace)
    calls: list = []
    stand_in = types.ModuleType('kumo.engine.skills')
    stand_in.warn_if_stale = lambda: calls.append(1)
    for name in ['kumo', 'kumo.engine']:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'kumo.engine.skills', stand_in)
    namespace['_warn_if_skill_stale']()
    assert calls == ([1] if checked else [])


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


class NotFound(LookupError):
    pass


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
        if path not in self.files:
            raise NotFound(f'Path ({path}) doesn\'t exist.')
        return io.BytesIO(self.files[path])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mods = {
            'databricks': types.ModuleType('databricks'),
            'databricks.sdk': types.SimpleNamespace(WorkspaceClient=lambda: self),
            'databricks.sdk.errors': types.SimpleNamespace(NotFound=NotFound),
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


def another_writer(ws: FakeDatabricks, monkeypatch: pytest.MonkeyPatch, walk: int,
                   write: dict[str, bytes]) -> None:
    """Writes `write` as the install starts its `walk`-th walk of the folder (1 is the prune)."""
    walks, listing = [], ws.list

    def list_(path: str, recursive: bool = False):
        if path == f'{GENIE}/kumo-data-science':
            walks.append(path)
            if len(walks) == walk:
                ws.files.update(write)
        yield from listing(path, recursive)
    monkeypatch.setattr(ws, 'list', list_)


def test_a_concurrent_genie_install_of_another_build_fails_one(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # B uploads and verifies between A's verify and prune; B's router needs b-only.md, which
    # A's prune deletes. A must not report success over a router missing its file.
    base = f'{GENIE}/kumo-data-science'
    ws = FakeDatabricks({})
    ws.install(monkeypatch)
    another_writer(ws, monkeypatch, 1, {f'{base}/SKILL.md': b'B router, see b-only.md',
                                        f'{base}/b-only.md': b'B'})
    with pytest.raises(RuntimeError, match='Another install changed'):
        skills.install_skills('genie')
    assert f'{base}/b-only.md' in ws.deleted


def test_genie_fails_when_a_file_appears_after_the_prune(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = f'{GENIE}/kumo-data-science'
    ws = FakeDatabricks({})
    ws.install(monkeypatch)
    another_writer(ws, monkeypatch, 2, {f'{base}/hosts/other/x.md': b'stray'})
    with pytest.raises(RuntimeError, match=f'changed {base} .*hosts/other'):
        skills.install_skills('genie')


class FakeRow(tuple):
    def __new__(cls, **fields: object) -> 'FakeRow':
        row = super().__new__(cls, fields.values())
        row.fields = fields
        return row

    def as_dict(self) -> dict:
        return self.fields


def unquote(literal: str) -> str:
    return re.sub(r'\\(.)', r'\1', literal)


class FakeSession:
    r"""Answers CURRENT_USER (``user``) and SHOW VERSIONS (live unless ``live=False``), and
    holds the files PUT, REMOVE and LIST act on, by path from the workspace's live version
    (``listed`` are there to begin with). Like Snowflake, LIST gives a file of ``s`` bytes
    the AES-padded size ``(s // 16 + 1) * 16``, and REMOVE matches by prefix."""

    def __init__(self, live: bool = True, listed: tuple[str, ...] = (),
                 user: str = 'ME') -> None:
        self.user = user
        self.sql_log: list[str] = []
        self.live = live
        self.files = {n: 16 for n in listed}

    @staticmethod
    def key(path: str) -> str:
        return path.split('/versions/live/', 1)[-1]

    def sql(self, text: str):
        self.sql_log.append(text)
        literals = [unquote(m) for m in re.findall(r"'((?:[^'\\]|\\.)*)'", text)]
        if text.startswith('SHOW VERSIONS'):
            rows = [FakeRow(name='VERSION$1', is_live='false')]
            rows += [FakeRow(name=None, is_live='true')] if self.live else []
        elif text.startswith('PUT'):
            local = Path(literals[0].removeprefix('file://'))
            size = local.stat().st_size
            self.files[self.key(literals[1]) + local.name] = (size // 16 + 1) * 16
            rows = []
        elif text.startswith('REMOVE'):
            prefix = self.key(literals[0] if literals else text.split()[1])
            self.files = {k: v for k, v in self.files.items() if not k.startswith(prefix)}
            rows = []
        elif text.startswith('LIST'):
            prefix = self.key(literals[0])
            rows = [FakeRow(name=f'/versions/live/{k}', size=v)
                    for k, v in sorted(self.files.items()) if k.startswith(prefix)]
        else:
            rows = [FakeRow(user=self.user)]
        return types.SimpleNamespace(collect=lambda: rows)


class FakeExtension(FakeSession):
    r"""Also keeps the versions of Cortex Extensions: each version records the stage folder it
    was added from, and its scan passes after ``scans`` polls (``status`` then)."""

    def __init__(self, scans: int = 0, status: str = 'SCAN_PASSED') -> None:
        super().__init__()
        self.versions: dict[str, list[dict]] = {}
        self.scans, self.status = scans, status

    def _add(self, ext: str, source: str) -> None:
        versions = self.versions.setdefault(ext, [])
        for v in versions:
            v['is_default'] = 'false'
        versions.append(dict(name=f'VERSION${len(versions) + 1}', is_default='true',
                             source_location_uri=source, polls=0))

    def sql(self, text: str):
        if m := re.fullmatch(r'SHOW VERSIONS IN CORTEX EXTENSION (\S+)', text):
            self.sql_log.append(text)
            if m[1] not in self.versions:
                raise RuntimeError(f'Cortex Extension {m[1]} does not exist.')
            rows = []
            for v in self.versions[m[1]]:
                v['polls'] += 1
                status = 'SCANNING' if v['polls'] <= self.scans else self.status
                rows.append(FakeRow(**{k: x for k, x in v.items() if k != 'polls'},
                                    effective_status=status, scan_findings=''))
            return types.SimpleNamespace(collect=lambda: rows)
        if m := re.fullmatch(r'CREATE CORTEX EXTENSION IF NOT EXISTS (\S+) FROM (\S+)', text):
            if m[1] not in self.versions:
                self._add(m[1], m[2])
        elif m := re.fullmatch(r'ALTER CORTEX EXTENSION (\S+) ADD VERSION FROM (\S+)', text):
            self._add(m[1], m[2])
        return super().sql(text)


@pytest.fixture(autouse=True)
def no_scan_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills, 'SCAN_POLL', 0)


def test_coco_extension(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    session = FakeExtension()
    assert skills.install_skills('coco', scope='extension', session=session) == [
        'KUMO_DATA_SCIENCE']
    log = session.sql_log
    assert "ENCRYPTION=(TYPE='SNOWFLAKE_SSE')" in log[1]
    puts = [s for s in log if s.startswith('PUT')]
    root = puts[0].split("'")[3].split('/skills/')[0]
    assert re.fullmatch(r'@KUMO_SKILL_PKG/kumo-ds-[0-9a-f]{16}-[0-9a-f]{8}', root)
    assert all(f"'{root}/skills/kumo-data-science" in s for s in puts)
    assert any('/hosts/snowflake-coco/' in s for s in puts)
    # Created from the package: no empty first version, and no second ADD VERSION.
    assert f'CREATE CORTEX EXTENSION IF NOT EXISTS KUMO_DATA_SCIENCE FROM {root}/' in log
    assert not any(s.startswith('ALTER CORTEX') for s in log)
    assert f'REMOVE {root}/' in log
    assert [v['name'] for v in session.versions['KUMO_DATA_SCIENCE']] == ['VERSION$1']
    assert '(VERSION$1)' in capsys.readouterr().out


def test_an_unchanged_extension_is_not_republished(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    session = FakeExtension()
    skills.install_skills('coco', scope='extension', session=session)
    skills.install_skills('coco', scope='extension', session=session)
    assert len(session.versions['KUMO_DATA_SCIENCE']) == 1
    assert '(unchanged, VERSION$1)' in capsys.readouterr().out
    # A changed package is published as a new version.
    session.versions['KUMO_DATA_SCIENCE'][0]['source_location_uri'] = '@S/kumo-ds-0-0/'
    skills.install_skills('coco', scope='extension', session=session)
    assert session.sql_log.count(
        next(s for s in session.sql_log if s.startswith('ALTER CORTEX'))) == 1
    assert len(session.versions['KUMO_DATA_SCIENCE']) == 2


def test_the_extension_install_waits_for_the_scan(home: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeExtension(scans=3)
    skills.install_skills('coco', scope='extension', session=session)
    assert session.versions['KUMO_DATA_SCIENCE'][0]['polls'] == 4
    monkeypatch.setattr(skills, 'SCAN_TIMEOUT', 0)
    with pytest.warns(UserWarning, match='still being scanned'):
        skills.install_skills('coco', scope='extension', session=FakeExtension(scans=99))
    with pytest.raises(RuntimeError, match='rejected .* SCAN_FAILED'):
        skills.install_skills('coco', scope='extension',
                              session=FakeExtension(status='SCAN_FAILED'))


def test_an_unchanged_extension_still_waits_for_its_scan(
        home: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    # A retry after the scan-timeout warning finds the same package as the default version.
    monkeypatch.setattr(skills, 'SCAN_TIMEOUT', 0)
    session = FakeExtension(scans=99)
    with pytest.warns(UserWarning, match='still being scanned'):
        skills.install_skills('coco', scope='extension', session=session)
    with pytest.warns(UserWarning, match='still being scanned'):
        skills.install_skills('coco', scope='extension', session=session)
    assert '(unchanged, VERSION$1, scanning)' in capsys.readouterr().out
    session.scans, session.status = 0, 'SCAN_FAILED'
    with pytest.raises(RuntimeError, match='rejected .* SCAN_FAILED'):
        skills.install_skills('coco', scope='extension', session=session)
    assert len(session.versions['KUMO_DATA_SCIENCE']) == 1


def test_concurrent_extension_installs_use_their_own_folders(home: Path) -> None:
    first, second = FakeExtension(), FakeExtension()
    skills.install_skills('coco', scope='extension', session=first)
    skills.install_skills('coco', scope='extension', session=second, extension='OTHER')
    removes = [[s for s in log if s.startswith('REMOVE')]
               for log in (first.sql_log, second.sql_log)]
    assert len(removes[0]) == len(removes[1]) == 1 and removes[0] != removes[1]


def test_a_failed_extension_install_removes_its_folder(home: Path) -> None:
    class Failing(FakeExtension):
        def sql(self, text: str):
            if text.startswith('CREATE CORTEX'):
                raise RuntimeError('create failed')
            return super().sql(text)

    session = Failing()
    with pytest.raises(RuntimeError, match='create failed'):
        skills.install_skills('coco', scope='extension', session=session)
    assert session.sql_log[-1].startswith('REMOVE @KUMO_SKILL_PKG/kumo-ds-')


def test_no_agent_forwards_the_snowflake_destination(home: Path) -> None:
    (home / '.snowflake' / 'cortex').mkdir(parents=True)
    session = FakeExtension()
    assert skills.install_skills(scope='extension', session=session, extension='MY_EXT',
                                 stage='MY_STAGE') == ['MY_EXT']
    assert any(s.startswith('CREATE CORTEX EXTENSION IF NOT EXISTS MY_EXT FROM @MY_STAGE/')
               for s in session.sql_log)
    [root] = skills.install_skills(scope='workspace', session=FakeSession(), workspace='WS')
    assert root.startswith('snow://workspace/"USER$ME".PUBLIC."WS"/')


def test_no_agent_sends_account_scopes_to_coco_alone(home: Path, monkeypatch) -> None:
    (home / '.claude').mkdir()
    (home / '.snowflake' / 'cortex').mkdir(parents=True)
    assert skills.install_skills(scope='extension', session=FakeExtension(), extension='MY_EXT',
                                 stage='MY_STAGE') == ['MY_EXT']
    assert not (home / '.claude' / 'skills').exists()
    # In a Snowflake notebook, detection refuses; a named account scope does not need it.
    monkeypatch.setattr(skills, '_detect', lambda: pytest.fail('detected'))
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


WS_ROOT = '.snowflake/cortex/skills/kumo-data-science'


def test_coco_workspace_lists_again_and_finds_its_build(home: Path) -> None:
    session = FakeSession(listed=(f'{WS_ROOT}/old-notes.md', f'{WS_ROOT}/hosts/.folder'))
    skills.install_skills('coco', scope='workspace', session=session, workspace='WS')
    lists = [i for i, s in enumerate(session.sql_log) if s.startswith('LIST')]
    assert len(lists) == 2 and lists[1] == len(session.sql_log) - 1
    assert any(s.startswith('REMOVE') for s in session.sql_log[lists[0]:lists[1]])
    with skills.skill_package('coco') as pkg:
        assert {k.removeprefix(WS_ROOT + '/'): v for k, v in session.files.items()} == {
            p.relative_to(pkg).as_posix(): (p.stat().st_size // 16 + 1) * 16
            for p in skills._files(pkg)} | {'hosts/.folder': 16}


class ConcurrentSession(FakeSession):
    """Calls ``change(files)`` as the install lists the workspace again after its prune."""

    def __init__(self, change) -> None:
        super().__init__()
        self.change = change

    def sql(self, text: str):
        if text.startswith('LIST') and sum(s.startswith('LIST') for s in self.sql_log) == 1:
            self.change(self.files)
        return super().sql(text)


@pytest.mark.parametrize('change, path', [
    (lambda files: files.update({f'{WS_ROOT}/SKILL.md': files[f'{WS_ROOT}/SKILL.md'] + 16}),
     'SKILL.md'),
    (lambda files: files.update({f'{WS_ROOT}/hosts/other/x.md': 32}), 'hosts/other/x.md'),
    (lambda files: files.pop(f'{WS_ROOT}/hosts/snowflake-coco/adapter.md'),
     'hosts/snowflake-coco/adapter.md'),
])
def test_a_concurrent_coco_workspace_install_of_another_build_fails(
        home: Path, change, path: str) -> None:
    # Another build's file of another size, a file this build does not ship, a file gone.
    session = ConcurrentSession(change)
    with pytest.raises(RuntimeError, match=f'Another install changed .*kumo-data-science while '
                                           rf'this one ran \(e\.g\. {re.escape(path)}\)'):
        skills.install_skills('coco', scope='workspace', session=session, workspace='WS')


def test_a_coco_workspace_folder_marker_is_not_a_change(home: Path) -> None:
    session = ConcurrentSession(lambda files: files.update({f'{WS_ROOT}/hosts/.folder': 16}))
    skills.install_skills('coco', scope='workspace', session=session, workspace='WS')


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
    assert names == ['kumo-data-science']


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


class Connected(FakeExtension):
    closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def connect(monkeypatch: pytest.MonkeyPatch) -> list:
    """Records each session the CLI opens, by connection name, in place of Snowpark."""
    opened: list = []

    def _connect(name: str) -> Connected:
        opened.append((name, Connected()))
        return opened[-1][1]

    monkeypatch.setattr(skills, '_connect', _connect)
    return opened


def test_cli_account_scopes_sign_in_with_a_connection(
        home: Path, connect: list, capsys: pytest.CaptureFixture) -> None:
    skills.main(['coco', '--scope', 'workspace', '--workspace', 'WS', '--connection', 'mine'])
    [(name, session)] = connect
    assert name == 'mine' and session.closed
    assert any(s.startswith('PUT') and '"USER$ME".PUBLIC."WS"' in s for s in session.sql_log)
    assert ' -> snow://workspace/"USER$ME".PUBLIC."WS"/' in capsys.readouterr().out
    skills.main(['--scope', 'extension', '--extension', 'EXT', '--stage', 'STG',
                 '--connection', 'other'])
    name, session = connect[1]
    assert name == 'other' and session.closed
    assert any(s.startswith('CREATE CORTEX EXTENSION IF NOT EXISTS EXT FROM @STG/')
               for s in session.sql_log)


def test_cli_closes_the_session_when_the_install_fails(home: Path, connect: list,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError('install failed')

    monkeypatch.setattr(skills, 'install_skills', fail)
    with pytest.raises(SystemExit) as exc:
        skills.main(['coco', '--scope', 'extension', '--connection', 'mine'])
    assert exc.value.code == 1
    assert connect[0][1].closed


def test_cli_reports_a_failed_connection_without_a_traceback(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def refuse(name: str):
        raise RuntimeError(f"Invalid connection_name {name!r}")

    monkeypatch.setattr(skills, '_connect', refuse)
    with pytest.raises(SystemExit) as exc:
        skills.main(['coco', '--scope', 'extension', '--connection', 'nope'])
    assert exc.value.code == 1
    assert "error: --connection nope: Invalid connection_name 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize('argv, message', [
    (['coco', '--scope', 'workspace', '--workspace', 'WS'], 'needs --connection'),
    (['--scope', 'extension'], 'needs --connection'),
    (['coco', '--scope', 'workspace', '--connection', 'mine'], 'needs --workspace'),
    (['claude', '--connection', 'mine'], '--connection is only for coco'),
    (['coco', '--scope', 'project', '--connection', 'mine'], '--connection is only for coco'),
    (['coco', '--scope', 'extension', '--path', 'x', '--connection', 'mine'],
     '--connection is only for coco'),
])
def test_cli_connection_errors(home: Path, connect: list, capsys: pytest.CaptureFixture,
                               argv: list, message: str) -> None:
    with pytest.raises(SystemExit):
        skills.main(argv)
    assert message in capsys.readouterr().err
    assert connect == []   # rejected before connecting


def test_genie_code_cli_shares_the_codex_directory(home: Path) -> None:
    # Genie Code CLI reads ~/.agents/skills; a second copy elsewhere would load as a rival skill.
    [target] = skills.install_skills('genie-cli')
    assert Path(target) == (home / '.agents' / 'skills' / 'kumo-data-science').resolve()
    assert hosts(Path(target)) == ['generic']
    (home / '.genie').mkdir()
    assert [Path(t).parts[-3] for t in skills.install_skills()] == ['.agents']
    (home / '.codex').mkdir()
    assert len(skills.install_skills()) == 1


def build(target: Path, stamp: str) -> None:
    md = target / 'SKILL.md'
    md.write_text(re.sub(r'SKILL-BUILD: [^`*\s]+', f'SKILL-BUILD: {stamp}', md.read_text()))


def test_an_install_reports_the_build_it_replaced(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    [target] = skills.install_skills('claude')
    assert capsys.readouterr().out.endswith('(new)\n')
    skills.install_skills('claude')
    assert capsys.readouterr().out.endswith('(reinstalled)\n')
    build(Path(target), 'older')
    skills.install_skills('claude')
    assert capsys.readouterr().out.endswith('(replaced older)\n')


def test_genie_reports_the_build_it_replaced(
        home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ws = FakeDatabricks({f'{GENIE}/kumo-data-science/SKILL.md': b'`SKILL-BUILD: older`'})
    ws.install(monkeypatch)
    skills.install_skills('genie')
    assert capsys.readouterr().out.endswith('(replaced older)\n')


def test_stale_installs_are_found(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    package_skill.bundle(tmp_path / '_skills')
    monkeypatch.setattr(skills, '_PACKAGER', tmp_path / 'missing.py')
    monkeypatch.setattr(skills, '_BUNDLED', tmp_path / '_skills')
    [claude] = skills.install_skills('claude')
    [codex] = skills.install_skills('codex', scope='project')
    assert skills.stale_skills() == {}
    with pytest.raises(SystemExit) as exc:
        skills.main(['--check'])
    assert exc.value.code == 0
    build(Path(codex), 'older')
    assert skills.stale_skills() == {codex: 'older'}
    with pytest.warns(UserWarning, match=f'{re.escape(codex)} \\(older\\)'):
        skills.warn_if_stale()
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        skills.main(['--check'])
    assert exc.value.code == 1
    assert f'{codex}: older (stale; run' in capsys.readouterr().out


def test_a_stale_install_never_fails_the_import(
        home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `import kumo.engine` runs the check; with warnings as errors it must still import.
    package_skill.bundle(tmp_path / '_skills')
    monkeypatch.setattr(skills, '_PACKAGER', tmp_path / 'missing.py')
    monkeypatch.setattr(skills, '_BUNDLED', tmp_path / '_skills')
    [target] = skills.install_skills('claude')
    build(Path(target), 'older')
    with pytest.warns(skills.SkillStaleWarning):
        skills.warn_if_stale()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        skills.warn_if_stale()


def test_a_source_checkout_is_never_stale(home: Path) -> None:
    [target] = skills.install_skills('claude')
    build(Path(target), 'older')
    assert skills.bundled_build() is None and skills.stale_skills() == {}


def test_the_stale_warning_names_the_call_that_updates_each_install(
        home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    package_skill.bundle(tmp_path / '_skills')
    monkeypatch.setattr(skills, '_PACKAGER', tmp_path / 'missing.py')
    monkeypatch.setattr(skills, '_BUNDLED', tmp_path / '_skills')
    (home / '.claude').mkdir()
    [target] = skills.install_skills('claude', scope='project')
    build(Path(target), 'older')
    call = "install_skills('claude', scope='project')"
    with pytest.warns(UserWarning, match=re.escape(call)) as caught:
        skills.warn_if_stale()
    # A project install is not shared, so the advice to make one is left out.
    assert 'shared by every Python environment' not in str(caught[0].message)
    with pytest.raises(SystemExit):
        skills.main(['--check'])
    assert call in capsys.readouterr().out
    # Following the advice clears it.
    skills.install_skills('claude', scope='project')
    assert skills.stale_skills() == {}
    # A user-scope install is shared by every environment; the advice says so.
    [target] = skills.install_skills('claude')
    build(Path(target), 'older')
    with pytest.warns(UserWarning, match="shared by every Python environment.*scope='project'"):
        skills.warn_if_stale()


def test_detection_skips_the_coco_copy_claude_code_would_shadow(
        home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (home / '.claude').mkdir()
    (home / '.snowflake' / 'cortex').mkdir(parents=True)
    targets = skills.install_skills()
    assert [Path(t).parts[-3] for t in targets] == ['.claude']
    assert not (home / '.snowflake' / 'cortex' / 'skills').exists()
    out = capsys.readouterr().out.splitlines()
    assert out[-1].endswith("scope='extension')") and 'skipped: Cortex Code reads' in out[-1]
