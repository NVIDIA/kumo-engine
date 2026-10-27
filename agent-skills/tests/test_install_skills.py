# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`kumo.engine.install_skills()`: the wheel carries every host package and installs the right one.

`kumo/engine/skills.py` uses only the standard library, so it is loaded here by path, without
importing `kumo.engine` and the training stack behind it.
"""
import importlib.util
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

    class NotFound(Exception):
        pass

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.current_user = types.SimpleNamespace(
            me=lambda: types.SimpleNamespace(user_name='someone@example.com'))
        self.workspace = self

    def delete(self, path: str, recursive: bool = False) -> None:
        gone = [p for p in self.files if p.startswith(path + '/')]
        if not gone:
            raise self.NotFound(path)
        assert recursive
        for p in gone:
            del self.files[p]

    def mkdirs(self, path: str) -> None:
        pass

    def upload(self, path: str, content, format, overwrite: bool) -> None:
        assert format == 'AUTO' and overwrite
        self.files[path] = content.read()

    def download(self, path: str, format):
        import io
        return io.BytesIO(self.files[path])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mods = {
            'databricks': types.ModuleType('databricks'),
            'databricks.sdk': types.SimpleNamespace(WorkspaceClient=lambda: self),
            'databricks.sdk.errors': types.SimpleNamespace(NotFound=self.NotFound),
            'databricks.sdk.service': types.ModuleType('databricks.sdk.service'),
            'databricks.sdk.service.workspace': types.SimpleNamespace(
                ImportFormat=types.SimpleNamespace(AUTO='AUTO'),
                ExportFormat=types.SimpleNamespace(AUTO='AUTO')),
        }
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)


def test_genie_on_databricks(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skills_dir = '/Users/someone@example.com/.assistant/skills'
    old = {f'{skills_dir}/kumo-data-science/hosts/generic/adapter.md': b'stale',
           f'{skills_dir}/tabular/SKILL.md': b'another skill'}
    ws = FakeDatabricks(dict(old))
    ws.install(monkeypatch)
    monkeypatch.setenv('DATABRICKS_RUNTIME_VERSION', '17.3')
    assert skills.install_skills() == [f'{skills_dir}/kumo-data-science']
    shipped = {p.split('/kumo-data-science/')[1] for p in ws.files if '/kumo-data-science/' in p}
    assert 'SKILL.md' in shipped and 'hosts/databricks-genie/adapter.md' in shipped
    assert not any(p.startswith('hosts/generic') for p in shipped)
    assert ws.files[f'{skills_dir}/tabular/SKILL.md'] == b'another skill'


def test_genie_fails_when_a_file_does_not_read_back(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = FakeDatabricks({})
    ws.install(monkeypatch)
    monkeypatch.setattr(ws, 'download', lambda path, format: __import__('io').BytesIO(b'x'))
    with pytest.raises(RuntimeError, match='did not read back'):
        skills.install_skills('genie')


class FakeSession:
    def __init__(self) -> None:
        self.sql_log: list[str] = []

    def sql(self, text: str):
        self.sql_log.append(text)
        return types.SimpleNamespace(collect=lambda: [('ME',)])


def test_coco_extension(home: Path) -> None:
    session = FakeSession()
    assert skills.install_skills('coco', scope='extension', session=session) == [
        'KUMO_DATA_SCIENCE']
    log = session.sql_log
    assert "ENCRYPTION=(TYPE='SNOWFLAKE_SSE')" in log[0]
    puts = [s for s in log if s.startswith('PUT')]
    assert puts and all("'@KUMO_SKILL_PKG/kumo-ds/skills/kumo-data-science" in s for s in puts)
    assert any('/hosts/snowflake-coco/' in s for s in puts)
    assert log[-1] == 'ALTER CORTEX EXTENSION KUMO_DATA_SCIENCE ADD VERSION FROM @KUMO_SKILL_PKG/kumo-ds/'


def test_coco_workspace(home: Path) -> None:
    session = FakeSession()
    [root] = skills.install_skills('coco', scope='workspace', session=session, workspace='MY_WS')
    assert root == ('snow://workspace/USER$ME.PUBLIC."MY_WS"/versions/live/'
                    '.snowflake/cortex/skills/kumo-data-science')
    assert all(f"'{root}" in s for s in session.sql_log if s.startswith('PUT'))
    with pytest.raises(ValueError, match='needs workspace'):
        skills.install_skills('coco', scope='workspace', session=session)


def test_snowflake_notebook_asks_for_a_destination(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path.exists
    monkeypatch.setattr(Path, 'exists',
                        lambda self: str(self) == '/snowflake/session/token' or real(self))
    with pytest.raises(ValueError, match='name the destination'):
        skills.install_skills()
