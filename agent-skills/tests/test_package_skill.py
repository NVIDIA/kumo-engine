# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'package_skill.py'
spec = importlib.util.spec_from_file_location('package_skill', SCRIPT)
assert spec and spec.loader
package_skill = importlib.util.module_from_spec(spec)
sys.modules['package_skill'] = package_skill
spec.loader.exec_module(package_skill)


@pytest.mark.parametrize('host', sorted(package_skill.HOSTS))
def test_every_reference_resolves(tmp_path: Path, host: str) -> None:
    pkg = package_skill.build(host, tmp_path)
    assert (pkg / 'SKILL.md').is_file()
    assert package_skill.unresolved(pkg) == []


def test_coco_nests_everything_inside_the_router(tmp_path: Path) -> None:
    pkg = package_skill.build('coco', tmp_path)
    assert pkg == tmp_path / 'coco' / 'kumo-ds' / 'skills' / 'kumo-data-science'
    assert [p.name for p in pkg.parent.iterdir()] == ['kumo-data-science']
    for name in ['kumoai-reference', *package_skill.CAPABILITY]:
        assert (pkg / name / 'SKILL.md').is_file()
    assert (pkg / 'hosts' / 'snowflake-coco' / 'adapter.md').is_file()


def test_build_stamps_the_marker(tmp_path: Path) -> None:
    source = (package_skill.SKILLS / 'kumo-data-science' / 'SKILL.md').read_text()
    assert '**`SKILL-BUILD: source`**' in source
    built = (package_skill.build('coco', tmp_path) / 'SKILL.md').read_text()
    marker = package_skill.MARKER.findall(built)
    assert marker == [f'SKILL-BUILD: {package_skill.build_id()}']
    assert re.fullmatch(r'SKILL-BUILD: (\d{4}-\d{2}-\d{2}-[0-9a-f]+(-dirty)?|unknown)', marker[0])


def test_broken_references_are_reported(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    (pkg / 'accept').mkdir(parents=True)
    (pkg / 'accept' / 'real.md').write_text('')
    (pkg / 'SKILL.md').write_text('\n'.join([
        '[ok](accept/real.md) [gone](accept/missing.md) [web](https://example.com/x.md)',
        '`accept/real.md` `kumo-data-science/accept/real.md` `accept/nope.md`',
        '`kumo/engine/trainer.py` is a source path, not a package path',
    ]))
    assert package_skill.unresolved(pkg) == [
        'SKILL.md: link (accept/missing.md) does not resolve',
        'SKILL.md: `accept/nope.md` does not resolve from the package root',
    ]


def test_generic_ships_only_its_own_host(tmp_path: Path) -> None:
    pkg = package_skill.build('generic', tmp_path)
    assert pkg == tmp_path / 'generic' / 'skills' / 'kumo-data-science'
    assert [p.name for p in (pkg / 'hosts').iterdir()] == ['generic']
    leaks = [str(p.relative_to(pkg)) for p in pkg.rglob('*.md')
             if 'snowflake-coco' in p.read_text() or 'ml_job.md' in p.read_text()]
    assert leaks == []


def test_genie_ships_only_its_own_host(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    assert pkg == tmp_path / 'genie' / 'skills' / 'kumo-data-science'
    assert [p.name for p in (pkg / 'hosts').iterdir()] == ['databricks-genie']
    assert sorted(p.name for p in (pkg / 'hosts' / 'databricks-genie').iterdir()) == [
        'adapter.md', 'job.md', 'runtime.md']
    for name in package_skill.CAPABILITY:
        assert (pkg / name / 'SKILL.md').is_file()
    leaks = [str(p.relative_to(pkg)) for p in pkg.rglob('*.md')
             if re.search(r'snowflake-coco|hosts/generic|ml_job\.md', p.read_text())]
    assert leaks == []


def test_genie_names_no_secret_by_default() -> None:
    """The notebook runs as its user; a service principal appears only as the opt-in form."""
    runtime = (package_skill.SKILLS / 'hosts' / 'databricks-genie' / 'runtime.md').read_text()
    code = '\n'.join(re.findall(r'```python\n(.*?)```', runtime, re.DOTALL))
    assert 'kumo.DatabricksIdentity()' in code
    assert not re.search(r'dbutils|client_secret|token', code)
    assert "__kumo__" in code
    assert "wait_timeout='50s'" in code and 'get_statement' in code


def test_genie_maps_every_snowflake_mechanism() -> None:
    """The capability skills and batch inference name Snowflake mechanisms; the Genie adapter
    maps each to its Databricks counterpart, so the Snowflake case groups are covered."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    adapter, runtime, job = ((host / n).read_text() for n in ('adapter.md', 'runtime.md', 'job.md'))
    section = adapter[adapter.index('## 6. Capability skills'):adapter.index('## 7.')]
    for snowflake, here in [('model registry', 'registered model'), ('stage', 'volume'),
                            ('SHOW VERSIONS', 'version listing'), ('Snowpark', 'run_sql'),
                            ('ML Job', 'job.md'), ('ARRAY', 'kumo_databricks.types')]:
        row = next((line for line in section.splitlines() if snowflake in line), '')
        assert here in row, f'adapter §6 does not map {snowflake!r} to {here!r}'
    assert 'metric view' in adapter and 'semantic_view.md' in adapter
    assert 'ModelRegistryLocation(model_name=MODEL_NAME)' in runtime   # UC assigns the version
    assert 'sink_backend.list_models(registry_connector)' in runtime
    assert 'existing_cluster_id' in job and 'PauseStatus.PAUSED' in job
    assert not re.search(r'clusters\.create|new_cluster', job)


def test_genie_sql_quotes_identifiers_and_names_the_staging_volume() -> None:
    """Unity Catalog names may contain hyphens, and a write creates `_kumo_staging`."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    adapter, runtime = ((host / n).read_text() for n in ('adapter.md', 'runtime.md'))
    code = '\n'.join(re.findall(r'```python\n(.*?)```', runtime, re.DOTALL))
    statements = re.findall(r"run_sql\(f'(.*?)'\)", code, re.DOTALL) + re.findall(
        r"f'(?:CREATE|SELECT)[^']*'", code)
    assert statements
    for st in statements:
        assert not re.search(r'\{(?!q\()[A-Z_]+\}\.', st), f'bare identifier in SQL: {st}'
    assert 'resolve, do not ask' not in adapter
    for text in (adapter, runtime):
        assert '_kumo_staging' in text and 'CREATE VOLUME' in adapter


def max_links_to_predict() -> int:
    source = (package_skill.SKILLS.parents[1] / 'packages' / 'kumo-core' / 'src' / 'kumo_core'
              / 'prediction_options.py')
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'MAX_LINKS_TO_PREDICT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'MAX_LINKS_TO_PREDICT is no longer defined in {source}')


def test_scoring_options_follow_the_sdk() -> None:
    """Rows of batch_inference.md's table: the cap is read from kumo_core.prediction_options;
    the rest is the wording the table must keep."""
    cap = str(max_links_to_predict())
    text = (package_skill.SKILLS / 'kumo-data-science' / 'accept' / 'batch_inference.md').read_text()
    rows = {r.split('|')[1].strip(): r for r in text.splitlines() if r.startswith('| ')}
    assert 'required' in rows['Temporal or static link prediction'] and cap in rows['Temporal or static link prediction']
    assert f'No {cap} cap' in rows['Multilabel ranking']
    assert 'RANK TOP K' in rows['Multiclass, multilabel classification']
    assert '0.5' in rows['Multiclass, multilabel classification']
    assert rows['Regression, forecasting'].split('|')[2].strip().startswith('none')


class FakeWorkspace:
    """Records imports and serves them back, as the Workspace API would."""

    def __init__(self, existing: dict[str, bytes] | None = None, corrupt: str = '',
                 leak: str = '') -> None:
        self.files = dict(existing or {})
        self.corrupt = corrupt
        self.leak = leak      # a path the listing returns whatever is asked, as a buggy API might
        self.deleted: list[str] = []

    def me(self) -> str:
        return 'someone@example.com'

    def api(self, method: str, path: str, body: dict | None = None,
            query: dict | None = None) -> dict:
        if path.endswith('/import'):
            assert body and body['overwrite'] is True
            data = package_skill.base64.b64decode(body['content'])
            self.files[body['path']] = b'x' if self.corrupt in body['path'] and self.corrupt else data
        if path.endswith('/export'):
            assert query
            return dict(content=package_skill.base64.b64encode(self.files[query['path']]).decode())
        return {}

    def listing(self, path: str, dirs: bool = False) -> list[str]:
        files = [p for p in self.files if p.startswith(path + '/') or p == self.leak]
        if not dirs:
            return files
        found = {f.rsplit('/', i)[0] for f in files for i in range(1, f[len(path):].count('/'))}
        return sorted(found - {path}, key=len, reverse=True)

    def delete(self, path: str) -> None:
        assert path in self.files or not self.listing(path), f'non-empty directory {path}'
        self.files.pop(path, None)
        self.deleted.append(path)


def test_deploy_reads_every_file_back(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pkg = package_skill.build('genie', tmp_path)
    target = '/Users/someone@example.com/.assistant/skills/kumo-data-science'
    ws = FakeWorkspace({f'{target}/old-notes.md': b'stale'})
    assert package_skill.deploy_genie(pkg, ws) == target  # type: ignore[arg-type]
    shipped = {f'{target}/{p.relative_to(pkg).as_posix()}': p.read_bytes()
               for p in pkg.rglob('*') if p.is_file()}
    assert {k: ws.files[k] for k in shipped} == shipped
    out = capsys.readouterr().out
    assert 'read back byte for byte' in out
    assert f'left in place: {target}/old-notes.md' in out


def test_prune_deletes_only_what_the_package_does_not_ship(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    skills = '/Users/someone@example.com/.assistant/skills'
    target = f'{skills}/kumo-data-science'
    old = {f'{target}/hosts/databricks/adapter.md': b'old', f'{target}/future/a.md': b'old',
           f'{skills}/tabular/SKILL.md': b'another skill'}
    ws = FakeWorkspace(old)
    package_skill.deploy_genie(pkg, ws, prune=True)  # type: ignore[arg-type]
    assert f'{target}/hosts/databricks/adapter.md' not in ws.files
    assert f'{target}/future/a.md' not in ws.files
    assert ws.files[f'{skills}/tabular/SKILL.md'] == b'another skill'
    assert f'{target}/hosts/databricks-genie/adapter.md' in ws.files
    assert all(d.startswith(target + '/') for d in ws.deleted)


def test_prune_refuses_a_listing_outside_the_target(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    skills = '/Users/someone@example.com/.assistant/skills'
    other = f'{skills}/tabular/SKILL.md'
    ws = FakeWorkspace({other: b'another skill',
                        f'{skills}/kumo-data-science/future/a.md': b'old'}, leak=other)
    with pytest.raises(SystemExit, match='outside it'):
        package_skill.deploy_genie(pkg, ws, prune=True)  # type: ignore[arg-type]
    assert ws.deleted == [] and ws.files[other] == b'another skill'


def test_prune_runs_before_the_upload(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    target = '/Users/someone@example.com/.assistant/skills/kumo-data-science'
    legacy = f'{target}/hosts/databricks/adapter.md'

    class FailingDelete(FakeWorkspace):
        def delete(self, path: str) -> None:
            raise SystemExit(f'delete failed: {path}')

    ws = FailingDelete({legacy: b'old adapter'})
    with pytest.raises(SystemExit, match='delete failed'):
        package_skill.deploy_genie(pkg, ws, prune=True)  # type: ignore[arg-type]
    assert set(ws.files) == {legacy}      # no second host directory was written


def test_deploy_refuses_a_second_host_without_prune(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    target = '/Users/someone@example.com/.assistant/skills/kumo-data-science'
    legacy = f'{target}/hosts/databricks/adapter.md'
    ws = FakeWorkspace({legacy: b'old adapter'})
    with pytest.raises(SystemExit, match='--prune'):
        package_skill.deploy_genie(pkg, ws)  # type: ignore[arg-type]
    assert set(ws.files) == {legacy}      # refused before anything was written


def test_deploy_fails_on_a_mismatch(tmp_path: Path) -> None:
    pkg = package_skill.build('genie', tmp_path)
    with pytest.raises(SystemExit, match='deploy check failed'):
        package_skill.deploy_genie(pkg, FakeWorkspace(corrupt='runtime.md'))  # type: ignore[arg-type]


def test_core_names_no_host() -> None:
    """The portable core reaches its host as `hosts/<host>/`, never by a host's own name."""
    hosts = sorted(p.name for p in (package_skill.SKILLS / 'hosts').iterdir() if p.is_dir())
    named = re.compile(r'hosts/(?:' + '|'.join(map(re.escape, hosts)) + r')\b')
    hits = [f'{p.relative_to(package_skill.SKILLS)}: {m.group(0)}'
            for p in sorted(package_skill.SKILLS.rglob('*.md'))
            if p.relative_to(package_skill.SKILLS).parts[0] != 'hosts'
            for m in named.finditer(p.read_text())]
    assert hits == []


@pytest.mark.parametrize('host', sorted(package_skill.HOSTS))
def test_adapter_covers_what_the_core_defers(host: str) -> None:
    """Preflight §0, the intake contract and batch inference defer these to the adapter."""
    adapter = next(rel for rel in package_skill.HOSTS[host]['hosts'] if rel.endswith('adapter.md'))
    text = (package_skill.SKILLS / 'hosts' / adapter).read_text()
    for section in ['Source scope', 'Access', 'Destinations', 'reflight',
                    'untime column constraints' if host != 'coco' else 'Runtime-dependent']:
        assert section in text, f'{adapter} does not cover {section}'


@pytest.mark.parametrize('host', sorted(package_skill.HOSTS))
def test_every_host_ships_a_runtime(host: str) -> None:
    """The router reads `hosts/<host>/runtime.md` at Step 2."""
    assert any(rel.endswith('/runtime.md') for rel in package_skill.HOSTS[host]['hosts'])


def test_workspace_requires_https() -> None:
    assert package_skill.Workspace('dbc.example.com', 't').host == 'https://dbc.example.com'
    assert package_skill.Workspace('https://dbc.example.com/', 't').host == 'https://dbc.example.com'
    with pytest.raises(SystemExit):
        package_skill.Workspace('http://dbc.example.com', 't')
