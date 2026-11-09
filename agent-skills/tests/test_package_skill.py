# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
import re
import shutil
import sys
import zipfile
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
    for name in package_skill.CAPABILITY:
        assert (pkg / name / 'SKILL.md').is_file()
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
    assert re.fullmatch(r'SKILL-BUILD: (\d{4}-\d{2}-\d{2}-[0-9a-f]{8}(-dirty)?|unknown-[0-9a-f]{12})',
                        marker[0])


def test_a_build_without_git_is_named_by_its_content(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two archives built without git must not share a stamp, or `--check` takes one for the other.
    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')

    skills = tmp_path / 'skills'
    shutil.copytree(package_skill.SKILLS, skills)
    monkeypatch.setattr(package_skill, 'SKILLS', skills)
    monkeypatch.setattr(package_skill.subprocess, 'run', no_git)
    first = package_skill.build_id()
    assert re.fullmatch(r'unknown-[0-9a-f]{12}', first)
    assert package_skill.build_id() == first
    router = skills / 'kumo-data-science' / 'SKILL.md'
    router.write_text(router.read_text() + '\n', encoding='utf-8')
    assert package_skill.build_id() != first


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


def test_genie_gates_hold_for_routed_requests_jobs_and_saved_models() -> None:
    """Live Genie Code skipped the adapter on job and population follow-ups and acted without
    confirmation; the router sends every request through the adapter, which names those acts."""
    router = (package_skill.SKILLS / 'kumo-data-science' / 'SKILL.md').read_text()
    assert '## Read the host adapter first' in router
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    adapter, job, runtime = (' '.join((host / n).read_text().split())
                             for n in ('adapter.md', 'job.md', 'runtime.md'))
    gate = adapter[adapter.index('## 0.'):adapter.index('## 1.')]
    for act in ('unpausing', 'submitting', 'capability skill', 'not confirmation'):
        assert act in gate, act
    assert 'one gate' in adapter and 'Never propose retraining' in adapter
    assert 'creates its own cluster' in job and 'adds no libraries' in job
    assert 'names the destination verbatim and nothing else' in runtime


def test_genie_pause_runs_at_once_and_everything_else_stays_confirmed() -> None:
    """A plain-words pause of a named (or the only matching) job runs without a confirmation turn
    and is reported after; unpausing and cancelling a run still need confirmation, and job.md and
    adapter.md say the same."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    adapter, job = (' '.join((host / n).read_text().split()) for n in ('adapter.md', 'job.md'))
    section = job[job.index('### Pausing'):job.index('## 9.')]
    for text in (section, adapter[adapter.index('## 0.'):adapter.index('## 1.')]):
        assert 'plain words' in text and 'exactly one job' in text
        assert 'at once' in text and 'previous state' in text
    assert 'no confirmation turn' in section and 'pause_status=PAUSED' in section
    assert 'list the candidates and ask; pause nothing' in section
    assert 'unpausing it needs the user\'s confirmation' in section
    assert 'Cancelling an active run discards work' in section
    assert 'ask to confirm cancelling the run' in section
    assert 'only after the user confirms' in job   # cancel_run at the deadline
    assert 'scheduling or unpausing a job' in adapter and 'unpausing or pausing' not in adapter
    # no other skill file says a pause needs confirmation
    for path in package_skill.SKILLS.rglob('*.md'):
        if path.name not in ('job.md', 'adapter.md'):
            assert 'unpausing or pausing' not in ' '.join(path.read_text().split()), path


def test_genie_saved_model_template_carries_every_line_other_files_require() -> None:
    """The §6.1 template is the one the agent copies for scoring; a line other files rely on
    (a confirmed staging volume, embeddings) must be in it, and its tags must be defined."""
    skills = package_skill.SKILLS
    host = skills / 'hosts' / 'databricks-genie'
    flat = lambda path: ' '.join(path.read_text().split())  # noqa: E731
    adapter, runtime = flat(host / 'adapter.md'), flat(host / 'runtime.md')
    batch = flat(skills / 'kumo-data-science' / 'accept' / 'batch_inference.md')
    section = adapter[adapter.index('### 6.1'):adapter.index('## 7.')]
    template = section[section.index('```'):section.index('```', section.index('```') + 3)]
    assert 'Staging volume:' in runtime and '- Embeddings:' in batch and '- Output:' in batch
    for line in ('- Output:', '- Staging volume:', '- Embeddings:'):
        assert line in template, f'§6.1 template lacks {line!r}'
    assert '[recorded]' in template and 'marks a value read from' in section
    assert 'refines `batch_inference.md` Node 1c and 1d' in section


def test_genie_gate_blocks_every_mutation_and_scoring_keeps_what_the_user_named() -> None:
    """Before confirmation every workspace or local mutation is blocked (read-only preflight is
    the exception); the §6.1 template restates a user-named source, subset or date first and the
    recorded or default value only otherwise."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    adapter = ' '.join((host / 'adapter.md').read_text().split())
    gate = adapter[adapter.index('## 0.'):adapter.index('## 1.')]
    for verb in ('create, update, edit, rename, move, upload, import or delete',
                 'cluster, warehouse, table, volume or model'):
        assert verb in gate, verb
    assert 'Preflight reads' in gate and 'need no confirmation' in gate
    assert 'plain words' in gate and 'that has a Node 1d gate' in gate
    assert 'any field the Node 1d restatement lists' in gate
    section = adapter[adapter.index('### 6.1'):adapter.index('## 7.')]
    template = section[section.index('```'):section.index('```', section.index('```') + 3)]
    assert 'the user\'s source, columns checked against the bundle\'s graph [supplied]' in template
    assert 'the bundle\'s source tables [recorded], only when none was named' in template
    assert 'the user\'s subset/date [supplied]' in template
    assert '[default], only when none was given' in template
    assert 'each table\'s columns with its keys, time column and links' in section
    batch = ' '.join((package_skill.SKILLS / 'kumo-data-science' / 'accept'
                      / 'batch_inference.md').read_text().split())
    assert 'a source the user named' in batch and 'If the user names a subset' in batch


def test_genie_scoring_reads_the_graph_from_the_published_bundle() -> None:
    """A Kumo bundle carries `training-spec.json` (query + `graph_config`), so scoring reads that
    through the Files API and the training notebook writes no separate record."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    flat = lambda path: ' '.join(path.read_text().split())  # noqa: E731
    adapter, runtime, job = (flat(host / n) for n in ('adapter.md', 'runtime.md', 'job.md'))
    section = adapter[adapter.index('### 6.1'):adapter.index('## 7.')]
    for text in ('training-spec.json', '`graph_config`', '`predictive_query`',
                 'databricks api get /api/2.0/fs/files/Models/<catalog>/<schema>/<model>/<N>/training-spec.json',
                 '/api/2.0/fs/directories/Models/', 'artifacts/published-model/',
                 'Read-only calls only', 'never read a credential file',
                 'rebuilt from the version\'s bundle', 'read from the version\'s bundle',
                 'spec carries no `graph_config`', 'Never propose retraining'):
        assert text in section, text
    # `[recorded]` stays on the graph, source and query fields
    assert '[recorded]' in section
    publish = runtime[runtime.index('### 5.1'):runtime.index('### 5.2')]
    assert 'Nothing else is written' in publish
    for gone in ('record.json', 'graph.py', 'GRAPH_CODE', 'records/', 'CREATE VOLUME',
                 'model_versions.update', 'kumo training record', 'ws.model_versions'):
        assert gone not in publish, gone
    for name, text in (('adapter', adapter), ('runtime', runtime), ('job', job)):
        for gone in ('kumo training record', 'training record', 'record.json', 'record files',
                     'dbfs:/Volumes'):
            assert gone not in text, (name, gone)
    assert 'bundle\'s `graph_config` tables, column lists and edges rebuilt' in job
    assert 're-read that version\'s bundle' in job
    for gone in ('mlflow', 'credentials-for-read', 'list-artifacts', 'curl', 'signed_uri'):
        assert gone not in section.lower().replace('wrapped for mlflow', ''), gone



def test_genie_threshold_survives_in_the_bundle_and_scoring_reads_it() -> None:
    """With no separate record, a confirmed threshold must go into the bundle as
    `scoring-defaults.json` (`fit(scoring_defaults=...)`) and §6.1 must read it back."""
    host = package_skill.SKILLS / 'hosts' / 'databricks-genie'
    flat = lambda path: ' '.join(path.read_text().split())  # noqa: E731
    adapter, runtime, job = (flat(host / n) for n in ('adapter.md', 'runtime.md', 'job.md'))
    publish = runtime[runtime.index('### 5.1'):runtime.index('### 5.2')]
    assert 'scoring_defaults=kumo.ScoringDefaults(binary_classification_threshold=THRESHOLD)' \
        in publish
    assert 'only when the user confirmed a threshold' in publish
    section = adapter[adapter.index('### 6.1'):adapter.index('## 7.')]
    assert 'Read `scoring-defaults.json` from the same directory' in section
    assert 'the threshold recorded with the model' in section
    # A version with no spec cannot load: that is the unresolved-address ask-back, not a graph ask
    assert 'it is not a Kumo bundle and cannot be loaded at all' in section
    assert 'Kumo bundle whose spec carries no `graph_config`' in section
    assert '§0 restatement' not in job and '§6.1 restatement' in job


def test_custom_populations_allow_signed_temporal_link_weights_after_rescaling() -> None:
    """kumo_core allows signed weights only for temporal link prediction; an approved
    rescaling must not be held to a stricter rule than the builder."""
    reference = ' '.join((package_skill.SKILLS / 'kumo-custom-populations' / 'references'
                          / 'populations.md').read_text().split())
    assert 'temporal link tasks permit signed weights' in reference
    assert ('finite and nonnegative, except the signed weights temporal link prediction '
            'permits above') in reference
    assert 'disallowed negative weight' in reference

def test_genie_no_code_rule_names_its_scope_and_agrees_with_inspection_skills() -> None:
    """Explanations ask for plotting code in the reply and graph visualization needs no intake;
    the adapter's no-code rule must scope itself to requests that have a Node 1d gate."""
    skills = package_skill.SKILLS
    adapter = ' '.join((skills / 'hosts' / 'databricks-genie' / 'adapter.md').read_text().split())
    gate = adapter[adapter.index('## 0.'):adapter.index('## 1.')]
    assert 'that has a Node 1d gate' in gate
    assert ('An inspection request has no Node 1d — explanations, training diagnostics, '
            'graph visualization') in gate
    assert 'code in the reply included' in gate
    assert 'explanations;' not in gate   # no longer listed among the gated capability skills
    assert 'stop at each' in gate and 'single combined restatement' in gate
    explanations = ' '.join((skills / 'kumo-model-explanations' / 'SKILL.md').read_text().split())
    assert 'the plotting code in the reply' in explanations


def test_custom_populations_treat_weights_like_splits() -> None:
    """Weights and splits share one rule: show, approve, apply read-only. Neither file may
    still say to decline weights or forbid a read-only transform outright."""
    base = package_skill.SKILLS / 'kumo-custom-populations'
    skill = ' '.join((base / 'SKILL.md').read_text().split())
    reference = ' '.join((base / 'references' / 'populations.md').read_text().split())
    for text in (skill, reference):
        assert 'decline' not in text and 'even after an earlier confirmation' not in text
        assert 'not in a read-only query either' not in text
        assert 'exact formula' in text and 'explicit approval' in text
        assert 'read-only `connector.sql(...)`' in text
        assert 'is not approval' in text and 'CREATE TABLE AS' in text
    assert 'Weights follow the same rule as splits' in reference
    assert 'resulting weight distribution' in reference
    assert 'Validation still applies to the result' in skill


def test_genie_job_fix_offer_does_not_bundle_a_resubmit() -> None:
    """One yes must not approve a notebook edit and a new GPU run."""
    job = ' '.join((package_skill.SKILLS / 'hosts' / 'databricks-genie' / 'job.md')
                   .read_text().split())
    assert 'then resubmit?' not in job
    assert 'Resubmitting is a separate confirmation' in job
    assert 'the one edit of an existing notebook the user can approve here' in job


def test_explanations_checklist_keeps_the_conditions_the_body_gives() -> None:
    text = ' '.join((package_skill.SKILLS / 'kumo-model-explanations' / 'SKILL.md')
                    .read_text().split())
    checklist = text[text.index('## Return a useful answer'):]
    assert 'normalized across the graph as observed' in checklist
    assert 'when the total is positive' in checklist
    assert 'when the file records one, mean target' in checklist


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


def test_archive_zips_each_host_as_its_bundle(tmp_path: Path) -> None:
    """A download zip is the directory install_skills() installs, at the zip's root."""
    zips = package_skill.archive(tmp_path / 'zips')
    assert [z.name for z in zips] == [f'kumo-data-science-{h}.zip' for h in sorted(package_skill.HOSTS)]
    package_skill.bundle(tmp_path / 'bundle')
    for host, z in zip(sorted(package_skill.HOSTS), zips):
        root = tmp_path / 'bundle' / host
        shipped = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
        with zipfile.ZipFile(z) as zf:
            assert sorted(zf.namelist()) == shipped
            assert {n.split('/')[0] for n in zf.namelist()} == {'kumo-data-science'}
            for name in shipped:
                assert zf.read(name) == (root / name).read_bytes()
        hosts = {n.split('/')[2] for n in shipped if n.startswith('kumo-data-science/hosts/')}
        assert len(hosts) == 1, f'{host} zip carries {sorted(hosts)}'


def test_archive_is_reproducible(tmp_path: Path) -> None:
    first = package_skill.archive(tmp_path / 'a')
    second = package_skill.archive(tmp_path / 'b')
    assert [z.read_bytes() for z in first] == [z.read_bytes() for z in second]


def test_archive_is_the_same_built_on_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A zip entry records the system that wrote it, 0 on Windows unless set.
    first = package_skill.archive(tmp_path / 'a')
    monkeypatch.setattr(zipfile.sys, 'platform', 'win32')
    second = package_skill.archive(tmp_path / 'b')
    assert [z.read_bytes() for z in first] == [z.read_bytes() for z in second]


def test_workspace_requires_https() -> None:
    assert package_skill.Workspace('dbc.example.com', 't').host == 'https://dbc.example.com'
    assert package_skill.Workspace('https://dbc.example.com/', 't').host == 'https://dbc.example.com'
    with pytest.raises(SystemExit):
        package_skill.Workspace('http://dbc.example.com', 't')


def test_text_io_names_its_encoding() -> None:
    """The skill holds non-ASCII text; Windows' default (cp1252) cannot decode it."""
    sources = [SCRIPT, SCRIPT.parents[2] / 'packages' / 'kumo-engine' / 'kumo' / 'engine' / 'skills.py']
    unnamed = [
        f'{src.name}:{node.lineno}' for src in sources
        for node in ast.walk(ast.parse(src.read_text(encoding='utf-8')))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ('read_text', 'write_text')
        and not any(k.arg == 'encoding' for k in node.keywords)
    ]
    assert not unnamed, unnamed


def test_followup_review_fixes_are_pinned() -> None:
    """A MODEL_VERSION repair differs from a version change; the runtime's no-edit rule carves
    out the job notebook; a mapping is never a derived table; the adapter's read list is full."""
    skills = package_skill.SKILLS
    flat = lambda p: ' '.join(p.read_text().split())
    job = flat(skills / 'hosts' / 'databricks-genie' / 'job.md')
    assert 'Tell a repair from a change first' in job and 're-read that version' in job
    assert 'resubmit only after the user confirms that restatement' in job
    runtime = flat(skills / 'hosts' / 'databricks-genie' / 'runtime.md')
    assert 'apart from the one edit `job.md` §7 allows' in runtime
    refs = flat(skills / 'kumo-custom-populations' / 'references' / 'populations.md')
    assert 'create no mapped copy or derived table' in refs and 'approves writing one' not in refs
    adapter = flat(skills / 'hosts' / 'databricks-genie' / 'adapter.md')
    assert 'probe — nothing else' not in adapter and 'read clusters, registered models, runs' in adapter
    router = flat(skills / 'kumo-data-science' / 'SKILL.md')
    assert 'Where it has a section on what you may do before the user confirms' in router
