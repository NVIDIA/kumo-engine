# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-function Snowflake ML Job submission without a live account."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kumo_core import runtime_progress
from kumo_core.callbacks import (
    AutoTrainingProgressLogger,
    PredictionProgressLogger,
)
from kumo_snowflake import (
    SnowflakeMLJob,
    SnowflakeMLJobConfigurationError,
    SnowflakeMLJobError,
    SnowflakeMLJobFailed,
    SnowflakeMLJobPermissionError,
    SnowflakeMLJobPreflightError,
    SnowflakeMLJobResult,
    SnowflakeMLJobStatus,
    SnowflakeMLJobSubmissionError,
    SnowflakeMLJobTransientError,
    get_ml_job,
    submit_ml_job,
)
from kumo_snowflake._ml_job_runtime import (
    _decode_argument,
    _json_value,
    _RemoteObserver,
    execute_user_function,
)
from kumo_snowflake._ml_job_stage import StageStore, validate_stage_name
from kumo_snowflake._ml_job_wheel import (
    discover_installed_wheel,
    stage_wheel,
    validate_wheel,
)
from kumo_snowflake.ml_job_contracts import _TRANSPORT_KEY
from kumo_snowflake.ml_jobs import (
    _decode_result_value,
    _detach_callable,
    _exception_kind,
    _json_input,
    _validate_callable,
)
from kumoapi.common import JobStatus
from kumoapi.jobs import (
    AutoTrainerProgress,
    EpochMetrics,
    JobEventLogEntry,
    JobStatusReport,
    PredictionProgress,
    TrainerProgress,
)
from kumoapi.json_serde import to_json_dict
from kumoapi.plugin import PublishedModel

STAGE = '@TEST_DB.PUBLIC.KUMO_JOBS'
MODEL_DIGEST = 'a' * 64


def _published_model(
    training_job_id: str = 'trainingjob-1',
    content_sha256: str = MODEL_DIGEST,
) -> PublishedModel:
    name = 'KUMO_ML_JOB_TEST_EXECUTION'
    version = f'V_{content_sha256}'
    return PublishedModel(
        name=name,
        version=version,
        uri=f'TEST_DB.PUBLIC.{name}/{version}',
        metadata={
            'schema_version': '1',
            'training_job_id': training_job_id,
            'content_sha256': content_sha256,
        },
    )


def user_function(value: int = 1) -> dict[str, int]:
    return {'value': value}


def failing_user_function() -> None:
    raise LookupError('user training failed')


def function_with_prediction_progress(log_dir: str | None) -> str:
    callback = PredictionProgressLogger(log_dir, 0, 5)
    callback.progress.completed_iterations = 5
    callback.on_predict_end(None, None)
    return 'done'


def function_with_training_progress(log_dir: str) -> str:
    callback = AutoTrainingProgressLogger(log_dir, 2)
    callback.set_current_trainer(
        SimpleNamespace(
            version='0',
            training_progress_logger=SimpleNamespace(
                progress=TrainerProgress(), ),
        ))
    callback.on_train_start(None, None)
    callback.on_train_end(None, None)
    return 'done'


def function_with_sequential_progress(log_dir: str) -> str:
    function_with_prediction_progress(log_dir)
    function_with_training_progress(log_dir)
    return 'done'


def function_with_training_then_prediction(log_dir: str) -> str:
    callback = AutoTrainingProgressLogger(log_dir, 1)
    callback.set_current_trainer(
        SimpleNamespace(
            version='0',
            training_progress_logger=SimpleNamespace(
                progress=TrainerProgress(), ),
        ))
    callback.on_train_start(None, None)
    callback.on_train_end(None, None)
    function_with_prediction_progress(log_dir)
    return 'done'


def function_with_transport_named_argument(
    __kumo_ml_job_transport__: str, ) -> str:
    return __kumo_ml_job_transport__


class _Query:
    def __init__(self, session: '_Session', sql: str,
                 params: list[Any] | None) -> None:
        self.session = session
        self.sql = sql
        self.params = params

    def collect(self) -> list[Any]:
        self.session.statements.append((self.sql, self.params))
        for prefix, error in self.session.query_errors.items():
            if self.sql.startswith(prefix):
                raise error
        if self.sql.startswith('SHOW COMPUTE POOLS'):
            return self.session.compute_pools
        if self.sql.startswith('SHOW NOTEBOOK PROJECTS'):
            return ([{
                'name': 'ML_JOB_TESTS'
            }] if self.session.notebook_project else [])
        if self.sql.startswith('DESCRIBE NOTEBOOK PROJECT'):
            return [{
                'property':
                'default_version_location_uri',
                'value':
                'snow://notebook project/DB.PUBLIC.PROJECT/versions/v1',
            }]
        if self.sql.startswith('SHOW TAGS'):
            return self.session.tags
        if self.sql.startswith('LIST '):
            uri = self.sql.removeprefix('LIST ').strip("'")
            return [{
                'name': key.removeprefix('@'),
                'size': len(data),
                'md5': hashlib.md5(data, usedforsecurity=False).hexdigest(),
            } for key, data in self.session.objects.items()
                    if key.startswith(uri)]
        if self.sql.startswith('REMOVE '):
            prefix = self.sql.removeprefix('REMOVE ')
            for key in list(self.session.objects):
                if key.startswith(prefix):
                    del self.session.objects[key]
            return []
        if self.sql.startswith('COPY FILES'):
            matched = re.match(r'COPY FILES INTO (.+?) FROM (.+?) FILES',
                               self.sql)
            assert matched is not None
            target, source = matched.groups()
            source = source.strip().strip("'")
            filename = re.search(r"FILES = \('([^']+)'\)", self.sql).group(1)
            source_uri = source + filename
            if source_uri not in self.session.objects:
                return []
            self.session.objects[target +
                                 filename] = self.session.objects[source_uri]
            return [{'file': filename}]
        return []


class _Files:
    def __init__(self, session: '_Session') -> None:
        self.session = session

    def put_stream(self, stream: Any, target: str, **kwargs: Any) -> Any:
        del kwargs
        data = stream.read()
        self.session.puts.append(target)
        self.session.writes.append((target, data))
        self.session.objects[target] = data
        return SimpleNamespace(target=target, status='UPLOADED')

    def get_stream(self, target: str, **kwargs: Any) -> io.BytesIO:
        del kwargs
        self.session.gets.append(target)
        if target not in self.session.objects:
            raise RuntimeError(f'{target} does not exist')
        return io.BytesIO(self.session.objects[target])


class _Session:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.statements: list[tuple[str, Any]] = []
        self.puts: list[str] = []
        self.gets: list[str] = []
        self.writes: list[tuple[str, bytes]] = []
        self.file = _Files(self)
        self.query_errors: dict[str, BaseException] = {}
        self.compute_pools = [{
            'name': 'KUMO_GPU_POOL',
            'instance_family': 'GPU_NV_S',
        }]
        self.tags = [{'name': 'REQUEST'}]
        self.notebook_project = False
        self.close_calls = 0
        self.close_error: BaseException | None = None

    def sql(self, sql: str, params: list[Any] | None = None) -> _Query:
        return _Query(self, sql, params)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _Remote:
    def __init__(self, service_id: str = 'DB.PUBLIC.MLJOB_1') -> None:
        self.id = service_id
        self.status = 'PENDING'
        self.cancelled = False
        self.cancel_error: BaseException | None = None
        self.value: Any = None

    def cancel(self) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled = True

    def get_logs(self, **kwargs: Any) -> str:
        del kwargs
        return 'training log'

    def result(self, timeout: float = -1) -> Any:
        del timeout
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _Jobs:
    def __init__(self, remote: _Remote | None = None) -> None:
        self.handle = remote or _Remote()
        self.options: dict[str, Any] = {}
        self.payload: Any = None
        self.arguments: tuple[Any, ...] | None = None
        self.get_error: BaseException | None = None

    def remote(self, pool: str, **options: Any) -> Any:
        self.options = {'compute_pool': pool, **options}

        def decorate(payload: Any) -> Any:
            self.payload = payload

            def invoke(*args: Any) -> _Remote:
                self.arguments = args
                return self.handle

            return invoke

        return decorate

    def get_job(self, service_id: str, session: Any) -> _Remote:
        del service_id, session
        if self.get_error is not None:
            raise self.get_error
        return self.handle


@pytest.fixture
def fat_wheel(tmp_path: Path) -> Any:
    path = tmp_path / 'kumo_fat_bundle-3.1.0-cp310-cp310-linux_x86_64.whl'
    components = [
        'bbhash',
        'diskgraph',
        'kumo-api',
        'kumo-core',
        'kumo-pql',
        'kumo-snowflake',
        'kumoai',
        'kumoml-enterprise',
        # The build records the normalized distribution spelling from wheel
        # metadata. Validation must treat underscores and hyphens equally.
        'pyg_lib',
        'torch-geometric',
        'torch_scatter',
    ]
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('kumoai/__init__.py', '')
        archive.writestr('kumo_core/__init__.py', '')
        archive.writestr('kumo_snowflake/__init__.py', '')
        archive.writestr(
            'kumo_bundle_manifest.json',
            json.dumps({
                'components': [{
                    'name': name
                } for name in components],
                'requires_at_install': ['pytorch-frame==0.3.0'],
            }))
        archive.writestr('kumo_fat_bundle-3.1.0.dist-info/WHEEL',
                         'Wheel-Version: 1.0\n')
    return validate_wheel(path)


def _patch_submission(monkeypatch: Any, wheel: Any, jobs: _Jobs) -> None:
    monkeypatch.setattr('kumo_snowflake.ml_jobs.discover_installed_wheel',
                        lambda: wheel)
    monkeypatch.setattr('kumo_snowflake.ml_jobs._load_jobs_api', lambda: jobs)


def test_submit_preflights_stages_one_wheel_and_returns_handle(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    _patch_submission(monkeypatch, fat_wheel, jobs)

    job = submit_ml_job(
        user_function,
        compute_pool='KUMO_GPU_POOL',
        stage_name=STAGE,
        args=(4, ),
        tags={'TEST_DB.PUBLIC.REQUEST': 'run-1'},
        artifact_repositories=['SNOWFLAKE.SNOWPARK.PYPI_SHARED_REPOSITORY'],
        session=session,
    )

    assert job.service_id == jobs.handle.id
    assert jobs.options['pip_requirements'] == [f'./{fat_wheel.name}']
    assert 'runtime_environment' not in jobs.options
    assert jobs.payload.__name__ == 'execute_user_function'
    assert jobs.arguments[0] is not user_function
    assert jobs.arguments[0](4) == user_function(4)
    assert jobs.arguments[1] == [4]
    assert any(
        sql.startswith('SHOW COMPUTE POOLS') for sql, _ in session.statements)
    assert any(sql.startswith('ALTER SERVICE') for sql, _ in session.statements)
    assert any('KUMO_SUBMITTED_BY' in jobs.options['env_vars'] for _ in [None])
    assert 'PIP_NO_DEPS' not in jobs.options['env_vars']
    job.close()
    assert session.close_calls == 0


def test_submit_resolves_quoted_identifiers_at_every_sql_boundary(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session = _Session()
    session.compute_pools = [{
        'name': 'GPU.pool',
        'instance_family': 'GPU_NV_S',
    }]
    session.tags = [{'name': 'request.tag'}]
    jobs = _Jobs(_Remote('"job.db"."job schema"."job.service"'), )
    _patch_submission(monkeypatch, fat_wheel, jobs)

    job = submit_ml_job(
        user_function,
        compute_pool='"GPU.pool"',
        stage_name='@"job.db"."job schema"."job.stage"',
        tags={'"tag.db"."tag schema"."request.tag"': 'run-1'},
        session=session,
    )

    assert jobs.options['compute_pool'] == '"GPU.pool"'
    statements = [sql for sql, _ in session.statements]
    assert "SHOW COMPUTE POOLS LIKE 'GPU.pool'" in statements
    assert ('SHOW TAGS IN SCHEMA "tag.db"."tag schema"' in statements)
    assert any(
        sql.startswith(
            'ALTER SERVICE "job.db"."job schema"."job.service" SET TAG '
            '"tag.db"."tag schema"."request.tag" = ?') for sql in statements)
    job.close()


def test_submit_serializes_published_model_as_json_metadata(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    _patch_submission(monkeypatch, fat_wheel, jobs)
    model = _published_model()

    submit_ml_job(
        user_function,
        compute_pool='KUMO_GPU_POOL',
        stage_name=STAGE,
        args=(model, ),
        session=session,
    )

    assert jobs.arguments is not None
    assert _decode_argument(jobs.arguments[1][0]) == model


def test_submit_preserves_transport_named_keyword_argument(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    _patch_submission(monkeypatch, fat_wheel, jobs)

    submit_ml_job(
        function_with_transport_named_argument,
        compute_pool='KUMO_GPU_POOL',
        stage_name=STAGE,
        kwargs={_TRANSPORT_KEY: 'ordinary argument'},
        session=session,
    )

    assert jobs.arguments is not None
    assert _decode_argument(jobs.arguments[2]) == {
        _TRANSPORT_KEY: 'ordinary argument',
    }


def test_content_addressed_wheel_is_reused_without_second_put(
        fat_wheel: Any) -> None:
    session = _Session()
    store = StageStore(session, STAGE)
    first = stage_wheel(store, fat_wheel)
    puts = list(session.puts)
    second = stage_wheel(store, fat_wheel)
    assert first == second
    assert session.puts == puts


def test_forged_wheel_cache_marker_cannot_reuse_changed_content(
        fat_wheel: Any) -> None:
    session = _Session()
    store = StageStore(session, STAGE)
    staged = stage_wheel(store, fat_wheel)
    directory = f'_kumo_ml_jobs/wheels/sha256/{staged.sha256}'
    target = store.uri(f'{directory}/{staged.name}')
    changed = b'not the installed Kumo wheel'
    session.objects[target] = changed
    store.put_json(
        f'{directory}/wheel.json', {
            'schema_version': 1,
            'name': staged.name,
            'sha256': staged.sha256,
            'source_md5': fat_wheel.md5,
            'stage_md5': hashlib.md5(changed,
                                     usedforsecurity=False).hexdigest(),
            'source_size': fat_wheel.size,
            'stage_size': len(changed),
        })
    wheel_puts_before = session.puts.count(target)

    stage_wheel(store, fat_wheel)

    assert session.puts.count(target) == wheel_puts_before + 1
    assert session.objects[target] == fat_wheel.path.read_bytes()


def test_staged_wheel_uses_snowflake_object_checksum_for_reuse(
        fat_wheel: Any) -> None:
    session = _Session()
    original_sql = session.sql
    staged_md5 = 'snowflake-managed-object-checksum'

    def sql(statement: str, params: list[Any] | None = None) -> _Query:
        query = original_sql(statement, params)
        if statement.startswith('LIST ') and fat_wheel.sha256 in statement:

            def collect() -> list[dict[str, Any]]:
                rows = original_sql(statement, params).collect()
                for row in rows:
                    if str(row['name']).endswith(fat_wheel.name):
                        row['md5'] = staged_md5
                        row['size'] += 5
                return rows

            query.collect = collect
        return query

    session.sql = sql
    store = StageStore(session, STAGE)
    first = stage_wheel(store, fat_wheel)
    puts = list(session.puts)
    second = stage_wheel(store, fat_wheel)

    assert first == second
    assert session.puts == puts


def test_workspace_wheel_is_streamed_directly_without_local_copy(
    fat_wheel: Any, ) -> None:
    session = _Session()
    stage_wheel(StageStore(session, STAGE), fat_wheel)
    assert session.objects[
        f'{STAGE}/_kumo_ml_jobs/wheels/sha256/'
        f'{fat_wheel.sha256}/{fat_wheel.name}'] == fat_wheel.path.read_bytes()


def test_wheel_validation_buffers_workspace_archive_reads(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    import builtins

    opened_with: list[int] = []
    real_open = builtins.open

    def buffered_open(path: Any, mode: str, *, buffering: int) -> Any:
        opened_with.append(buffering)
        return real_open(path, mode, buffering=buffering)

    monkeypatch.setattr('kumo_snowflake._ml_job_wheel.open', buffered_open,
                        raising=False)

    validate_wheel(fat_wheel.path)

    assert opened_with
    assert set(opened_with) == {64 * 1024 * 1024}


def test_workspace_wheel_uses_server_side_notebook_project_copy(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session = _Session()
    session.notebook_project = True
    source = ('snow://notebook project/DB.PUBLIC.PROJECT/versions/v1/'
              f'{fat_wheel.name}')
    session.objects[source] = fat_wheel.path.read_bytes()
    monkeypatch.setattr('kumo_snowflake._ml_job_wheel._workspace_path',
                        lambda path: True)
    monkeypatch.setenv('NOTEBOOK_PROJECT_DATABASE', 'DB')
    monkeypatch.setenv('NOTEBOOK_PROJECT_SCHEMA', 'PUBLIC')
    monkeypatch.setenv('SNOWFLAKE_NON_INTERACTIVE_NOTEBOOK_NAME', 'test.ipynb')

    staged = stage_wheel(StageStore(session, STAGE), fat_wheel)

    assert staged.sha256 == fat_wheel.sha256
    assert not any(target.endswith(fat_wheel.name) for target in session.puts)
    copies = [
        sql for sql, _ in session.statements if sql.startswith('COPY FILES')
    ]
    assert len(copies) == 2


@pytest.mark.parametrize('failing_query', [
    'SHOW NOTEBOOK PROJECTS',
    'DESCRIBE NOTEBOOK PROJECT',
])
def test_notebook_project_discovery_failure_uses_verified_upload(
    monkeypatch: Any,
    fat_wheel: Any,
    failing_query: str,
) -> None:
    session = _Session()
    session.notebook_project = True
    session.query_errors[failing_query] = RuntimeError('project changed')
    monkeypatch.setattr('kumo_snowflake._ml_job_wheel._workspace_path',
                        lambda path: True)
    monkeypatch.setenv('NOTEBOOK_PROJECT_DATABASE', 'DB')
    monkeypatch.setenv('NOTEBOOK_PROJECT_SCHEMA', 'PUBLIC')
    monkeypatch.setenv('SNOWFLAKE_NON_INTERACTIVE_NOTEBOOK_NAME', 'test.ipynb')

    staged = stage_wheel(StageStore(session, STAGE), fat_wheel)

    assert staged.name == fat_wheel.name
    assert any(target.endswith(fat_wheel.name) for target in session.puts)


def test_invalid_and_thin_wheels_are_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / 'corrupt-1-py3-none-any.whl'
    corrupt.write_bytes(b'not a zip')
    with pytest.raises(RuntimeError, match='not a valid Kumo fat wheel'):
        validate_wheel(corrupt)

    thin = tmp_path / 'kumoai-1-py3-none-any.whl'
    with zipfile.ZipFile(thin, 'w') as archive:
        archive.writestr('kumoai/__init__.py', '')
        archive.writestr('kumo_bundle_manifest.json',
                         json.dumps({
                             'components': [{
                                 'name': 'kumoai'
                             }],
                         }))
    with pytest.raises(RuntimeError, match='not self-contained'):
        validate_wheel(thin)


def test_wheel_missing_required_manifest_component_is_rejected(
        tmp_path: Path, fat_wheel: Any) -> None:
    incomplete = tmp_path / 'incomplete-3.1.0-py3-none-any.whl'
    with zipfile.ZipFile(fat_wheel.path) as source:
        with zipfile.ZipFile(incomplete, 'w') as target:
            for member in source.namelist():
                if member == 'kumo_bundle_manifest.json':
                    manifest = json.loads(source.read(member))
                    manifest['components'] = [
                        component for component in manifest['components']
                        if component['name'] != 'bbhash'
                    ]
                    target.writestr(member, json.dumps(manifest))
                else:
                    target.writestr(member, source.read(member))

    with pytest.raises(RuntimeError, match=r"components \['bbhash'\]"):
        validate_wheel(incomplete)


def test_wheel_missing_required_install_dependency_is_rejected(
        tmp_path: Path, fat_wheel: Any) -> None:
    incomplete = tmp_path / 'incomplete-requirement-3.1.0-py3-none-any.whl'
    with zipfile.ZipFile(fat_wheel.path) as source:
        with zipfile.ZipFile(incomplete, 'w') as target:
            for member in source.namelist():
                if member == 'kumo_bundle_manifest.json':
                    manifest = json.loads(source.read(member))
                    manifest['requires_at_install'] = []
                    target.writestr(member, json.dumps(manifest))
                else:
                    target.writestr(member, source.read(member))

    with pytest.raises(
            RuntimeError, match=(r"runtime dependencies "
                                 r"\['pytorch-frame==0.3.0'\]")):
        validate_wheel(incomplete)


def test_wheel_discovery_uses_retained_pep610_fat_wheel(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    class Distribution:
        files = [Path('kumo_bundle_manifest.json')]

        @staticmethod
        def read_text(name: str) -> str | None:
            if name != 'direct_url.json':
                return None
            return json.dumps({
                'url': fat_wheel.path.as_uri(),
                'archive_info': {
                    'hashes': {
                        'sha256': fat_wheel.sha256
                    }
                },
            })

    monkeypatch.setattr(
        'kumo_snowflake._ml_job_wheel.importlib.metadata.distributions',
        lambda: [Distribution()],
    )
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_wheel._hashes',
        lambda handle: pytest.fail(
            'PEP 610 discovery must not reread the workspace wheel'),
    )
    discovered = discover_installed_wheel()
    assert discovered.path == fat_wheel.path
    assert discovered.sha256 == fat_wheel.sha256
    assert discovered.size == fat_wheel.size
    assert discovered.md5 == ''


@pytest.mark.parametrize(('error', 'kind'), [
    (ConnectionError('offline'), 'transient'),
    (TimeoutError('slow'), 'transient'),
    (type('OperationalError', (Exception, ), {})('retry'), 'transient'),
    (type('SnowparkSQLException', (Exception, ), {
        'sqlstate': '42501',
    })('denied'), 'permission'),
    (type('SnowparkSQLException', (Exception, ), {})('bad sql'), 'sql'),
    (RuntimeError('bug'), 'unexpected'),
])
def test_preflight_exception_classifier(error: BaseException,
                                        kind: str) -> None:
    assert _exception_kind(error) == kind


@pytest.mark.parametrize(('error', 'expected'), [
    (ConnectionError('offline'), SnowflakeMLJobTransientError),
    (type('SnowparkSQLException', (Exception, ), {
        'sqlstate': '42501',
    })('denied'), SnowflakeMLJobPermissionError),
    (type('SnowparkSQLException',
          (Exception, ), {})('bad sql'), SnowflakeMLJobPreflightError),
    (RuntimeError('client bug'), SnowflakeMLJobPreflightError),
])
def test_submit_preserves_preflight_failure_class_and_cause(
    monkeypatch: Any,
    fat_wheel: Any,
    error: BaseException,
    expected: type[SnowflakeMLJobPreflightError],
) -> None:
    session, jobs = _Session(), _Jobs()
    session.query_errors['SHOW COMPUTE POOLS'] = error
    _patch_submission(monkeypatch, fat_wheel, jobs)
    with pytest.raises(expected) as caught:
        submit_ml_job(
            user_function,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
            session=session,
        )
    assert caught.value.__cause__ is error
    assert jobs.payload is None


def test_invalid_and_capturing_callables_are_rejected_before_session_lookup(
        monkeypatch: Any) -> None:
    value = 3

    def nested() -> int:
        return value

    called = False

    def factory() -> Any:
        nonlocal called
        called = True
        return _Session()

    monkeypatch.setattr('kumo_snowflake.ml_jobs._active_session', factory)

    with pytest.raises(SnowflakeMLJobConfigurationError, match='top level'):
        submit_ml_job(nested, compute_pool='KUMO_GPU_POOL', stage_name=STAGE)
    assert not called


_UNSAFE_GLOBAL = object()


def function_with_unsafe_global() -> str:
    return str(_UNSAFE_GLOBAL)


def helper_with_unsafe_global() -> str:
    return str(_UNSAFE_GLOBAL)


def function_with_indirect_unsafe_global() -> str:
    return helper_with_unsafe_global()


def test_live_global_is_rejected_before_session_lookup(
        monkeypatch: Any) -> None:
    called = False

    def factory() -> Any:
        nonlocal called
        called = True
        return _Session()

    monkeypatch.setattr('kumo_snowflake.ml_jobs._active_session', factory)

    with pytest.raises(SnowflakeMLJobConfigurationError, match='unsafe global'):
        submit_ml_job(
            function_with_unsafe_global,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
        )
    assert not called


def test_indirect_live_global_is_rejected_before_session_lookup(
        monkeypatch: Any) -> None:
    called = False

    def factory() -> Any:
        nonlocal called
        called = True
        return _Session()

    monkeypatch.setattr('kumo_snowflake.ml_jobs._active_session', factory)

    with pytest.raises(SnowflakeMLJobConfigurationError, match='unsafe global'):
        submit_ml_job(
            function_with_indirect_unsafe_global,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
        )
    assert not called


def test_missing_tag_object_fails_before_dispatch(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    _patch_submission(monkeypatch, fat_wheel, jobs)
    original = session.sql

    def sql(statement: str, params: Any = None) -> _Query:
        query = original(statement, params)
        if statement.startswith('SHOW TAGS'):
            query.collect = lambda: []
        return query

    session.sql = sql
    with pytest.raises(SnowflakeMLJobConfigurationError,
                       match='does not exist'):
        submit_ml_job(
            user_function,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
            tags={'TEST_DB.PUBLIC.MISSING': 'x'},
            session=session,
        )
    assert jobs.payload is None


def test_initial_tag_failure_cancels_and_persists_service_id(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    tag_error = RuntimeError('APPLY TAG privilege was revoked')
    session.query_errors['ALTER SERVICE'] = tag_error
    _patch_submission(monkeypatch, fat_wheel, jobs)

    with pytest.raises(SnowflakeMLJobSubmissionError) as caught:
        submit_ml_job(
            user_function,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
            tags={'TEST_DB.PUBLIC.REQUEST': 'run-1'},
            session=session,
        )

    assert caught.value.service_id == jobs.handle.id
    assert caught.value.__cause__ is tag_error
    assert jobs.handle.cancelled
    manifests = [
        json.loads(data) for target, data in session.writes
        if target.endswith('/manifest.json')
    ]
    assert manifests[-1]['service_id'] == jobs.handle.id
    assert manifests[-1]['submission_state'] == 'failed'


def test_initial_tag_and_cancel_failure_preserve_both_diagnostics(
    monkeypatch: Any,
    fat_wheel: Any,
) -> None:
    session, jobs = _Session(), _Jobs()
    tag_error = RuntimeError('tag write failed')
    cancel_error = RuntimeError('cancel failed')
    session.query_errors['ALTER SERVICE'] = tag_error
    jobs.handle.cancel_error = cancel_error
    _patch_submission(monkeypatch, fat_wheel, jobs)

    with pytest.raises(SnowflakeMLJobSubmissionError) as caught:
        submit_ml_job(
            user_function,
            compute_pool='KUMO_GPU_POOL',
            stage_name=STAGE,
            tags={'TEST_DB.PUBLIC.REQUEST': 'run-1'},
            session=session,
        )
    assert caught.value.service_id == jobs.handle.id
    assert caught.value.__cause__ is tag_error
    assert 'cancel failed' in str(caught.value)


def _progress_payload(execution_id: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    trial = TrainerProgress(
        num_epochs_completed=1,
        metrics={
            0:
            EpochMetrics(
                train_metrics={'train_loss': 0.8},
                validation_metrics={'val_loss': 0.7},
            )
        },
    )
    training = AutoTrainerProgress(
        start_time=now,
        total_trials=5,
        completed_trials=2,
        elapsed_training_time=60,
        estimated_training_time=150,
        trial_progress={'0': trial},
    )
    report = JobStatusReport(
        status=JobStatus.RUNNING,
        tracking_url='',
        start_time=now,
        event_log=[JobEventLogEntry('Training', now)],
    )
    return {
        'schema_version': 1,
        'execution_id': execution_id,
        'updated_at': now.isoformat(),
        'operation_kind': 'training',
        'operation_id': 'trainingjob-1',
        'status_report': to_json_dict(report),
        'training': to_json_dict(training),
        'prediction': None,
        'current_metrics': {
            'val_loss': 0.7
        },
        'best_metrics': {
            'val_loss': 0.7
        },
    }


def test_status_progress_logs_and_cancel_use_the_snowflake_handle() -> None:
    session, jobs = _Session(), _Jobs()
    execution_id = 'e' * 32
    remote = jobs.handle
    remote.status = 'RUNNING'
    store = StageStore(session, STAGE)
    store.put_json('progress.json', _progress_payload(execution_id))
    job = SnowflakeMLJob(
        remote.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
        handle=remote,
    )
    assert job.status() is SnowflakeMLJobStatus.RUNNING
    assert job.progress().training.completed_trials == 2
    assert job.logs() == 'training log'
    job.cancel()
    assert remote.cancelled


def test_live_progress_reads_durable_result_only_once() -> None:
    session, jobs = _Session(), _Jobs()
    jobs.handle.status = 'RUNNING'
    execution_id = 'e' * 32
    StageStore(session, STAGE).put_json('progress.json',
                                        _progress_payload(execution_id))
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
        handle=jobs.handle,
    )

    assert job.progress().status_report.status is JobStatus.RUNNING
    assert session.gets.count(f'{STAGE}/result.json') == 1


def test_unknown_service_status_fails_loudly() -> None:
    session, jobs = _Session(), _Jobs()
    jobs.handle.status = 'NEW_SNOWFLAKE_STATE'
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id='e',
        progress_location='p.json',
        result_location='r.json',
        session=session,
        jobs_api=jobs,
        handle=jobs.handle,
    )
    with pytest.raises(SnowflakeMLJobError, match='unrecognized'):
        job.status()


def test_durable_result_survives_service_history_removal() -> None:
    session, jobs = _Session(), _Jobs()
    jobs.get_error = RuntimeError('service was deleted')
    execution_id = 'durable-1'
    store = StageStore(session, STAGE)
    store.put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'SUCCEEDED',
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'value': {
                'model': '@STAGE/model'
            },
            'diagnostic': None,
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
    )
    assert job.result().value == {'model': '@STAGE/model'}
    assert job.status() is SnowflakeMLJobStatus.SUCCEEDED


def test_durable_terminal_status_wins_over_lagging_service_status() -> None:
    session, jobs = _Session(), _Jobs()
    jobs.handle.status = 'RUNNING'
    execution_id = 'durable-status'
    StageStore(session, STAGE).put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'SUCCEEDED',
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'value': 'done',
            'diagnostic': None,
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
        handle=jobs.handle,
    )

    assert job.status() is SnowflakeMLJobStatus.SUCCEEDED


def test_durable_result_rejects_nonterminal_status() -> None:
    session, jobs = _Session(), _Jobs()
    execution_id = 'invalid-status'
    StageStore(session, STAGE).put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'RUNNING',
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'value': None,
            'diagnostic': None,
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
    )

    with pytest.raises(SnowflakeMLJobError, match='non-terminal status'):
        job.result()


def test_durable_result_rejects_mismatched_status_report() -> None:
    session, jobs = _Session(), _Jobs()
    execution_id = 'mismatched-report'
    now = datetime.now(timezone.utc)
    report = JobStatusReport(
        status=JobStatus.FAILED,
        tracking_url='',
        start_time=now,
        end_time=now,
        event_log=[JobEventLogEntry('Failed', now)],
    )
    StageStore(session, STAGE).put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'SUCCEEDED',
            'finished_at': now.isoformat(),
            'value': None,
            'status_report': to_json_dict(report),
            'diagnostic': None,
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
    )

    with pytest.raises(SnowflakeMLJobError, match='disagrees'):
        job.result()


def test_durable_result_rejects_invalid_diagnostic_types() -> None:
    session, jobs = _Session(), _Jobs()
    execution_id = 'invalid-diagnostic'
    StageStore(session, STAGE).put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'FAILED',
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'value': None,
            'diagnostic': {
                'type': 'RemoteError',
                'message': 'failed',
                'code': 'not-an-integer',
            },
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
    )

    with pytest.raises(SnowflakeMLJobError, match='code'):
        job.result()


def test_raise_for_failure_uses_public_exception() -> None:
    now = datetime.now(timezone.utc)
    failed = SnowflakeMLJobResult(
        execution_id='failed-execution',
        service_id='DB.PUBLIC.FAILED_JOB',
        status=SnowflakeMLJobStatus.FAILED,
        finished_at=now,
    )

    with pytest.raises(SnowflakeMLJobFailed) as caught:
        failed.raise_for_failure()
    assert caught.value.result is failed


def test_terminal_result_decodes_durable_published_model() -> None:
    session, jobs = _Session(), _Jobs()
    execution_id = 'model-result'
    StageStore(session, STAGE).put_json(
        'result.json', {
            'schema_version': 1,
            'execution_id': execution_id,
            'status': 'SUCCEEDED',
            'finished_at': datetime.now(timezone.utc).isoformat(),
            'value': _json_value(_published_model()),
            'diagnostic': None,
        })
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id=execution_id,
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
    )

    assert job.result().value == _published_model()


def test_worker_round_trip_preserves_published_model_sidecar(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    model = _published_model()
    execute_user_function(lambda: model, [], {}, {
        'schema_version': 1,
        'execution_id': 'published-model-result',
        'stage_name': STAGE,
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })
    job = SnowflakeMLJob(
        'DB.PUBLIC.PUBLISHED_MODEL_RESULT',
        stage_name=STAGE,
        execution_id='published-model-result',
        progress_location='progress.json',
        result_location='result.json',
        session=session,
    )

    result = job.result()
    assert result.value == model
    assert result.published_model == model
    assert result.status_report is not None
    assert result.status_report.status is JobStatus.DONE


def test_nested_dataclass_preserves_published_model_type() -> None:
    @dataclass
    class Envelope:
        model: PublishedModel

    model = _published_model()
    payload = _json_value(Envelope(model))

    assert _decode_argument(payload)['model'] == model


def test_reserved_transport_key_round_trips_as_user_data() -> None:
    marker_shaped_value = {
        'kind': 'training_job_result',
        'payload': {
            'job_id': 'ordinary-user-data',
            'project_id': 'ordinary-user-data',
        },
    }
    value = {
        _TRANSPORT_KEY: marker_shaped_value,
        'other': 'unchanged',
    }

    assert _decode_result_value(_json_value(value)) == value
    assert _decode_argument(_json_input(value, 'args[0]')) == value


def test_dataclass_reserved_transport_field_round_trips_as_user_data() -> None:
    @dataclass
    class Envelope:
        __kumo_ml_job_transport__: dict[str, Any]

    marker_shaped_value = {
        'kind': 'published_model',
        'payload': {
            'model_uri': 'not-an-sdk-object',
        },
    }
    payload = _json_value(Envelope(marker_shaped_value))

    assert _decode_result_value(payload) == {
        _TRANSPORT_KEY: marker_shaped_value,
    }


@pytest.mark.parametrize('service_status', ['RUNNING', 'SUCCEEDED'])
def test_transient_result_error_is_not_reclassified_as_job_failure(
    service_status: str, ) -> None:
    session, jobs = _Session(), _Jobs()
    jobs.handle.status = service_status
    result_error = ConnectionError('temporary result transport failure')
    jobs.handle.value = result_error
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id='execution-1',
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
        handle=jobs.handle,
    )
    with pytest.raises(ConnectionError) as caught:
        job.result()
    assert caught.value is result_error


def test_terminal_service_failure_without_payload_has_diagnostic() -> None:
    session, jobs = _Session(), _Jobs()
    jobs.handle.status = 'FAILED_WITH_ERROR'
    jobs.handle.value = RuntimeError('remote container failed')
    job = SnowflakeMLJob(
        jobs.handle.id,
        stage_name=STAGE,
        execution_id='execution-1',
        progress_location='progress.json',
        result_location='result.json',
        session=session,
        jobs_api=jobs,
        handle=jobs.handle,
    )
    result = job.result()
    assert result.status is SnowflakeMLJobStatus.FAILED
    assert result.diagnostic is not None
    assert result.diagnostic.message == 'remote container failed'


def test_reattach_uses_durable_service_index() -> None:
    session = _Session()
    service_id = 'DB.PUBLIC.MLJOB_1'
    digest = hashlib.sha256(service_id.encode()).hexdigest()
    StageStore(session, STAGE).put_json(
        f'_kumo_ml_jobs/services/{digest}.json', {
            'schema_version': 1,
            'execution_id': 'execute-1',
            'service_id': service_id,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })
    job = get_ml_job(service_id, stage_name=STAGE, session=session)
    assert job.execution_id == 'execute-1'
    assert session.close_calls == 0
    job.close()
    assert session.close_calls == 0


def test_remote_wrapper_writes_durable_success_before_return(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    result = execute_user_function(user_function, [7], {}, {
        'schema_version': 1,
        'execution_id': 'execution-1',
        'stage_name': STAGE,
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })
    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable == result
    assert result['status'] == 'SUCCEEDED'
    assert result['value'] == {'value': 7}
    assert result['published_model'] is None
    assert result['progress'] is None
    assert result['status_report']['status'] == 'DONE'


def test_lost_success_ack_preserves_the_completed_result(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    calls = 0
    statuses: list[str] = []
    original_put = session.file.put_stream

    def commit_then_lose_ack(stream: Any, target: str, **kwargs: Any) -> Any:
        result = original_put(stream, target, **kwargs)
        if target.endswith('/result.json'):
            statuses.append(json.loads(session.objects[target])['status'])
            if len(statuses) == 1:
                raise ConnectionError('terminal write acknowledgement lost')
        return result

    monkeypatch.setattr(session.file, 'put_stream', commit_then_lose_ack)

    def completed_workload() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {'completed': True}

    result = execute_user_function(completed_workload, [], {}, {
        'schema_version': 1,
        'execution_id': 'lost-success-ack',
        'stage_name': STAGE,
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })

    assert calls == 1
    assert statuses == ['SUCCEEDED']
    assert result['status'] == 'SUCCEEDED'
    assert result['value'] == {'completed': True}
    assert StageStore(session, STAGE).get_json('result.json') == result


def test_unconfirmed_success_publication_is_explicit_and_not_failed(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    calls = 0
    original_put = session.file.put_stream

    def refuse_terminal_write(stream: Any, target: str, **kwargs: Any) -> Any:
        if target.endswith('/result.json'):
            raise ConnectionError('terminal stage unavailable')
        return original_put(stream, target, **kwargs)

    monkeypatch.setattr(session.file, 'put_stream', refuse_terminal_write)

    def completed_workload() -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return {'completed': True}

    with pytest.raises(RuntimeError, match='workload completed'):
        execute_user_function(completed_workload, [], {}, {
            'schema_version': 1,
            'execution_id': 'unconfirmed-success',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })

    assert calls == 1
    assert StageStore(session, STAGE).get_json('result.json',
                                               missing_ok=True) is None


def test_nonfinite_metrics_do_not_break_terminal_result_round_trip(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def train_with_nonfinite_metric() -> str:
        with runtime_progress.operation('training', 'trainingjob-1',
                                        'Training'):
            runtime_progress.progress_updated(
                'training',
                AutoTrainerProgress(
                    total_trials=1,
                    completed_trials=1,
                    elapsed_training_time=float('inf'),
                    estimated_training_time=float('nan'),
                    trial_progress={
                        '0':
                        TrainerProgress(
                            num_epochs_completed=1,
                            metrics={
                                0:
                                EpochMetrics(
                                    train_metrics={
                                        'loss': float('nan'),
                                        'accuracy': 0.75,
                                    }, )
                            },
                        )
                    },
                ),
            )
        return 'trained'

    execute_user_function(train_with_nonfinite_metric, [], {}, {
        'schema_version': 1,
        'execution_id': 'nonfinite-progress',
        'stage_name': STAGE,
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })
    job = SnowflakeMLJob(
        'DB.PUBLIC.NONFINITE',
        stage_name=STAGE,
        execution_id='nonfinite-progress',
        progress_location='progress.json',
        result_location='result.json',
        session=session,
    )

    result = job.result()
    assert result.succeeded
    assert result.value == 'trained'
    assert isinstance(result.progress, AutoTrainerProgress)
    epoch = result.progress.trial_progress['0'].metrics[0]
    assert epoch.train_metrics == {'accuracy': 0.75}
    assert result.progress.elapsed_training_time == 0.0
    assert result.progress.estimated_training_time is None


def test_remote_wrapper_preserves_user_error_and_durable_diagnostic(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    with pytest.raises(LookupError, match='user training failed'):
        execute_user_function(failing_user_function, [], {}, {
            'schema_version': 1,
            'execution_id': 'execution-1',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })
    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable['status'] == 'FAILED'
    assert durable['diagnostic']['type'] == 'LookupError'
    assert durable['diagnostic']['message'] == 'user training failed'


def test_remote_failure_coerces_non_json_diagnostic_attributes(
    monkeypatch: Any, ) -> None:
    class StrangeError(RuntimeError):
        errno = object()
        sqlstate = ['not', 'a', 'scalar']
        sfqid = {'query': 'id'}

    def fail_with_strange_diagnostic() -> None:
        raise StrangeError('strange failure')

    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    with pytest.raises(StrangeError, match='strange failure'):
        execute_user_function(fail_with_strange_diagnostic, [], {}, {
            'schema_version': 1,
            'execution_id': 'strange-diagnostic',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })

    durable = StageStore(session, STAGE).get_json('result.json')
    diagnostic = durable['diagnostic']
    assert diagnostic['code'] is None
    assert diagnostic['sqlstate'] == "['not', 'a', 'scalar']"
    assert diagnostic['sfqid'] == "{'query': 'id'}"


def test_remote_failure_preserves_exception_with_broken_string(
    monkeypatch: Any, ) -> None:
    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise ValueError('message formatting failed')

    error = UnprintableError()

    def fail_without_printable_message() -> None:
        raise error

    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def fail_to_format_traceback(error: BaseException) -> list[str]:
        del error
        raise ValueError('traceback formatting failed')

    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime.traceback.format_exception',
        fail_to_format_traceback,
    )
    with pytest.raises(UnprintableError) as caught:
        execute_user_function(fail_without_printable_message, [], {}, {
            'schema_version': 1,
            'execution_id': 'unprintable-diagnostic',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })

    assert caught.value is error
    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable['status'] == 'FAILED'
    assert durable['diagnostic']['type'] == 'UnprintableError'
    assert durable['diagnostic'][
        'message'] == '<UnprintableError message unavailable>'
    assert durable['diagnostic']['traceback'].endswith(
        'UnprintableError: <UnprintableError message unavailable>')


def test_remote_failure_preserves_final_native_progress(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def failed_training() -> None:
        with runtime_progress.operation('training', 'trainingjob-1',
                                        'Training'):
            runtime_progress.progress_updated(
                'training',
                AutoTrainerProgress(
                    total_trials=5,
                    completed_trials=2,
                    elapsed_training_time=20,
                    estimated_training_time=50,
                ),
            )
            raise LookupError('training failed after two trials')

    with pytest.raises(LookupError, match='after two trials'):
        execute_user_function(failed_training, [], {}, {
            'schema_version': 1,
            'execution_id': 'execution-1',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })
    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable['progress_kind'] == 'training'
    assert durable['progress']['completed_trials'] == 2
    assert durable['status_report']['status'] == 'FAILED'


def test_non_native_final_operation_does_not_reuse_prediction_progress(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def prediction_then_graph() -> str:
        with runtime_progress.operation('prediction', 'bp-job-1', 'Predicting'):
            runtime_progress.progress_updated(
                'prediction',
                PredictionProgress(
                    total_iterations=10,
                    completed_iterations=10,
                ),
            )
        with runtime_progress.operation('graph', 'snapshot-1',
                                        'Preparing graph snapshot'):
            pass
        return 'done'

    result = execute_user_function(prediction_then_graph, [], {}, {
        'schema_version': 1,
        'execution_id': 'execution-1',
        'stage_name': STAGE,
        'progress_location': 'progress.json',
        'result_location': 'result.json',
    })
    assert result['value'] == 'done'
    assert result['progress'] is None
    assert result['progress_kind'] is None


def test_non_published_sdk_results_are_not_transport_special_cases() -> None:
    class TrainingJobResult:
        job_id = 'not-a-transport-contract'

    with pytest.raises(TypeError, match='unsupported type'):
        _json_value(TrainingJobResult())


def test_same_named_user_class_does_not_trigger_model_publication() -> None:
    class TrainingJobResult:
        job_id = 'not-a-kumo-job'

    with pytest.raises(TypeError, match='unsupported type'):
        _json_value(TrainingJobResult())


def test_prediction_progress_round_trips() -> None:
    now = datetime.now(timezone.utc)
    progress = PredictionProgress(
        start_time=now,
        total_iterations=50,
        completed_iterations=14,
    )
    assert progress.completed_iterations == 14


def test_remote_observer_publishes_multiple_training_updates_and_metrics(
) -> None:
    session = _Session()
    observer = _RemoteObserver(StageStore(session, STAGE), {
        'execution_id': 'execution-1',
        'progress_location': 'progress.json',
    })
    observer.operation_started('training', 'trainingjob-1', 'Training')
    for completed, loss in ((1, 0.8), (2, 0.5), (3, 0.6)):
        # The production debounce is time-based; force each synthetic unit to
        # cross it so this test observes the actual serialized intermediates.
        observer._last_publish = 0.0
        observer.progress_updated(
            'training',
            AutoTrainerProgress(
                total_trials=5,
                completed_trials=completed,
                elapsed_training_time=float(completed * 10),
                estimated_training_time=50.0,
                trial_progress={
                    str(completed):
                    TrainerProgress(
                        num_epochs_completed=1,
                        metrics={
                            0:
                            EpochMetrics(validation_metrics={'val_loss': loss})
                        },
                    ),
                },
            ))
    observer.operation_finished('training', 'trainingjob-1', None)

    payload = StageStore(session, STAGE).get_json('progress.json')
    assert payload['training']['completed_trials'] == 3
    assert payload['training']['total_trials'] == 5
    assert payload['current_metrics'] == {'val_loss': 0.6}
    assert payload['best_metrics'] == {'val_loss': 0.5}
    assert [
        event['stage_name'] for event in payload['status_report']['event_log']
    ] == ['Starting user function', 'Training', 'Training complete']
    intermediates = [
        json.loads(data)['training']['completed_trials']
        for target, data in session.writes if target.endswith('/progress.json')
        and json.loads(data).get('training') is not None
    ]
    assert intermediates == [1, 2, 3, 3]


def test_remote_observer_publishes_prediction_iterations() -> None:
    session = _Session()
    observer = _RemoteObserver(StageStore(session, STAGE), {
        'execution_id': 'execution-1',
        'progress_location': 'progress.json',
    })
    observer.operation_started('prediction', 'bp-job-1', 'Predicting')
    observer._last_publish = 0.0
    observer.progress_updated(
        'prediction',
        PredictionProgress(
            total_iterations=50,
            completed_iterations=14,
        ))
    payload = StageStore(session, STAGE).get_json('progress.json')
    assert payload['prediction']['completed_iterations'] == 14
    assert payload['prediction']['total_iterations'] == 50


def test_prediction_callback_does_not_require_legacy_log_file(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    result = execute_user_function(
        function_with_prediction_progress,
        [None],
        {},
        {
            'schema_version': 1,
            'execution_id': 'prediction-without-file',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        },
    )

    assert result['status'] == 'SUCCEEDED'
    assert result['progress_kind'] == 'prediction'
    assert result['progress']['completed_iterations'] == 5
    assert result['progress']['total_iterations'] == 5


@pytest.mark.parametrize(
    ('function', 'kind', 'active_stage', 'expected_progress'),
    [
        (
            function_with_prediction_progress,
            'prediction',
            'Predicting',
            {
                'completed_iterations': 5,
                'total_iterations': 5,
            },
        ),
        (
            function_with_training_progress,
            'training',
            'Training',
            {
                'completed_trials': 1,
                'total_trials': 2,
            },
        ),
    ],
)
def test_real_progress_callback_starts_remote_operation(
    monkeypatch: Any,
    tmp_path: Path,
    function: Any,
    kind: str,
    active_stage: str,
    expected_progress: dict[str, int],
) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    result = execute_user_function(
        function,
        [str(tmp_path)],
        {},
        {
            'schema_version': 1,
            'execution_id': f'{kind}-execution',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        },
    )

    assert result['status'] == 'SUCCEEDED'
    assert result['progress_kind'] == kind
    assert {
        field: result['progress'][field]
        for field in expected_progress
    } == expected_progress
    stages = [
        event['stage_name'] for event in result['status_report']['event_log']
    ]
    assert stages == [
        'Starting user function',
        active_stage,
        f'{kind.title()} complete',
        'Completed',
    ]


def test_real_progress_callbacks_support_sequential_operations(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    result = execute_user_function(
        function_with_sequential_progress,
        [str(tmp_path)],
        {},
        {
            'schema_version': 1,
            'execution_id': 'sequential-execution',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        },
    )

    assert result['status'] == 'SUCCEEDED'
    assert result['progress_kind'] == 'training'
    assert result['progress']['completed_trials'] == 1
    assert [
        event['stage_name'] for event in result['status_report']['event_log']
    ] == [
        'Starting user function',
        'Predicting',
        'Prediction complete',
        'Training',
        'Training complete',
        'Completed',
    ]


def test_real_progress_callbacks_support_training_then_prediction(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    result = execute_user_function(
        function_with_training_then_prediction,
        [str(tmp_path)],
        {},
        {
            'schema_version': 1,
            'execution_id': 'training-then-prediction',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        },
    )

    assert result['status'] == 'SUCCEEDED'
    assert result['progress_kind'] == 'prediction'
    assert result['progress']['completed_iterations'] == 5
    assert [
        event['stage_name'] for event in result['status_report']['event_log']
    ] == [
        'Starting user function',
        'Training',
        'Training complete',
        'Predicting',
        'Prediction complete',
        'Completed',
    ]


def test_automatic_progress_ignores_late_regression_after_completion() -> None:
    session = _Session()
    observer = _RemoteObserver(
        StageStore(session, STAGE),
        {
            'execution_id': 'late-regression',
            'progress_location': 'progress.json',
        },
    )
    progress = PredictionProgress(total_iterations=2, completed_iterations=2)
    observer.progress_updated('prediction', progress)
    progress.completed_iterations = 1
    observer.progress_updated('prediction', progress)

    payload = observer.terminal(True)
    assert payload['prediction']['completed_iterations'] == 2
    assert [
        event['stage_name'] for event in payload['status_report']['event_log']
    ] == [
        'Starting user function',
        'Predicting',
        'Prediction complete',
        'Completed',
    ]


def test_automatic_progress_rejects_sequential_same_kind_operations() -> None:
    session = _Session()
    observer = _RemoteObserver(
        StageStore(session, STAGE),
        {
            'execution_id': 'sequential-prediction',
            'progress_location': 'progress.json',
        },
    )
    first = PredictionProgress(total_iterations=2, completed_iterations=2)
    second = PredictionProgress(total_iterations=3, completed_iterations=1)

    observer.progress_updated('prediction', first)
    with pytest.raises(RuntimeError, match='at most one prediction'):
        observer.progress_updated('prediction', second)


def test_changed_automatic_progress_source_fails_durable_result(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def change_source_while_active() -> str:
        runtime_progress.progress_updated(
            'prediction',
            PredictionProgress(total_iterations=3, completed_iterations=1),
        )
        runtime_progress.progress_updated(
            'prediction',
            PredictionProgress(total_iterations=3, completed_iterations=2),
        )
        return 'must not succeed'

    with pytest.raises(RuntimeError, match='progress source changed'):
        execute_user_function(change_source_while_active, [], {}, {
            'schema_version': 1,
            'execution_id': 'changed-progress-source',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })

    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable['status'] == 'FAILED'
    assert durable['diagnostic']['message'].endswith(
        'progress source changed during an active automatic operation')


@pytest.mark.parametrize('kind', ['training', 'prediction'])
def test_repeated_automatic_operation_fails_durable_result(
    monkeypatch: Any,
    kind: str,
) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )

    def progress(completed: int, total: int) -> Any:
        if kind == 'training':
            return AutoTrainerProgress(
                total_trials=total,
                completed_trials=completed,
            )
        return PredictionProgress(
            total_iterations=total,
            completed_iterations=completed,
        )

    def run_twice() -> str:
        runtime_progress.progress_updated(
            kind,
            progress(2, 2),
        )
        runtime_progress.progress_updated(
            kind,
            progress(1, 3),
        )
        return 'must not succeed'

    with pytest.raises(RuntimeError, match=f'at most one {kind}'):
        execute_user_function(run_twice, [], {}, {
            'schema_version': 1,
            'execution_id': f'repeated-{kind}',
            'stage_name': STAGE,
            'progress_location': 'progress.json',
            'result_location': 'result.json',
        })

    durable = StageStore(session, STAGE).get_json('result.json')
    assert durable['status'] == 'FAILED'
    assert durable['diagnostic']['message'].endswith(
        'at most one prediction operation')


def test_terminal_progress_waits_for_in_flight_worker_callback(
    monkeypatch: Any, ) -> None:
    session = _Session()
    monkeypatch.setattr(
        'kumo_snowflake._ml_job_runtime._active_job_session',
        lambda: session,
    )
    callback_started = threading.Event()
    release_callback = threading.Event()
    callback_workers: list[threading.Thread] = []
    original_progress_updated = _RemoteObserver.progress_updated

    def blocked_progress_updated(
        observer: _RemoteObserver,
        kind: str,
        progress: Any,
    ) -> None:
        callback_started.set()
        assert release_callback.wait(timeout=2)
        original_progress_updated(observer, kind, progress)

    monkeypatch.setattr(
        _RemoteObserver,
        'progress_updated',
        blocked_progress_updated,
    )

    def workload_with_late_progress() -> str:
        worker = threading.Thread(
            target=runtime_progress.progress_updated,
            args=(
                'prediction',
                PredictionProgress(
                    total_iterations=2,
                    completed_iterations=1,
                ),
            ),
        )
        callback_workers.append(worker)
        worker.start()
        assert callback_started.wait(timeout=1)
        return 'done'

    outcome: dict[str, Any] = {}

    def execute() -> None:
        try:
            outcome['result'] = execute_user_function(
                workload_with_late_progress,
                [],
                {},
                {
                    'schema_version': 1,
                    'execution_id': 'late-progress',
                    'stage_name': STAGE,
                    'progress_location': 'progress.json',
                    'result_location': 'result.json',
                },
            )
        except Exception as error:
            outcome['error'] = error

    execution = threading.Thread(target=execute)
    execution.start()
    assert callback_started.wait(timeout=1)
    time.sleep(0.05)
    release_callback.set()
    execution.join(timeout=2)
    callback_workers[0].join(timeout=2)

    assert not execution.is_alive()
    assert 'error' not in outcome
    assert outcome['result']['status'] == 'SUCCEEDED'
    durable = StageStore(session, STAGE).get_json('progress.json')
    assert durable['status_report']['status'] == 'DONE'
    assert durable['status_report']['event_log'][-1][
        'stage_name'] == 'Completed'


def test_progress_transport_failure_does_not_abort_user_operation() -> None:
    class FailingStore(StageStore):
        def __init__(self) -> None:
            super().__init__(_Session(), STAGE)

        def put_json(self, location: str, payload: Any) -> None:
            del location, payload
            raise ConnectionError('stage temporarily unavailable')

    observer = _RemoteObserver(FailingStore(), {
        'execution_id': 'execution-1',
        'progress_location': 'progress.json',
    })
    observer.operation_started('training', 'trainingjob-1', 'Training')
    observer.progress_updated(
        'training', AutoTrainerProgress(
            total_trials=2,
            completed_trials=1,
        ))
    observer.operation_finished('training', 'trainingjob-1', None)


def test_overlapping_top_level_operations_are_rejected() -> None:
    observer = _RemoteObserver(StageStore(_Session(), STAGE), {
        'execution_id': 'execution-1',
        'progress_location': 'progress.json',
    })
    observer.operation_started('training', 'trainingjob-1', 'Training')
    with pytest.raises(RuntimeError, match='only one top-level'):
        observer.operation_started('prediction', 'bp-job-1', 'Predicting')


def test_engine_snowflake_extra_requires_ml_jobs_client_version() -> None:
    sdk_pyproject = (Path(__file__).parents[2] / 'kumo-engine' /
                     'pyproject.toml').read_text()
    sdk_extra = re.search(r'^snowflake=\[(.*?)^\]$', sdk_pyproject,
                          re.MULTILINE | re.DOTALL)
    assert sdk_extra is not None
    assert '"kumo-snowflake[sdk]>=0.1.0.dev1"' in sdk_extra.group(1)

    plugin_pyproject = (Path(__file__).parents[1] /
                        'pyproject.toml').read_text()
    plugin_extra = re.search(r'^sdk = \[(.*?)^\]$', plugin_pyproject,
                             re.MULTILINE | re.DOTALL)
    assert plugin_extra is not None
    assert '"snowflake-ml-python>=1.51,<2"' in plugin_extra.group(1)


def test_stage_name_rejects_empty_segments_and_bounds_input() -> None:
    assert validate_stage_name(f'{STAGE}/models/v1/') == f'{STAGE}/models/v1'
    with pytest.raises(ValueError, match='fully-qualified'):
        validate_stage_name(f'{STAGE}//models')
    with pytest.raises(ValueError, match='fully-qualified'):
        validate_stage_name(f'{STAGE}/' + 'a' * 4096)


@pytest.mark.parametrize('missing',
                         [None, 'package', 'component', 'requirement'])
@pytest.mark.parametrize('requirements_key',
                         ['requires', 'requires_at_install'])
def test_engine_bundle_validation(tmp_path: Path, missing: str | None,
                                  requirements_key: str) -> None:
    path = tmp_path / 'kumo_engine-1.0.0-cp311-abi3-linux_x86_64.whl'
    components = [
        'kumo-engine',
        'kumo-core',
        'kumo-pql',
        'kumo-snowflake',
        'diskgraph',
        'torch-geometric',
    ]
    if missing == 'component':
        components.remove('diskgraph')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('kumo/engine/__init__.py', '')
        archive.writestr('kumo_core/__init__.py', '')
        if missing != 'package':
            archive.writestr('kumo_snowflake/__init__.py', '')
        archive.writestr(
            'kumo_bundle_manifest.json',
            json.dumps({
                'bundle': {
                    'name': 'kumo-engine'
                },
                'components': [{
                    'name': name
                } for name in components],
                requirements_key:
                ([] if missing == 'requirement' else ['pytorch-frame==0.3.0']),
                'extras': {
                    'snowflake': ['snowflake-ml-python>=1.51,<2']
                },
            }))
    if missing is None:
        assert validate_wheel(path).path == path
    else:
        expected = {
            'package': 'not self-contained',
            'component': 'missing required fat-wheel components',
            'requirement': 'missing required runtime dependencies',
        }[missing]
        with pytest.raises(RuntimeError, match=expected):
            validate_wheel(path)


def test_explicit_training_boundary_accepts_reconstructed_progress() -> None:
    observer = _RemoteObserver(StageStore(_Session(), STAGE), {
        'execution_id': 'execution-1',
        'progress_location': 'progress.json',
    })
    token = runtime_progress.install(observer)
    try:
        with runtime_progress.operation('training', 'fit-1', 'Training'):
            for _ in range(2):
                runtime_progress.progress_updated(
                    'training',
                    AutoTrainerProgress(total_trials=1, completed_trials=1))
        with runtime_progress.operation('prediction', 'predict-1',
                                        'Predicting'):
            runtime_progress.progress_updated(
                'prediction',
                PredictionProgress(total_iterations=1, completed_iterations=1))
        observer.raise_for_policy_violation()
    finally:
        runtime_progress.reset(token)
    assert observer._payload(
        JobStatus.DONE)['prediction']['completed_iterations'] == 1


@pytest.mark.parametrize('source,argument,expected', [
    ("def train(value):\n    return value + 1\n", 3, 4),
    ("FACTOR = 3\ndef helper(value):\n    return value * FACTOR\n"
     "alias = helper\ndef train(value):\n"
     "    def nested(item):\n        return alias(item)\n"
     "    return nested(value)\n", 3, 9),
    ("def train(value):\n    return value * train(value - 1) if value else 1\n",
     5, 120),
    ("def even(value):\n    return odd(value - 1) if value else True\n"
     "def odd(value):\n    return even(value - 1) if value else False\n"
     "def train(value):\n    return even(value)\n", 6, True),
    ("def helper(value):\n    return value\n"
     "alias = helper\ndef train(value):\n"
     "    return [helper is alias, helper(value) + alias(value)]\n", 3,
     [True, 6]),
])
def test_imported_callable_runs_without_source_module(
    tmp_path: Path,
    source: str,
    argument: int,
    expected: Any,
) -> None:
    import cloudpickle

    name = 'kumo_test_user_entrypoint'
    module_path = tmp_path / f'{name}.py'
    module_path.write_text(source +
                           "\nUNUSED_SECRET = 'must-stay-local-9851'\n")
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        original_globals = dict(vars(module))
        _validate_callable(module.train)
        detached = _detach_callable(module.train)
        payload = cloudpickle.dumps(((detached, [argument], {}, {}), {}))
        assert b'must-stay-local-9851' not in payload
        assert all(
            vars(module)[key] is value
            for key, value in original_globals.items())
        assert detached.__globals__ is not module.train.__globals__
        assert detached.__module__ == module.train.__module__
        assert detached.__qualname__ == module.train.__qualname__
    finally:
        del sys.modules[name]
        module_path.unlink()
    payload_path = tmp_path / 'args.pkl'
    payload_path.write_bytes(payload)
    child = subprocess.run([
        sys.executable,
        '-I',
        '-c',
        "import cloudpickle, json, pathlib, sys; "
        "args, kwargs = cloudpickle.loads(pathlib.Path(sys.argv[1]).read_bytes()); "
        "function, args, kwargs, config = args; "
        "print(json.dumps(function(*args, **kwargs)))",
        str(payload_path),
    ], check=True, capture_output=True, text=True, cwd=tmp_path)
    assert json.loads(child.stdout) == expected


def test_detached_callable_preserves_metadata() -> None:
    def entry(value: int = 3, *, offset: int = 2) -> int:
        """Metadata must survive detachment."""
        return value + offset

    setattr(entry, 'description', 'training entrypoint')
    detached = _detach_callable(entry)
    assert detached() == 5
    assert detached.__defaults__ == entry.__defaults__
    assert detached.__kwdefaults__ == entry.__kwdefaults__
    assert detached.__annotations__ == entry.__annotations__
    assert detached.__dict__ == entry.__dict__
    assert detached.__doc__ == entry.__doc__


def _function_with_nested_unsafe_global() -> Any:
    def nested() -> Any:
        return _UNSAFE_GLOBAL

    return nested()


def test_callable_validation_checks_nested_code_globals(
        monkeypatch: Any) -> None:
    monkeypatch.setitem(globals(), '_UNSAFE_GLOBAL', object())
    with pytest.raises(SnowflakeMLJobConfigurationError, match='unsafe global'):
        _validate_callable(_function_with_nested_unsafe_global)
