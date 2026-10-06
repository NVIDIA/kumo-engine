# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import os.path as osp
import uuid
from typing import Any

from kumo_core.diskgraph.sidecars import job_sidecar_root, safe_relative_path

from io_utils import fsutil, logging
from io_utils.json import load_json, save_json
from kumo.diskgraph._diskgraph_legacy import canonical_graph_snapshot_id

_DATAPLANE_DIR = 'diskgraph_job_sidecars'
_MANIFEST_DIR = 'manifests'
_GENERATIONS_DIR = 'generations'
_RUNTIME_MANIFEST_FILENAME = 'runtime_manifest.json'
_KEEP_LOCAL_ENV = 'KUMO_TTNG_KEEP_JOB_SIDECARS'
_LOCAL_LOCK_DIR = '.kumo_sidecar_locks'
_TT_EXTENSION_SENTINEL = 'diskgraph_manifest.json'
_TASK_DATASET_SENTINELS = ('manifest.json', _RUNTIME_MANIFEST_FILENAME)

log = logging.getLogger(__name__)


@contextlib.contextmanager
def prediction_materialization_ownership(index_path: str, job_id: str):
    """Exclusively own one prediction extension and task bundle.

    Activity retries can run in separate processes on the same placement.
    These locks intentionally use the same placement-local lock namespace as
    sidecar restore and cleanup, so no producer, consumer, or cleanup can
    mutate either final path while materialization and publication are in
    progress.
    """
    relpaths = sorted((
        safe_relative_path(osp.join('train_tables', f'pred_{job_id}')),
        safe_relative_path(osp.join('task_datasets', job_id)),
    ))
    locks: list[_SharedSidecarLock] = []
    try:
        for relpath in relpaths:
            lock = _SharedSidecarLock(index_path, relpath)
            lock.acquire_exclusive()
            locks.append(lock)
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _should_keep_local_job_sidecars() -> bool:
    value = os.environ.get(_KEEP_LOCAL_ENV, '')
    return value.lower() in {'1', 'true', 'yes', 'on'}


def upload_job_sidecars_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
) -> str | None:
    config = _load_config(config_path, prediction=prediction)
    manifest = _manifest_from_local_config(snapshot_id, config)
    if manifest is None:
        return None
    remote_root = _dataplane_root(snapshot_id,
                                  data_persist_dir=data_persist_dir)
    generation_relpath = _safe_relpath(
        osp.join(_GENERATIONS_DIR, manifest['job_id'], uuid.uuid4().hex))
    generation_root = osp.join(remote_root, generation_relpath)
    # Successful older generations may still have active readers and are
    # intentionally retained. Reclaiming unreferenced generations requires a
    # separate ownership-aware dataplane GC.
    index_path = _require_config_path(config, 'diskgraph_index_path')
    try:
        for relpath in manifest['relpaths']:
            _copy_relpath(index_path, generation_root, relpath)
        manifest = {
            **manifest,
            'schema_version': 2,
            'generation_relpath': generation_relpath,
        }
        manifest_path = _manifest_path(snapshot_id,
                                       data_persist_dir=data_persist_dir,
                                       job_id=manifest['job_id'])
        _publish_manifest_pointer(manifest, manifest_path)
    except BaseException:
        try:
            fsutil.rm(generation_root, ignore_errors=True)
        except Exception:
            log.warning('Failed to clean unpublished TT-NG sidecar generation',
                        exc_info=True)
        raise
    return remote_root


def restore_job_sidecars_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
) -> str | None:
    restored = _restore_job_sidecars_from_config_path(
        snapshot_id,
        config_path,
        prediction=prediction,
        data_persist_dir=data_persist_dir,
        hold_shared_sidecars=False,
    )
    return restored.remote_root if restored is not None else None


def restore_job_sidecars_for_use_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
) -> RestoredJobSidecars | None:
    return _restore_job_sidecars_from_config_path(
        snapshot_id,
        config_path,
        prediction=prediction,
        data_persist_dir=data_persist_dir,
        hold_shared_sidecars=True,
    )


def _restore_job_sidecars_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
    hold_shared_sidecars: bool,
) -> RestoredJobSidecars | None:
    config = _load_config(config_path, prediction=prediction)
    if _config_bundle_dir(config) is None:
        return None
    manifest = _load_dataplane_manifest(
        snapshot_id,
        config,
        data_persist_dir=data_persist_dir,
    )
    index_path = _require_config_path(config, 'diskgraph_index_path')
    remote_root = _remote_generation_root(
        snapshot_id,
        manifest,
        data_persist_dir=data_persist_dir,
    )
    shared_relpaths, owned_relpaths = _split_shared_sidecar_relpaths(
        manifest, prediction=prediction)
    locks: list[tuple[_SharedSidecarLock, str]] = []
    try:
        for relpath in shared_relpaths:
            if hold_shared_sidecars:
                lock = _SharedSidecarLock(index_path, relpath)
                lock.acquire_shared()
                if _shared_sidecar_complete(index_path, relpath):
                    locks.append((lock, relpath))
                    continue
                lock.release()

            lock = _SharedSidecarLock(index_path, relpath)
            lock.acquire_exclusive()
            try:
                _restore_shared_sidecar(remote_root, index_path, relpath)
                if hold_shared_sidecars:
                    lock.acquire_shared()
                    locks.append((lock, relpath))
                    lock = None
            finally:
                if lock is not None:
                    lock.release()

        for relpath in owned_relpaths:
            _copy_relpath(remote_root, index_path, relpath)
    except BaseException:
        for lock, _relpath in locks:
            lock.release()
        raise
    return RestoredJobSidecars(
        remote_root=remote_root,
        index_path=index_path,
        relpaths=manifest['relpaths'],
        shared_relpaths=shared_relpaths,
        shared_locks=locks,
        cleanup_enabled=not _should_keep_local_job_sidecars(),
    )


def cleanup_local_job_sidecars_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
) -> None:
    try:
        config = _load_config(config_path, prediction=prediction)
        if _should_keep_local_job_sidecars() or _config_bundle_dir(
                config) is None:
            return
        try:
            manifest = _manifest_from_local_config(snapshot_id, config)
        except FileNotFoundError:
            manifest = _load_dataplane_manifest(
                snapshot_id,
                config,
                data_persist_dir=data_persist_dir,
            )
        if manifest is None:
            return
        index_path = _require_config_path(config, 'diskgraph_index_path')
        shared_relpaths, _owned_relpaths = _split_shared_sidecar_relpaths(
            manifest, prediction=prediction)
        _cleanup_local_job_sidecars(index_path, manifest['relpaths'],
                                    shared_relpaths=shared_relpaths)
    except Exception:
        log.warning(
            'TT-NG job sidecar cleanup failed for config=%s (best-effort)',
            config_path,
            exc_info=True,
        )


class RestoredJobSidecars:
    def __init__(
        self,
        *,
        remote_root: str,
        index_path: str,
        relpaths: list[str],
        shared_relpaths: list[str],
        shared_locks: list[tuple['_SharedSidecarLock', str]],
        cleanup_enabled: bool,
    ) -> None:
        self.remote_root = remote_root
        self._index_path = index_path
        self._relpaths = relpaths
        self._shared_relpaths = shared_relpaths
        self._shared_locks = shared_locks
        self._cleanup_enabled = cleanup_enabled
        self._released = False

    def cleanup(self) -> None:
        if self._released:
            return
        shared_relpaths: list[str] = []
        try:
            try:
                shared_relpaths = self._shared_relpaths
                shared_set = set(shared_relpaths)
                owned_relpaths = [
                    relpath for relpath in self._relpaths
                    if relpath not in shared_set
                ]
                if self._cleanup_enabled:
                    _cleanup_local_job_sidecars(self._index_path,
                                                owned_relpaths,
                                                shared_relpaths=[])
            except Exception:
                log.warning(
                    'TT-NG restored job sidecar cleanup failed '
                    '(best-effort)',
                    exc_info=True,
                )
        finally:
            self.release()
        try:
            if self._cleanup_enabled:
                _cleanup_local_job_sidecars(
                    self._index_path,
                    shared_relpaths,
                    shared_relpaths=shared_relpaths,
                )
        except Exception:
            log.warning(
                'TT-NG restored shared job sidecar cleanup failed '
                '(best-effort)',
                exc_info=True,
            )

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for lock, _relpath in self._shared_locks:
            try:
                lock.release()
            except Exception:
                log.warning(
                    'TT-NG shared job sidecar lock release failed '
                    '(best-effort)',
                    exc_info=True,
                )
        self._shared_locks = []


def dataplane_job_sidecar_bundle_dir_from_config_path(
    snapshot_id: str,
    config_path: str,
    *,
    prediction: bool,
    data_persist_dir: str,
) -> str | None:
    config = _load_config(config_path, prediction=prediction)
    if _config_bundle_dir(config) is None:
        return None
    manifest = _load_dataplane_manifest(
        snapshot_id,
        config,
        data_persist_dir=data_persist_dir,
    )
    bundle_relpath = manifest.get('bundle_relpath')
    if not isinstance(bundle_relpath, str):
        raise RuntimeError('TT-NG job sidecar manifest missing bundle_relpath')
    return osp.join(
        _remote_generation_root(
            snapshot_id,
            manifest,
            data_persist_dir=data_persist_dir,
        ),
        _safe_relpath(bundle_relpath),
    )


def _manifest_from_local_config(snapshot_id: str, config: Any) -> dict | None:
    index_path = _config_path(config, 'diskgraph_index_path')
    bundle_dir = _config_bundle_dir(config)
    if not index_path or not bundle_dir:
        return None

    runtime_manifest = load_json(
        osp.join(bundle_dir, _RUNTIME_MANIFEST_FILENAME),
        dict,
    )
    relpaths = {_relpath_under(index_path, bundle_dir)}
    stages = runtime_manifest.get('stages')
    if not isinstance(stages, dict):
        raise RuntimeError(
            f'{_RUNTIME_MANIFEST_FILENAME} is missing dict-valued stages')
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        stage_index_path = stage.get('index_path')
        if stage_index_path:
            relpaths.add(_relpath_under(index_path, stage_index_path))

    job_id = _job_id_from_bundle_dir(bundle_dir)
    return {
        'schema_version': 1,
        'snapshot_id': canonical_graph_snapshot_id(snapshot_id),
        'job_id': job_id,
        'bundle_relpath': _relpath_under(index_path, bundle_dir),
        'relpaths': sorted(relpaths),
    }


def _load_dataplane_manifest(
    snapshot_id: str,
    config: Any,
    *,
    data_persist_dir: str,
) -> dict:
    bundle_dir = _config_bundle_dir(config)
    if bundle_dir is None:
        raise ValueError('TT-NG sidecar config is missing bundle dir')
    manifest = load_json(
        _manifest_path(snapshot_id, data_persist_dir=data_persist_dir,
                       job_id=_job_id_from_bundle_dir(bundle_dir)),
        dict,
    )
    expected_job_id = _job_id_from_bundle_dir(bundle_dir)
    if manifest.get('job_id') != expected_job_id:
        raise RuntimeError(
            'TT-NG job sidecar manifest is not bound to the requested job')
    if manifest.get('snapshot_id') != canonical_graph_snapshot_id(snapshot_id):
        raise RuntimeError(
            'TT-NG job sidecar manifest is not bound to the requested snapshot')
    relpaths = manifest.get('relpaths')
    if not isinstance(relpaths, list) or not all(
            isinstance(relpath, str) for relpath in relpaths):
        raise RuntimeError('TT-NG job sidecar manifest has invalid relpaths')
    manifest['relpaths'] = [_safe_relpath(relpath) for relpath in relpaths]
    return manifest


def _dataplane_root(
    snapshot_id: str,
    *,
    data_persist_dir: str,
) -> str:
    return job_sidecar_root(canonical_graph_snapshot_id(snapshot_id),
                            data_persist_dir)


def _manifest_path(snapshot_id: str, *, data_persist_dir: str,
                   job_id: str) -> str:
    return osp.join(
        _dataplane_root(snapshot_id, data_persist_dir=data_persist_dir),
        _MANIFEST_DIR,
        f'{job_id}.json',
    )


def _remote_generation_root(
    snapshot_id: str,
    manifest: dict,
    *,
    data_persist_dir: str,
) -> str:
    remote_root = _dataplane_root(snapshot_id,
                                  data_persist_dir=data_persist_dir)
    schema_version = manifest.get('schema_version')
    if schema_version == 1:
        return remote_root
    if schema_version != 2:
        raise RuntimeError(
            f'TT-NG job sidecar manifest has unsupported schema_version '
            f'{schema_version!r}')
    generation_relpath = manifest.get('generation_relpath')
    if not isinstance(generation_relpath, str):
        raise RuntimeError(
            'TT-NG v2 job sidecar manifest missing generation_relpath')
    generation_relpath = _safe_relpath(generation_relpath)
    expected_prefix = _safe_relpath(
        osp.join(_GENERATIONS_DIR, str(manifest.get('job_id', ''))))
    if (generation_relpath == expected_prefix or not generation_relpath.startswith(
            expected_prefix + os.sep)):
        raise RuntimeError(
            'TT-NG v2 job sidecar generation is not bound to its job_id')
    return osp.join(remote_root, generation_relpath)


def _publish_manifest_pointer(manifest: dict, manifest_path: str) -> None:
    manifest_dir = osp.dirname(manifest_path)
    fsutil.makedirs(manifest_dir, exist_ok=True)
    attempt_path = osp.join(
        manifest_dir,
        f'.{osp.basename(manifest_path)}.attempt-{uuid.uuid4().hex}',
    )
    try:
        save_json(manifest, attempt_path)
        if fsutil.is_local(manifest_path) and '://' not in manifest_path:
            os.replace(attempt_path, manifest_path)
        else:
            try:
                fsutil.copy(
                    attempt_path,
                    manifest_path,
                    overwrite=True,
                    raise_on_exists=False,
                )
            except Exception:
                # Object stores may commit the new pointer but lose the put
                # acknowledgement. Preserve the selected generation only when
                # a read-after-error proves the complete pointer is exactly the
                # one this attempt intended to publish.
                if _manifest_pointer_matches(manifest, manifest_path):
                    log.warning(
                        'TT-NG sidecar manifest publication raised after the '
                        'intended pointer became durable; treating it as '
                        'successful', exc_info=True)
                    return
                raise
    finally:
        fsutil.rm(attempt_path, ignore_errors=True)


def _manifest_pointer_matches(manifest: dict, manifest_path: str) -> bool:
    try:
        return load_json(manifest_path, dict) == manifest
    except Exception:
        return False


def _copy_relpath(src_root: str, dst_root: str, relpath: str) -> None:
    relpath = _safe_relpath(relpath)
    src = osp.join(src_root, relpath)
    dst = osp.join(dst_root, relpath)
    if not fsutil.exists(src):
        raise FileNotFoundError(f'TT-NG job sidecar missing at {src}')
    fsutil.makedirs(osp.dirname(dst), exist_ok=True)
    fsutil.copy(src, dst, overwrite=True, raise_on_exists=False)


def _relpath_under(root: str, path: str) -> str:
    root_abs = osp.abspath(root)
    path_abs = osp.abspath(path)
    return _safe_relpath(osp.relpath(path_abs, root_abs))


def _safe_relpath(relpath: str) -> str:
    return safe_relative_path(relpath)


def _load_config(config_path: str, *, prediction: bool) -> Any:
    from kumo_core.job_config import JobConfig, PredictionConfig
    return (PredictionConfig.load(config_path)
            if prediction else JobConfig.load(config_path))


def _config_bundle_dir(config: Any) -> str | None:
    return _config_path(config, 'diskgraph_task_dataset_bundle_dir')


def _require_config_path(config: Any, field_name: str) -> str:
    value = _config_path(config, field_name)
    if not value:
        raise ValueError(f'TT-NG sidecar config is missing {field_name}')
    return value


def _config_path(config: Any, field_name: str) -> str | None:
    value = getattr(config, field_name, None)
    return value if isinstance(value, str) and value else None


def _job_id_from_bundle_dir(bundle_dir: str) -> str:
    return osp.basename(osp.normpath(bundle_dir))


class _SharedSidecarLock:
    def __init__(self, index_path: str, relpath: str) -> None:
        lock_dir = osp.join(index_path, _LOCAL_LOCK_DIR)
        fsutil.makedirs(lock_dir, exist_ok=True)
        digest = hashlib.sha256(relpath.encode('utf-8')).hexdigest()
        self.path = osp.join(lock_dir, f'{digest}.lock')
        self._file = open(self.path, 'a+', encoding='utf-8')

    def acquire_exclusive(self, *, blocking: bool = True) -> bool:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._file.fileno(), flags)
        except BlockingIOError:
            return False
        return True

    def acquire_shared(self) -> None:
        fcntl.flock(self._file.fileno(), fcntl.LOCK_SH)

    def release(self) -> None:
        if self._file.closed:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


def _split_shared_sidecar_relpaths(
    manifest: dict,
    *,
    prediction: bool,
) -> tuple[list[str], list[str]]:
    relpaths = manifest['relpaths']
    bundle_relpath = manifest.get('bundle_relpath')
    if not isinstance(bundle_relpath, str):
        bundle_relpath = None
    elif prediction:
        bundle_relpath = _safe_relpath(bundle_relpath)

    shared: list[str] = []
    owned: list[str] = []
    for relpath in relpaths:
        if _is_shared_train_table_relpath(relpath) or (
                prediction and bundle_relpath == relpath):
            shared.append(relpath)
        else:
            owned.append(relpath)
    return shared, owned


def _is_shared_train_table_relpath(relpath: str) -> bool:
    parts = _safe_relpath(relpath).split(os.sep)
    return (len(parts) == 2 and parts[0] == 'train_tables'
            and not parts[1].startswith('.'))


def _train_table_extension_complete(index_path: str, relpath: str) -> bool:
    return fsutil.exists(osp.join(index_path, relpath, _TT_EXTENSION_SENTINEL))


def _task_dataset_bundle_complete(index_path: str, relpath: str) -> bool:
    root = osp.join(index_path, relpath)
    return all(
        fsutil.exists(osp.join(root, name)) for name in _TASK_DATASET_SENTINELS)


def _shared_sidecar_complete(index_path: str, relpath: str) -> bool:
    return _shared_sidecar_complete_at(
        index_path,
        relpath,
        expected_relpath=relpath,
    )


def _shared_sidecar_complete_at(
    index_path: str,
    relpath: str,
    *,
    expected_relpath: str,
) -> bool:
    if _is_shared_train_table_relpath(expected_relpath):
        return _train_table_extension_complete(index_path, relpath)
    return _task_dataset_bundle_complete(index_path, relpath)


def _restore_shared_sidecar(
    remote_root: str,
    index_path: str,
    relpath: str,
) -> None:
    if _shared_sidecar_complete(index_path, relpath):
        return

    src = osp.join(remote_root, relpath)
    dst = osp.join(index_path, relpath)
    if not fsutil.exists(src):
        raise FileNotFoundError(f'TT-NG job sidecar missing at {src}')

    parent = osp.dirname(dst)
    fsutil.makedirs(parent, exist_ok=True)
    tmp = osp.join(parent, f'.{osp.basename(dst)}.restore-{uuid.uuid4().hex}')
    try:
        fsutil.rm(tmp, ignore_errors=True)
        _copy_path(src, tmp)
        if not _shared_sidecar_complete_at(
                index_path,
                _relpath_under(index_path, tmp),
                expected_relpath=relpath,
        ):
            raise RuntimeError(
                f'TT-NG shared job sidecar restored incomplete: {src}')
        if fsutil.exists(dst):
            fsutil.rm(dst, ignore_errors=True)
        os.rename(tmp, dst)
    except BaseException:
        fsutil.rm(tmp, ignore_errors=True)
        raise


def _cleanup_local_job_sidecars(
    index_path: str,
    relpaths: list[str],
    *,
    shared_relpaths: list[str],
) -> None:
    shared_set = set(shared_relpaths)
    for relpath in sorted(relpaths, key=len, reverse=True):
        if relpath in shared_set:
            lock = _SharedSidecarLock(index_path, relpath)
            try:
                if lock.acquire_exclusive(blocking=False):
                    fsutil.rm(osp.join(index_path, relpath), ignore_errors=True)
            finally:
                lock.release()
            continue
        fsutil.rm(osp.join(index_path, relpath), ignore_errors=True)


def _copy_path(src: str, dst: str) -> None:
    if not fsutil.exists(src):
        raise FileNotFoundError(f'TT-NG job sidecar missing at {src}')
    fsutil.makedirs(osp.dirname(dst), exist_ok=True)
    fsutil.copy(src, dst, overwrite=True, raise_on_exists=False)
