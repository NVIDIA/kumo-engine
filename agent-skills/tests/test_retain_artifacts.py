# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / 'skills' / 'kumo-model-persistence' / 'scripts' /
          'retain_artifacts.py')
spec = importlib.util.spec_from_file_location('retain_artifacts', SCRIPT)
assert spec and spec.loader
retain = importlib.util.module_from_spec(spec)
sys.modules['retain_artifacts'] = retain
spec.loader.exec_module(retain)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / 'run'
    (root / 'trials' / '0').mkdir(parents=True)
    (root / 'metrics.json').write_text('{"auc": 0.9}')
    (root / 'trials' / '0' / 'progress.json').write_bytes(b'\x00' * 3000)
    return root


def forge(path: Path, members: dict[str, bytes], manifest: object = None) -> str:
    """Write an archive by hand, bypassing create_archive's checks, and return its SHA256."""
    if manifest is None:
        manifest = [dict(path=n, size=len(b), sha256=hashlib.sha256(b).hexdigest())
                    for n, b in members.items()]
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
        bundle.writestr('retention-manifest.json', json.dumps(manifest))
    return retain.sha256(path)


def test_round_trip(run: Path, tmp_path: Path) -> None:
    files = ['metrics.json', 'trials/0/progress.json']
    digest = retain.create_archive(run, files, tmp_path / 'a.zip')
    out = retain.restore_archive(tmp_path / 'a.zip', tmp_path / 'restored', digest)
    for name in files:
        assert (out / name).read_bytes() == (run / name).read_bytes()
    assert not (out / 'retention-manifest.json').exists()


@pytest.mark.parametrize('name', [
    '../escape.txt', 'a/../../escape.txt', '/etc/passwd', './metrics.json', 'a//b',
    'C:/temp/payload', 'C:payload', 'a\\b', '', 'retention-manifest.json',
])
def test_unsafe_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        retain._relative(name)


@pytest.mark.parametrize('name', ['../escape.txt', 'C:/temp/payload', '/abs.txt'])
def test_restore_rejects_unsafe_member(tmp_path: Path, name: str) -> None:
    digest = forge(tmp_path / 'evil.zip', {name: b'x'})
    with pytest.raises(ValueError):
        retain.restore_archive(tmp_path / 'evil.zip', tmp_path / 'out', digest)
    assert not (tmp_path / 'out').exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_create_rejects_symlink_and_escape(run: Path, tmp_path: Path) -> None:
    (tmp_path / 'secret.txt').write_text('s')
    (run / 'link.txt').symlink_to(tmp_path / 'secret.txt')
    with pytest.raises(ValueError, match='Symlink'):
        retain.create_archive(run, ['link.txt'], tmp_path / 'a.zip')
    with pytest.raises(ValueError):
        retain.create_archive(run, ['../secret.txt'], tmp_path / 'b.zip')
    assert not (tmp_path / 'a.zip').exists() and not (tmp_path / 'b.zip').exists()


def test_create_rejects_duplicates_and_existing_archive(run: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='duplicates'):
        retain.create_archive(run, ['metrics.json', 'metrics.json'], tmp_path / 'a.zip')
    (tmp_path / 'exists.zip').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        retain.create_archive(run, ['metrics.json'], tmp_path / 'exists.zip')
    assert (tmp_path / 'exists.zip').read_bytes() == b'keep'


def test_archive_checksum_mismatch(run: Path, tmp_path: Path) -> None:
    retain.create_archive(run, ['metrics.json'], tmp_path / 'a.zip')
    with pytest.raises(ValueError, match='Archive SHA256 mismatch'):
        retain.restore_archive(tmp_path / 'a.zip', tmp_path / 'out', '0' * 64)
    assert not (tmp_path / 'out').exists()


def test_member_checksum_mismatch(tmp_path: Path) -> None:
    manifest = [dict(path='f.txt', size=4, sha256=hashlib.sha256(b'good').hexdigest())]
    digest = forge(tmp_path / 'a.zip', {'f.txt': b'evil'}, manifest)
    with pytest.raises(ValueError, match='File checksum mismatch'):
        retain.restore_archive(tmp_path / 'a.zip', tmp_path / 'out', digest)
    assert not (tmp_path / 'out').exists()


def test_unlisted_member_and_size_budget(tmp_path: Path) -> None:
    manifest = [dict(path='a.txt', size=1, sha256=hashlib.sha256(b'a').hexdigest())]
    digest = forge(tmp_path / 'extra.zip', {'a.txt': b'a', 'b.txt': b'b'}, manifest)
    with pytest.raises(ValueError, match='Unexpected or missing'):
        retain.restore_archive(tmp_path / 'extra.zip', tmp_path / 'out', digest)
    digest = forge(tmp_path / 'big.zip', {'a.txt': b'a' * 100})
    with pytest.raises(ValueError, match='storage budget'):
        retain.restore_archive(tmp_path / 'big.zip', tmp_path / 'out', digest, max_bytes=50)


def test_restore_never_overwrites(run: Path, tmp_path: Path) -> None:
    digest = retain.create_archive(run, ['metrics.json'], tmp_path / 'a.zip')
    (tmp_path / 'out').mkdir()
    with pytest.raises(FileExistsError):
        retain.restore_archive(tmp_path / 'a.zip', tmp_path / 'out', digest)
    assert list((tmp_path / 'out').iterdir()) == []
