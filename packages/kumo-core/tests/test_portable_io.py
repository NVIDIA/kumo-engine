# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

import fsspec
import pandas as pd
import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from kumo_core.io import filesystem
from kumo_core.io.csv import save_csv
from kumo_core.io.json import load_json, save_json
from kumo_core.train import io as train_io


@dataclass
class Payload:
    name: str
    value: int


@pytest.mark.parametrize('path', [
    'C:/work/automl/version_0',
    'D:/work/test runner/automl/version_0',
    r'C:\work\automl\version_0',
])
def test_windows_listing_has_no_authority(monkeypatch, path: str) -> None:
    fs = LocalFileSystem()
    monkeypatch.setattr(filesystem, '_resolve', lambda _: (fs, 'unused'))
    monkeypatch.setattr(fs, 'ls', lambda _, detail: [path])

    uri, = filesystem.listdir('unused')
    parsed = urlparse(uri)
    assert parsed.scheme == 'file'
    assert parsed.netloc == ''
    assert PureWindowsPath(parsed.path.removeprefix('/')) == PureWindowsPath(
        path)


@pytest.mark.parametrize('path', [
    '/tmp/result.csv',
    'C:relative/result.csv',
    '//server/share/result.csv',
    '//?/C:/result.csv',
])
def test_other_local_paths_preserve_protocol(path: str) -> None:
    fs = LocalFileSystem()
    assert filesystem._path_uri(fs, path) == fs.unstrip_protocol(path)


def test_windows_copy_uses_local_uri(monkeypatch) -> None:
    fs = LocalFileSystem()
    monkeypatch.setattr(filesystem, '_resolve', lambda _: (fs, 'C:/source'))
    monkeypatch.setattr(fs, 'isdir', lambda _: True)
    monkeypatch.setattr(fs, 'find', lambda _: ['C:/source/artifact.txt'])
    copied = []
    monkeypatch.setattr(
        filesystem, 'copy_file',
        lambda src, dst, overwrite: copied.append((src, dst)))

    filesystem.copy('C:/source', 'memory://copy-target')

    assert copied == [('file:///C:/source/artifact.txt',
                       'memory://copy-target/artifact.txt')]


def test_listed_csv_roundtrip(tmp_path: Path) -> None:
    for directory in (str(tmp_path), 'memory://listing-roundtrip'):
        filesystem.makedirs(directory, exist_ok=True)
        save_csv({'value': 3}, f'{directory}/result.csv')
        uri, = filesystem.listdir(directory)
        assert pd.read_csv(uri).to_dict(orient='list') == {'value': [3]}


def test_json_round_trip_on_local_and_remote_filesystems(
    tmp_path: Path, ) -> None:
    value = Payload(name="run", value=3)
    paths = [str(tmp_path / "payload.json"), "memory://run/payload.json"]

    for path in paths:
        save_json(value, path)
        assert load_json(path, Payload) == value


def test_registered_provider_adapter_routes_portable_artifacts(
        monkeypatch) -> None:
    provider_root = "abfss://container@account/run"
    memory_root = "memory://provider/run"
    original_get_filesystem_class = fsspec.get_filesystem_class

    def get_filesystem_class(protocol: str):
        if protocol == "abfss":
            raise ValueError("Protocol not known: abfss")
        return original_get_filesystem_class(protocol)

    monkeypatch.setattr(fsspec, "get_filesystem_class", get_filesystem_class)
    monkeypatch.setattr(
        filesystem,
        "_fsspec_fallback",
        lambda path: path.replace(provider_root, memory_root, 1),
    )

    value = Payload(name="provider", value=7)
    save_json(value, f"{provider_root}/payload.json")
    save_csv({"loss": 0.25}, f"{provider_root}/metrics.csv")
    train_io.write_binary(f"{provider_root}/payload.bin", b"payload")
    train_io.write_frame(pd.DataFrame({"value": [1]}),
                         f"{provider_root}/frame.parquet")

    assert load_json(f"{provider_root}/payload.json", Payload) == value
    assert train_io.read_binary(f"{provider_root}/payload.bin") == b"payload"
    with fsspec.open(f"{memory_root}/metrics.csv", "rt") as source:
        assert source.read() == "loss\n0.250000\n"
    with fsspec.open(f"{memory_root}/frame.parquet", "rb") as source:
        assert pd.read_parquet(source).to_dict(orient="list") == {"value": [1]}


def test_copy_directory_preserves_remote_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.txt").write_text("artifact", encoding="utf-8")

    filesystem.copy(str(source), "memory://run/copied", overwrite=True)

    with fsspec.open("memory://run/copied/artifact.txt", "rt") as result:
        assert result.read() == "artifact"


def test_local_path_resolves_file_uri() -> None:
    assert filesystem.local_path("file:/Volumes/example") == Path(
        "/Volumes/example")


def test_local_path_rejects_remote_uri() -> None:
    try:
        filesystem.local_path("memory://run/artifact")
    except ValueError as error:
        assert "not on a local filesystem" in str(error)
    else:
        raise AssertionError("remote URI must not resolve to a local path")


def test_torch_and_csv_writes_support_remote_filesystems() -> None:
    filesystem.torch_save({"value": torch.tensor([1])}, "memory://run/value.pt")
    with fsspec.open("memory://run/value.pt", "rb") as source:
        assert torch.equal(torch.load(source)["value"], torch.tensor([1]))

    save_csv(pd.DataFrame({"value": [1]}), "memory://run/value.csv",
             overwrite=True)
    with fsspec.open("memory://run/value.csv", "rt") as source:
        assert source.read() == "value\n1\n"


def test_csv_write_treats_scalar_mapping_as_one_row() -> None:
    save_csv({
        "loss": 0.25,
        "accuracy": 0.75
    }, "memory://run/metrics.csv", overwrite=True)
    with fsspec.open("memory://run/metrics.csv", "rt") as source:
        assert source.read() == "loss,accuracy\n0.250000,0.750000\n"
