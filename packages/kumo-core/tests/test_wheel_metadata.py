# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.version import Version


def _build_wheel(tmp_path: Path) -> Path:
    package_root = Path(__file__).parents[1]
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--out-dir",
            str(tmp_path),
            str(package_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("kumo_core-*.whl"))
    return wheel


def _wheel_requirements(tmp_path: Path) -> dict[str, Requirement]:
    wheel = _build_wheel(tmp_path)
    with ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist()
                             if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
    requirements = (Requirement(value)
                    for value in metadata.get_all("Requires-Dist", []))
    return {requirement.name: requirement for requirement in requirements}


def test_wheel_owns_training_dependency_closure(tmp_path: Path) -> None:
    requirements = _wheel_requirements(tmp_path)

    assert {
        "diskgraph",
        "filelock",
        "fsspec",
        "kumo-api",
        "kumo-pql",
        "matplotlib",
        "numpy",
        "optuna",
        "pandas",
        "pyarrow",
        "pydantic",
        "pydantic-core",
        "psutil",
        "pytorch-frame",
        "pytorch-lightning",
        "tabulate",
        "tensorboard",
        "torch",
        "torch-geometric",
        "torchmetrics",
        "typing-extensions",
    } <= set(requirements)
    torch_versions = requirements["torch"].specifier
    assert Version("2.7.0") in torch_versions
    assert Version("2.9.0") in torch_versions
    assert Version("2.12.0") in torch_versions
    assert Version("2.13.0") in torch_versions
    assert Version("2.6.0") not in torch_versions
    assert Version("3.0.0") not in torch_versions
    assert str(requirements["numpy"].specifier) == "<3,>=1.26"
    assert str(requirements["pandas"].specifier) == "<3,>=1.5"
    assert str(requirements["pyarrow"].specifier) == "<25,>=15"
    assert Version("3.13.1") in requirements["filelock"].specifier
    assert Version("1.6.1") in requirements["scikit-learn"].specifier
    fsspec_versions = requirements["fsspec"].specifier
    assert Version("2023.5.0") in fsspec_versions
    assert Version("2025.3.0") in fsspec_versions
    assert Version("2025.5.1") in fsspec_versions
    assert Version("2025.5.2") in fsspec_versions
    assert str(requirements["pytorch-lightning"].specifier) == "==2.6.5"
    tensorboard_versions = requirements["tensorboard"].specifier
    assert Version("2.16.2") in tensorboard_versions
    assert Version("2.17.1") in tensorboard_versions
    assert Version("2.19.0") in tensorboard_versions
    assert Version("2.20.0") in tensorboard_versions
    assert Version("3.0.0") not in tensorboard_versions
    assert Version("3.10.0") in requirements["matplotlib"].specifier
    assert not {"pyg-lib", "torch-scatter"} & set(requirements)
    assert str(requirements["pytorch-frame"].specifier) == "==0.3.0"
    assert str(requirements["torch-geometric"].specifier) == ">=2.8.0"
    assert str(requirements["diskgraph"].specifier) == ">=0.1.0"
    assert requirements["diskgraph"].marker is None
    assert not {"bbhash", "lmdb", "safetensors", "xxhash"} & set(requirements)
    assert "kumo" + "ml-enterprise" not in requirements
    assert all(requirement.url is None for requirement in requirements.values())


def test_wheel_contains_migrated_runtime_but_not_test_support(
    tmp_path: Path,
) -> None:
    wheel = _build_wheel(tmp_path)
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    expected_modules = {
        "kumo_core/encoders/tensor_mapper.py",
        "kumo_core/evaluation/suite.py",
        "kumo_core/explain/subgraph_collector.py",
        "kumo_core/features/stores/feature.py",
        "kumo_core/modeling/gnn/hetero.py",
        "kumo_core/training/supervision/link_prediction.py",
    }
    assert expected_modules <= names
    assert "kumo_core/pyg_lib_trace.py" not in names
    assert "kumo_core/sampling/metapath.py" not in names
    assert "kumo_core/features/stores/in_memory_graph.py" not in names
    assert not any(name.startswith("kumo_core/online_serving/") for name in names)
    assert not any(name.startswith("kumo" + "ml/") for name in names)
    assert not any(name.startswith("tests/") for name in names)
