# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend for DiskGraph's Python wheel.

The Python package is built by maturin, but the wheel also carries the
standalone Rust `ingest` binary used by Kumo's subprocess runner. Keeping this
step inside the build backend makes VCS installs work from a plain Git URL.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import maturin

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
INGEST_NAME = "ingest.exe" if os.name == "nt" else "ingest"
STAGED_INGEST = PROJECT_ROOT / "python" / "diskgraph" / "bin" / INGEST_NAME


def get_requires_for_build_wheel(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    requirements = maturin.get_requires_for_build_wheel(config_settings)
    if _use_zig():
        requirements = [*requirements, "maturin[zig]>=1,<2", "ziglang>=0.10.0"]
    return requirements


def build_wheel(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    return _build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
        editable=False,
    )


def get_requires_for_build_editable(
    config_settings: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    return get_requires_for_build_wheel(config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
) -> str:
    use_zig = _use_zig()
    settings = _maturin_config_settings(config_settings, use_zig=use_zig)
    with _patched_environ(_maturin_env()):
        return maturin.prepare_metadata_for_build_editable(
            metadata_directory, settings)


def build_editable(
    wheel_directory: str,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
) -> str:
    return _build_wheel(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
        editable=True,
    )


def _build_wheel(
    wheel_directory: str,
    *,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
    editable: bool,
) -> str:
    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    use_zig = _use_zig()

    _stage_ingest_binary(use_zig=use_zig)
    wheel_name = _maturin_build_wheel(
        wheel_dir,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
        use_zig=use_zig,
        editable=editable,
    )

    if editable:
        # Editable wheels are a `.pth` redirect to this source tree, not an
        # archive of its files -- `_stage_ingest_binary` above already placed
        # the binary where the redirect points, so there's nothing to verify
        # inside the (data-file-free) wheel zip.
        _verify_staged_ingest_present()
    else:
        _verify_wheel_contains_ingest(wheel_dir / wheel_name)
    return wheel_name


def _stage_ingest_binary(*, use_zig: bool) -> None:
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is required to build diskgraph")

    env = _cargo_env()
    target_dir = Path(env["CARGO_TARGET_DIR"])
    cmd = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "diskgraph-ingest" / "Cargo.toml"),
        "--bin",
        "ingest",
    ]

    if use_zig:
        if shutil.which("cargo-zigbuild") is None:
            raise RuntimeError(
                "DISKGRAPH_BUILD_ZIG requires cargo-zigbuild; install it with "
                "`cargo install cargo-zigbuild --locked`"
            )
        target = _linux_gnu_target()
        cmd[1] = "zigbuild"
        env.update(_zig_env())
        cmd.extend(["--target", target])
        built_ingest = target_dir / target / "release" / INGEST_NAME
    else:
        built_ingest = target_dir / "release" / INGEST_NAME

    _run(cmd, env=env)
    if not built_ingest.is_file():
        raise RuntimeError(f"cargo did not produce {built_ingest}")

    STAGED_INGEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_ingest, STAGED_INGEST)
    STAGED_INGEST.chmod(
        STAGED_INGEST.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def _maturin_build_wheel(
    wheel_dir: Path,
    *,
    config_settings: Optional[Mapping[str, Any]] = None,
    metadata_directory: Optional[str] = None,
    use_zig: bool,
    editable: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    settings = _maturin_config_settings(config_settings, use_zig=use_zig)
    if env is None:
        env = _maturin_env()
    else:
        env = _maturin_env(env)
    build_fn = maturin.build_editable if editable else maturin.build_wheel
    with _patched_environ(env):
        return build_fn(str(wheel_dir), settings, metadata_directory)


def _verify_wheel_contains_ingest(wheel: Path) -> None:
    expected = f"diskgraph/bin/{INGEST_NAME}"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    if expected not in names:
        raise RuntimeError(f"{wheel} does not contain {expected}")


def _verify_staged_ingest_present() -> None:
    if not STAGED_INGEST.is_file() or not os.access(STAGED_INGEST, os.X_OK):
        raise RuntimeError(
            f"diskgraph ingest binary missing or not executable: {STAGED_INGEST}"
        )


def _cargo_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(REPO_ROOT / "target"))
    rustflags = _default_rustflags()
    if rustflags:
        env.setdefault("RUSTFLAGS", rustflags)
    _pin_bundled_aws_lc(env)
    return env


def _pin_bundled_aws_lc(env: Dict[str, str]) -> None:
    # A detected system AWS-LC would replace the bundled source, making the
    # wheel depend on the build host (and on its libcrypto.so if no .a exists).
    # Setting AWS_LC_SYS_SYSTEM_DIR while this default leaves
    # AWS_LC_SYS_USE_SYSTEM=0 is a hard build error, so using a system AWS-LC
    # deliberately also needs AWS_LC_SYS_USE_SYSTEM=1. The target-specific
    # AWS_LC_SYS_USE_SYSTEM_<target> takes precedence over the plain variable.
    env.setdefault("AWS_LC_SYS_USE_SYSTEM", "0")


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _use_zig() -> bool:
    value = os.environ.get("DISKGRAPH_BUILD_ZIG", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _maturin_config_settings(
    config_settings: Optional[Mapping[str, Any]],
    *,
    use_zig: bool,
) -> Optional[Dict[str, Any]]:
    if not use_zig:
        return dict(config_settings) if config_settings is not None else None

    settings = dict(config_settings or {})
    key = (
        "maturin.build-args"
        if "maturin.build-args" in settings or "build-args" not in settings
        else "build-args"
    )
    settings[key] = _append_build_args(
        settings.get(key),
        ["--zig", "--compatibility", "manylinux2014"],
    )
    return settings


def _append_build_args(existing: Any, extra: List[str]) -> Any:
    if existing is None:
        return extra
    if isinstance(existing, str):
        return " ".join([existing, *extra])
    return [*existing, *extra]


def _maturin_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = (base_env or os.environ).copy()
    script_dir = "Scripts" if os.name == "nt" else "bin"
    path_parts = [
        str(Path(sys.prefix) / script_dir),
        str(Path(sys.executable).resolve().parent),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join(path_parts)
    rustflags = _default_rustflags()
    if rustflags:
        env.setdefault("RUSTFLAGS", rustflags)
    _pin_bundled_aws_lc(env)
    return env


def _default_rustflags() -> str:
    if os.environ.get("RUSTFLAGS") or _use_zig():
        return ""

    flags: List[str] = []
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        flags.extend(["-C", "target-feature=+aes,+sse2"])
    elif machine in {"aarch64", "arm64"}:
        flags.extend(["-C", "target-feature=+aes,+neon"])

    if sys.platform == "linux" and shutil.which("ld.lld") is not None:
        flags.extend(["-C", "link-arg=-fuse-ld=lld", "-C", "debuginfo=line-tables-only"])

    return " ".join(flags)


@contextmanager
def _patched_environ(env: Dict[str, str]) -> Iterator[None]:
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _linux_gnu_target() -> str:
    if sys.platform != "linux":
        raise RuntimeError("DISKGRAPH_BUILD_ZIG is only supported on Linux")

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64-unknown-linux-gnu"
    if machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-gnu"
    raise RuntimeError(f"Unsupported Zig build architecture: {platform.machine()}")


def _zig_env() -> Dict[str, str]:
    zig_bin = _zig_bin()
    zig_dir = zig_bin.parent
    return {"PATH": f"{zig_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


def _zig_bin() -> Path:
    try:
        import ziglang  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DISKGRAPH_BUILD_ZIG requires zig; install build dependencies with "
            '`python -m pip install "maturin[zig]>=1,<2" ziglang`'
        ) from exc

    name = "zig.exe" if os.name == "nt" else "zig"
    path = Path(ziglang.__file__).resolve().parent / name
    if not path.is_file():
        raise RuntimeError(f"Unable to find zig executable at {path}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path
