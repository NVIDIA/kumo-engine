# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build/cache one matching unpublished native-loader wheel set for CPU CI.

Run with Python 3.12, uv 0.13.7 and rustup available. No project imports or
runtime dependencies are needed. The pinned upstream Rust toolchain is read
from its checkout. Cache hits verify every wheel hash before reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_CONSTRAINTS = Path(__file__).with_name("native-build-constraints.txt")
PACKAGES = {
    "diskgraph": "packages/diskgraph",
    "kumo_settings": "packages/kumo-settings",
    "kumo_pytorch": "packages/experimental/kumo-pytorch",
    "kumo_api": "packages/kumo-api",
    "kumo_pql": "packages/kumo-pql",
    "kumo_core": "packages/kumo-core",
}


def digest(path: Path) -> str:
    """SHA-256 of a built artifact, independent of its modification time."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity() -> tuple[str, str]:
    """Require the same upstream Git repository/revision for all six components."""
    sources = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]
    identities = {
        (sources[name.replace("_", "-")]["git"], sources[name.replace("_", "-")]["rev"])
        for name in PACKAGES
    }
    if len(identities) != 1:
        raise ValueError("Kumo packages must use one matching source revision")
    return identities.pop()


def valid_cache(path: Path, identity: dict[str, str]) -> dict[str, str] | None:
    """Reject stale source/build settings, missing wheels, and altered bytes."""
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        saved = json.loads(manifest.read_text())
        wheels = saved["wheels"]
        if (
            not isinstance(wheels, dict)
            or saved["identity"] != identity
            or len(wheels) != len(PACKAGES)
        ):
            return None
        if {name.split("-")[0] for name in wheels} != set(PACKAGES):
            return None
        for name, checksum in wheels.items():
            if (
                Path(name).name != name
                or not name.endswith(".whl")
                or not (path / name).is_file()
                or digest(path / name) != checksum
            ):
                return None
        return wheels
    except (ValueError, KeyError, TypeError):
        return None


def cargo_jobs() -> str:
    """Honor Cargo's caller override; cap only our automatic default."""
    return os.environ.get("CARGO_BUILD_JOBS", str(min(os.cpu_count() or 1, 8)))


def cargo_cache_key(source: Path, revision: str, channel: str) -> str:
    """Separate native targets by immutable source, lockfile, toolchain and flags."""
    inputs = {
        "revision": revision,
        "toolchain": channel,
        "lock": digest(source / "packages/diskgraph/Cargo.lock"),
        "platform": f"{sys.platform}-{platform.machine()}",
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
        "constraints": digest(BUILD_CONSTRAINTS),
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("native-wheels"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/kumo-native-wheels"))
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--cargo-cache", type=Path, default=Path(".cache/kumo-native-cargo"))
    args = parser.parse_args()
    url, revision = source_identity()
    identity = {
        "git": url,
        "revision": revision,
        "builder": digest(Path(__file__)),
        "build_constraints": digest(BUILD_CONSTRAINTS),
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
        "platform": f"{sys.platform}-{platform.machine()}-py{sys.version_info.major}.{sys.version_info.minor}",
    }
    cache = args.cache.resolve() / revision
    cache.mkdir(parents=True, exist_ok=True)
    wheels = valid_cache(cache, identity)
    if wheels is None:
        with tempfile.TemporaryDirectory(prefix="tfm-kumo-build-") as tmp:
            source = Path(tmp) / "source"
            env = {
                **os.environ,
                "GIT_LFS_SKIP_SMUDGE": "1",
                "CARGO_BUILD_JOBS": cargo_jobs(),
                "CARGO_INCREMENTAL": "0",
                "CARGO_PROFILE_RELEASE_DEBUG": "0",
                "CARGO_PROFILE_RELEASE_STRIP": "debuginfo",
            }

            def run(*cmd: str, cwd: Path | None = None) -> None:
                subprocess.run(cmd, cwd=cwd, env=env, check=True)

            run("git", "init", str(source))
            run("git", "remote", "add", "origin", url, cwd=source)
            run("git", "fetch", "--depth", "1", "origin", revision, cwd=source)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            if actual != revision:
                raise ValueError("Fetched native source does not match the pinned revision")
            channel = tomllib.loads(
                (source / "packages/diskgraph/rust-toolchain.toml").read_text()
            )["toolchain"]["channel"]
            run("rustup", "toolchain", "install", channel, "--profile", "minimal")
            env["RUSTUP_TOOLCHAIN"] = channel
            # A fresh checkout keeps its source mtimes newer than cached project
            # outputs. Cargo may reuse unchanged registry dependencies, but a
            # different source revision/toolchain/lock/flag set gets its own target.
            target = args.cargo_cache.resolve() / cargo_cache_key(source, revision, channel)
            target.mkdir(parents=True, exist_ok=True)
            env["CARGO_TARGET_DIR"] = str(target)
            stage = Path(tmp) / "wheels"
            for package in PACKAGES.values():
                run(
                    args.uv,
                    "build",
                    "--wheel",
                    "--build-constraint",
                    str(BUILD_CONSTRAINTS),
                    "--python",
                    sys.executable,
                    "--out-dir",
                    str(stage),
                    package,
                    cwd=source,
                )
            wheels = {p.name: digest(p) for p in stage.glob("*.whl")}
            if len(wheels) != len(PACKAGES):
                raise ValueError("Native build did not produce exactly the requested wheel set")
            for name in wheels:
                shutil.copy2(stage / name, cache / name)
            (cache / "manifest.json").write_text(
                json.dumps({"identity": identity, "wheels": wheels}, indent=2) + "\n"
            )
    output = args.output.resolve() / revision
    output.mkdir(parents=True, exist_ok=True)
    for name in wheels:
        shutil.copy2(cache / name, output / name)
    shutil.copy2(cache / "manifest.json", output / "manifest.json")
    print(f"Verified native wheel set {revision}: {', '.join(sorted(wheels))}")


if __name__ == "__main__":
    main()
