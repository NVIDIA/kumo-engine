#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the complete kumo-engine wheel from the monorepo.

Build every first-party package except the hosted SDK, build native DiskGraph,
and merge the components with their provenance and licenses. Third-party
runtime dependencies remain installer requirements. Release builds use Zig on
Linux; source installations select native compilation through --no-zig.
"""

import argparse
import importlib.util
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger(__name__)
NOTICE_ARTIFACTS = (
    "diskgraph-extension",
    "diskgraph-ingest",
    "kumo-core-bundled-source",
)


def _load_module(path: Path) -> ModuleType:
    name = f"_kumo_build_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load build module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(
    command: list[str], environment: dict[str, str], *, capture: bool = False
) -> str:
    result = subprocess.run(
        command,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else sys.stderr,
    )
    return result.stdout if capture else ""


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        logger.warning("Git metadata unavailable: %s", result.stderr.strip())
        return ""
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=os.getenv("OUTPUT_DIR") or None
    )
    parser.add_argument("--release", action="store_true", help="Use the public VERSION")
    parser.add_argument("--version", default=os.getenv("BUNDLE_VERSION") or None)
    parser.add_argument("--platform-tag", default=os.getenv("PLATFORM_TAG", ""))
    parser.add_argument(
        "--kumo-repo", type=Path, default=Path(os.getenv("KUMO_REPO") or REPO_ROOT)
    )
    parser.add_argument(
        "--no-zig",
        action="store_true",
        help="Build DiskGraph natively instead of through cargo-zigbuild",
    )
    parser.add_argument(
        "--verify-image",
        default=os.getenv("VERIFY_IMAGE", ""),
        help="Install and smoke-test the wheel in a Docker image",
    )
    parser.add_argument("--offline-report", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    return parser


def _record(
    wheels: Path, pattern: str, manifest: Path, classification: str, origin: str
) -> Path:
    wheel = max(wheels.glob(pattern), default=None)
    if wheel is None:
        raise RuntimeError(f"Expected a wheel matching {pattern} in {wheels}")
    with manifest.open("a") as output:
        output.write(f"{wheel.name}\t{classification}\t{origin}\n")
    return wheel


def _check_glibc(
    wheel: Path, ceiling: str, environment: dict[str, str], work: Path
) -> None:
    dumper = shutil.which("objdump") or shutil.which("readelf")
    if dumper is None:
        logger.warning("No objdump/readelf; skipping the glibc floor check")
        return
    arguments = ["-T"] if Path(dumper).name == "objdump" else ["-W", "--dyn-syms"]
    worst = (0, 0)
    with ZipFile(wheel) as archive:
        binaries = [
            name
            for name in archive.namelist()
            if name == "diskgraph/bin/ingest"
            or (name.startswith("diskgraph/_c") and name.endswith(".so"))
        ]
        for name in binaries:
            path = work / Path(name).name
            path.write_bytes(archive.read(name))
            symbols = _run([dumper, *arguments, str(path)], environment, capture=True)
            versions = [
                (int(major), int(minor))
                for major, minor in re.findall(r"GLIBC_(\d+)\.(\d+)", symbols)
            ]
            floor = max(versions, default=(0, 0))
            logger.info("%s needs glibc >= %s.%s", path.name, *floor)
            worst = max(worst, floor)
    if worst > tuple(map(int, ceiling.split("."))):
        raise RuntimeError(
            f"DiskGraph needs glibc {worst[0]}.{worst[1]}, above the {ceiling} ceiling"
        )
    logger.info("DiskGraph glibc floor: %s.%s (ceiling %s)", *worst, ceiling)


def _assemble(
    args: argparse.Namespace, environment: dict[str, str], work: Path
) -> Path:
    repository = args.kumo_repo.resolve()
    local = repository / "packages/kumo-engine"
    diskgraph = repository / "packages/diskgraph"
    host = platform.system()
    use_zig = (
        not args.no_zig
        and (environment.get("USE_ZIG") or ("0" if host == "Darwin" else "1")) != "0"
    )
    if host == "Darwin":
        target = environment.get("MACOSX_DEPLOYMENT_TARGET") or "14.0"
        if not re.fullmatch(r"\d+(?:\.\d+)*", target) or int(target.split(".")[0]) < 14:
            raise RuntimeError("macOS wheels require MACOSX_DEPLOYMENT_TARGET >= 14.0")
        if use_zig:
            raise RuntimeError(
                "macOS wheels require a native DiskGraph build (USE_ZIG=0)"
            )
        environment["MACOSX_DEPLOYMENT_TARGET"] = target
    python = environment.get("PYTHON") or sys.executable
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo is required to build DiskGraph")
    _run([python, "-m", "pip", "--version"], environment)
    if not local.is_dir() or not (diskgraph / "diskgraph-pybind/Cargo.toml").is_file():
        raise RuntimeError(f"No complete kumo monorepo at {repository}")
    license_file = repository / "LICENSE"
    notice = Path(
        environment.get("THIRD_PARTY_NOTICE") or repository / "THIRD-PARTY.txt"
    )
    for path in (license_file, notice):
        if not path.is_file():
            raise RuntimeError(f"Missing license/notice: {path}")
    generate_notice = not environment.get("THIRD_PARTY_NOTICE")
    if generate_notice and repository != REPO_ROOT:
        raise RuntimeError(
            "Automatic notice generation requires this script's checkout; "
            "set THIRD_PARTY_NOTICE to a reviewed notice"
        )
    head = _git(repository, "rev-parse", "--short", "HEAD") or "unknown"
    version = args.version
    if not version:
        version = "".join((repository / "VERSION").read_text().split())
        if not args.release:
            version += f"+{head}"
    if args.release and "+" in version:
        raise RuntimeError(f"--release needs a public version, got {version}")
    output = (args.output_dir or local / "dist").resolve()
    wheels = work / "wheels"
    wheels.mkdir()
    manifest = work / "components.tsv"
    manifest.touch()
    dirty = (
        "+local changes"
        if _git(repository, "status", "--porcelain", "--", "packages")
        else ""
    )
    for package in sorted((repository / "packages").iterdir()):
        if package.name in {"kumo-sdk", "diskgraph"} or not any(
            (package / name).is_file() for name in ("pyproject.toml", "setup.py")
        ):
            continue
        logger.info("building %s from %s", package.name, package)
        component = (
            ["--config-settings=kumo-component=true"]
            if package.name == "kumo-engine"
            else []
        )
        _run(
            [
                python,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "-q",
                *component,
                "-w",
                str(wheels),
                str(package),
            ],
            environment,
        )
        _record(
            wheels,
            f"{package.name.replace('-', '_')}-*.whl",
            manifest,
            "kumo-first-party",
            f"{package} @ {head}{dirty}",
        )
    if use_zig and shutil.which("cargo-zigbuild") is None:
        logger.info("Installing cargo-zigbuild for the manylinux build")
        _run(["cargo", "install", "cargo-zigbuild", "--locked"], environment)
    logger.info("building diskgraph from %s", diskgraph)
    _run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-q",
            "-w",
            str(wheels),
            str(diskgraph),
        ],
        dict(environment, DISKGRAPH_BUILD_ZIG=str(int(use_zig))),
    )
    native = _record(
        wheels,
        "diskgraph-*.whl",
        manifest,
        "diskgraph-native",
        f"in-tree source {diskgraph}",
    )
    ceiling = environment.get("MAX_GLIBC") or "2.34"
    tag = args.platform_tag
    if host == "Darwin":
        native_tag = native.stem.rsplit("-", 1)[-1]
        if not native_tag.startswith("macosx_") or not native_tag.endswith(
            f"_{platform.machine()}"
        ):
            raise RuntimeError(f"Expected a native macOS DiskGraph wheel, got {native}")
        if tag and tag != native_tag:
            raise RuntimeError(
                f"Platform tag {tag} does not match DiskGraph's {native_tag}"
            )
        tag = native_tag
    else:
        tag = tag or f"manylinux_{ceiling.replace('.', '_')}_x86_64"
        _check_glibc(native, ceiling, environment, work)
    if use_zig and "manylinux" not in native.name:
        raise RuntimeError(
            f"Expected a manylinux DiskGraph wheel from Zig, got {native.name}"
        )
    legal = _load_module(repository / "scripts/gen_third_party.py")
    inventory = legal.load_json(legal.DEFAULT_INVENTORY)
    policy = legal.load_json(legal.DEFAULT_POLICY)
    if generate_notice:
        if notice.read_bytes() != legal.render(inventory, policy):
            raise RuntimeError("The committed third-party notice is stale")
        notice = work / "THIRD-PARTY.txt"
        notice.write_bytes(
            legal.render(
                inventory,
                policy,
                wheels_dir=wheels,
                components_path=manifest,
                artifact_ids=NOTICE_ARTIFACTS,
            )
        )
    expression = legal.selected_license_expression(
        inventory,
        policy,
        artifact_ids=NOTICE_ARTIFACTS,
        base_licenses=("Apache-2.0",),
    )
    merger = _load_module(Path(__file__).with_name("merge_wheels.py"))
    wheel: Path = merger.merge_wheels(
        merger.MergeOptions(
            wheels_dir=wheels,
            components_tsv=manifest,
            out_dir=output,
            version=version,
            platform_tag=tag,
            notice=notice,
            license=license_file,
            license_expression=expression,
            offline_report=args.offline_report,
        )
    )
    logger.info("wrote %s (%s MiB)", wheel, wheel.stat().st_size // 1024**2)
    if args.verify_image:
        verification = work / "verify.py"
        verification.write_text(VERIFY_INSTALL + VERIFY_SOURCE)
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{wheel.parent}:/wheel:ro",
                "-v",
                f"{work}:/work:ro",
                "--entrypoint",
                "python",
                args.verify_image,
                "/work/verify.py",
                "kumo-engine",
                f"/wheel/{wheel.name}",
            ],
            environment,
        )
    return wheel


def main() -> int:
    """Build a bundle using CLI flags and environment defaults.

    Returns:
        Zero on success, the failing tool's status, or one for invalid build inputs.
    """
    logging.basicConfig(level=logging.INFO, format="[wheel] %(levelname)s: %(message)s")
    args = _parser().parse_args()
    work = Path(tempfile.mkdtemp(prefix="kumo-wheel-"))
    try:
        wheel = _assemble(args, dict(os.environ), work)
        print(wheel)
        return 0
    except subprocess.CalledProcessError as error:
        logger.error("Build command failed: %s", error)
        if error.stdout:
            logger.error("%s", error.stdout)
        return error.returncode
    except (RuntimeError, OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
    finally:
        if args.keep_work:
            logger.info("scratch kept at %s", work)
        else:
            shutil.rmtree(work)


VERIFY_INSTALL = """\
import pathlib
import subprocess
import sys

for directory in ('user', 'system'):
    pathlib.Path('/var/log/managedservices', directory, 'mlrs').mkdir(
        parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                '--prefix', sys.prefix, sys.argv[2]], check=True)
"""

VERIFY_SOURCE = """\
import importlib.util
import os
import pathlib
import sys
from importlib.metadata import distribution

import diskgraph
import kumo_core  # noqa: F401
import kumoapi  # noqa: F401
import torch
import torch_geometric
from kumo.engine import FileConnector, Graph, Table, Trainer  # noqa: F401
from kumo.engine import SnowflakeConnector  # noqa: F401  (import is lazy)
from kumo.engine import __version__
from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.version import get_kumo_version

dist = distribution(sys.argv[1])
assert dist.version == __version__ == get_kumo_version(), (
    dist.version, __version__, get_kumo_version())
extras = set(dist.metadata.get_all('Provides-Extra') or [])
print('extras:', ', '.join(sorted(extras)))

# Provider code ships in every install, their third-party dependencies only
# with the matching extra: the entry points are registered, but loading one
# must fail on the missing dependency rather than succeed.
sources = {e.name for e in dist.entry_points
           if e.group == 'kumoai_studio.sources'}
if 'snowflake' in sources:
    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        pass
    else:
        raise SystemExit('snowflake-connector-python is installed without '
                         'the snowflake extra')
    try:
        import kumo_snowflake.backend  # noqa: F401
    except ImportError:
        pass
    else:
        raise SystemExit('kumo_snowflake.backend imported without the '
                         'snowflake extra')

ingest = pathlib.Path(diskgraph.__file__).parent / 'bin' / 'ingest'
assert ingest.is_file() and os.access(ingest, os.X_OK), ingest
if importlib.util.find_spec('kumo_glove_6b_50d'):  # only with the `text` extra
    EmbeddingModelStore().get('glove.6B-50d').vectors()
print('wheel OK', __version__, '| torch', torch.__version__, '| pyg',
      torch_geometric.__version__)
"""


if __name__ == "__main__":
    sys.exit(main())
