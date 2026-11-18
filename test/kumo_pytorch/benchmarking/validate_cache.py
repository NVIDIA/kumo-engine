# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate file-cache conditioning on an owned temporary Linux file."""

import argparse
import hashlib
import json
import mmap
import platform
import tempfile
import traceback
from pathlib import Path

from .file_cache import condition_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")
    result = {
        "kernel": platform.release(),
        "platform": platform.platform(),
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("file_cache.py").read_bytes()
        ).hexdigest(),
        "conditions": [],
    }
    try:
        with tempfile.TemporaryDirectory(dir=args.workdir) as directory:
            path = Path(directory) / "pages.bin"
            contents = bytes(range(256)) * (4 * 1024 * 1024 // 256)
            path.write_bytes(contents)
            for mode in ("warm", "cold", "warm"):
                observation = condition_files([path], mode)
                result["conditions"].append(observation)
                if observation["status"] != "verified":
                    raise AssertionError(
                        f"File-page-cache {mode} residency was not verified"
                    )
            with (
                path.open("rb") as stream,
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping,
            ):
                assert mapping[:] == contents  # fault every page into the live mapping
                observation = condition_files([path], "cold", release_mappings=True)
                result["conditions"].append(observation)
                assert observation["status"] == "verified"
                assert observation["files"][0]["released_readonly_mappings"] >= 1
                assert (
                    mapping[:] == contents
                )  # reader can fault back without corruption
            with (
                path.open("r+b") as stream,
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE),
            ):
                try:
                    condition_files([path], "cold", release_mappings=True)
                except RuntimeError as error:
                    assert "writable" in str(error)
                else:
                    raise AssertionError("Writable mapping was not rejected")
            if (
                hashlib.sha256(path.read_bytes()).digest()
                != hashlib.sha256(contents).digest()
            ):
                raise AssertionError("Conditioning changed file contents")
        result["status"] = "passed"
    except Exception:  # noqa: BLE001 -- unavailable kernel support must fail explicitly
        result.update(status="failed", traceback=traceback.format_exc())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
