# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the official AdventureWorks tables used by the Linux examples."""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE_URL = ("https://raw.githubusercontent.com/microsoft/sql-server-samples/"
            "master/samples/databases/adventure-works/oltp-install-script")
FILES = {
    "Customer.csv":
    "4704aa5b46aed326599014524c821aca50e938587e6a1133c0f6d9aae5c5efc6",
    "Product.csv":
    "df0379a7cba8b11b97ea5d104cf1313ef9f683137b6426f2fac6bb8c74c2a2ea",
    "SalesOrderHeader.csv":
    "cf637c7d58006ebbc13daf0b8ed6cf7f738165d298ae0b5479a59237429c1882",
    "SalesOrderDetail.csv":
    "220129977aa63e34fa4a1d66c2604bd2184373151efb09c98b343371ba77ab8a",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(destination):
    destination = destination.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-",
                         dir=destination.parent))
    try:
        for name, expected in FILES.items():
            path = staging / name
            with urlopen(f"{BASE_URL}/{name}", timeout=60) as response:
                with path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            actual = digest(path)
            if actual != expected:
                raise ValueError(
                    f"{name}: expected SHA-256 {expected}, received {actual}")
            print(f"Verified {name}: {actual}", flush=True)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        download(args.destination)
    except (HTTPError, URLError, OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
