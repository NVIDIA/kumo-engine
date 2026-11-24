# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a connector table as a flat directory of local parquet parts."""

import os
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import unquote

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kumoapi.typing import Dtype

from kumo.engine.connector.source_table import SourceTable


def stage_source_table(
    source_table: SourceTable,
    destination: Path,
    *,
    dtypes: Optional[Mapping[str, Dtype]] = None,
) -> Tuple[Path, Optional[str]]:
    """Stage ``source_table``'s parquet parts; returns them and their version.

    ``destination`` must not exist yet. The connector's backend stages into
    ``destination / "source"``, and each file it reports there is moved, in
    order, into ``destination / "parts"``.

    ``dtypes`` is passed to the backend's ``prepare_ingest``, which uses it to
    select columns. Its effect on types depends on the backend: an object
    store, for example, keeps Parquet's stored types only when it's given.
    Read the staged schema rather than assume ``dtypes``.
    """
    if not isinstance(source_table, SourceTable):
        raise TypeError("source_table must be a kumo.engine SourceTable")
    destination.mkdir(parents=True, exist_ok=False)
    staged = destination.resolve() / "source"
    prepared = source_table.connector.source_backend.prepare_ingest(
        source_table.connector,
        source_table.table_name,
        str(destination / "source"),
        dtypes=dtypes,
        sql=source_table.sql,
    )
    if prepared.credentials is not None:
        raise NotImplementedError(
            f"Staging {source_table.table_name} needs a source backend that "
            "stages local parquet without scoped ingest credentials.")
    if prepared.source.get("type") != "parquet":
        raise NotImplementedError(
            f"Staging {source_table.table_name} needs a source backend that "
            f"stages local parquet; got {prepared.source.get('type')!r}.")
    paths = [Path(path) for path in prepared.source.get("paths", ())]
    if not paths:
        path = prepared.source.get("path")
        if path:
            paths = [Path(path)]
    if not paths or any(not path.exists() for path in paths):
        raise RuntimeError(
            f"Staging {source_table.table_name} produced no parquet files.")
    # DiskGraph accepts one file or a flat directory of parquet parts. Flatten
    # connector directories without re-encoding or combining their row groups.
    parts = [
        Path(fragment.path) for path in paths
        for fragment in ds.dataset(str(path), format="parquet").get_fragments()
    ]
    if not parts:
        raise RuntimeError(
            f"Staging {source_table.table_name} produced no parquet files.")
    # Move only what the backend wrote for this call, never a source's files.
    parts = list(dict.fromkeys(part.parent.resolve() / part.name
                               for part in parts))
    outside = [str(part) for part in parts if not part.is_relative_to(staged)]
    if outside:
        raise RuntimeError(
            f"Staging {source_table.table_name} reported files outside "
            f"{staged}: {outside}")
    # Flattening drops Hive key=value directories, so each key must already
    # be a column of the file under it.
    for part in parts:
        keys = {
            unquote(name.split("=", 1)[0])
            for name in part.relative_to(staged).parent.parts if "=" in name
        }
        missing = keys - set(pq.read_schema(part).names) if keys else set()
        if missing:
            raise RuntimeError(
                f"Staging {source_table.table_name} left partition keys "
                f"{sorted(missing)} in directory names only: {part}")
    output = destination / "parts"
    output.mkdir()
    for index, part in enumerate(parts):
        os.replace(part, output / f"part-{index:05d}.parquet")
    return output, prepared.version
