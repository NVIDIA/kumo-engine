# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a connector table as a flat directory of local parquet parts."""

import errno
import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, Tuple

import pyarrow.dataset as ds
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
    ``destination / "source"``, and ``destination / "parts"`` links to each
    file it reports, in order.

    ``dtypes`` is passed to the backend's ``prepare_ingest``, which uses it to
    select columns. Its effect on types depends on the backend: an object
    store, for example, keeps Parquet's stored types only when it's given.
    Read the staged schema rather than assume ``dtypes``.
    """
    if not isinstance(source_table, SourceTable):
        raise TypeError("source_table must be a kumo.engine SourceTable")
    destination.mkdir(parents=True, exist_ok=False)
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
    output = destination / "parts"
    output.mkdir()
    parts = [
        Path(fragment.path) for path in paths
        for fragment in ds.dataset(str(path), format="parquet").get_fragments()
    ]
    if not parts:
        raise RuntimeError(
            f"Staging {source_table.table_name} produced no parquet files.")
    for index, part in enumerate(parts):
        target = output / f"part-{index:05d}.parquet"
        try:
            # Relative links keep the retained source and parts directory
            # together if the staging tree moves.
            target.symlink_to(os.path.relpath(part.resolve(), output.resolve()))
        except OSError as error:
            if error.errno not in (
                    errno.EPERM, errno.EACCES, errno.EOPNOTSUPP, errno.ENOSYS):
                raise
            shutil.copyfile(part, target)
    return output, prepared.version
