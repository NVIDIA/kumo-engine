# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""`configure(...)`: the one process-wide, opt-in setup call this package
has.

Everything else in `kumo.engine` is constructed and threaded through directly, no
`Backend`/`kumoai_studio.init(...)` needed. Scratch-directory placement is the
one exception: `Graph.snapshot()`'s DiskGraph ingest and `Trainer.fit`/`predict`
all read and write a shared local filesystem root
(`kumo.engine._scratch.get_scratch_dir()`, default `~/.cache/kumo-engine`)
directly, for the same reason `kumo_core.config.NativeRuntimeConfig` keeps
`scratch_root` explicit rather than hardcoded -- so it can be pointed at a
local NVMe SSD mount instead.

Call `configure(...)` once, early -- before constructing any `Graph`/
`Trainer`/etc. -- not mid-session: nothing here re-reads it after a
scratch-directory path has already been computed and used by an
in-progress `Graph`/`Trainer`.
"""

from pathlib import Path
from typing import Optional, Union


def configure(
    *,
    scratch_dir: Optional[Union[str, Path]] = None,
) -> None:
    r"""Sets process-wide configuration for this package.

    Args:
        scratch_dir: where `Graph.snapshot()`'s DiskGraph ingest and
            `Trainer.fit`/`predict`'s training/prediction runs read and
            write their working data -- e.g. a local NVMe SSD mount, for
            faster ingest/training than the default `~/.cache/kumo-engine`
            (typically the boot disk). Must be a local filesystem path --
            DiskGraph and GNN training read and write it directly, so a
            remote URI (an `s3://`/`gs://` path, a Snowflake stage) isn't
            accepted here. `None` leaves the current setting unchanged.

    Raises:
        ValueError: if `scratch_dir` isn't a local filesystem path.
    """
    if scratch_dir is not None:
        from kumo.engine._scratch import set_scratch_dir

        set_scratch_dir(scratch_dir)
