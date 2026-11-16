# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def hold_prediction_materialization_ownership(index_path, job_id, entered,
                                              release):
    from kumo.diskgraph.job_sidecars import (
        prediction_materialization_ownership,
    )

    with prediction_materialization_ownership(index_path, job_id):
        entered.set()
        assert release.wait(timeout=60)
    # Imported application modules can own background threads; the lock has
    # been released and this process exists only to exercise OS-level flock.
    os._exit(0)


def hold_prediction_extension_shared_lock(index_path, job_id, entered, release):
    from kumo.diskgraph.job_sidecars import _SharedSidecarLock

    lock = _SharedSidecarLock(index_path,
                              os.path.join('train_tables', f'pred_{job_id}'))
    lock.acquire_shared()
    entered.set()
    assert release.wait(timeout=60)
    lock.release()
    os._exit(0)
