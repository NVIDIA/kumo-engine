# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot-backed FM preparation and retained request lifecycle."""
from ._snapshot import SnapshotFMLoader as FMLoader


class PreparedRequest:
    """Resolved anchors; every load call samples exactly once per included split."""

    def __init__(self, loader, splits):
        self._loader = loader
        self._splits = splits
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._closed = True

    def load(self):
        if self._closed or self._loader._closed:
            raise RuntimeError("Prepared request or loader is closed")
        return self._loader._load(self._splits)


def prepare(*, engine, sample, target, task, features, classes=None,
            batch_size=1024, budgets=None, schema=None):
    """Prepare FM over one adopted public PyTorch snapshot engine."""
    return FMLoader(engine=engine, sample=sample, target=target, task=task,
                    features=features, classes=classes, batch_size=batch_size,
                    budgets=budgets, schema=schema)
