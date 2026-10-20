# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import diskgraph


def test_format_versions_names_every_on_disk_format():
    versions = diskgraph.format_versions()
    assert set(versions) == {
        "layout",
        "shard_commit",
        "temp_meta",
        "ingest_report",
        "manifest_schema",
    }
    assert all(type(value) is int and value > 0 for value in versions.values())
