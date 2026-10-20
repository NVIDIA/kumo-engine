# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import diskgraph


def test_format_versions_names_every_on_disk_format():
    assert diskgraph.format_versions() == {
        "layout": 4,
        "shard_commit": 1,
        "temp_meta": 2,
        "ingest_report": 2,
        "manifest_schema": 1,
    }
