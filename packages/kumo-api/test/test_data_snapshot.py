# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.data_snapshot import TableSnapshotResource, TableSnapshotStage
from kumoapi.json_serde import from_json, to_json_dict
from kumoapi.source_table import TableStats


def test_table_snapshot_resource_preserves_public_table_stats() -> None:
    resource = from_json(
        {
            'id': 'table_fingerprint@data-version',
            'stats': {
                'size_bytes': 128,
                'num_rows': 5,
            },
            'stages': {},
        },
        TableSnapshotResource,
    )

    assert resource.stats == TableStats(size_bytes=128, num_rows=5)
    assert to_json_dict(resource)['stats'] == {
        'size_bytes': 128,
        'num_rows': 5,
    }


def test_table_snapshot_resource_allows_stats_before_ingest_completes(
) -> None:
    resource = from_json(
        {
            'id': 'table_fingerprint@data-version',
            'stages': {
                TableSnapshotStage.INGEST.value: {
                    'status': 'RUNNING',
                },
            },
        },
        TableSnapshotResource,
    )

    assert resource.stats is None
