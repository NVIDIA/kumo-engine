"""Direct-fetch handles and payloads exposed by `DiskgraphTaskDataset`.

Matches `runtime-dataset-contracts.md` §Contract 2 (post-rework).  The
canonical direct-access handle is `TaskRowHandle(table_name, node_id)`
where `table_name` is the ingested per-split DG table and `node_id`
is its table-local row identifier.  In the mock, the table is backed by
an `ingested_stage_tables` manifest ref with `storage_kind="mock_parquet"`.
The real implementation should use the same stage -> table-ref shape with
`storage_kind="diskgraph_table"`.  Kumo should treat the table name, not
the parquet path or stage string, as the stable identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskRowHandle:
    """Authoritative direct-access handle within one materialization.

    `node_id` is **table-local**, not graph-global.  In the mock
    implementation `node_id` is the row position within the parquet file
    referenced by the table's `ingested_stage_tables` manifest entry.
    """
    table_name: str
    node_id: int


@dataclass
class RawFeaturePayload:
    """Optional inspection-oriented raw feature payload returned by
    `DiskgraphTaskDataset.get_input(..., include_raw_features=True)`.
    The mock leaves this empty by default; the real runtime fills it
    from sampled feature blocks."""
    key: str
    rows: Any                                   # pyarrow.Table / RecordBatch
    row_count: int
    columns: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectTaskInput:
    """Direct-fetch result.  Task-semantic projection of one ingested
    stage-table row, intended for debugging / explainability /
    inspection — *not* a sampled training batch."""
    handle: TaskRowHandle

    task_family: str
    seed_ref: dict[str, Any]
    supervision: dict[str, Any]
    identity: dict[str, Any]
    task_payload: dict[str, Any]

    # Optional preserved pre-ingest identifiers for lineage/debugging.
    # In the mock, this is `{"input_id": <pre-ingest TT row position>}`
    # when the writer stamped one.
    source_row_ids: dict[str, Any] = field(default_factory=dict)

    # Optional raw feature payloads.  Mock leaves these empty.
    raw_features: dict[str, RawFeaturePayload] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
