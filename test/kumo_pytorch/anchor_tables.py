"""Persist bounded correctness-probe anchors; DG owns their loading and ordering.

Population benchmarks use population_anchors.project_population directly. These
small probes retain independent expected records for target/delivery assertions.
"""

from pathlib import Path

import kumo.pytorch.api as r
import pyarrow as pa
import pyarrow.parquet as pq


def write_probe_anchors(expected, path):
    path = Path(path).resolve()
    metadata = sorted({key for row in expected for key in row.metadata})
    targets = sorted({target.name for row in expected for target in row.targets})
    records = []
    for row in expected:
        assert not row.candidates, "Probe projection does not support candidate sets"
        records.append(
            {
                "sample_id": row.sample_id,
                "entity_id": row.entity_id,
                "as_of_time": row.as_of_time,
                "split": row.split,
                **{key: row.metadata.get(key) for key in metadata},
                **{
                    f"target_{name}": next(
                        (t.value for t in row.targets if t.name == name), None
                    )
                    for name in targets
                },
            }
        )
    table = pa.Table.from_pylist(records)
    if path.exists():
        existing = pq.read_table(path)
        assert existing.equals(table.cast(existing.schema)), (
            "Anchor path reused with different records"
        )
    else:
        pq.write_table(table, path)
    return r.AnchorTable(
        r.Parquet(str(path)),
        "entity_id",
        "as_of_time",
        sample_id="sample_id",
        split="split",
        targets={name: f"target_{name}" for name in targets},
        metadata=metadata,
    )


def load_probe_anchors(binding):
    assert not binding["candidates"], "Probe projection does not support candidate sets"
    return r.AnchorTable(**{**binding, "source": r.Parquet(binding["source"]["path"])})
