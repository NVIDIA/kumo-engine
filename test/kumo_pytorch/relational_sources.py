# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream declared public Parquet sources into root/event tables.

Ownership is explicit: direct root events, events on an owned entity, or a
physical event with two owned endpoints. No time or target filtering occurs.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import safe_path, verify_file


def source_paths(root, declaration, files):
    keys = declaration.get("paths")
    if keys is None:
        prefix = declaration["prefix"]
        if not prefix.endswith("/"):
            raise ValueError("Source table prefixes must end with a slash")
        keys = sorted(
            k for k in files if k.startswith(prefix) and k.endswith(".parquet")
        )
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Each source table requires distinct pinned Parquet files")
    return [safe_path(root, key) for key in keys]


def _required_integers(values, name):
    if (
        values.type != pa.int64()
        or values.null_count
        or pc.any(pc.less(values, 0)).as_py()
    ):
        raise ValueError(f"{name} must contain nonnegative int64 identities")


def prepare_sources(data_dir, output, definition, files, roots):
    from kumo.pytorch.api import Dataset, Parquet

    options = definition["source_options"]
    paths = {
        name: source_paths(data_dir, entry, files)
        for name, entry in options["tables"].items()
    }
    for group in paths.values():
        for path in group:
            verify_file(path, files[path.relative_to(data_dir).as_posix()])
    output.mkdir(parents=True, exist_ok=False)
    root = options["root"]
    owner = options["ownership"]
    level = options.get("root_level", "owner")
    if level not in {"owner", "owned"}:
        raise ValueError(f"Unknown root ownership level: {level}")
    root_rows = pq.read_table(paths[root["table"]], columns=[root["key"], root["time"]])
    root_ids = root_rows[root["key"]].combine_chunks()
    _required_integers(root_ids, root["key"])
    if len(pc.unique(root_ids)) != len(root_ids):
        raise ValueError("Duplicate root identity")
    ownership = pq.read_table(
        paths[owner["table"]], columns=[owner["key"], owner["root"]]
    )
    child_ids = ownership[owner["key"]].combine_chunks()
    owners = ownership[owner["root"]].combine_chunks()
    _required_integers(child_ids, owner["key"])
    _required_integers(owners, owner["root"])
    parent_ids = root_ids
    if level == "owned":
        parent = options["parent"]
        parent_ids = pq.read_table(paths[parent["table"]], columns=[parent["key"]])[
            parent["key"]
        ].combine_chunks()
        _required_integers(parent_ids, parent["key"])
        if len(pc.unique(parent_ids)) != len(parent_ids) or set(
            child_ids.to_pylist()
        ) != set(root_ids.to_pylist()):
            raise ValueError("Duplicate parent or root/owned identities disagree")
    if (
        len(pc.unique(child_ids)) != len(child_ids)
        or pc.any(pc.invert(pc.is_in(owners, value_set=parent_ids))).as_py()
    ):
        raise ValueError("Duplicate owned identity or unknown root owner")
    selected = (
        root_ids
        if roots is None
        else pa.array(sorted(int(v) for v in roots), pa.int64())
    )
    if pc.any(pc.invert(pc.is_in(selected, value_set=root_ids))).as_py():
        raise ValueError("Unknown selected root")
    root_rows = root_rows.filter(pc.is_in(root_ids, value_set=selected))
    clock = root_rows[root["time"]]
    if clock.type != pa.int64() or clock.null_count:
        raise ValueError("Root availability requires integral Unix seconds")
    pq.write_table(
        pa.table(
            {
                "root_id": root_rows[root["key"]],
                "available_at": pc.cast(clock, pa.timestamp("s")),
            }
        ),
        output / "roots.parquet",
    )

    # Slots use the complete ownership table, preserving IDs in bounded runs.
    fanout, members = 1, None
    if level == "owned":
        ordered = pa.table({"__parent": owners, "__member": child_ids}).sort_by(
            [("__parent", "ascending"), ("__member", "ascending")]
        )
        slots, previous, slot = [], None, 0
        for parent_id in ordered["__parent"].to_pylist():
            slot = slot + 1 if parent_id == previous else 0
            slots.append(slot)
            previous = parent_id
        fanout = max(slots, default=0) + 1
        members = ordered.append_column("__slot", pa.array(slots, pa.int64())).filter(
            pc.is_in(ordered["__member"], value_set=selected)
        )

    def owner_ids(values):
        indices = pc.index_in(values, value_set=child_ids)
        unknown = pc.and_(pc.is_valid(values), pc.is_null(indices))
        if pc.any(unknown).as_py():
            raise ValueError("Unknown owned event endpoint")
        return pc.take(owners, indices)

    for family, spec in options["events"].items():
        ownership_spec = spec["ownership"]
        columns = [spec["key"], spec["time"], *spec["fields"].values()]
        columns += ownership_spec.get("columns", [ownership_spec.get("column")])
        if ownership_spec.get("verify_root"):
            columns.append(ownership_spec["verify_root"])
        columns = list(dict.fromkeys(c for c in columns if c))
        writer = None
        try:
            for path in paths[family]:
                source = pq.ParquetFile(path)
                batches = source.iter_batches(columns=columns, batch_size=65536)
                if source.metadata.num_rows == 0:
                    batches = (
                        pa.record_batch(
                            [
                                pa.array([], source.schema_arrow.field(c).type)
                                for c in columns
                            ],
                            names=columns,
                        ),
                    )
                for batch in batches:
                    data = pa.Table.from_batches([batch])
                    kind = ownership_spec["kind"]
                    multiplier = fanout if level == "owned" and kind == "direct" else 2
                    identity, stamp = data[spec["key"]], data[spec["time"]]
                    _required_integers(identity, spec["key"])
                    maximum = (2**63 - multiplier) // multiplier
                    if pc.any(pc.greater(identity, maximum)).as_py():
                        raise ValueError(
                            f"{family}.{spec['key']} exceeds {maximum}; "
                            f"identity mapping requires {multiplier} int64 slots per source row"
                        )
                    if stamp.type != pa.int64() or stamp.null_count:
                        raise ValueError("Event clocks require integral Unix seconds")
                    if level == "owned" and kind == "direct":
                        parents = data[ownership_spec["column"]]
                        _required_integers(parents, "event owner")
                        if pc.any(
                            pc.invert(pc.is_in(parents, value_set=parent_ids))
                        ).as_py():
                            raise ValueError("Unknown parent event owner")
                        data = data.append_column("__parent", parents).join(
                            members, keys="__parent", join_type="inner"
                        )
                    identity, stamp = data[spec["key"]], data[spec["time"]]
                    if kind == "endpoints":
                        endpoints = [data[c] for c in ownership_spec["columns"]]
                        parent_endpoints = [owner_ids(values) for values in endpoints]
                        sender, receiver = (
                            endpoints if level == "owned" else parent_endpoints
                        )
                        same = pc.fill_null(pc.equal(sender, receiver), False)
                        emissions = [
                            (
                                sender,
                                pc.is_valid(sender),
                                pc.if_else(same, "self", "outgoing"),
                                0,
                            ),
                            (
                                receiver,
                                pc.and_(pc.is_valid(receiver), pc.invert(same)),
                                pa.repeat("incoming", len(data)),
                                1,
                            ),
                        ]
                    elif level == "owned" and kind == "direct":
                        emissions = [
                            (
                                data["__member"],
                                pa.repeat(True, len(data)),
                                None,
                                data["__slot"],
                            )
                        ]
                    else:
                        values = data[ownership_spec["column"]]
                        parent_target = owner_ids(values) if kind == "owned" else values
                        target = values if level == "owned" else parent_target
                        _required_integers(target, "event owner")
                        if kind not in {"owned", "direct"}:
                            raise ValueError(f"Unknown ownership kind: {kind}")
                        if pc.any(
                            pc.invert(pc.is_in(target, value_set=root_ids))
                        ).as_py():
                            raise ValueError("Unknown direct event owner")
                        if ownership_spec.get("verify_root"):
                            declared = data[ownership_spec["verify_root"]]
                            if declared.null_count or (
                                len(data)
                                and not pc.all(
                                    pc.equal(parent_target, declared)
                                ).as_py()
                            ):
                                raise ValueError(
                                    "Event ownership disagrees with its owned entity"
                                )
                        emissions = [(target, pa.repeat(True, len(data)), None, 0)]
                    for target, eligible, direction, side in emissions:
                        # Endpoint/member slots retain stable physical identities.
                        event_id = pc.add_checked(
                            pc.multiply_checked(identity, multiplier), side
                        )
                        values = {
                            "event_id": event_id,
                            "root_id": target,
                            "source_id": identity,
                            "event_time": pc.cast(stamp, pa.timestamp("s")),
                            "event_time_seconds": stamp,
                            "source_kind": pa.repeat(family, len(data)),
                            **{
                                alias: data[column]
                                for alias, column in spec["fields"].items()
                            },
                        }
                        if direction is not None:
                            values["direction"] = direction
                        mapped = pa.table(values).filter(
                            pc.and_(eligible, pc.is_in(target, value_set=selected))
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(
                                output / f"{family}.parquet",
                                mapped.schema,
                                compression="zstd",
                            )
                        writer.write_table(mapped)
            if writer is None:
                raise ValueError(f"Source family has no row groups: {family}")
        finally:
            if writer is not None:
                writer.close()
    dataset = Dataset(
        {
            name: Parquet(str(output / f"{name}.parquet"))
            for name in definition["tables"]
        }
    )
    mapping = {
        "scope": {
            "kind": "full_source" if roots is None else "selected_roots",
            "complete_selected_root_histories": True,
            "time_filter": None,
        },
        "sources": {
            name: [
                {**files[path.relative_to(data_dir).as_posix()], "path": str(path)}
                for path in group
            ]
            for name, group in paths.items()
        },
        "tables": {
            name: {
                "rows": pq.read_metadata(source.path).num_rows,
                "bytes": Path(source.path).stat().st_size,
            }
            for name, source in dataset.tables.items()
        },
    }
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return dataset, mapping
