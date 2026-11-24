# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small relational fixtures and a Python oracle independent of FM internals."""
import base64
import json
import random
from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from kumo.pytorch import api

from kumo import fm


def instant(second):
    return datetime.fromtimestamp(second, tz=UTC)


def make_case(path, *, string_keys=False, parent_first=False, seed=391,
              entities=4, events_per_entity=12, ignored_dtype=None, amount_dtype=None):
    path.mkdir(parents=True, exist_ok=True)
    keys = ([f"用户-{i:04d}" for i in range(entities)] if string_keys else
            [-(2**63), -1, 2**53 + 1, 2**63 - 1] + list(range(entities)))[:entities]
    rng = random.Random(seed)
    events = []
    for position, key in enumerate(keys[:-1]):  # Last entity has no history.
        times = [9, 10, 11, 19, 20, 21, 29, 30, 31]
        times += rng.sample(range(40, 100000), max(0, events_per_entity - len(times)))
        for second in times[:events_per_entity]:
            identifier = len(events) + 1
            events.append({"id": identifier, "user": key, "at": instant(second),
                           "text": None if identifier % 3 == 0 else 'é😀\\"\n',
                           "amount": None if identifier % 2 else identifier / 4,
                           "flag": None if identifier % 5 == 0 else identifier % 2 == 0})
    if amount_dtype == pa.int32():
        for event in events:
            if event["amount"] is not None:
                event["amount"] = event["id"]
    rng.shuffle(events)  # Physical input order must not select the recent rows.
    key_type = pa.string() if string_keys else pa.int64()
    frames = {
        "users": pa.table({"id": pa.array(keys, key_type)}),
        "events": pa.Table.from_pylist(events, schema=pa.schema([
            ("id", pa.int64()), ("user", key_type), ("at", pa.timestamp("us", tz="UTC")),
            ("text", pa.string()), ("amount", amount_dtype or pa.float64()), ("flag", pa.bool_())])),
    }
    if ignored_dtype is not None:
        frames["events"] = frames["events"].append_column(
            "unused", pa.nulls(len(events), type=ignored_dtype))
    tables, sources = [], {}
    for name, frame in frames.items():
        filename = path / f"{name}.parquet"
        pq.write_table(frame, filename, row_group_size=3)
        sources[name] = api.Parquet(str(filename))
        tables.append(api.TableSpec(name, "id", {
            field.name: api.DType.from_arrow(field.type) for field in frame.schema},
            time_column="at" if name == "events" else api.NON_TEMPORAL))
    relation = (api.RelationSpec("activity", "users", "id", "events", "user") if parent_first
                else api.RelationSpec("activity", "events", "user", "users", "id"))
    schema = api.DataSchema(tuple(tables), (relation,))
    sample = sample_for(parent_first=parent_first)
    contract = api.DataLoaderSpec(schema, sample, api.BatchSpec(inputs={
        "history": api.Input(api.NeighborhoodSource("history"), api.Graph(), None)}))
    producer = api.DiskGraphEngine(num_threads=2)
    with api.prepare(contract=contract, dataset=api.Dataset(sources), engine=producer):
        pass
    engine = api.DiskGraphEngine.from_snapshot(
        SimpleNamespace(index_path=producer.index, id="correctness"), num_threads=2)
    return engine, keys, events


def sample_for(*, parent_first=False, fanout=3, strategy="recent", classification=False):
    future = api.Aggregate(api.Rows(
        path=("activity",), time=api.TimeWindow("at", start=0, stop=10)), "count")
    if classification:
        future = api.Compare(future, "greater_than", 0)
    return api.SampleSpec(entity=api.EntitySpec("users"), targets={"future": future},
                          neighborhoods={"history": api.NeighborhoodSpec((api.GraphLayer((
                              api.GraphSample("activity", direction="forward" if parent_first else "reverse",
                                              fanout=fanout, strategy=strategy),)),))})


def prepare(engine, **options):
    defaults = {"sample": sample_for(), "target": "future", "task": "regression",
                "features": {"users": [], "events": ["text", "amount", "flag", "at"]}}
    defaults.update(options)
    return fm.prepare(engine=engine, **defaults)


def assert_oracle(batch, events, splits, *, fanout=3, classification=False):
    """Check values, occurrence ownership, recent selection, and selected edges."""
    for split, pairs in splits.items():
        frames = batch.tables(split)
        instances = frames["instance_table"].to_pylist()
        assert [(r["instance_id"], r["__fm_entity_key"], r["__fm_anchor"])
                for r in instances] == [(identifier, key, instant(second))
                                        for key, second, identifier in pairs]
        expected = []
        for row, (key, second, identifier) in zip(instances, pairs, strict=True):
            if split == "context":
                count = sum(e["user"] == key and instant(second) < e["at"] <= instant(second + 10)
                            for e in events)
                expected_label = ("yes" if count else "no") if classification else float(count)
                assert row["__fm_target"] == expected_label
            else:
                assert "__fm_target" not in row
            history = sorted((e for e in events if e["user"] == key and e["at"] <= instant(second)),
                             key=lambda e: e["at"], reverse=True)[:fanout]
            expected.extend({"instance_id": identifier, "id": e["id"],
                             "text": e["text"], "amount": e["amount"], "flag": e["flag"],
                             "at": e["at"], "__fm_edge_activity": key}
                            for e in sorted(history, key=lambda e: e["id"]))
        assert frames["related_tables"]["events"].to_pylist() == expected
        assert frames["related_tables"]["users"].to_pylist() == [
            {"instance_id": identifier, "id": key} for key, _, identifier in pairs]


def assert_wire_roundtrip(batch):
    for encoding in ("arrays", "arrow_ipc"):
        body = batch.encode(encoding)
        assert body == batch.encode(encoding)
        payload = json.loads(body)
        for split in batch.splits:
            direct = batch.tables(split)
            pairs = [(payload[split]["instance_table"], direct["instance_table"])]
            pairs += [(payload[split]["related_tables"][name], table)
                      for name, table in direct["related_tables"].items()]
            for encoded, table in pairs:
                if encoding == "arrow_ipc":
                    decoded = pa.ipc.open_stream(base64.b64decode(encoded["payload_base64"])).read_all()
                    assert decoded.equals(table)
                else:
                    def cell(value):
                        if isinstance(value, datetime):
                            return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        if type(value) is int and abs(value) > 2**53 - 1:
                            return str(value)
                        return value
                    assert encoded["rows"] == [[cell(value) for value in row.values()]
                                               for row in table.to_pylist()]
