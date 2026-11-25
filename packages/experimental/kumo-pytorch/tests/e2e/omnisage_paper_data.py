# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time Pin, board and user examples for OmniSage §§3.1–3.4."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from kumo.pytorch import api
from tests.test_scored_walks import star

PB_TOP_K = {"pins": 25, "boards": 75}
PP_TOP_K = {"pins": 50}
RESTART = 0.25
WALK_BUDGET = 1024
FUTURE_WINDOW = 7  # seconds in the synthetic activity log; both endpoints are excluded


@dataclass
class PaperFixture:
    root: Path
    schema: api.DataSchema
    dataset: api.Dataset
    selectors: dict
    sizes: dict
    steps: int
    batch_size: int
    history_length: int
    random_negatives: int
    cutoffs: tuple[int, ...] = ()
    future_window_seconds: int = FUTURE_WINDOW
    first_seen: dict | None = None


@dataclass
class PaperRequests:
    root_ids: dict
    times: dict
    pairs: dict
    features: dict
    negatives: dict
    history: torch.Tensor
    future: torch.Tensor


@dataclass
class PaperBatch(PaperRequests):
    graphs: dict


def _source(root, name, table, client):
    path = root / f"{name}.parquet"
    pq.write_table(table, path)
    if client is None:
        return api.Parquet(str(path))
    key = f"paper/{name}.parquet"
    client.upload_file(str(path), "omnisage-fixture", key)
    return api.Parquet(f"s3://omnisage-fixture/{key}")


def _vectors(ids, width, seed):
    # The two feature types have independent values; the model splits pin width at 1024.
    rng = np.random.default_rng(seed)
    value = rng.standard_normal((len(ids), width), dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(value.ravel()), width)


def _at(seconds):
    return pa.array(seconds, type=pa.timestamp("s", tz="UTC"))


def _table(name, columns, *, temporal=False):
    integer = api.DType("int64")
    fields = {"id": integer, **dict.fromkeys(columns, integer)}
    if temporal:
        fields["at"] = api.DType("timestamp", unit="s", timezone="UTC")
    return api.TableSpec(name, "id", fields, time_column="at" if temporal else api.NON_TEMPORAL)


def _paper_schema():
    base, _ = star()
    original = {t.name: t for t in base.tables}
    return api.DataSchema(
        tables=(
            replace(
                original["pins"],
                columns={**original["pins"].columns, "image": api.DType("float32", shape=(1152,))},
            ),
            replace(
                original["boards"],
                columns={
                    **original["boards"].columns,
                    "annotation": api.DType("float32", shape=(128,)),
                },
            ),
            original["membership"],
            _table("related", ("src", "dst"), temporal=True),
            _table("users", ()),
            _table("activity", ("user", "pin", "second", "unit_weight"), temporal=True),
        ),
        relations=(
            *base.relations,
            api.RelationSpec("related_from", "pins", "id", "related", "src"),
            api.RelationSpec("related_to", "pins", "id", "related", "dst"),
            api.RelationSpec("user_events", "users", "id", "activity", "user"),
        ),
    )


def _selector(root, name, entity, clock, supplied, client):
    anchor = {"sid": np.arange(len(entity)), "entity": entity, "at": _at(clock)}
    candidates = {}
    if name != "users":
        anchor["logged"] = pa.array(supplied, type=pa.list_(pa.struct([("pin", pa.int64())])))
        candidates["logged"] = api.CandidateColumn("logged", "pin")
    source = _source(root, name + "_examples", pa.table(anchor), client)
    return api.AnchorTable(source, "entity", "at", sample_id="sid", candidates=candidates)


def make_paper_fixture(
    root, *, pins, steps, batch_size, history_length, random_negatives, seed=42, client=None
):
    """Persist graph edges, synthetic engagement logs, user activity and independent anchors."""
    if pins < 32 or min(steps, batch_size, random_negatives) < 1 or history_length < 2:
        raise ValueError("need at least 32 pins, history length >=2 and positive batch sizes")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    boards = (pins + 2) // 3  # At most three Pins per offset, hence nine edges per board.
    users = max(12, batch_size * 2)
    schema = _paper_schema()
    pids, bids = np.arange(pins), np.arange(boards)
    tables = {
        "pins": pa.table({"id": pids, "image": _vectors(pids, 1152, seed + 1)}),
        "boards": pa.table({"id": bids, "annotation": _vectors(bids, 128, seed + 2)}),
        "users": pa.table({"id": range(users)}),
    }
    # Low-degree subgraphs exercise the paper's exemption from high-degree pruning.
    # The late edges permit direct checks that a history token cannot see its later graph.
    qs = np.repeat(pids, 3)
    offsets = np.tile([1, 3, 7], pins)
    when = np.tile([1, 6, 13], pins)
    tables["membership"] = pa.table(
        {"id": range(len(qs)), "pin": qs, "board": (qs + offsets) % boards, "at": _at(when)}
    )
    tables["related"] = pa.table(
        {"id": range(len(qs)), "src": qs, "dst": (qs + offsets) % pins, "at": _at(when)}
    )
    # Delayed starts exercise padding; deliberate repeat actions make row-vs-ID sampling observable.
    maximum_cutoff = 2 * history_length + 9
    activity = [
        (user, (user * 17 + i // 2 * 3) % pins, 2 * i + user % 3 * 2 + 1)
        for user in range(users)
        for i in range(maximum_cutoff // 2 + FUTURE_WINDOW + 2)
        if i >= (history_length // 2 if user % 3 == 1 else 0)
    ]
    au, ap, sec = zip(*activity, strict=True)
    tables["activity"] = pa.table(
        {
            "id": range(len(au)),
            "user": au,
            "pin": ap,
            "at": _at(sec),
            "second": sec,
            "unit_weight": np.ones(len(au), dtype=np.int64),
        }
    )
    sources = {name: _source(root, name, table, client) for name, table in tables.items()}
    sizes = {"pins": pins, "boards": boards, "users": users}
    selectors = {}
    # Separate random streams for the engagement, graph-pair and sequence examples.
    for i, name in enumerate(sizes):
        rng = np.random.default_rng(seed + 19 + i)
        entity, clock, supplied = [], [], []
        for step in range(steps):
            cutoff = _cutoff(step, history_length)
            for origin in ("sequence",) if name == "users" else ("logged", "graph"):
                for _ in range(batch_size):
                    query = int(rng.integers(sizes[name]))
                    entity.append(query)
                    clock.append(cutoff)
                    if origin == "logged":
                        label = (
                            query * (3 if name == "boards" else 1) + int(rng.integers(1, 7))
                        ) % pins
                        supplied.append([{"pin": label}])
                    else:
                        supplied.append([])
        selectors[name] = _selector(root, name, entity, clock, supplied, client)
    return PaperFixture(
        root,
        schema,
        api.Dataset(sources),
        selectors,
        sizes,
        steps,
        batch_size,
        history_length,
        random_negatives,
    )


def _cutoff(step, length):
    return 2 * length + 1 + 2 * (step % 4)


def _walks(table):
    banks = [api.WalkSample("pb", ("membership",), PB_TOP_K, RESTART, WALK_BUDGET)]
    if table == "pins":
        banks.append(api.WalkSample("pp", ("related",), PP_TOP_K, RESTART, WALK_BUDGET))
    return (api.GraphLayer(tuple(banks)),)


def _fields(*names):
    return api.Fields(tuple(api.Encode(name, api.Raw()) for name in names))


def paper_graph_spec(schema, table, batch_size):
    links = {
        "pins": {"degree": "pin_links", "from_degree": "related_from", "to_degree": "related_to"},
        "boards": {"degree": "board_links"},
    }
    values = {
        node: {
            name: api.Aggregate(api.Rows((link,), api.TimeWindow("at", stop=0)), "count")
            for name, link in names.items()
        }
        for node, names in links.items()
    }
    graph = api.NeighborhoodSpec(_walks(table), node_values=values)
    rep = api.Graph(
        {"pins": _fields("image", *links["pins"]), "boards": _fields("annotation", "degree")}
    )
    return api.DataLoaderSpec(
        schema,
        api.SampleSpec(api.EntitySpec(table), neighborhoods={"graph": graph}),
        api.BatchSpec(
            inputs={"graph": api.Input(api.NeighborhoodSource("graph"), rep, None)},
            batching=api.Batching(max_samples=batch_size),
        ),
    )


def _select_spec(fixture, table):
    events, hoods, draws = {}, {}, {}
    inputs = {}
    if table == "users":
        events["actions"] = api.EventSpec(
            api.Rows(("user_events",), api.TimeWindow("at", stop=0)),
            {"pin": api.Field("pin"), "second": api.Field("second")},
        )
        inputs["history"] = api.Input(
            api.EventSource(
                ("actions",), max_events=fixture.history_length, keep="last", order="ascending"
            ),
            _fields("pin", "second"),
            api.Padded(),
        )
        # A weight of one per physical interaction yields probability proportional to the
        # number of future actions on a Pin, even if several rows reference the same Pin.
        draws["future"] = (
            "pins",
            api.DrawRows(
                api.Rows(
                    ("user_events",),
                    api.TimeWindow("at", start=0, stop=fixture.future_window_seconds - 1),
                ),
                "pin",
                count=1,
                weight="unit_weight",
            ),
        )
    else:
        hoods["graph"] = api.NeighborhoodSpec(_walks(table))
        draws["logged"] = ("pins", None)
        if table == "pins":
            draws["pp"] = (
                "pins",
                api.DrawNeighbors("graph", ("pp.pins",), count=1, weight="score"),
            )
            draws["pb"] = (
                "boards",
                api.DrawNeighbors("graph", ("pb.boards",), count=1, weight="score"),
            )
        else:
            draws["bp"] = (
                "pins",
                api.DrawNeighbors("graph", ("pb.pins",), count=1, weight="score"),
            )
    sets = {}
    for name, (dest, draw) in draws.items():
        sets[name] = api.CandidateSetSpec(
            api.EntitySpec(dest), values={"id": api.Field("id")}, draw=draw
        )
        inputs[name] = api.Input(api.CandidateSource(name), _fields("id"), api.Packed())
    return api.DataLoaderSpec(
        fixture.schema,
        api.SampleSpec(
            api.EntitySpec(table), events=events, neighborhoods=hoods, candidate_sets=sets
        ),
        api.BatchSpec(inputs=inputs, batching=api.Batching(max_samples=fixture.batch_size)),
    )


def _roots(batch):
    return batch.metadata["entity_ids"].tolist()


def paper_selection(fixture, engine, seed=42):
    """Sample an ID/time schedule using only native, source-backed DiskGraph batches.

    This finite run fixes the sample identities for paired timings. The schedule contains
    neither gathered features nor model graphs; both backends do that work during iteration.
    """
    rng = np.random.default_rng(seed + 317)
    result = []
    with ExitStack() as stack:
        streams = {}
        for table in fixture.sizes:
            plan = stack.enter_context(
                api.prepare(_select_spec(fixture, table), fixture.dataset, engine)
            )
            task = stack.enter_context(plan.materialize(fixture.selectors[table]))
            streams[table] = iter(task.batches(output_format="model", shuffle=False, seed=seed))
        for step in range(fixture.steps):
            at = fixture.cutoffs[step] if fixture.cutoffs else _cutoff(step, fixture.history_length)
            lookup = {"pins": {}, "boards": {}}

            def index(table, pk, t=at):
                entries = lookup[table]
                return entries.setdefault((int(pk), int(t)), len(entries))

            pairs = {}
            for table in ("pins", "boards"):
                logged = next(streams[table])
                graph = next(streams[table])
                draws = (
                    (("logged", logged, "pins"), ("pp", graph, "pins"), ("pb", graph, "boards"))
                    if table == "pins"
                    else (("logged", logged, "pins"), ("bp", graph, "pins"))
                )
                for label, batch, dest in draws:
                    q, p = [], []
                    for pk, group in zip(
                        _roots(batch), batch.metadata["candidate_entity_ids"][label], strict=True
                    ):
                        if group:
                            q.append(index(table, pk))
                            p.append(index(dest, group[0]))
                    pairs[f"{table}_{label}"] = (
                        table,
                        torch.tensor(q, dtype=torch.long),
                        dest,
                        torch.tensor(p, dtype=torch.long),
                    )
            users = next(streams["users"])
            seq = users.inputs["history"]
            history = torch.full((len(_roots(users)), fixture.history_length), -1, dtype=torch.long)
            for i, (ids, times, valid) in enumerate(
                zip(seq["pin"]["values"], seq["second"]["values"], seq["pin"]["valid"], strict=True)
            ):
                for j, (pin, t) in enumerate(
                    zip(ids[valid].tolist(), times[valid].tolist(), strict=True)
                ):
                    history[i, j] = index("pins", pin, t)
            future = torch.tensor(
                [
                    index("pins", p[0]) if p else -1
                    for p in users.metadata["candidate_entity_ids"]["future"]
                ]
            )

            active = {
                t: fixture.sizes[t]
                if fixture.first_seen is None
                else int(np.searchsorted(fixture.first_seen[t], at, side="right"))
                for t in lookup
            }

            def random_roots(count):
                return {
                    table: torch.tensor(
                        [index(table, pk) for pk in rng.integers(active[table], size=count)]
                    )
                    for table in lookup
                }

            features = random_roots(fixture.batch_size)
            negatives = random_roots(fixture.random_negatives)
            ids = {
                t: torch.tensor([pk for pk, _ in m], dtype=torch.long) for t, m in lookup.items()
            }
            ticks = {
                t: torch.tensor([clock for _, clock in m], dtype=torch.long)
                for t, m in lookup.items()
            }
            result.append(PaperRequests(ids, ticks, pairs, features, negatives, history, future))
    return result


@contextmanager
def paper_batches(
    fixture, *, num_threads=1, seed=42, engine=None, requests=None, graph_loader_factory=None
):
    """Fetch native model graphs for one finite pass over the selected root occurrences."""
    if engine is None:
        engine = api.DiskGraphEngine(num_threads=num_threads, random_seed=seed)
    if requests is None:
        requests = paper_selection(fixture, engine, seed)
    with ExitStack() as stack:
        if graph_loader_factory is not None:
            fetch = stack.enter_context(
                graph_loader_factory(engine.index, fixture, num_threads, engine.random_seed)
            )
        else:
            tasks = {}
            stage = Path(stack.enter_context(TemporaryDirectory()))
            for table in ("pins", "boards"):
                ids = torch.cat([r.root_ids[table] for r in requests])
                times = torch.cat([r.times[table] for r in requests])
                names = [str(i) for i, r in enumerate(requests) for _ in r.root_ids[table]]
                source = _source(
                    stage,
                    "scheduled_" + table,
                    pa.table(
                        {
                            "id": ids.numpy(),
                            "at": _at(times.numpy()),
                            "sid": range(len(ids)),
                            "step": names,
                        }
                    ),
                    None,
                )
                anchor = api.AnchorTable(source, "id", "at", sample_id="sid", split="step")
                spec = paper_graph_spec(
                    fixture.schema, table, max(len(r.root_ids[table]) for r in requests)
                )
                plan = stack.enter_context(api.prepare(spec, fixture.dataset, engine))
                tasks[table] = stack.enter_context(
                    plan.materialize(anchor, fit_split=api.ALL_ANCHORS)
                )

        def events():
            for step, req in enumerate(requests):
                graphs = {}
                for table in ("pins", "boards"):
                    ids, times = req.root_ids[table], req.times[table]
                    if graph_loader_factory is not None:
                        graph = fetch(table, ids, times)
                    else:
                        batch = next(
                            iter(
                                tasks[table].batches(
                                    split=str(step),
                                    batch_size=len(ids),
                                    output_format="model",
                                    shuffle=False,
                                )
                            )
                        )
                        graph = batch.inputs["graph"]
                        fields = graph["nodes"]["pins"]["features"]
                        fields["degree"]["values"] = (
                            fields["degree"]["values"]
                            + fields.pop("from_degree")["values"]
                            + fields.pop("to_degree")["values"]
                        )
                    if not torch.equal(graph["nodes"][table]["ids"][: len(ids)], ids):
                        raise AssertionError("model graph roots changed schedule order")
                    graphs[table] = graph
                yield PaperBatch(**vars(req), graphs=graphs)

        yield events()
