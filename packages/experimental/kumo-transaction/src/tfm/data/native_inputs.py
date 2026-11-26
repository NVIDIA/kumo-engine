# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source tables and one AnchorTable feed continuous model-ready native batches."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.config import FeaturesConfig, RelationalTaskConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.table_source import TableSource
from tfm.data.loaders import resolve_loader
from tfm.data.native_batches import NativeBatches
from tfm.data.native_source import (
    NativeSourceFiles,
    file_sha256,
    source_entity_id,
    validate_feature_bindings,
)
from tfm.data.native_spec import anchor_fitted_engine, history_contract, with_graph
from tfm.data.native_tensors import NativeTensorAdapter
from tfm.data.native_views import SourceRows, source_view
from tfm.extras.kumo_pytorch import SOURCE_REVISION, native_api

Mode = Literal["history", "sequence", "graph", "hybrid", "flat"]


class NativeInputs(AbstractContextManager["NativeInputs"]):
    """Own source bindings, fitted native tasks and bounded CPU batch producers.

    ``anchors`` is a Parquet table with the root's original key columns plus
    ``sample_id``, UTC ``cutoff`` and ``split``. Repeated roots are allowed;
    sample IDs must be unique. Optional entity targets and their availability
    travel on metadata columns, never through model feature encoders.

    History controls open a history-only task, avoiding unused graph retrieval
    and transfer. Its fitted history state must exactly match the hybrid's.
    Shared native resources close after every producer, including on failure.
    """

    def __init__(
        self,
        source: TableSource,
        *,
        root: str,
        loader: str = "kumo_pytorch",
        history_paths: Mapping[str, tuple[str, ...]],
        anchors: Path | None = None,
        task: RelationalTaskConfig | None = None,
        anchor_factory: Callable[[NativeSourceFiles, Path], Path] | None = None,
        fit_split: str | tuple[str, ...] = "train",
        target_level: Literal["entity", "sequence"] = "entity",
        views: Mapping[str, Mapping[str, SourceRows]] | None = None,
        fit_view: str = "full",
        fixed_vocabularies: Mapping[str, tuple[str, ...]] | None = None,
        split_families: Mapping[str, str] | None = None,
        split_namespaces: Mapping[str, str] | None = None,
        split_views: Mapping[str, str] | None = None,
        exclusive_terminal_paths: bool = False,
        work_dir: Path,
        max_events: int,
        batch_size: int,
        depth: int = 6,
        fanout: int = 10,
        observation_seconds: int | None = None,
        amount_key: str = "amount",
        num_threads: int = 1,
        features: FeaturesConfig | None = None,
    ) -> None:
        if task is not None and task.labels is not None:
            label_table = next(
                (table for table in source.tables if table.name == task.labels.table), None
            )
            if label_table is None or label_table.role != "labels":
                raise SchemaError(f"Label path {task.labels.table}: declare a labels-role table")
            validate_feature_bindings(source, extra_forbidden=frozenset({task.labels.target}))
        self.loader_name = loader
        self.loader = resolve_loader(loader)
        if self.loader.engine_factory is None:
            raise ValueError("Source-table NativeInputs requires an engine-backed loader")
        self.resources = ExitStack()
        self._closed = False
        self.root, self.depth, self.fanout = root, depth, fanout
        self.max_events, self.batch_size = max_events, batch_size
        self.fit_split = fit_split
        self.split_families = dict(split_families or {})
        self.split_namespaces = dict(split_namespaces or {})
        self.views = {name: dict(rules) for name, rules in (views or {}).items()}
        if any(
            re.fullmatch(r"[A-Za-z0-9_-]+", name) is None or name == "full" for name in self.views
        ):
            raise ValueError("Native view names must be simple identifiers; full is reserved")
        self.fit_view = fit_view
        if fit_view != "full" and fit_view not in self.views:
            raise ValueError("Native fit view must be a declared source view")
        self.split_views = dict(split_views or {})
        if set(self.split_views.values()) - {"full", *self.views}:
            raise ValueError("Native split views must name declared source views")
        self._view_keys = {
            name: hashlib.sha256(
                json.dumps(
                    {
                        table: {**asdict(rule), "partitions": sorted(set(rule.partitions))}
                        for table, rule in rules.items()
                        if rule.before_us is not None or rule.partition_column is not None
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for name, rules in {"full": {}, **self.views}.items()
        }
        self._view_sources: dict[tuple[str, str], TableSource] = {}
        self.target_level: Literal["entity", "sequence"] = target_level
        self.original, self.paths = source, dict(history_paths)
        self.exclusive_terminal_paths = exclusive_terminal_paths
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.directory = Path(
                self.resources.enter_context(
                    TemporaryDirectory(prefix="source-native-", dir=work_dir)
                )
            )
            self.normalized = self.resources.enter_context(
                NativeSourceFiles(source, self.directory / "tables")
            )
            self.normalized.canonical_graph_order = (
                self.loader.provenance.get("graph_order") == "canonical_key_asc_uint32_v1"
            )
            self.source = self.normalized.source
            if exclusive_terminal_paths:
                if set(history_paths.values()) != {("events",), ("window_events",)}:
                    raise UnsupportedCombinationError(
                        "Exclusive native histories currently require the fixed owner/window projection"
                    )
                if (
                    self.normalized.connection.execute(
                        "SELECT 1 FROM links a JOIN links b ON a.child=b.child AND a.parent=b.parent WHERE a.role='events' AND b.role='window_events' LIMIT 1"
                    ).fetchone()
                    is not None
                ):
                    raise SchemaError("Native owner/window history paths overlap for one root")
            self.api = native_api()
            if anchors is None and anchor_factory is not None:
                anchors = anchor_factory(
                    self.normalized, self.directory / "generated-anchors.parquet"
                )
            if anchors is None:
                if task is None:
                    raise ValueError("Native inputs require anchors or a relational task")
                from tfm.data.native_anchors import task_anchors

                anchors = task_anchors(
                    self.normalized, task, self.directory / "generated-anchors.parquet"
                )
            self.original_anchors = anchors
            self.binding = self._anchors(anchors)
            self.anchors_sha256 = file_sha256(self.directory / "anchors.parquet")
            self._history_options: dict[str, Any] = {
                "root": root,
                "fixed_vocabularies": fixed_vocabularies,
                "exclusive_terminal_paths": exclusive_terminal_paths,
                "paths": dict(history_paths),
                "max_events": max_events,
                "batch_size": batch_size,
                "observation_seconds": observation_seconds,
                "amount_key": amount_key,
                "categorical_keys": frozenset(
                    () if features is None else features.categorical_keys
                ),
                "excluded_keys": frozenset(() if features is None else features.exclude_keys),
            }
            self._tasks: dict[str, Any] = {}
            self._indexes: dict[str, Any] = {}
            self._fitted_directory: Path | None = None
            self._expected_tasks: dict[str, dict[str, str]] = {}
            self._num_threads = num_threads
            self._history_state: dict[str, Any] | None = None
            self._history_source: TableSource | None = None
        except BaseException:
            self.resources.close()
            raise

    def _view_for_split(self, split: str) -> str:
        selected = (self.fit_split,) if isinstance(self.fit_split, str) else self.fit_split
        default = split if split in self.views else self.fit_view if split in selected else "full"
        return self.split_views.get(split, default)

    def _anchors(self, path: Path) -> Any:
        table = next((t for t in self.original.tables if t.name == self.root), None)
        if table is None:
            raise SchemaError(f"Native root {self.root!r} is undeclared")
        output = self.directory / "anchors.parquet"
        raw_output = self.directory / "replay-anchors.parquet"
        raw_schema = pq.read_schema(path)
        for name in ("view", "view_identity"):
            index = raw_schema.get_field_index(name)
            field = pa.field(name, pa.string())
            raw_schema = raw_schema.append(field) if index == -1 else raw_schema.set(index, field)
        writer = raw_writer = None
        connection = self.normalized.connection
        connection.execute(
            "CREATE TABLE anchors (sample_id TEXT PRIMARY KEY, root INTEGER, cutoff INTEGER, split TEXT, view TEXT, view_identity TEXT)"
        )
        connection.execute("CREATE TABLE anchor_labels(sample_id TEXT PRIMARY KEY,target REAL)")
        schema = pa.schema(
            [
                ("sample_id", pa.string()),
                ("root", pa.int64()),
                ("cutoff", pa.timestamp("s", tz="UTC")),
                ("split", pa.string()),
                ("entity_id", pa.string()),
                ("view", pa.string()),
                ("view_identity", pa.string()),
                ("target", pa.float32()),
                ("label_available_us", pa.int64()),
                ("label_cutoff_us", pa.int64()),
            ]
        )
        try:
            writer = pq.ParquetWriter(output, schema)
            raw_writer = pq.ParquetWriter(raw_output, raw_schema)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                rows, raw_rows = [], []
                for row in batch.to_pylist():
                    key = tuple(row[c] for c in table.key)
                    root = self.normalized.code(self.root, key)
                    cutoff = row.get("cutoff")
                    offset = cutoff.utcoffset() if isinstance(cutoff, datetime) else None
                    if (
                        not isinstance(cutoff, datetime)
                        or cutoff.tzinfo is None
                        or offset is None
                        or offset.total_seconds() != 0
                        or cutoff.microsecond
                    ):
                        raise UnsupportedCombinationError(
                            "anchors.cutoff: native anchors require whole-second UTC timestamps"
                        )
                    if (
                        self.original.observed_through is not None
                        and cutoff > self.original.observed_through
                    ):
                        raise SchemaError("Native anchor cutoff exceeds declared source coverage")
                    if row.get("observation_start") is not None:
                        raise UnsupportedCombinationError(
                            "anchors.observation_start: per-anchor history lower bounds require a supported native projection"
                        )
                    sid, split = row.get("sample_id"), row.get("split")
                    if (
                        not isinstance(sid, str)
                        or not sid.strip()
                        or not isinstance(split, str)
                        or not split.strip()
                    ):
                        raise SchemaError("Native anchors require nonempty sample_id and split")
                    view = self._view_for_split(split)
                    view_identity = self._view_keys[view]
                    if (row.get("view") is not None and row["view"] != view) or (
                        row.get("view_identity") is not None
                        and row["view_identity"] != view_identity
                    ):
                        raise SchemaError(
                            "Replay anchor view differs from the configured split view"
                        )
                    seconds = int((cutoff - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds())
                    known_at = connection.execute(
                        "SELECT available_us FROM identities WHERE table_name=? AND code=?",
                        (self.root, root),
                    ).fetchone()[0]
                    if known_at is not None and known_at > seconds * 1_000_000:
                        raise SchemaError(f"Native anchor {sid}: root is not observable at cutoff")
                    # SQLite uniqueness keeps validation bounded for large anchor
                    # populations, including replayed root/cutoff identities.

                    try:
                        connection.execute(
                            "INSERT INTO anchors VALUES (?,?,?,?,?,?)",
                            (sid, root, seconds, split, view, view_identity),
                        )
                    except sqlite3.IntegrityError as error:
                        raise SchemaError(
                            "Native anchors require unique sample/root/cutoff/split bindings"
                        ) from error
                    target = row.get("target")
                    known, label_cutoff = row.get("label_available_us"), row.get("label_cutoff_us")
                    if target is not None:
                        if (
                            target not in (0, 1)
                            or known is None
                            or label_cutoff is None
                            or int(known) > int(label_cutoff)
                        ):
                            raise SchemaError(
                                f"Native anchor {sid}: entity target unavailable at fitting cutoff"
                            )
                    elif known is not None or label_cutoff is not None:
                        raise SchemaError(
                            f"Native anchor {sid}: label availability without a target"
                        )
                    raw_rows.append({**row, "view": view, "view_identity": view_identity})
                    connection.execute("INSERT INTO anchor_labels VALUES (?,?)", (sid, target))
                    rows.append(
                        {
                            "sample_id": sid,
                            "root": root,
                            "cutoff": cutoff,
                            "split": split,
                            "entity_id": source_entity_id(key),
                            "view": view,
                            "view_identity": view_identity,
                            "target": target,
                            "label_available_us": known,
                            "label_cutoff_us": label_cutoff,
                        }
                    )
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                raw_writer.write_table(pa.Table.from_pylist(raw_rows, schema=raw_schema))
            connection.commit()
        finally:
            if writer is not None:
                writer.close()
            if raw_writer is not None:
                raw_writer.close()
        if not connection.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]:
            raise SchemaError("Native anchors cannot be empty")
        connection.execute("CREATE TABLE split_families(name TEXT PRIMARY KEY,family TEXT)")
        connection.executemany(
            "INSERT INTO split_families VALUES (?,?)", self.split_families.items()
        )
        connection.execute("CREATE TABLE split_namespaces(name TEXT PRIMARY KEY,namespace TEXT)")
        connection.executemany(
            "INSERT INTO split_namespaces VALUES (?,?)", self.split_namespaces.items()
        )
        connection.commit()
        if (
            connection.execute(
                "SELECT root,cutoff FROM anchors a LEFT JOIN split_families f ON f.name=a.split LEFT JOIN split_namespaces n ON n.name=a.split GROUP BY root,cutoff,COALESCE(n.namespace,'default'),a.view_identity HAVING COUNT(DISTINCT COALESCE(f.family,a.split))>1 LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise SchemaError("Native root/cutoff cannot occur in multiple splits within one view")
        self.original_anchors = raw_output
        return self.api.AnchorTable(
            self.api.Parquet(str(output)),
            entity_id="root",
            as_of_time="cutoff",
            sample_id="sample_id",
            split="split",
            metadata=(
                "entity_id",
                "target",
                "label_available_us",
                "label_cutoff_us",
                "view",
                "view_identity",
            ),
        )

    def task(self, mode: Mode, *, view: str | None = None) -> Any:
        """Fit a mode once from train anchors; retain only native task/index state."""
        if self._closed:
            raise ValueError("Native inputs are closed")
        view = self.fit_view if view is None else view
        if view != "full" and view not in self.views:
            raise ValueError(f"Unknown native source view {view!r}")
        base_name = "history" if mode in {"sequence", "history", "flat"} else mode
        name = base_name if view == self.fit_view else f"{base_name}@{view}"
        if name in self._tasks:
            return self._tasks[name]
        source, contract = self.source, None
        if base_name != "graph":
            if self._history_source is None:
                from tfm.data.native_projection import history_projection

                self._history_source, paths = history_projection(
                    self.normalized,
                    root=self.root,
                    paths={
                        **self.paths,
                        **{
                            f"__profile_{key}": value
                            for key, value in self.source.profile_paths.items()
                        },
                    },
                    directory=self.directory / "projections",
                    exclusive_terminal_paths=self.exclusive_terminal_paths,
                )
                # Aliases to the same physical terminal now resolve through
                # one union projection. Both paths contribute associations,
                # but each root/event enters the encoded stream only once.
                compiled_paths, seen_paths = {}, set()
                for key in self.paths:
                    if paths[key] not in seen_paths:
                        compiled_paths[key] = paths[key]
                        seen_paths.add(paths[key])
                self._history_options["paths"] = compiled_paths
                if not self.exclusive_terminal_paths:
                    relations = {role.name: role for role in self.source.relationships}
                    event_names = {}
                    for key in compiled_paths:
                        terminal = self.root
                        for role_name in self.paths[key]:
                            role = relations[role_name]
                            terminal = role.child if terminal == role.parent else role.parent
                        event_names[key] = terminal
                    self._history_options["event_names"] = event_names
                self._history_source = replace(
                    self._history_source,
                    profile_paths={
                        key: paths[f"__profile_{key}"] for key in self.source.profile_paths
                    },
                )
            source = self._history_source
            contract = history_contract(
                source,
                **self._history_options,
                equivalent_clocks=frozenset(
                    name for name, same in self.normalized.same_clocks.items() if same
                ),
            )
        if base_name in {"graph", "hybrid"}:
            contract = with_graph(
                contract,
                self.source,
                root=self.root,
                depth=self.depth,
                fanout=self.fanout,
                forward_degrees=self.normalized.forward_degrees,
            )
        assert contract is not None
        if view != "full":
            rules = dict(self.views[view])
            if base_name != "graph" and self._history_source is not None:
                relations = {role.name: role for role in source.relationships}
                original_roles = {role.name: role for role in self.source.relationships}
                projected = {
                    table.name for table in source.tables if table.name.startswith("__tfm_history_")
                }
                handled = set()
                for name, path in self._history_options["paths"].items():
                    terminal = self.root
                    for role_name in path:
                        role = relations[role_name]
                        terminal = role.child if terminal == role.parent else role.parent
                    if not terminal.startswith("__tfm_history_"):
                        continue
                    handled.add(terminal)
                    original_path = self.paths[name]
                    if len(original_path) != 1:
                        raise UnsupportedCombinationError(
                            "Physical source prefixes with projected multi-hop histories need an explicit projected prefix"
                        )
                    role = original_roles[original_path[0]]
                    original = role.child if role.parent == self.root else role.parent
                    if original in rules:
                        rules[terminal] = rules[original]
                if projected - handled:
                    raise UnsupportedCombinationError(
                        "Physical source prefixes with projected profile facts need an explicit projected prefix"
                    )
            layout = "source" if base_name == "graph" else "history"
            key = (view, layout)
            if key not in self._view_sources:
                self._view_sources[key] = source_view(
                    source, rules, self.directory / f"view-{view}-{layout}"
                )
            source = self._view_sources[key]
        dataset = self.api.Dataset(
            {
                table.name: self.api.Parquet(str(source.files[table.name]))
                for table in source.tables
                if table.role == "input"
                and table.name in {entry.name for entry in contract.schema.tables}
            }
        )
        fitted = None if view == self.fit_view else self.task(mode, view=self.fit_view).fitted_state
        expected = self._expected_tasks.get(name)
        if self._fitted_directory is not None and expected is not None:
            try:
                artifact = self.api.Artifact.load(self._fitted_directory / name)
            except (FileNotFoundError, ValueError):
                # Source/anchor identity was checked against the checkpoint.
                # A refit may repair a corrupt sidecar only if its final fitted
                # fingerprint exactly matches the metadata bound to the weights.
                artifact = None
            if artifact is not None:
                if (
                    artifact.contract.fingerprint != expected["contract"]
                    or artifact.fitted_state.fingerprint != expected["fitted_state"]
                ):
                    raise ValueError("Native fitted artifact differs from checkpoint identity")
                fitted = artifact.fitted_state
        if expected is not None and contract.fingerprint != expected["contract"]:
            raise ValueError("Native input contract differs from checkpoint identity")
        schema_key = json.dumps([view, contract.to_dict()["schema"]], sort_keys=True)
        snapshot = self._indexes.get(schema_key)
        if snapshot is None:
            factory = self.loader.engine_factory
            assert factory is not None
            engine = factory(self.resources, work_dir=self.directory, num_threads=self._num_threads)
        else:
            engine = self.api.DiskGraphEngine.from_snapshot(snapshot, num_threads=self._num_threads)
            dataset = None
        if any(
            not callable(getattr(engine, method, None))
            for method in ("open", "materialize_anchors")
        ):
            raise UnsupportedCombinationError(
                f"data.loader {self.loader_name!r}: engine requires open() and source AnchorTable materialize_anchors()"
            )
        plan = self.resources.enter_context(
            self.api.prepare(contract, dataset, anchor_fitted_engine(engine))
        )
        if snapshot is None and self.loader_name == "kumo_pytorch":
            self._indexes[schema_key] = SimpleNamespace(index_path=engine.index, id=None)
        task = self.resources.enter_context(
            plan.materialize(
                self.binding, fit_split=self.fit_split if fitted is None else None, fitted=fitted
            )
        )
        if expected is not None and task.fitted_state.fingerprint != expected["fitted_state"]:
            raise ValueError("Native fitted encodings differ from checkpoint identity")
        if base_name != "graph":
            state = task.fitted_state.to_dict()
            history = {
                "encoders": {
                    key: value
                    for key, value in state["encoders"].items()
                    if not key.startswith('["inputs","graph"')
                },
                "spaces": state["spaces"],
                "field_names": state["field_names"],
                "inputs": {key: value for key, value in state["inputs"].items() if key != "graph"},
            }
            if self._history_state is not None and self._history_state != history:
                raise SchemaError(
                    "Native history-control encoding differs from hybrid fitted inputs"
                )
            self._history_state = history
        self._tasks[name] = task
        return task

    def batches(
        self,
        split: str | tuple[str, ...],
        *,
        mode: Mode,
        batch_size: int | None = None,
        max_tokens: int = 128,
        seed: int = 0,
        shuffle: bool = False,
        require_labels: bool = False,
        sharing: Literal["shared", "disjoint"] = "disjoint",
        hop_trim: bool = True,
        decorate: Callable[[Any], Any] | None = None,
        view: str | None = None,
        history_limit: int | None = None,
    ) -> NativeBatches[Any]:
        """One native seeded iterator per epoch, with one owned CPU lookahead."""
        if sharing == "shared" and mode in {"graph", "hybrid"}:
            selected = (split,) if isinstance(split, str) else split
            cutoffs = max(
                (
                    self.normalized.connection.execute(
                        "SELECT COUNT(DISTINCT cutoff) FROM anchors WHERE split=?", (side,)
                    ).fetchone()[0]
                    for side in selected
                ),
                default=0,
            )
            if cutoffs > 1:
                raise UnsupportedCombinationError(
                    "Shared native graph stages require one cutoff per split"
                )
        selected = (split,) if isinstance(split, str) else split
        assigned = {self._view_for_split(side) for side in selected}
        if len(assigned) != 1 or (view is not None and view not in assigned):
            raise SchemaError("Native execution view differs from the stored split view")
        view = next(iter(assigned))
        task = self.task(mode, view=view)
        adapter = NativeTensorAdapter(
            self.source,
            root=self.root,
            mode=mode,
            depth=self.depth,
            max_tokens=max_tokens,
            require_labels=require_labels,
            sharing=sharing,
            hop_trim=hop_trim,
            target_level=self.target_level,
            history_limit=history_limit,
        )
        return self.resources.enter_context(
            NativeBatches(
                task,
                adapter if decorate is None else lambda value: decorate(adapter(value)),
                split=split,
                batch_size=batch_size or self.batch_size,
                seed=seed,
                shuffle=shuffle,
            )
        )

    def manifest(self) -> dict[str, Any]:
        """Content and fitted-encoding identity, independent of temporary file paths."""
        return {
            "name": self.loader_name,
            **(
                {"engine_provider": dict(self.loader.provenance)}
                if self.loader_name != "kumo_pytorch"
                else {}
            ),
            "target_builder": "python_reference",
            "revision": SOURCE_REVISION,
            "pipeline": "source_tables_native_tensors_v1",
            "source_sha256": self.normalized.source_identity,
            "source_schema": self.original.manifest(),
            "fit_split": self.fit_split,
            "fit_view": self.fit_view,
            "split_families": self.split_families,
            "split_namespaces": self.split_namespaces,
            "split_views": self.split_views,
            "exclusive_terminal_paths": self.exclusive_terminal_paths,
            "history_tie_policy": (
                "prepared_window_order_v1"
                if self.exclusive_terminal_paths
                else "event_time_then_typed_event_identity_v1"
            ),
            "views": {
                name: {table: asdict(rule) for table, rule in rules.items()}
                for name, rules in self.views.items()
            },
            "target_level": self.target_level,
            "anchors_sha256": self.anchors_sha256,
            "tasks": {
                mode: {
                    "contract": task.contract.fingerprint,
                    "fitted_state": task.fitted_state.fingerprint,
                }
                for mode, task in self._tasks.items()
            },
            "graph_tie_policy": (
                "canonical_key_asc"
                if self.normalized.canonical_graph_order
                else "native index-local recent ties"
            ),
        }

    def restore_fitted(self, directory: Path, expected: Mapping[str, Mapping[str, str]]) -> None:
        """Reuse native encoders only after verifying source/anchor checkpoint identity."""
        if self._tasks:
            raise ValueError("Restore fitted state before preparing native tasks")
        self._fitted_directory = directory
        self._expected_tasks = {name: dict(identity) for name, identity in expected.items()}

    def save_anchors(self, path: Path) -> None:
        """Persist raw-key replay anchors, independently of temporary native codes."""
        if self._closed:
            raise ValueError("Native inputs are closed")
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".anchors-", dir=path.parent) as folder:
            staged = Path(folder) / "anchors.parquet"
            shutil.copyfile(self.original_anchors, staged)
            staged.replace(path)

    def save_fitted(self, directory: Path) -> None:
        """Persist each mode's native contract/encoders and verified identity."""
        if self._closed:
            raise ValueError("Native inputs are closed")
        for name, task in self._tasks.items():
            task.to_artifact().save(directory / name)

    def close(self) -> None:
        self._closed = True
        self.resources.close()

    def __exit__(self, *args: Any) -> None:
        self.close()
