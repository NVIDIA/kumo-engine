# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate FutureSubgraph reference targets, cached on disk by native anchor.

Historical graph retrieval stays native. Target construction uses window-end
membership, canonical FK fanout, and uncapped terminal sets in SQLite; it never
retains a corpus of sampled input graphs or exposes future rows to the model.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

import torch

from tfm.contracts.graph import STATIC_TIME, FutureSubgraphTargets, GraphAnchor, utc_us
from tfm.data.graph import STATISTICS, FutureTargetTransform
from tfm.data.native_source import NativeSourceFiles, source_entity_id


class NativeFutureTargets(AbstractContextManager["NativeFutureTargets"]):
    """Build target vectors before training; producer threads read owned copies."""

    def __init__(
        self,
        source: NativeSourceFiles,
        *,
        root: str,
        depth: int,
        fanout: int,
        horizon_us: int,
        observed_through_us: int,
        partition_end_us: int,
        label_delay_us: int = 0,
    ) -> None:
        if depth < 1 or fanout < 1 or horizon_us <= 0 or label_delay_us < 0:
            raise ValueError(
                "Future target depth/fanout/horizon must be positive; delay nonnegative"
            )
        self.source, self.root, self.depth, self.fanout = source, root, depth, fanout
        self.horizon_us, self.observed_us, self.partition_us, self.delay = (
            horizon_us,
            observed_through_us,
            partition_end_us,
            label_delay_us,
        )
        declaration = source.source
        self.tables = {
            table.name: table
            for table in declaration.tables
            if table.role == "input"
            and (declaration.graph_numeric is None or table.name in declaration.graph_numeric)
        }
        self.roles = {
            role.name: role
            for role in declaration.relationships
            if role.parent in self.tables and role.child in self.tables
        }
        self.directed = tuple(
            (
                role,
                forward,
                role.parent if forward else role.child,
                role.child if forward else role.parent,
                role.name if forward else f"rev_{role.name}",
            )
            for role in self.roles.values()
            for forward in (True, False)
        )
        self.groups = tuple(
            (hop, item)
            for hop in range(1, depth + 1)
            for item in self.directed
            if not self.tables[item[3]].immutable
        )
        self.targets = {
            name: tuple((f"raw:{alias}", alias) for alias in table.numeric)
            if declaration.future_numeric is None
            else tuple(declaration.future_numeric.get(name, {}).items())
            for name, table in self.tables.items()
        }
        self.columns = tuple(
            f"{hop}/{parent}/{name}/{child}/{field}"
            for hop, (_, _, parent, child, name) in self.groups
            for field in (
                "count",
                *(f"{column}/{stat}" for column, _ in self.targets[child] for stat in STATISTICS),
            )
        )
        self.db = source.connection
        self.db.execute(
            "CREATE TABLE future_cache (sample_id TEXT PRIMARY KEY,values_blob BLOB,valid_blob BLOB,available INTEGER)"
        )
        self.db.execute(
            "CREATE TEMP TABLE future_terminal (code INTEGER PRIMARY KEY,available INTEGER)"
        )
        self._readers: list[sqlite3.Connection] = []
        self._lock = RLock()
        self._closed = False

    def _membership(self, code: int, end: int) -> dict[tuple[str, int], int]:
        root = self.db.execute(
            "SELECT available_us FROM identities WHERE table_name=? AND code=?", (self.root, code)
        ).fetchone()
        if root is None or root[0] is not None and root[0] > end:
            raise ValueError("Future root is missing or unavailable at the window end")
        distances = {(self.root, code): 0}
        queue = deque(distances)
        while queue:
            table, parent = queue.popleft()
            distance = distances[table, parent]
            if distance >= self.depth - 1:
                continue
            for role, forward, start, stop, _ in self.directed:
                if start != table:
                    continue
                source, target = ("parent", "child") if forward else ("child", "parent")
                rows = self.db.execute(
                    f"SELECT l.{target} FROM links l JOIN identities n ON n.table_name=? AND n.code=l.{target} WHERE l.role=? AND l.{source}=? AND COALESCE(l.available_us,{STATIC_TIME})<=? AND COALESCE(n.available_us,{STATIC_TIME})<=? ORDER BY COALESCE(n.event_us,{STATIC_TIME}) DESC,n.source_key ASC LIMIT ?",
                    (stop, role.name, parent, end, end, self.fanout),
                )
                for (child,) in rows:
                    key = stop, child
                    if key not in distances:
                        distances[key] = distance + 1
                        queue.append(key)
        return distances

    def _raw(self, code: int, cutoff: int) -> tuple[list[float], list[bool], int]:
        end = cutoff + self.horizon_us
        if end > self.observed_us:
            raise ValueError("Incomplete future window beyond declared source coverage")
        distances = self._membership(code, end)
        values, valid = [], []
        available = end + self.delay
        for hop, (role, forward, parent, child, _) in self.groups:
            self.db.execute("DELETE FROM future_terminal")
            source, target = ("parent", "child") if forward else ("child", "parent")
            for (table, node), distance in distances.items():
                if table != parent or distance != hop - 1:
                    continue
                self.db.execute(
                    f"INSERT INTO future_terminal SELECT l.{target},MAX(COALESCE(n.available_us,{STATIC_TIME}),COALESCE(l.available_us,{STATIC_TIME})) FROM links l JOIN identities n ON n.table_name=? AND n.code=l.{target} WHERE l.role=? AND l.{source}=? AND COALESCE(l.available_us,{STATIC_TIME})<=? AND n.event_us>? AND n.event_us<=? ON CONFLICT(code) DO UPDATE SET available=MAX(available,excluded.available)",
                    (child, role.name, node, end, cutoff, end),
                )
            count, known = self.db.execute(
                "SELECT COUNT(*),MAX(available) FROM future_terminal"
            ).fetchone()
            values.append(float(count))
            valid.append(True)
            if known is not None:
                available = max(available, known + self.delay)
            for _, field in self.targets[child]:
                query = "SELECT v.value FROM future_terminal t JOIN numbers v ON v.table_name=? AND v.code=t.code AND v.field=? WHERE v.value IS NOT NULL ORDER BY t.code"
                parameters = child, field
                n = self.db.execute("SELECT COUNT(*) FROM (" + query + ")", parameters).fetchone()[
                    0
                ]
                total = math.fsum(row[0] for row in self.db.execute(query, parameters))
                mean = total / n if n else 0.0
                low, high = self.db.execute(
                    "SELECT MIN(value),MAX(value) FROM (" + query + ")", parameters
                ).fetchone()
                variance = (
                    math.fsum((row[0] - mean) ** 2 for row in self.db.execute(query, parameters))
                    / (n - 1)
                    if n > 1
                    else 0.0
                )
                values.extend(
                    (
                        float(n),
                        total,
                        mean,
                        0.0 if low is None else low,
                        0.0 if high is None else high,
                        math.sqrt(variance),
                    )
                )
                valid.extend((True, not count or n > 0, n > 0, n > 0, n > 0, n > 1))
        if available > self.observed_us:
            raise ValueError("Target availability exceeds declared source coverage")
        return values, valid, available

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self, splits: tuple[str, ...] = ("train", "validation")) -> None:
        """Materialize only target vectors, one anchor at a time, before consumers."""
        for split in splits:
            cursor = self.db.execute(
                "SELECT sample_id,root,cutoff FROM anchors WHERE split=? ORDER BY rowid", (split,)
            )
            for sid, code, cutoff in cursor:
                values, valid, available = self._raw(code, cutoff * 1_000_000)
                if available > self.partition_us:
                    raise ValueError("Future target unavailable inside declared partition")
                tensor = torch.tensor(values, dtype=torch.float32)
                mask = torch.tensor(valid, dtype=torch.bool)
                if not bool(torch.isfinite(tensor[mask]).all()):
                    raise ValueError("Future target statistics exceed finite float32 range")
                self.db.execute(
                    "INSERT INTO future_cache VALUES (?,?,?,?)",
                    (sid, tensor.numpy().tobytes(), mask.numpy().tobytes(), available),
                )
        self.db.commit()

    def _reader(self) -> sqlite3.Connection:
        # raw() serializes this small cache read. A bounded shared reader avoids
        # retaining a connection for every short-lived epoch producer thread.
        if self._closed:
            raise ValueError("Future target cache is closed")
        if not self._readers:
            self._readers.append(
                sqlite3.connect(
                    f"{(self.source.directory / 'keys.sqlite').resolve().as_uri()}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                )
            )
        return self._readers[0]

    @torch.inference_mode(False)
    @torch.no_grad()
    def raw(self, anchors: tuple[GraphAnchor, ...]) -> FutureSubgraphTargets:
        """Read independent CPU tensors, preserving target mutation guards."""
        with self._lock:
            return self._read(anchors)

    def _read(self, anchors: tuple[GraphAnchor, ...]) -> FutureSubgraphTargets:
        db = self._reader()
        values, masks, availability = [], [], []
        for anchor in anchors:
            row = db.execute(
                "SELECT c.values_blob,c.valid_blob,c.available,a.cutoff,i.source_key FROM future_cache c JOIN anchors a USING(sample_id) JOIN identities i ON i.table_name=? AND i.code=a.root WHERE sample_id=?",
                (self.root, anchor.sample_id),
            ).fetchone()
            if (
                row is None
                or row[3] * 1_000_000 != utc_us(anchor.cutoff)
                or source_entity_id(tuple(json.loads(row[4]))) != anchor.entity_id
            ):
                raise ValueError("Future target cache is missing the exact requested anchor")
            # Copy read-only persisted bytes before exposing a tensor. PyTorch
            # never aliases cached buffers or another emitted minibatch.
            values.append(torch.frombuffer(bytearray(row[0]), dtype=torch.float32).clone())
            masks.append(torch.frombuffer(bytearray(row[1]), dtype=torch.bool).clone())
            availability.append(row[2])
        return FutureSubgraphTargets(
            anchors,
            torch.stack(values),
            torch.stack(masks),
            self.columns,
            self.horizon_us,
            self.delay,
            torch.tensor(availability, dtype=torch.long),
            self.partition_us,
        )

    def _chunks(self, split: str | tuple[str, ...]) -> Iterator[FutureSubgraphTargets]:
        splits = (split,) if isinstance(split, str) else split
        placeholders = ",".join("?" for _ in splits)
        cursor = self.db.execute(
            f"SELECT a.sample_id,i.source_key,a.cutoff FROM anchors a JOIN future_cache USING(sample_id) JOIN identities i ON i.table_name=? AND i.code=a.root WHERE a.split IN ({placeholders}) ORDER BY a.rowid",
            (self.root, *splits),
        )
        while rows := cursor.fetchmany(128):
            anchors = tuple(
                GraphAnchor(
                    sid,
                    source_entity_id(tuple(json.loads(code))),
                    datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=cutoff),
                )
                for sid, code, cutoff in rows
            )
            yield self.raw(anchors)

    @torch.inference_mode(False)
    @torch.no_grad()
    def fit(self, split: str | tuple[str, ...] = "train") -> FutureTargetTransform:
        """Two bounded passes fit train-only signed-log means and scales."""
        total = torch.zeros(len(self.columns), dtype=torch.float64)
        count = torch.zeros_like(total)
        for batch in self._chunks(split):
            batch.assert_validated()
            x = torch.where(batch.valid, batch.values.double(), 0.0)
            total += (x.sign() * x.abs().log1p()).sum(0)
            count += batch.valid.sum(0)
        mean = total / count.clamp_min(1)
        variance = torch.zeros_like(mean)
        for batch in self._chunks(split):
            batch.assert_validated()
            x = batch.values.double()
            x = x.sign() * x.abs().log1p()
            variance += torch.where(batch.valid, (x - mean).square(), 0.0).sum(0)
        variance /= count.clamp_min(1)
        selected = ((count >= 2) & (variance.sqrt() > 1e-6)).nonzero().flatten()
        if not selected.numel():
            raise ValueError("FutureSubgraph has no nonconstant training targets")
        return FutureTargetTransform(
            self.columns,
            tuple(selected.tolist()),
            mean[selected].float(),
            variance[selected].sqrt().float(),
        )

    def close(self) -> None:
        """Call after batch producers stop; close readers before source removal."""
        with self._lock:
            self._closed = True
            for reader in self._readers:
                reader.close()
            self._readers.clear()

    def __exit__(self, *args: Any) -> None:
        self.close()
