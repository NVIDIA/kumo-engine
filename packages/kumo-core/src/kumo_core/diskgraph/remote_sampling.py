# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume complete DiskGraph task batches from a Flight gateway.

Workers own epoch shards and push complete batches over credit-controlled Flight
exchanges. Task extensions are transferred once before opening stage streams.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
from diskgraph.runtime_batch_iterator import RuntimeIndex
from pyarrow import flight

from kumo_core.diskgraph.sampling_protocol import (
    ENVELOPE_SCHEMA,
    PUSH_VERSION,
    PUSH_WINDOW,
    decode_frame,
    encode_frame,
)
from kumo_core.diskgraph.sampling_task import (
    task_files,
    upload_task,
)

logger = logging.getLogger(__name__)

_NATIVE_ARGUMENTS = (
    "stage_table",
    "sampling_plan_json",
    "num_workers",
    "worker_id",
    "shuffle",
    "rnd_seed",
    "epoch",
    "batch_size",
    "drop_last",
    "fused",
    "swap_edge_direction",
    "neg_rhs_node_type",
    "neg_amount_m",
    "neg_rnd_seed",
    "replace",
    "disjoint",
    "adaptive",
    "entity_node_type",
    "target_node_type",
    "rhs_seed_node_type",
    "rhs_seed_time",
    "group_by_timestamp",
    "weighted_seed_sampling_json",
    "selected_stage_node_ids",
    "use_seed_timestamps",
    "pseudo_seed_time",
    "disable_pseudo_time",
    "projections",
)


def _decode(value: Any, components: list[bytes]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__arrow__", "array"} and type(value["array"]) is bool:
            slot = value["__arrow__"]
            if type(slot) is not int or not 0 <= slot < len(components):
                raise ValueError("Invalid Arrow component reference")
            with pa.ipc.open_stream(components[slot]) as reader:
                batches = list(reader)
            if len(batches) != 1:
                raise ValueError("Expected one record batch per Arrow component")
            batch = batches[0]
            if value["array"]:
                if batch.num_columns != 1:
                    raise ValueError("Array component must contain one column")
                return batch.column(0)
            return batch
        return {key: _decode(item, components) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, components) for item in value]
    return value


class RemoteTaskIterator(Iterator[dict[str, Any]]):
    """Own one Flight stream and reject incomplete epochs.

    Args:
        endpoint: Flight URI of the gateway serving the immutable graph.
        request: Versioned native iterator request.

    Raises:
        ValueError: If framing, version, or completion metadata is invalid.
        TypeError: If frame metadata is not an object.
        flight.FlightError: If the gateway rejects or interrupts the request.
    """

    def __init__(
        self,
        endpoint: str,
        request: dict[str, Any],
        *,
        client: flight.FlightClient | None = None,
    ) -> None:
        """Open a stream and validate its initial metadata."""
        self._owns_client = client is None
        self._client = client or flight.FlightClient(
            endpoint,
            generic_options=[("grpc.max_receive_message_length", 128 * 1024**2)],
        )
        self._reader: flight.FlightStreamReader | None = None
        self._writer: flight.FlightStreamWriter | None = None
        self._expected_batches: int | None = None
        self._closed = False
        self._batches = 0
        self._owner: RemoteRuntimeIndex | None = None
        try:
            payload = orjson.dumps(request, option=orjson.OPT_SERIALIZE_NUMPY)
            self._writer, self._reader = self._client.do_exchange(
                flight.FlightDescriptor.for_command(payload),
                options=flight.FlightCallOptions(timeout=3600),
            )
            self._writer.begin(ENVELOPE_SCHEMA)
            metadata, components = self._read_frame()
            if (
                metadata.get("kind") != "iterator"
                or metadata.get("version") != PUSH_VERSION
                or components
                or type(metadata.get("num_samples")) is not int
                or metadata["num_samples"] < 0
            ):
                raise ValueError("Invalid task iterator metadata")
            self._num_samples: int = metadata["num_samples"]
            if self._writer is not None:
                count = metadata.get("num_batches")
                if type(count) is not int or count < 0:
                    raise ValueError("Invalid push iterator batch count")
                self._expected_batches = count
                if count:
                    self._writer.write_batch(
                        encode_frame(
                            {"kind": "credit", "batches": min(PUSH_WINDOW, count)}, []
                        )
                    )
        except BaseException:
            self.close()
            raise

    def _read_frame(self) -> tuple[dict[str, Any], list[bytes]]:
        assert self._reader is not None
        try:
            chunk = self._reader.read_chunk()
        except StopIteration as error:
            raise ValueError("Task stream ended without a completion frame") from error
        return decode_frame(chunk.data)

    def num_samples(self) -> int:
        """Return the server's sample count for this stage and shard."""
        return self._num_samples

    def __next__(self) -> dict[str, Any]:
        """Return a complete batch, stopping only after verified completion."""
        if self._closed:
            raise StopIteration
        try:
            if self._writer is not None:
                assert self._expected_batches is not None
                if (
                    self._batches
                    and self._batches + PUSH_WINDOW <= self._expected_batches
                ):
                    self._writer.write_batch(
                        encode_frame({"kind": "credit", "batches": 1}, [])
                    )
            metadata, components = self._read_frame()
            if metadata.get("kind") == "end":
                if (
                    type(metadata.get("batches")) is not int
                    or metadata.get("batches") != self._batches
                    or components
                    or (
                        self._expected_batches is not None
                        and self._batches != self._expected_batches
                    )
                ):
                    raise ValueError("Invalid task stream completion count")
                if self._writer is not None:
                    self._writer.done_writing()
                # Observe the final RPC status as well as the logical end frame.
                assert self._reader is not None
                try:
                    self._reader.read_chunk()
                except StopIteration:
                    self.close()
                    raise
                raise ValueError("Unexpected data after task stream completion")
            if (
                metadata.get("kind") != "batch"
                or type(metadata.get("batch_id")) is not int
                or metadata.get("batch_id") != self._batches
                or (
                    self._expected_batches is not None
                    and self._batches >= self._expected_batches
                )
            ):
                raise ValueError("Invalid task batch sequence")
            value = _decode(metadata.get("value"), components)
            if not isinstance(value, dict) or set(value) != {
                "task",
                "graphs",
                "raw_features",
            }:
                raise ValueError("Invalid task batch contents")
            self._batches += 1
            return value
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> RemoteTaskIterator:
        """Return this stream iterator."""
        return self

    def close(self) -> None:
        """Cancel outstanding sampling and release the client; safe to repeat."""
        if self._closed:
            return
        self._closed = True
        if self._owner is not None:
            self._owner._streams.discard(self)
            self._owner = None
        try:
            if self._reader is not None:
                self._reader.cancel()
        finally:
            if self._writer is not None:
                try:
                    self._writer.close()
                except (pa.ArrowException, flight.FlightError, OSError) as error:
                    logger.debug("Closed push sampling stream: %s", error)
            if self._owns_client:
                self._client.close()


class RemotePushIterator(Iterator[dict[str, Any]]):
    """Merge independently produced native epoch shards with bounded prefetch.

    Args:
        owner: Runtime owning replica connections and prepared task identities.
        extension_ref: Relative native training-table extension.
        options: Normalized native iterator settings.
    Raises:
        ValueError: If task transfer, framing or shard completion is invalid.
        flight.FlightError: If a feeder fails; incomplete epochs are not retried.
    """

    def __init__(
        self, owner: RemoteRuntimeIndex, extension_ref: str, options: dict[str, Any]
    ) -> None:
        self._owner = owner
        self._closed = False
        self._streams: list[RemoteTaskIterator] = []
        self._pending: deque[tuple[RemoteTaskIterator, Future[dict[str, Any]]]] = (
            deque()
        )
        self._num_samples = 0
        self._executor = ThreadPoolExecutor(
            max_workers=len(owner.endpoints), thread_name_prefix="sampling-push"
        )
        try:
            endpoints = owner.endpoints
            if options.get("rhs_seed_node_type") is not None:
                endpoints = endpoints[:1]
            owner._prepare_task(extension_ref, endpoints)
            replicas = len(endpoints)
            options = dict(options)
            weighted = options.get("weighted_seed_sampling_json")
            if weighted is not None:
                weighted = orjson.loads(weighted)
                root = Path(owner._parent_root) / extension_ref
                for bucket in weighted["buckets"]:
                    for key in ("node_ids_path", "cdf_path"):
                        if bucket.get(key) is not None:
                            bucket[key] = Path(bucket[key]).relative_to(root).as_posix()
                options["weighted_seed_sampling_json"] = orjson.dumps(weighted).decode()
            for replica, endpoint in enumerate(endpoints):
                shard = {
                    **options,
                    "num_workers": options["num_workers"] * replicas,
                    "worker_id": options["worker_id"]
                    + options["num_workers"] * replica,
                }
                request = {
                    "kind": "push_iter",
                    "version": PUSH_VERSION,
                    "extension_ref": extension_ref,
                    "engine_seed": owner._rnd_seed,
                    "num_threads": owner._num_threads or 1,
                    "options": shard,
                }
                stream = RemoteTaskIterator(
                    endpoint,
                    request,
                    client=owner._client_for(endpoint),
                )
                self._streams.append(stream)
                self._num_samples += stream.num_samples()
            self._pending.extend(
                (stream, self._executor.submit(next, stream))
                for stream in self._streams
            )
            owner._streams.add(self)
        except BaseException:
            self.close()
            raise

    def num_samples(self) -> int:
        """Return the total number of examples in the replica shards."""
        return self._num_samples

    def __iter__(self) -> RemotePushIterator:
        """Return this epoch iterator."""
        return self

    def __next__(self) -> dict[str, Any]:
        """Yield shards in deterministic round-robin order, propagating failures."""
        if self._closed:
            raise StopIteration
        try:
            while self._pending:
                stream, future = self._pending.popleft()
                try:
                    batch = future.result()
                except StopIteration:
                    continue
                self._pending.append((stream, self._executor.submit(next, stream)))
                return batch
            self.close()
            raise StopIteration
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Cancel every producer before joining blocked reader threads."""
        if self._closed:
            return
        self._closed = True
        self._owner._streams.discard(self)
        errors: list[Exception] = []
        try:
            for stream in self._streams:
                try:
                    stream.close()
                except (pa.ArrowException, flight.FlightError, OSError) as error:
                    errors.append(error)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._streams.clear()
            self._pending.clear()
        if errors:
            raise ExceptionGroup("Failed to close push sampling streams", errors)


class _RemoteNativeIndex:
    def __init__(self, owner: RemoteRuntimeIndex, extension_ref: str) -> None:
        self._owner = owner
        self._extension_ref = extension_ref

    def iter(self, *args: Any) -> RemotePushIterator:
        if len(args) != len(_NATIVE_ARGUMENTS):
            raise ValueError("Unsupported DiskGraph native iterator signature")
        return RemotePushIterator(
            self._owner,
            self._extension_ref,
            dict(zip(_NATIVE_ARGUMENTS, args, strict=True)),
        )


class RemoteRuntimeIndex(RuntimeIndex):
    """Reuse DiskGraph's manifest resolution while sampling on a gateway.

    Args:
        dataset_ref: Trainer-local portable runtime manifest.
        endpoint: Flight URI or list of URIs serving identical graph replicas.
            Workers stream complete batches from disjoint native epoch shards.
        rnd_seed: Engine and default shuffle seed.
        num_threads: Server engine thread count, affecting reproducibility.

    Raises:
        ValueError: If the native wrapper contract or replica endpoints are invalid.
        flight.FlightError: If a request fails. Requests are never retried.
    """

    def __init__(
        self,
        dataset_ref: str,
        endpoint: str | list[str],
        rnd_seed: int | None = None,
        num_threads: int | None = None,
    ) -> None:
        """Initialize manifest resolution without opening a graph or connection."""
        super().__init__(dataset_ref, rnd_seed=rnd_seed, num_threads=num_threads)
        self.endpoints = (endpoint,) if isinstance(endpoint, str) else tuple(endpoint)
        if not self.endpoints or any(not value for value in self.endpoints):
            raise ValueError("At least one nonempty sampling endpoint is required")
        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("Sampling endpoints must be unique")
        self.endpoint = self.endpoints[0]
        self._streams: set[RemotePushIterator | RemoteTaskIterator] = set()
        self._tasks: dict[str, tuple[str, list[tuple[str, int]]]] = {}
        self._prepared: set[tuple[str, str]] = set()
        self._clients: dict[str, flight.FlightClient] = {}

    @property
    def client(self) -> flight.FlightClient:
        """Return the first replica's connection for single-endpoint callers."""
        return self._client_for(self.endpoint)

    def _client_for(self, endpoint: str) -> flight.FlightClient:
        client = self._clients.get(endpoint)
        if client is None:
            client = flight.FlightClient(
                endpoint,
                generic_options=[("grpc.max_receive_message_length", 128 * 1024**2)],
            )
            self._clients[endpoint] = client
        return client

    def _prepare_task(self, extension_ref: str, endpoints: tuple[str, ...]) -> None:
        root = Path(self._parent_root) / extension_ref
        identity = self._tasks.get(extension_ref)
        for endpoint in endpoints:
            key = (endpoint, extension_ref)
            if key not in self._prepared:
                client = self._client_for(endpoint)
                actions = {action.type for action in client.list_actions()}
                if "task_status" in actions:
                    if identity is None:
                        identity = task_files(root)
                        self._tasks[extension_ref] = identity
                    upload_task(client, root, extension_ref, identity)
                else:
                    raise ValueError(
                        "Sampling endpoint does not support task epoch streaming"
                    )
                self._prepared.add(key)

    def _index_for(self, extension_ref: str) -> _RemoteNativeIndex:
        return _RemoteNativeIndex(self, extension_ref)

    def close(self) -> None:
        """Cancel stage iteration, close replica streams and join sampling workers.

        Raises:
            ExceptionGroup: If any stream failed to release its resources.
        """
        errors: list[Exception] = []
        for stream in tuple(self._streams):
            try:
                stream.close()
            except (
                pa.ArrowException,
                flight.FlightError,
                OSError,
                ExceptionGroup,
            ) as error:
                errors.append(error)
        self._streams.clear()
        self._tasks.clear()
        self._prepared.clear()
        for client in self._clients.values():
            try:
                client.close()
            except (pa.ArrowException, flight.FlightError, OSError) as error:
                errors.append(error)
        self._clients.clear()
        if errors:
            raise ExceptionGroup("Failed to close remote sampling streams", errors)
