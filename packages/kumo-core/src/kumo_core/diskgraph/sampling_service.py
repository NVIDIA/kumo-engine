# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process Arrow Flight task sampling over the native DiskGraph iterator."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
from diskgraph._c import RuntimeIndex
from pyarrow import flight

from kumo_core.diskgraph.remote_sampling import _NATIVE_ARGUMENTS
from kumo_core.diskgraph.sampling_protocol import (
    ENVELOPE_SCHEMA,
    MAX_BATCH_BYTES,
    PUSH_VERSION,
    PUSH_WINDOW,
    TASK_CHUNK_BYTES,
    decode_frame,
    encode_batch,
    encode_frame,
)
from kumo_core.diskgraph.sampling_task import (
    task_files,
)

logger = logging.getLogger("kumo.sampling")


@dataclass(frozen=True)
class SamplingServiceConfig:
    """Immutable root, bind address and native concurrency budget.

    Args:
        index_path: Complete graph index; task extensions are registered before
            epoch streaming.
        host: Listening interface; use 0.0.0.0 for cross-node access.
        port: Listening port; zero selects a free port atomically.
        max_streams: Maximum simultaneously active sampling streams.
        max_threads: Maximum native threads allowed per requested index.
    """

    index_path: Path
    host: str = "127.0.0.1"
    port: int = 0
    max_streams: int = 1
    max_threads: int = 2


class SamplingService(flight.FlightServerBase):
    """Own a Flight listener and native iterators in the caller's process.

    Construct after graph preparation. The listener starts during
    construction; calling the blocking Flight ``serve()`` is unnecessary.
    Worker lifecycle methods can run while Flight threads serve batches.

    Args:
        config: Worker-local immutable graph and resource budget.
    Raises:
        ValueError: If the index or concurrency limits are invalid.
        OSError: If the graph root cannot be resolved or the socket cannot bind.
    """

    def __init__(self, config: SamplingServiceConfig) -> None:
        """Validate a ready index and bind an in-process Flight listener."""
        root = config.index_path.resolve(strict=True)
        if not (root / "index_metadata.json").is_file():
            raise ValueError("Sampling requires a completed graph index")
        if min(config.max_streams, config.max_threads) < 1:
            raise ValueError("Sampling stream and thread limits must be positive")
        self._config = config
        self._root = root
        self._indexes: dict[tuple[str, int | None, int], RuntimeIndex] = {}
        self._lock = threading.Lock()
        self._permits = threading.BoundedSemaphore(config.max_streams)
        self._stopping = threading.Event()
        self._active = 0
        self._requests = 0
        self._task_requests = 0
        self._batches = 0
        self._exchange_streams = 0
        self._tasks: dict[str, str] = {}
        self._upload_lock = threading.Lock()
        super().__init__(location=(config.host, config.port))
        logger.info(
            "SAMPLING_LISTENER_READY pid=%d host=%s port=%d max_streams=%d max_threads=%d",
            os.getpid(),
            config.host,
            self.port,
            config.max_streams,
            config.max_threads,
        )

    def endpoint(self, advertised_host: str | None = None) -> str:
        """Return the bound endpoint, optionally advertising a routable node IP.

        Args:
            advertised_host: Address clients should use instead of the bind host.
        Returns:
            A Flight URI with the actual bound port.
        """
        return f"grpc://{advertised_host or self._config.host}:{self.port}"

    def health(self) -> dict[str, str | int | bool]:
        """Return process identity, request counters and readiness.

        Returns:
            Lightweight counters and identity for worker lifecycle calls.
        """
        with self._lock:
            return {
                "ready": not self._stopping.is_set(),
                "has_training_tables": (self._root / "train_tables").exists(),
                "pid": os.getpid(),
                "active_streams": self._active,
                "requests": self._requests,
                "task_requests": self._task_requests,
                "batches": self._batches,
                "exchange_streams": self._exchange_streams,
                "port": self.port,
            }

    def close(self) -> None:
        """Stop accepting work, finish in-flight batches and release the listener.

        Active generators check cancellation between native batches. This method
        must be called by the owner, rather than from a Flight request callback.
        """
        self._stopping.set()
        logger.info("SAMPLING_LISTENER_STOPPING pid=%d port=%d", os.getpid(), self.port)
        self.shutdown()
        with self._lock:
            self._indexes.clear()
            batches, requests, streams = (
                self._batches,
                self._requests,
                self._exchange_streams,
            )
        logger.info(
            "SAMPLING_LISTENER_STOPPED pid=%d port=%d batches=%d "
            "requests=%d exchange_streams=%d",
            os.getpid(),
            self.port,
            batches,
            requests,
            streams,
        )

    def _record_batch(self) -> None:
        with self._lock:
            self._batches += 1
            batches = self._batches
            if batches != 1 and batches % 100:
                return
            requests, active, streams = (
                self._task_requests,
                self._active,
                self._exchange_streams,
            )
        logger.info(
            "SAMPLING_PROGRESS pid=%d port=%d batches=%d task_requests=%d "
            "active_streams=%d exchange_streams=%d",
            os.getpid(),
            self.port,
            batches,
            requests,
            active,
            streams,
        )

    def do_action(self, context: Any, action: flight.Action) -> Iterator[flight.Result]:
        """Check a registered task identity or report readiness.

        Args:
            context: Flight call context supplied by Arrow.
            action: Immutable task identity, or ``health`` with no payload.
        Yields:
            One metadata result or confirmed artifact identity.
        Raises:
            NotImplementedError: If an unknown action is requested.
            ValueError: If task paths or identities are invalid.
            flight.FlightUnavailableError: If registration starts during shutdown.
        """
        if action.type == "task_status":
            request = orjson.loads(action.body.to_pybytes())
            relative = self._task_ref(request["extension_ref"])
            if self._stopping.is_set():
                raise flight.FlightUnavailableError("Sampling service is stopping")
            with self._upload_lock:
                key = self._tasks.get(relative)
                root = self._root / relative
                if key is None and root.exists():
                    key, _ = task_files(root)
                    self._tasks[relative] = key
                if key is not None and key != request["sha256"]:
                    raise ValueError("Task extension identity mismatch")
            yield flight.Result((key or "").encode())
            return
        if action.type != "health":
            raise NotImplementedError(f"Unsupported sampling action: {action.type}")
        yield flight.Result(orjson.dumps(self.health()))

    def list_actions(
        self, context: flight.ServerCallContext
    ) -> Iterator[flight.ActionType]:
        """Advertise task preparation and health actions.

        Args:
            context: Flight call context.
        Yields:
            Supported actions for capability negotiation.
        """
        yield flight.ActionType("task_status", "Check an immutable task identity")
        yield flight.ActionType("health", "Read worker readiness and sampling counters")

    def _task_ref(self, relative: str) -> str:
        path = Path(relative)
        if (
            path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] not in {"train_tables", "pred_tables"}
            or path.parts[1] in {".", ".."}
            or path.parts[1].startswith(".")
            or not (self._root / path).resolve().is_relative_to(self._root)
        ):
            raise ValueError("Expected train_tables/<id> or pred_tables/<id>")
        return path.as_posix()

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.FlightMetadataWriter,
    ) -> None:
        """Receive and atomically publish a checksum-verified task extension.

        Args:
            context: Flight cancellation context.
            descriptor: Version-four preparation command and task identity.
            reader: File headers and bounded binary chunk envelopes.
            writer: Checksum acknowledgement after successful publication.
        Raises:
            ValueError: If paths, byte lengths or checksums are invalid.
            OSError: If task files cannot be written.
            flight.FlightUnavailableError: If stopping or cancelled.
        """
        if descriptor.command is None or len(descriptor.command) > MAX_BATCH_BYTES:
            raise ValueError("Invalid task preparation descriptor")
        request = orjson.loads(descriptor.command)
        if (
            not isinstance(request, dict)
            or set(request) != {"kind", "version", "extension_ref", "sha256"}
            or request["kind"] != "prepare_task"
            or request["version"] != PUSH_VERSION
        ):
            raise ValueError("Unsupported task preparation protocol")
        relative = self._task_ref(request["extension_ref"])
        with self._upload_lock:
            destination = self._root / relative
            if destination.exists():
                key, _ = task_files(destination)
                if key != request["sha256"]:
                    raise ValueError("Task extension identity mismatch")
                staging = None
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = destination.parent / f".upload-{uuid.uuid4().hex}"
                staging.mkdir()
            key = self._receive_task(context, reader, staging)
            if key != request["sha256"]:
                raise ValueError("Task transfer checksum mismatch")
            if staging is not None:
                staging.rename(destination)
            self._tasks[relative] = key
            writer.write(key.encode())

    def _receive_task(
        self,
        context: flight.ServerCallContext,
        reader: flight.MetadataRecordBatchReader,
        staging: Path | None,
    ) -> str:
        digest = hashlib.sha256()
        previous = ""
        messages = iter(reader)
        for chunk in messages:
            if self._stopping.is_set() or context.is_cancelled():
                raise flight.FlightUnavailableError("Task transfer cancelled")
            metadata, components = decode_frame(chunk.data)
            if metadata == {"kind": "end"} and not components:
                if not previous or next(messages, None) is not None:
                    raise ValueError("Invalid task transfer completion")
                return digest.hexdigest()
            if (
                set(metadata) != {"kind", "path", "size"}
                or metadata["kind"] != "file"
                or components
            ):
                raise ValueError("Expected task file header")
            name, size = metadata["path"], metadata["size"]
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or Path(name).as_posix() != name
                or name <= previous
                or type(size) is not int
                or size < 0
            ):
                raise ValueError("Invalid task file path or length")
            previous = name
            digest.update(orjson.dumps([name, size]))
            if staging is not None:
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") if staging is not None else nullcontext() as output:
                remaining = size
                while remaining:
                    chunk = next(messages, None)
                    if chunk is None:
                        raise ValueError("Incomplete task file")
                    metadata, components = decode_frame(chunk.data)
                    if (
                        metadata != {"kind": "data"}
                        or len(components) != 1
                        or not 0
                        < len(components[0])
                        <= min(remaining, TASK_CHUNK_BYTES)
                    ):
                        raise ValueError("Invalid task file chunk")
                    if output is not None:
                        output.write(components[0])
                    digest.update(components[0])
                    remaining -= len(components[0])
        raise ValueError("Task transfer ended without completion")

    def _resolve(self, root: Path, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not relative:
            raise ValueError("Task paths must be relative to the served root")
        resolved = (root / path).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("Task path escapes the served root")
        return resolved

    def _open(self, request: dict[str, Any]) -> Any:
        allowed = {
            "kind",
            "version",
            "extension_ref",
            "engine_seed",
            "num_threads",
            "options",
        }
        if (
            set(request) - allowed
            or request.get("kind") != "push_iter"
            or type(request.get("version")) is not int
            or request["version"] != PUSH_VERSION
        ):
            raise ValueError("Unsupported task iterator protocol")
        extension = self._resolve(self._root, request["extension_ref"])
        relative = extension.relative_to(self._root)
        if len(relative.parts) != 2 or relative.parts[0] not in {
            "train_tables",
            "pred_tables",
        }:
            raise ValueError("Expected train_tables/<id> or pred_tables/<id>")
        threads = request.get("num_threads")
        if threads is None:
            threads = 1
        if type(threads) is not int or not 1 <= threads <= self._config.max_threads:
            raise ValueError("Requested native threads exceed the service budget")
        seed = request.get("engine_seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("Engine seed must be an unsigned 64-bit integer")
        options = request["options"]
        if not isinstance(options, dict) or set(options) - set(_NATIVE_ARGUMENTS):
            raise ValueError("Unsupported native iterator options")
        options = dict(options)
        if options.get("weighted_seed_sampling_json") is not None:
            weighted = orjson.loads(options["weighted_seed_sampling_json"])
            for bucket in weighted["buckets"]:
                for key in ("node_ids_path", "cdf_path"):
                    if bucket.get(key) is not None:
                        bucket[key] = str(self._resolve(extension, bucket[key]))
            options["weighted_seed_sampling_json"] = orjson.dumps(weighted).decode()
        cache_key = (str(relative), seed, threads)
        with self._lock:
            index = self._indexes.get(cache_key)
            if index is None:
                if len(self._indexes) >= 4:
                    raise ValueError("Task index cache is full")
                index = RuntimeIndex(str(self._root), str(relative), seed, threads)
                self._indexes[cache_key] = index
        return index.iter(**options)

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.MetadataRecordBatchWriter,
    ) -> None:
        """Push a native epoch shard, paced by bounded consumer credits.

        Args:
            context: Flight cancellation context.
            descriptor: Version-four normalized epoch and shard options.
            reader: Credit frames, containing no per-batch seed requests.
            writer: Iterator metadata, complete batches and verified completion.
        Raises:
            ValueError: If framing, sequence or sampling options are invalid.
            flight.FlightUnavailableError: If stopping or at stream capacity.
        """
        if descriptor.command is None or len(descriptor.command) > MAX_BATCH_BYTES:
            raise ValueError("Invalid push iterator descriptor")
        request = orjson.loads(descriptor.command)
        if not isinstance(request, dict) or (
            request.get("kind"),
            request.get("version"),
        ) != ("push_iter", PUSH_VERSION):
            raise ValueError("Unsupported push sampling protocol")
        version = request["version"]
        if self._stopping.is_set() or not self._permits.acquire(timeout=5.0):
            raise flight.FlightUnavailableError("Sampling service is busy or stopping")
        with self._lock:
            self._active += 1
            self._exchange_streams += 1
            stream_id = self._exchange_streams
        started = time.perf_counter()
        completed = 0
        outcome = "eof"
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                with self._lock:
                    self._active -= 1
                self._permits.release()

        logger.info(
            "SAMPLING_STREAM_OPEN pid=%d port=%d stream_id=%d",
            os.getpid(),
            self.port,
            stream_id,
        )
        try:
            iterator = self._open(request)
            expected = iterator.worker_batch_count()
            with self._lock:
                self._requests += 1
                self._task_requests += 1
            writer.begin(ENVELOPE_SCHEMA)
            writer.write_batch(
                encode_frame(
                    {
                        "kind": "iterator",
                        "version": version,
                        "num_samples": iterator.num_samples(),
                        "num_batches": expected,
                    },
                    [],
                )
            )
            while completed < expected:
                try:
                    chunk = reader.read_chunk()
                except StopIteration as error:
                    raise ValueError(
                        "Push stream ended before its shard completed"
                    ) from error
                metadata, components = decode_frame(chunk.data)
                if (
                    set(metadata) != {"kind", "batches"}
                    or metadata["kind"] != "credit"
                    or type(metadata["batches"]) is not int
                    or not 1 <= metadata["batches"] <= PUSH_WINDOW
                    or metadata["batches"] > expected - completed
                    or components
                ):
                    raise ValueError("Invalid push stream credits")
                for _ in range(metadata["batches"]):
                    if self._stopping.is_set() or context.is_cancelled():
                        outcome = "cancelled"
                        return
                    writer.write_batch(encode_batch(next(iterator), completed))
                    completed += 1
                    self._record_batch()
            if next(iterator, None) is not None:
                raise ValueError("Native shard exceeded its declared batch count")
            writer.write_batch(encode_frame({"kind": "end", "batches": completed}, []))
            del iterator
            release()
            try:
                reader.read_chunk()
            except StopIteration:
                pass
            else:
                raise ValueError("Unexpected credits after shard completion")
            outcome = "complete"
        except (
            pa.ArrowException,
            flight.FlightError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as error:
            outcome = "cancelled" if context.is_cancelled() else type(error).__name__
            if outcome != "cancelled":
                logger.error(
                    "SAMPLING_STREAM_FAILED pid=%d port=%d stream_id=%d error_type=%s",
                    os.getpid(),
                    self.port,
                    stream_id,
                    outcome,
                )
            raise
        finally:
            release()
            logger.info(
                "SAMPLING_STREAM_CLOSED pid=%d port=%d stream_id=%d "
                "batches=%d seconds=%.3f outcome=%s",
                os.getpid(),
                self.port,
                stream_id,
                completed,
                time.perf_counter() - started,
                outcome,
            )
