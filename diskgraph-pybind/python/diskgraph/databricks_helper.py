"""Databricks SQL control-plane helper for standalone DiskGraph ingest.

The helper deliberately stops at raw Cloud Fetch metadata.  It never downloads
signed objects and never decodes bulk Arrow data; the Rust ingest process owns
that data plane.  stdin/stdout are a serialized JSON-lines protocol and stderr
is reserved for sanitized diagnostics.
"""

from __future__ import annotations

import base64
import ctypes
import importlib.metadata
import json
import logging
import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any, IO, Mapping
from urllib.parse import urlsplit

SUPPORTED_CONNECTOR_VERSION = "4.3.0"
SUPPORTED_THRIFT_VERSION = "0.22.0"
PROTOCOL_VERSION = 1
CONTROL_SOCKET_TIMEOUT_SECONDS = 300
PARENT_PID_ENV = "DISKGRAPH_PARENT_PID"

logger = logging.getLogger("diskgraph.databricks_helper")


class ProtocolError(RuntimeError):
    """A caller-visible protocol or source-contract error."""


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def _require_private_config_file(path: str) -> str:
    config_path = Path(path)
    mode = stat.S_IMODE(config_path.stat().st_mode)
    if os.name == "posix" and mode & 0o077:
        raise ProtocolError("Databricks config file must have mode 0600")
    return os.fspath(config_path)


def _hostname(value: str) -> str:
    candidate = value if "://" in value else f"https://{value}"
    parsed = urlsplit(candidate)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProtocolError("Databricks workspace host must be an HTTPS hostname")
    return parsed.hostname.casefold()


def create_sdk_config(
    source: Mapping[str, Any], config_file: str | None
) -> Any:
    """Create one refreshable Databricks unified-authentication provider."""
    from databricks.sdk.core import Config

    profile_value = source.get("auth_profile")
    profile = (
        _require_nonempty(profile_value, "auth_profile")
        if profile_value is not None
        else None
    )
    kwargs: dict[str, Any] = {}
    if profile is not None:
        kwargs["profile"] = profile
    if config_file is not None:
        kwargs["config_file"] = _require_private_config_file(config_file)

    try:
        config = Config(**kwargs)
    except Exception as error:
        raise ProtocolError(
            "Databricks unified authentication could not be configured"
        ) from error

    expected_hostname = _hostname(
        _require_nonempty(source.get("server_hostname"), "server_hostname")
    )
    configured_host = getattr(config, "host", None)
    if not configured_host or _hostname(configured_host) != expected_hostname:
        raise ProtocolError(
            "Databricks authentication profile host does not match server_hostname"
        )

    return config


def _connector_versions() -> dict[str, str]:
    connector = importlib.metadata.version("databricks-sql-connector")
    thrift = importlib.metadata.version("thrift")
    if connector != SUPPORTED_CONNECTOR_VERSION or thrift != SUPPORTED_THRIFT_VERSION:
        raise ProtocolError(
            "unsupported Databricks connector stack; reinstall the pinned DiskGraph dependencies"
        )
    return {"connector": connector, "thrift": thrift}


class DatabricksControl:
    """One active query and one narrow adapter over connector 4.3.0 internals."""

    def __init__(self) -> None:
        self._versions = _connector_versions()
        self._connection: Any = None
        self._connection_key: tuple[Any, ...] | None = None
        self._cursor: Any = None
        self._query_id: str | None = None
        self._next_page_id = 0
        self._arrow_schema: bytes | None = None
        self._lz4_compressed: bool | None = None
        self._result_format: int | None = None

    def hello(self) -> dict[str, Any]:
        return {"protocol": PROTOCOL_VERSION, **self._versions}

    def open(
        self, source: Mapping[str, Any], config_file: str | None = None
    ) -> dict[str, Any]:
        if self._cursor is not None:
            raise ProtocolError("a Databricks query is already active")

        from databricks import sql

        hostname = _require_nonempty(source.get("server_hostname"), "server_hostname")
        warehouse_id = _require_nonempty(source.get("warehouse_id"), "warehouse_id")
        catalog = _require_nonempty(source.get("catalog"), "catalog")
        schema = _require_nonempty(source.get("schema"), "schema")
        statement = _require_nonempty(source.get("sql"), "sql")
        config = create_sdk_config(source, config_file)
        connection_key = (
            hostname,
            warehouse_id,
            catalog,
            schema,
            config.auth_type,
            config.profile,
            config.config_file,
        )

        kwargs: dict[str, Any] = {
            "server_hostname": hostname,
            "http_path": f"/sql/1.0/warehouses/{warehouse_id}",
            "catalog": catalog,
            "schema": schema,
            "use_cloud_fetch": True,
            "enable_query_result_lz4_compression": True,
            "enable_telemetry": False,
            # Connector 4.3.0 otherwise waits 15 minutes for a control-plane
            # socket read. FETCH_NEXT cannot be retried safely because the
            # server cursor may already have advanced, so fail the worker in
            # bounded time and let the ingest supervisor restart Encode.
            "_socket_timeout": CONTROL_SOCKET_TIMEOUT_SECONDS,
        }
        # The SQL connector asks this factory for fresh headers on every
        # request. Config handles notebook-native auth, OAuth U2M/M2M, PATs,
        # and workload identity federation without exposing tokens here.
        kwargs["credentials_provider"] = lambda: config.authenticate

        connection: Any = self._connection
        try:
            if connection is not None and self._connection_key != connection_key:
                connection.close()
                connection = None
                self._connection = None
                self._connection_key = None
            if connection is None:
                connection = sql.connect(**kwargs)
            cursor = connection.cursor()
            cursor.execute_async(statement)
            while cursor.is_query_pending():
                time.sleep(0.2)
            from databricks.sql.backend.types import CommandState

            state = cursor.get_query_state()
            if state != CommandState.SUCCEEDED:
                raise ProtocolError(f"Databricks query ended in state {state.name}")
        except Exception:
            try:
                if connection is not None:
                    connection.close()
            except Exception:
                pass
            self._connection = None
            self._connection_key = None
            raise

        self._connection = connection
        self._connection_key = connection_key
        self._cursor = cursor
        self._query_id = _require_nonempty(cursor.query_id, "query_id")
        self._next_page_id = 0
        self._arrow_schema = None
        self._lz4_compressed = None
        self._result_format = None
        return {"query_id": self._query_id}

    def schema(
        self, source: Mapping[str, Any], config_file: str | None = None
    ) -> dict[str, Any]:
        """Resolve the physical Arrow schema without scanning the result."""
        statement = _require_nonempty(source.get("sql"), "sql").rstrip().rstrip(";")
        prepared_source = dict(source)
        prepared_source["sql"] = (
            f"SELECT * FROM ({statement}) AS __diskgraph_schema LIMIT 0"
        )
        self.open(prepared_source, config_file)
        try:
            # ``execute_async`` does not initialize ``active_result_set`` for
            # a zero-row result in connector 4.3.0. Fetch the empty page so
            # Databricks returns its result metadata through the same raw,
            # non-retryable adapter used by normal pagination. LIMIT 0 makes
            # this a schema-only request; the one-byte/one-row budgets are a
            # defensive guard if the server ever violates that contract.
            page = self.next_page(
                max_rows=1, max_bytes=1, max_links=1, include_arrow_schema=True
            )
            return {"arrow_schema": page["arrow_schema"]}
        finally:
            self.close()

    def next_page(
        self,
        *,
        max_rows: int,
        max_bytes: int,
        max_links: int,
        include_arrow_schema: bool = False,
    ) -> dict[str, Any]:
        cursor = self._require_cursor()
        if max_rows <= 0 or max_bytes <= 0 or max_links <= 0:
            raise ProtocolError("next budgets must be positive")

        # This is the only connector-private adapter. FETCH_NEXT mutates the
        # server cursor, so retryable=False is load-bearing.
        from databricks.sql.thrift_api.TCLIService import ttypes

        command_id = cursor.active_command_id
        thrift_handle = command_id.to_thrift_handle()
        if thrift_handle is None:
            raise ProtocolError("active query has no operation handle")
        request = ttypes.TFetchResultsReq(
            operationHandle=thrift_handle,
            maxRows=max_rows,
            maxBytes=max_bytes,
            orientation=ttypes.TFetchOrientation.FETCH_NEXT,
            includeResultSetMetadata=True,
        )
        backend = cursor.backend
        response = backend.make_request(
            backend._client.FetchResults, request, retryable=False
        )
        row_set = response.results
        metadata = response.resultSetMetadata
        if row_set is None or metadata is None:
            raise ProtocolError("Databricks returned incomplete result metadata")

        result_format = int(metadata.resultFormat)
        if result_format not in (
            ttypes.TSparkRowSetType.URL_BASED_SET,
            ttypes.TSparkRowSetType.ARROW_BASED_SET,
        ):
            raise ProtocolError("Databricks returned an unsupported result format")
        if row_set.columns:
            raise ProtocolError("Databricks returned unsupported column results")

        links = row_set.resultLinks or []
        batches = row_set.arrowBatches or []
        if len(links) > max_links:
            raise ProtocolError("Databricks returned more links than the requested bound")
        if links and batches:
            raise ProtocolError("Databricks mixed Cloud Fetch links and inline batches")

        encoded_links = []
        advertised_rows = 0
        advertised_bytes = 0
        for link in links:
            fields = {
                "url": _require_nonempty(link.fileLink, "result link URL"),
                "start_row_offset": int(link.startRowOffset),
                "row_count": int(link.rowCount),
                "decoded_bytes": int(link.bytesNum),
                "headers": dict(link.httpHeaders or {}),
            }
            if fields["row_count"] < 0 or fields["decoded_bytes"] < 0:
                raise ProtocolError("Databricks returned negative link metadata")
            advertised_rows += fields["row_count"]
            advertised_bytes += fields["decoded_bytes"]
            encoded_links.append(fields)

        encoded_batches = []
        for batch in batches:
            row_count = int(batch.rowCount)
            if row_count < 0:
                raise ProtocolError("Databricks returned a negative inline row count")
            advertised_rows += row_count
            advertised_bytes += len(batch.batch)
            encoded_batches.append(
                {
                    "data": base64.b64encode(batch.batch).decode("ascii"),
                    "row_count": row_count,
                }
            )

        has_more_rows = bool(response.hasMoreRows)
        if advertised_rows > max_rows or advertised_bytes > max_bytes:
            raise ProtocolError("Databricks result page exceeded its requested bounds")
        if has_more_rows and advertised_rows == 0:
            raise ProtocolError("Databricks result pagination made no progress")

        arrow_schema = metadata.arrowSchema
        if not arrow_schema and metadata.schema is not None:
            arrow_schema = (
                backend._hive_schema_to_arrow_schema(
                    metadata.schema, self._connection.session.host
                )
                .serialize()
                .to_pybytes()
            )
        if not arrow_schema:
            raise ProtocolError("Databricks returned no Arrow schema")

        lz4_compressed = bool(metadata.lz4Compressed)
        if self._arrow_schema is None:
            self._arrow_schema = arrow_schema
            self._lz4_compressed = lz4_compressed
            self._result_format = result_format
        elif (
            arrow_schema != self._arrow_schema
            or lz4_compressed != self._lz4_compressed
            or result_format != self._result_format
        ):
            raise ProtocolError("Databricks changed result metadata between pages")

        page_id = self._next_page_id
        self._next_page_id += 1
        page = {
            "query_id": self._query_id,
            "page_id": page_id,
            "start_row_offset": int(row_set.startRowOffset),
            "has_more_rows": has_more_rows,
            "lz4_compressed": lz4_compressed,
            "links": encoded_links,
            "inline_batches": encoded_batches,
        }
        if include_arrow_schema or encoded_batches or not encoded_links:
            page["arrow_schema"] = base64.b64encode(arrow_schema).decode("ascii")
        return page

    def close(self, *, cancel: bool = False) -> dict[str, Any]:
        cursor, connection = self._cursor, self._connection
        self._cursor = None
        self._query_id = None
        self._next_page_id = 0
        self._arrow_schema = None
        self._lz4_compressed = None
        self._result_format = None
        first_error = None
        if cursor is not None:
            try:
                if cancel:
                    cursor.cancel()
                elif cursor.active_command_id is not None:
                    cursor.backend.close_command(cursor.active_command_id)
            except Exception as error:
                first_error = error
            try:
                cursor.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if cancel or first_error is not None:
            self._connection = None
            self._connection_key = None
            if connection is not None:
                try:
                    connection.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error
        return {}

    def _require_cursor(self) -> Any:
        if self._cursor is None:
            raise ProtocolError("no Databricks query is active")
        return self._cursor


def _dispatch(control: Any, request: Mapping[str, Any]) -> Any:
    op = request.get("op")
    if op == "hello":
        return control.hello()
    if op == "open":
        source = request.get("source")
        if not isinstance(source, dict):
            raise ProtocolError("open.source must be an object")
        return control.open(source, request.get("config_file"))
    if op == "schema":
        source = request.get("source")
        if not isinstance(source, dict):
            raise ProtocolError("schema.source must be an object")
        return control.schema(source, request.get("config_file"))
    if op == "next":
        return control.next_page(
            max_rows=int(request.get("max_rows", 0)),
            max_bytes=int(request.get("max_bytes", 0)),
            max_links=int(request.get("max_links", 0)),
        )
    if op == "close":
        return control.close(cancel=bool(request.get("cancel", False)))
    raise ProtocolError("unknown helper operation")


def serve(stdin: IO[str], stdout: IO[str], control: Any | None = None) -> None:
    control = control or DatabricksControl()
    expected_request_id = 1
    for line in stdin:
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ProtocolError("request must be an object")
            request_id = request.get("id")
            if (
                not isinstance(request_id, int)
                or isinstance(request_id, bool)
                or request_id != expected_request_id
            ):
                raise ProtocolError("request id is not the next expected integer")
            expected_request_id += 1
            result = _dispatch(control, request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as error:
            # Do not include exception text: connector exceptions can contain
            # SQL, response bodies, headers, or signed links.
            logger.warning("helper operation failed (%s)", type(error).__name__)
            response = {
                "id": request_id,
                "ok": False,
            }
        stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        stdout.flush()


def _disable_unsafe_logging() -> None:
    unsafe = logging.getLogger("databricks.sql.unsafe")
    unsafe.handlers.clear()
    unsafe.propagate = False
    unsafe.disabled = True


def _set_parent_death_signal() -> None:
    if sys.platform != "linux":
        return
    try:
        expected_parent_pid = int(os.environ[PARENT_PID_ENV])
    except (KeyError, ValueError) as error:
        raise RuntimeError("expected Databricks helper parent PID is unavailable") from error
    if expected_parent_pid <= 0:
        raise RuntimeError("expected Databricks helper parent PID is invalid")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    # The parent may have exited between fork/exec and PR_SET_PDEATHSIG.
    # PID 1 is valid when DiskGraph is the container's entrypoint.
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    _disable_unsafe_logging()
    _set_parent_death_signal()
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
