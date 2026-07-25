from __future__ import annotations

import base64
import io
import json
import os
import stat
from types import SimpleNamespace

import pytest

from diskgraph.databricks_helper import (
    CONTROL_SOCKET_TIMEOUT_SECONDS,
    DatabricksControl,
    ProtocolError,
    create_sdk_config,
    serve,
)


@pytest.fixture(autouse=True)
def default_unified_auth(monkeypatch):
    for name in (
        "DATABRICKS_CONFIG_FILE",
        "DATABRICKS_CONFIG_PROFILE",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_AUTH_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb.example.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")


def _write_config(path, *, profile="db", host="https://adb.example.net"):
    path.write_text(f"[{profile}]\nhost = {host}\ntoken = secret\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_default_unified_auth_uses_environment():
    config = create_sdk_config(_source(), None)
    assert config.auth_type == "pat"
    assert config.authenticate()["Authorization"] == "Bearer test-token"


def test_named_profile_uses_standard_databricks_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    monkeypatch.delenv("DATABRICKS_TOKEN")
    path = tmp_path / ".databrickscfg"
    _write_config(path)
    config = create_sdk_config(_source(auth_profile="db"), os.fspath(path))
    assert config.profile == "db"
    assert config.authenticate()["Authorization"] == "Bearer secret"


def test_config_file_requires_private_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    monkeypatch.delenv("DATABRICKS_TOKEN")
    path = tmp_path / ".databrickscfg"
    _write_config(path)
    path.chmod(0o644)
    with pytest.raises(ProtocolError, match="mode 0600"):
        create_sdk_config(_source(auth_profile="db"), os.fspath(path))
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    create_sdk_config(_source(auth_profile="db"), os.fspath(path))


def test_profile_host_must_match_source(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST")
    monkeypatch.delenv("DATABRICKS_TOKEN")
    path = tmp_path / ".databrickscfg"
    _write_config(path, host="https://other.example.net")
    with pytest.raises(ProtocolError, match="does not match"):
        create_sdk_config(_source(auth_profile="db"), os.fspath(path))


def test_auth_profile_cannot_be_empty():
    with pytest.raises(ProtocolError, match="auth_profile"):
        create_sdk_config(_source(auth_profile=""), None)


def _new_unversioned_control():
    control = object.__new__(DatabricksControl)
    control._versions = {}
    control._connection = None
    control._connection_key = None
    control._cursor = None
    control._query_id = None
    control._next_page_id = 0
    control._arrow_schema = None
    control._lz4_compressed = None
    control._result_format = None
    return control


def _source(*, schema="default", auth_profile=None):
    source = {
        "server_hostname": "adb.example.net",
        "warehouse_id": "warehouse",
        "catalog": "main",
        "schema": schema,
        "sql": "SELECT 1",
    }
    if auth_profile is not None:
        source["auth_profile"] = auth_profile
    return source


def test_control_reuses_only_an_exact_idle_connection(monkeypatch):
    from databricks import sql
    from databricks.sql.backend.types import CommandState

    class Cursor:
        active_command_id = None
        query_id = "query"

        def execute_async(self, statement):
            assert statement == "SELECT 1"

        def is_query_pending(self):
            return False

        def get_query_state(self):
            return CommandState.SUCCEEDED

        def close(self):
            pass

        def cancel(self):
            pass

    class Connection:
        def __init__(self):
            self.cursor_count = 0
            self.close_count = 0

        def cursor(self):
            self.cursor_count += 1
            return Cursor()

        def close(self):
            self.close_count += 1

    connections = []

    def connect(**kwargs):
        connection = Connection()
        connections.append((connection, kwargs))
        return connection

    monkeypatch.setattr(sql, "connect", connect)
    control = _new_unversioned_control()
    control.open(_source())
    control.close()
    control.open(_source())
    control.close()
    assert len(connections) == 1
    assert connections[0][0].cursor_count == 2
    assert connections[0][1]["_socket_timeout"] == CONTROL_SOCKET_TIMEOUT_SECONDS
    credentials_provider = connections[0][1]["credentials_provider"]
    assert credentials_provider()()["Authorization"] == "Bearer test-token"

    control.open(_source(schema="other"))
    assert len(connections) == 2
    assert connections[0][0].close_count == 1
    control.close(cancel=True)
    assert connections[1][0].close_count == 1
    assert control._connection is None


@pytest.mark.parametrize("failure", ["cancel", "cursor_close", "connection_close"])
def test_cancel_cleanup_closes_and_forgets_connection_after_errors(failure):
    events = []

    class Cursor:
        active_command_id = object()

        def cancel(self):
            events.append("cancel")
            if failure == "cancel":
                raise RuntimeError(failure)

        def close(self):
            events.append("cursor_close")
            if failure == "cursor_close":
                raise RuntimeError(failure)

    class Connection:
        def close(self):
            events.append("connection_close")
            if failure == "connection_close":
                raise RuntimeError(failure)

    control = _new_unversioned_control()
    control._connection = Connection()
    control._connection_key = ("cached",)
    control._cursor = Cursor()

    with pytest.raises(RuntimeError, match=failure):
        control.close(cancel=True)

    assert events == ["cancel", "cursor_close", "connection_close"]
    assert control._cursor is None
    assert control._connection is None
    assert control._connection_key is None


def test_normal_close_error_discards_connection():
    events = []

    class Backend:
        def close_command(self, _command_id):
            events.append("close_command")
            raise RuntimeError("close_command")

    class Cursor:
        active_command_id = object()
        backend = Backend()

        def close(self):
            events.append("cursor_close")

    class Connection:
        def close(self):
            events.append("connection_close")

    control = _new_unversioned_control()
    control._connection = Connection()
    control._connection_key = ("cached",)
    control._cursor = Cursor()

    with pytest.raises(RuntimeError, match="close_command"):
        control.close()

    assert events == ["close_command", "cursor_close", "connection_close"]
    assert control._connection is None
    assert control._connection_key is None


def test_schema_uses_limit_zero_and_reuses_the_connection(monkeypatch):
    from databricks import sql
    from databricks.sql.backend.types import CommandState

    executed = []

    class Cursor:
        active_command_id = None
        query_id = "query"

        def execute_async(self, statement):
            executed.append(statement)

        def is_query_pending(self):
            return False

        def get_query_state(self):
            return CommandState.SUCCEEDED

        def close(self):
            pass

        def cancel(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    connections = []

    def connect(**_kwargs):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(sql, "connect", connect)
    control = _new_unversioned_control()
    source = _source()
    source["sql"] = "SELECT id FROM customers;"
    page_requests = []

    def next_page(**budgets):
        page_requests.append(budgets)
        return {"arrow_schema": base64.b64encode(b"ipc-schema").decode("ascii")}

    monkeypatch.setattr(control, "next_page", next_page)

    result = control.schema(source)
    assert result == {"arrow_schema": base64.b64encode(b"ipc-schema").decode("ascii")}
    assert page_requests == [
        {
            "max_rows": 1,
            "max_bytes": 1,
            "max_links": 1,
            "include_arrow_schema": True,
        }
    ]
    assert executed == [
        "SELECT * FROM (SELECT id FROM customers) AS __diskgraph_schema LIMIT 0"
    ]
    assert control._cursor is None

    control.open(source)
    assert len(connections) == 1
    assert executed[-1] == "SELECT id FROM customers;"
    control.close()


class FakeControl:
    def __init__(self):
        self.calls = []

    def hello(self):
        self.calls.append(("hello",))
        return {"protocol": 1}

    def open(self, source, config_file):
        self.calls.append(("open", source, config_file))
        return {"query_id": "query"}

    def schema(self, source, config_file):
        self.calls.append(("schema", source, config_file))
        return {"arrow_schema": "schema"}

    def next_page(self, **budgets):
        self.calls.append(("next", budgets))
        return {"links": []}

    def close(self, *, cancel=False):
        self.calls.append(("close", cancel))
        return {}


def test_protocol_is_serialized_json_lines():
    requests = [
        {"id": 1, "op": "hello"},
        {
            "id": 2,
            "op": "schema",
            "source": {"type": "databricks"},
            "config_file": "/credentials",
        },
        {
            "id": 3,
            "op": "open",
            "source": {"type": "databricks"},
            "config_file": "/credentials",
        },
        {
            "id": 4,
            "op": "next",
            "max_rows": 10,
            "max_bytes": 20,
            "max_links": 2,
        },
        {"id": 5, "op": "close"},
    ]
    stdin = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
    stdout = io.StringIO()
    control = FakeControl()

    serve(stdin, stdout, control)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4, 5]
    assert all(response["ok"] for response in responses)
    assert control.calls == [
        ("hello",),
        ("schema", {"type": "databricks"}, "/credentials"),
        ("open", {"type": "databricks"}, "/credentials"),
        ("next", {"max_rows": 10, "max_bytes": 20, "max_links": 2}),
        ("close", False),
    ]


def test_protocol_error_does_not_echo_secret_or_exception_text():
    class FailingControl(FakeControl):
        def open(self, source, config_file):
            raise RuntimeError("secret-value https://signed.example/?token=secret")

    stdin = io.StringIO(
        json.dumps(
            {
                "id": 1,
                "op": "open",
                "source": {"token": "secret-value"},
            }
        )
        + "\n"
    )
    stdout = io.StringIO()
    serve(stdin, stdout, FailingControl())
    encoded = stdout.getvalue()
    assert "secret-value" not in encoded
    assert "signed.example" not in encoded
    response = json.loads(encoded)
    assert not response["ok"]
    assert set(response) == {"id", "ok"}


def test_protocol_rejects_out_of_order_request_ids():
    stdin = io.StringIO(json.dumps({"id": 2, "op": "hello"}) + "\n")
    stdout = io.StringIO()
    serve(stdin, stdout, FakeControl())
    response = json.loads(stdout.getvalue())
    assert response["id"] == 2
    assert not response["ok"]


def _control_with_raw_page(response):
    calls = []

    class CommandId:
        def to_thrift_handle(self):
            return SimpleNamespace(
                operationId=SimpleNamespace(guid=b"0" * 16, secret=b"1" * 16),
                operationType=0,
                hasResultSet=True,
                modifiedRowCount=None,
            )

    class Backend:
        _client = SimpleNamespace(FetchResults=object())

        def make_request(self, method, request, retryable=True):
            calls.append((method, request, retryable))
            return response

    control = _new_unversioned_control()
    control._connection = SimpleNamespace(
        session=SimpleNamespace(host="adb.example.net")
    )
    control._cursor = SimpleNamespace(
        active_command_id=CommandId(), backend=Backend()
    )
    control._query_id = "query"
    return control, calls


def test_raw_page_fetch_is_single_attempt_and_preserves_link_metadata():
    from databricks.sql.thrift_api.TCLIService import ttypes

    link = ttypes.TSparkArrowResultLink(
        fileLink="https://signed.example/object?token=secret",
        expiryTime=123,
        startRowOffset=0,
        rowCount=7,
        bytesNum=99,
        httpHeaders={"x-ms-encryption-key": "opaque"},
    )
    response = SimpleNamespace(
        results=SimpleNamespace(
            resultLinks=[link],
            arrowBatches=[],
            columns=None,
            startRowOffset=0,
        ),
        resultSetMetadata=SimpleNamespace(
            arrowSchema=b"arrow-schema",
            schema=None,
            lz4Compressed=True,
            resultFormat=ttypes.TSparkRowSetType.URL_BASED_SET,
        ),
        hasMoreRows=False,
    )
    control, calls = _control_with_raw_page(response)

    page = control.next_page(max_rows=10, max_bytes=100, max_links=2)

    assert len(calls) == 1
    assert calls[0][2] is False
    assert calls[0][1].operationHandle.hasResultSet is True
    assert page["query_id"] == "query"
    assert page["page_id"] == 0
    assert page["links"] == [
        {
            "url": "https://signed.example/object?token=secret",
            "start_row_offset": 0,
            "row_count": 7,
            "decoded_bytes": 99,
            "headers": {"x-ms-encryption-key": "opaque"},
        }
    ]
    assert "result_format" not in page
    assert "arrow_schema" not in page


def test_raw_page_rejects_column_result_format():
    from databricks.sql.thrift_api.TCLIService import ttypes

    response = SimpleNamespace(
        results=SimpleNamespace(
            resultLinks=[], arrowBatches=[], columns=[object()], startRowOffset=0
        ),
        resultSetMetadata=SimpleNamespace(
            arrowSchema=b"arrow-schema",
            schema=None,
            lz4Compressed=False,
            resultFormat=ttypes.TSparkRowSetType.COLUMN_BASED_SET,
        ),
        hasMoreRows=False,
    )
    control, calls = _control_with_raw_page(response)

    with pytest.raises(ProtocolError, match="unsupported result format"):
        control.next_page(max_rows=10, max_bytes=100, max_links=2)
    assert len(calls) == 1
