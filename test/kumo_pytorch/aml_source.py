# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create-only AML schema/type mappings from exact accepted source files.

The mapping preserves source identities, signed decimal values, nulls and UTC
clock interpretation. Optional root selection retains every physical transaction
at every time touching a selected root, plus endpoint accounts. Those endpoint
accounts do not acquire complete histories. This is a bounded execution view,
not a new accepted dataset or a frozen benchmark input specification.

Inputs and outputs are content-addressed in a completion manifest. An incomplete
output directory is retained on failure and is never reused or overwritten.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VARIANTS = frozenset({"hi-small", "hi-medium", "hi-large", "li-small", "li-medium", "li-large"})
MONEY_TYPE = pa.decimal128(38, 10)
TIME_TYPE = pa.timestamp("ms", tz="UTC")
ACCOUNTS_SCHEMA = pa.schema([
    pa.field("account_id", pa.int64(), nullable=False),
    pa.field("entity_id", pa.string(), nullable=False),
    pa.field("first_observed_time", TIME_TYPE, nullable=False),
])
_VALUES = [
    pa.field("amount_paid", MONEY_TYPE), pa.field("payment_currency", pa.string()),
    pa.field("amount_received", MONEY_TYPE), pa.field("receiving_currency", pa.string()),
    pa.field("payment_format", pa.string()),
]
TRANSACTIONS_SCHEMA = pa.schema([
    pa.field("transaction_id", pa.int64(), nullable=False),
    pa.field("event_time", TIME_TYPE, nullable=False),
    pa.field("event_time_seconds", pa.int64(), nullable=False),
    pa.field("sender_id", pa.int64(), nullable=False),
    pa.field("receiver_id", pa.int64(), nullable=False), *_VALUES,
])
ACCOUNT_EVENTS_SCHEMA = pa.schema([
    pa.field("event_id", pa.int64(), nullable=False),
    pa.field("account_id", pa.int64(), nullable=False),
    pa.field("transaction_id", pa.int64(), nullable=False),
    pa.field("event_time", TIME_TYPE, nullable=False),
    pa.field("event_time_seconds", pa.int64(), nullable=False),
    pa.field("source_row", pa.int64(), nullable=False),
    pa.field("direction", pa.string(), nullable=False), *_VALUES,
])
SOURCE_FIELDS = (
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format",
)
TABLE_SCHEMAS = {"accounts": ACCOUNTS_SCHEMA, "transactions": TRANSACTIONS_SCHEMA,
                 "account_events": ACCOUNT_EVENTS_SCHEMA}


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if "context" in candidate.parts or "context" in candidate.resolve().parts:
        raise ValueError("context/ reference material cannot be a source or destination")
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError(f"Symlinks are not accepted: {candidate}")
    return candidate


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _expected_sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected checksums must be lowercase hexadecimal SHA-256 values")


def _bank(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value.strip()) is None:
        raise ValueError("Bank IDs must be decimal digit strings")
    number = int(value.strip())
    if number > 2**127 - 1:
        raise ValueError("Bank ID exceeds the accepted HUGEINT key range")
    return str(number)


def _account_id(variant: str, bank: Any, account: Any) -> str:
    if not isinstance(account, str) or account == "":
        raise ValueError("Account numbers must be nonempty strings")
    return json.dumps([variant, "account", _bank(bank), account], ensure_ascii=False, separators=(",", ":"))


def _binding_id(value: Any, variant: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Binding entity_id must be a string")
    try:
        parts = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed account binding identity") from exc
    if not isinstance(parts, list) or len(parts) != 4 or parts[:2] != [variant, "account"]:
        raise ValueError("Account binding identity has the wrong variant or shape")
    if value != _account_id(variant, parts[2], parts[3]):
        raise ValueError("Account binding identity is not in the accepted canonical encoding")
    return value


def _clock(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.microsecond or getattr(value, "nanosecond", 0):
        raise ValueError("Account availability must be a present whole-second timestamp")
    value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    pa.scalar(value, type=TIME_TYPE)
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}", value) is None:
        raise ValueError("Timestamp must match %Y/%m/%d %H:%M exactly")
    try:
        result = datetime.strptime(value, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
        pa.scalar(result, type=TIME_TYPE)
        return result
    except (ValueError, OverflowError, pa.ArrowException) as exc:
        raise ValueError("Invalid or out-of-range timestamp") from exc


def _amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]{1,10})?", value) is None:
        raise ValueError("Amount must be exact finite decimal text with at most ten fractional digits, or null")
    try:
        with localcontext() as context:
            context.prec = max(50, len(value) + 10)
            result = Decimal(value).quantize(Decimal("0.0000000001"))
        if len(result.as_tuple().digits) > 38:
            raise ValueError("Amount exceeds Decimal128(38,10)")
        return result
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal amount") from exc


def _category(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise ValueError("Currency and payment format must be nonempty strings or null")
    return value


def _require_strings(schema: pa.Schema, fields: tuple[str, ...]) -> None:
    for name in fields:
        if schema.names.count(name) != 1 or schema.field(name).type != pa.string():
            raise ValueError(f"Expected one source STRING field named {name}")


def _load_binding(path: Path, variant: str, batch_size: int) -> dict[str, tuple[int, datetime]]:
    binding = pq.ParquetFile(path)
    _require_strings(binding.schema_arrow, ("entity_id",))
    if binding.schema_arrow.names.count("first_observed_time") != 1 or not pa.types.is_timestamp(binding.schema_arrow.field("first_observed_time").type):
        raise ValueError("Accepted account binding requires first_observed_time timestamp metadata")
    members = {}
    for batch in binding.iter_batches(batch_size=batch_size, columns=["entity_id", "first_observed_time"], use_threads=False):
        try:
            clocks = pc.cast(batch.column(1), pa.timestamp("s", tz="UTC"), safe=True)
        except pa.ArrowInvalid as exc:
            raise ValueError("Account availability must be a present whole-second timestamp") from exc
        batch = batch.set_column(1, "first_observed_time", clocks)
        for row in batch.to_pylist():
            identity = _binding_id(row["entity_id"], variant)
            if identity in members:
                raise ValueError("Duplicate account binding identity")
            members[identity] = _clock(row["first_observed_time"])
    return {identity: (rank, members[identity]) for rank, identity in enumerate(sorted(members, key=lambda key: key.encode("utf-8")))}


def load_account_id_map(accounts_binding_path: str | Path, variant: str, *, expected_sha256: str) -> dict[str, int]:
    """Read the exact accepted binding; rank its canonical UTF-8 identities.

    Ranks are zero-based and use the full binding, independent of input row order,
    transaction batching or a later root selection. The source checksum scopes
    these internal integer identities; canonical strings remain authoritative.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported AML variant: {variant}")
    _expected_sha256(expected_sha256)
    path = _safe_path(accounts_binding_path)
    if _sha256(path) != expected_sha256:
        raise ValueError("Source checksum mismatch for accounts_binding")
    result = {identity: item[0] for identity, item in _load_binding(path, variant, 16384).items()}
    if _sha256(path) != expected_sha256:
        raise ValueError("Source changed while reading accounts_binding")
    return result


def _table_record(path: Path, rows: int, schema: pa.Schema) -> dict[str, Any]:
    footer = pq.ParquetFile(path)
    if footer.metadata.num_rows != rows or not footer.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f"Written Parquet metadata disagrees with prepared rows/schema: {path}")
    return {"filename": path.name, "rows": rows, "bytes": path.stat().st_size, "sha256": _sha256(path),
            "schema": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema]}


def prepare_aml_sources(
    transactions_path: str | Path, accounts_binding_path: str | Path,
    output_dir: str | Path, variant: str, *, expected_checksums: Mapping[str, str],
    batch_size: int = 16384, root_entity_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Stream a complete source mapping or an explicitly bounded root probe view.

    ``root_entity_ids`` selects accounts, never time windows. All source rows are
    scanned, and every transaction touching a selected root is retained at every
    source time. Counterparty accounts are present for topology, but their other
    transactions are outside this view. Memory is O(accepted accounts + batch).
    Labels are neither read nor projected. Exact Decimal128 values are retained;
    numerical conversion inside Kumo encoders is recorded separately by execution.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported AML variant: {variant}")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if set(expected_checksums) != {"transactions", "accounts_binding"}:
        raise ValueError("Both exact source checksums are required")
    for digest in expected_checksums.values():
        _expected_sha256(digest)
    paths = {"transactions": _safe_path(transactions_path), "accounts_binding": _safe_path(accounts_binding_path)}
    destination = _safe_path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory is immutable and already exists: {destination}")
    sources = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected_checksums[role]:
            raise ValueError(f"Source checksum mismatch for {role}")
        sources[role] = {"path": str(path), "bytes": path.stat().st_size, "sha256": expected_checksums[role]}
    source = pq.ParquetFile(paths["transactions"])
    _require_strings(source.schema_arrow, SOURCE_FIELDS)
    if source.metadata.num_rows > 2**62:
        raise ValueError("Physical transaction rows exceed int64 event identity capacity")
    members = _load_binding(paths["accounts_binding"], variant, batch_size)
    if isinstance(root_entity_ids, (str, bytes)):
        raise TypeError("root_entity_ids must be an iterable of canonical account identities")
    roots = None if root_entity_ids is None else {_binding_id(value, variant) for value in root_entity_ids}
    if roots is not None and (not roots or roots - members.keys()):
        raise ValueError("Root selection must be nonempty and contained in the accepted account binding")
    root_numbers = None if roots is None else pa.array(sorted({json.loads(key)[3] for key in roots}), type=pa.string())
    included_accounts = set(members) if roots is None else set(roots)
    destination.mkdir(parents=True, exist_ok=False)
    accounts_path, transactions_output, events_path = (destination / f"{name}.parquet" for name in TABLE_SCHEMAS)
    scanned_rows = transaction_rows = event_rows = self_transfers = 0
    with (pq.ParquetWriter(transactions_output, TRANSACTIONS_SCHEMA, compression="zstd") as tx_writer,
          pq.ParquetWriter(events_path, ACCOUNT_EVENTS_SCHEMA, compression="zstd") as events_writer):
        for batch in source.iter_batches(batch_size=batch_size, columns=list(SOURCE_FIELDS), use_threads=False):
            if roots is None:
                indices, candidates = range(batch.num_rows), batch
            else:
                mask = pc.or_(pc.is_in(batch.column(batch.schema.get_field_index("from_account")), value_set=root_numbers),
                              pc.is_in(batch.column(batch.schema.get_field_index("to_account")), value_set=root_numbers))
                indices = pc.indices_nonzero(mask).to_pylist()
                candidates = batch.take(pa.array(indices, type=pa.int64()))
            tx_records, event_records = [], []
            for position, row in zip(indices, candidates.to_pylist(), strict=True):
                source_row = scanned_rows + position
                try:
                    sender_key = _account_id(variant, row["from_bank"], row["from_account"])
                    receiver_key = _account_id(variant, row["to_bank"], row["to_account"])
                    if roots is not None and not ({sender_key, receiver_key} & roots):
                        continue
                    if sender_key not in members or receiver_key not in members:
                        raise ValueError("Transaction endpoint is absent from the accepted account binding")
                    sender, receiver = members[sender_key][0], members[receiver_key][0]
                    time = _timestamp(row["timestamp"])
                    if time < members[sender_key][1] or time < members[receiver_key][1]:
                        raise ValueError("Accepted account availability is after a touching transaction")
                    stamp = int(time.timestamp())
                    values = {name: (_amount if name in {"amount_paid", "amount_received"} else _category)(row[name])
                              for name in ("amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format")}
                    tx_records.append({"transaction_id": source_row, "event_time": time, "event_time_seconds": stamp,
                                       "sender_id": sender, "receiver_id": receiver, **values})
                    touches = [(sender, "self")] if sender == receiver else [(sender, "outgoing"), (receiver, "incoming")]
                    for role_index, (account, direction) in enumerate(touches):
                        event_records.append({"event_id": 2 * source_row + role_index, "account_id": account,
                                              "transaction_id": source_row, "event_time": time, "event_time_seconds": stamp,
                                              "source_row": source_row, "direction": direction, **values})
                    included_accounts.update((sender_key, receiver_key))
                    self_transfers += int(sender == receiver)
                    transaction_rows += 1
                    event_rows += len(touches)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"Invalid transaction at physical source row {source_row}: {exc}") from exc
            if tx_records:
                tx_writer.write_table(pa.Table.from_pylist(tx_records, schema=TRANSACTIONS_SCHEMA))
                events_writer.write_table(pa.Table.from_pylist(event_records, schema=ACCOUNT_EVENTS_SCHEMA))
            scanned_rows += batch.num_rows
    with pq.ParquetWriter(accounts_path, ACCOUNTS_SCHEMA, compression="zstd") as writer:
        rows = []
        for identity, (rank, availability) in members.items():
            if identity not in included_accounts:
                continue
            rows.append({"account_id": rank, "entity_id": identity, "first_observed_time": availability})
            if len(rows) == batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=ACCOUNTS_SCHEMA))
                rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=ACCOUNTS_SCHEMA))
    if scanned_rows != source.metadata.num_rows or event_rows != 2 * transaction_rows - self_transfers:
        raise ValueError("Prepared row counts disagree with source scanning/one-touch rules")
    for role, path in paths.items():
        if _sha256(path) != expected_checksums[role]:
            raise ValueError(f"Source changed during preparation: {role}")
    manifest = {
        "schema_version": 2, "status": "complete", "input_contract_status": "execution_mapping_not_frozen_specification",
        "dataset_id": f"aml-data/{variant}", "dataset_version": "v8", "variant": variant, "view_version": "mapped-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "tables": {name: _table_record(destination / f"{name}.parquet", count, TABLE_SCHEMAS[name])
                   for name, count in (("accounts", len(included_accounts)), ("transactions", transaction_rows), ("account_events", event_rows))},
        "scope": {"kind": "full_source" if roots is None else "selected_root_full_histories",
                  "complete_root_histories": True, "complete_history_root_entity_ids": None if roots is None else sorted(roots),
                  "counterparty_scope": "all_source_histories" if roots is None else "endpoint_accounts_and_selected_root_transactions_only",
                  "source_transactions_scanned": scanned_rows, "accepted_accounts": len(members),
                  "time_filter": None, "cutoff_filter": None},
        "semantics": {
            "history": "Consumers must select event_time < anchor.as_of_time; selected roots retain all source times.",
            "account_key": "Zero-based UTF-8 canonical identity sort rank across the full checksum-pinned accepted account binding; entity_id preserves the exact canonical string.",
            "availability": "Accepted first_observed_time interpreted as UTC; sampler metadata only, not a predictive feature.",
            "transaction_key": "Zero-based immutable source physical row; unique within the checksum-pinned variant.",
            "event_key": "2 * source physical row + role index; outgoing/self 0, incoming 1.",
            "event_order": ["event_time", "source_row"], "self_transfers": "One account event with direction self; both transaction role relationships retained.",
            "timestamp": "Source minute timestamp interpreted as UTC; timestamp[ms, UTC] with whole-second values plus explicit integer epoch seconds. Source timezone is unspecified.",
            "amounts": "Signed exact nullable Decimal128(38,10); nonfinite/malformed/overflow/excess fractional precision rejected. Kumo numerical encoders perform their native floating-point conversion.",
            "missing": "Null remains null. PRAGMA omits missing fields; nuFormer missing_event=error is enforced by its unchanged encoder. No MISSING token or replacement value is introduced here.",
            "labels": "Source labels not read or copied; anchor targets, splits and validity stay external and unchanged.",
            "feature_exclusions": "Identity fields and availability are join/sampler metadata. No graph calendar or sender/receiver indicator columns added.",
        },
        "self_transfers": self_transfers,
        "provenance": {"module_sha256": _sha256(Path(__file__)), "python": platform.python_version(),
                       "pyarrow": pa.__version__, "batch_size": batch_size, "compression": "zstd"},
    }
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def aml_data_schema():
    """The mapped table schema accepted by the shared AML contract builders."""
    from kumo.pytorch.api import DataSchema, RelationSpec, TableSpec

    from .contracts import schema_columns
    keys = {"accounts": "account_id", "transactions": "transaction_id", "account_events": "event_id"}
    return DataSchema(
        tables=tuple(TableSpec(name, keys[name], schema_columns(schema),
                               time_column="first_observed_time" if name == "accounts" else "event_time")
                     for name, schema in TABLE_SCHEMAS.items()),
        relations=(RelationSpec("account_touches", "accounts", "account_id", "account_events", "account_id"),
                   RelationSpec("sender", "accounts", "account_id", "transactions", "sender_id"),
                   RelationSpec("receiver", "accounts", "account_id", "transactions", "receiver_id")),
    )


def _prepared_manifest(output_dir: str | Path) -> tuple[Path, dict]:
    directory = _safe_path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete" or set(manifest.get("tables", {})) != set(TABLE_SCHEMAS):
        raise ValueError("Expected a complete mapped AML source manifest")
    for name, schema in TABLE_SCHEMAS.items():
        record = manifest["tables"][name]
        if record["filename"] != f"{name}.parquet":
            raise ValueError("Mapped table filename is not canonical")
        path = _safe_path(directory / record["filename"])
        if _sha256(path) != record["sha256"] or _table_record(path, record["rows"], schema) != record:
            raise ValueError(f"Prepared table content or metadata changed: {name}")
    return directory, manifest


def prepared_dataset(output_dir: str | Path):
    """Verify mapped files and expose native Kumo Parquet sources, without copying."""
    from kumo.pytorch.api import Dataset, Parquet
    directory, manifest = _prepared_manifest(output_dir)
    return Dataset({name: Parquet(str(directory / record["filename"])) for name, record in manifest["tables"].items()})


def account_identity_map(output_dir: str | Path) -> dict[str, int]:
    """Verify the view and expose its reversible canonical-to-integer identities."""
    directory, _ = _prepared_manifest(output_dir)
    return {row["entity_id"]: row["account_id"] for row in pq.read_table(directory / "accounts.parquet").to_pylist()}


def normalize_sources(data_dir, directory, definition, files, roots):
    """Convert the declared raw source format; campaign/anchor logic lives elsewhere."""
    options = definition["source_options"]
    tx, accounts = options["transactions"], options["accounts_binding"]
    if directory.exists():
        _, mapping = _prepared_manifest(directory)
        expected_roots = None if roots is None else sorted(roots)
        if (
            mapping["variant"] != options["variant"]
            or mapping["scope"]["complete_history_root_entity_ids"] != expected_roots
        ):
            raise ValueError(
                "Prepared source belongs to another dataset or root population"
            )
        for role, key in (("transactions", tx), ("accounts_binding", accounts)):
            if mapping["sources"][role]["sha256"] != files[key]["sha256"]:
                raise ValueError("Prepared source checksum changed")
        return prepared_dataset(directory), mapping
    mapping = prepare_aml_sources(
        data_dir / tx,
        data_dir / accounts,
        directory,
        options["variant"],
        expected_checksums={
            "transactions": files[tx]["sha256"],
            "accounts_binding": files[accounts]["sha256"],
        },
        root_entity_ids=roots,
    )
    return prepared_dataset(directory), mapping
