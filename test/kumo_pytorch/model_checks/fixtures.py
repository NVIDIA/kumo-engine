"""Tiny synthetic AML capability probes, never substitutes for accepted data.

The touch view preserves one row per (account, physical transaction). Count
features use native relations; no sender/receiver indicator columns are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DAY = 86400
CUTOFF = datetime(2024, 3, 1, tzinfo=timezone.utc)
VALUE_FIELDS = (
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
)


@dataclass(frozen=True)
class AMLFixture:
    tables: dict
    schema: object
    training: tuple

    def memory_dataset(self):
        from kumo.pytorch.api import Dataset, InMemoryRows

        return Dataset({name: InMemoryRows(rows) for name, rows in self.tables.items()})

    def parquet_dataset(self, destination: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from kumo.pytorch.api import Dataset, Parquet

        destination.mkdir(parents=True, exist_ok=False)
        sources = {}
        for name, rows in self.tables.items():
            table = pa.Table.from_pylist(rows)
            for clock in ("event_time", "first_observed_time"):
                if clock in table.column_names:
                    index = table.schema.get_field_index(clock)
                    table = table.set_column(
                        index, clock, table[clock].cast(pa.timestamp("s", tz="UTC"))
                    )
            path = destination / f"{name}.parquet"
            pq.write_table(table, path)
            sources[name] = Parquet(str(path))
        return Dataset(sources)


def build_fixture() -> AMLFixture:
    from kumo.pytorch.api import (
        DataSchema,
        RelationSpec,
        SampleRequest,
        TableSpec,
        TargetValue,
    )

    def transaction(
        identity, sender, receiver, offset, paid, received, currency="USD", form="wire"
    ):
        stamp = CUTOFF + timedelta(seconds=offset)
        return dict(
            transaction_id=identity,
            sender_id=sender,
            receiver_id=receiver,
            event_time=stamp,
            event_time_seconds=int(stamp.timestamp()),
            amount_paid=paid,
            amount_received=received,
            payment_currency=currency,
            receiving_currency=currency,
            payment_format=form,
        )

    transactions = [
        transaction(101, 1, 2, -40 * DAY, -100.0, -99.0),
        transaction(102, 3, 1, -10 * DAY, 20.0, 19.0, "EUR"),
        transaction(103, 1, 1, -7 * DAY, 0.0, 0.0),
        transaction(104, 1, 2, -DAY, 40.0, 39.0),
        transaction(105, 2, 1, -3600, -5.0, -4.0),
        transaction(106, 1, 3, 0, 60.0, 59.0, "JPY", "new_format"),
        transaction(107, 1, 3, 60, 70.0, 69.0, "JPY", "new_format"),
        transaction(108, 2, 4, -1800, 80.0, 79.0),
        transaction(109, 1, 2, -3600, 9.0, 8.0),
        transaction(110, 4, 3, 60, None, 4.0, None, None),
        transaction(111, 1, 2, -1, 12.0, 11.0),
    ]
    first_observed = {}
    touches = []
    for row in transactions:
        sender, receiver = row["sender_id"], row["receiver_id"]
        for account in {sender, receiver}:
            first_observed[account] = min(
                first_observed.get(account, row["event_time"]), row["event_time"]
            )
        roles = (
            [(sender, "self")]
            if sender == receiver
            else [(sender, "outgoing"), (receiver, "incoming")]
        )
        for role_index, (account, direction) in enumerate(roles):
            touches.append(
                {
                    "event_id": row["transaction_id"] * 2 + role_index,
                    "account_id": account,
                    "transaction_id": row["transaction_id"],
                    "event_time": row["event_time"],
                    "event_time_seconds": row["event_time_seconds"],
                    "direction": direction,
                    **{name: row[name] for name in VALUE_FIELDS},
                }
            )
    accounts = [
        {"account_id": account, "first_observed_time": stamp}
        for account, stamp in sorted(first_observed.items())
    ]
    tables = {
        "accounts": accounts,
        "transactions": transactions,
        "account_events": touches,
    }
    from test.kumo_pytorch.contracts import schema_columns
    import pyarrow as pa

    schema = DataSchema(
        tables=tuple(
            TableSpec(
                name,
                {
                    "accounts": "account_id",
                    "transactions": "transaction_id",
                    "account_events": "event_id",
                }[name],
                schema_columns(pa.Table.from_pylist(rows).schema),
                time_column="first_observed_time"
                if name == "accounts"
                else "event_time",
            )
            for name, rows in tables.items()
        ),
        relations=(
            RelationSpec(
                "account_touches",
                "accounts",
                "account_id",
                "account_events",
                "account_id",
            ),
            RelationSpec(
                "sender", "accounts", "account_id", "transactions", "sender_id"
            ),
            RelationSpec(
                "receiver", "accounts", "account_id", "transactions", "receiver_id"
            ),
        ),
    )

    def request(name, account, cutoff, split="test", label=1.0):
        return SampleRequest(
            name, account, cutoff, targets=(TargetValue("label", label),), split=split
        )

    training = (
        request("fit-account-1", 1, CUTOFF, "train", 1.0),
        request("fit-account-2", 2, CUTOFF, "train", 0.0),
    )
    return AMLFixture(tables, schema, training)
