# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Databricks artifact export may land, and what it stages through.

The write direction's counterpart to :mod:`sql`: the rules a *destination* has
to satisfy, stated once so the hosted connector and the SDK's local sink
enforce the same ones. A table the cloud refuses is refused locally, and both
stage through the one volume an administrator has granted.

``ArtifactValidationError`` rather than a platform error type, because
neither caller shares one; each translates at its own boundary.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Callable
from uuid import uuid4

from kumo_databricks.errors import ArtifactValidationError
from kumo_databricks.sql import qualified_name

#: A Databricks output carries nowhere but the table field to put its schema,
#: so one string names both, joined by this.
SEPARATOR = "__kumo__"

#: One managed staging volume per schema. Fixed rather than configurable: it
#: is what an administrator grants on, and a write that chose its own would
#: land outside that grant.
STAGING_VOLUME = "_kumo_staging"

#: Everything an artifact export stages sits under this prefix, so a cleanup
#: that misses a file cannot be confused with the volume's other contents.
STAGING_PREFIX = "artifact-export"

#: What a permission failure says, in the two spellings Databricks uses.
_DENIED = ("PERMISSION_DENIED", "INSUFFICIENT_PRIVILEGES")

#: What a volume that is not there says, in the four spellings Databricks
#: uses. Anything else is the warehouse's own failure and is re-raised.
_MISSING = ("VOLUME_NOT_FOUND", "VOLUME_DOES_NOT_EXIST", "UC_VOLUME_NOT_FOUND",
            "NO_SUCH_VOLUME")


def split_output_table(name: str) -> tuple[str, str]:
    """Reads the schema and table an output names."""
    parts = name.split(SEPARATOR)
    if len(parts) != 2 or not all(parts):
        raise ArtifactValidationError(
            "Databricks output table must contain exactly one schema and one "
            "table name.")
    return parts[0], parts[1]


def within_schema(name: str, schema: str) -> tuple[str, str]:
    """Requires an output to name a table in the connector's own schema.

    The warehouse counterpart of :func:`kumo_core.sources.layout.within_root`:
    a connector reads from one catalog and schema, and writes to the same one,
    so a destination naming another is refused rather than written.
    """
    schema_name, table_name = split_output_table(name)
    if not schema or schema_name.casefold() != schema.casefold():
        raise ArtifactValidationError(
            "Databricks output table schema does not match the connector "
            "schema.")
    return schema_name, table_name


def staging_stage(catalog: str, schema: str,
                  export_id: str | None = None) -> str:
    """Returns the volume directory one export stages into.

    Derived from ``export_id`` where there is one, so a retry of the same
    export reuses its directory rather than leaving the first attempt's files
    behind; random otherwise.
    """
    stage_id = (sha256(export_id.encode("utf-8")).hexdigest()
                if export_id is not None else uuid4().hex)
    return (f"/Volumes/{catalog}/{schema}/{STAGING_VOLUME}/{STAGING_PREFIX}/"
            f"{stage_id}")


def ensure_staging_volume(execute: Callable[[str], Any], catalog: str,
                          schema: str) -> None:
    """Ensures the one managed staging volume this schema writes through.

    Described before it is created, because the identity that may *use* an
    existing volume often may not create one, and a bare
    ``CREATE VOLUME IF NOT EXISTS`` would refuse it on a volume already there.
    The two failures are told apart for that reason: one is a missing grant,
    the other a volume an administrator has to make.
    """
    coordinate = qualified_name(catalog, schema, STAGING_VOLUME)
    try:
        execute(f"DESCRIBE VOLUME {coordinate}")
        return
    except Exception as error:
        reported = str(error).upper()
        if not _missing(reported):
            if _denied(reported):
                raise ArtifactValidationError(
                    f"The Databricks connector cannot access its managed "
                    f"staging volume {catalog}.{schema}.{STAGING_VOLUME}. "
                    f"Grant USE CATALOG, USE SCHEMA, READ VOLUME, and WRITE "
                    f"VOLUME to the connector identity.") from None
            raise
    try:
        execute(f"CREATE VOLUME IF NOT EXISTS {coordinate}")
    except Exception as error:
        if _denied(str(error).upper()):
            raise ArtifactValidationError(
                f"The Databricks managed staging volume does not exist and "
                f"the connector identity cannot create it. Ask a Databricks "
                f"administrator to create {catalog}.{schema}."
                f"{STAGING_VOLUME}, then grant USE CATALOG, USE SCHEMA, READ "
                f"VOLUME, and WRITE VOLUME to the connector identity."
            ) from None
        raise


def _missing(reported: str) -> bool:
    return any(marker in reported
               for marker in _MISSING) or ("VOLUME" in reported
                                           and "DOES NOT EXIST" in reported)


def _denied(reported: str) -> bool:
    return any(marker in reported for marker in _DENIED)
