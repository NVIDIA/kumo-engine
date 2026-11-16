# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from io_utils import kumo_dataclass
from statedb.config.common import JobID, JobType
from statedb.config.job_tag import JobTagKey, JobTags, JobTagValue
from statedb.config.timestamp import (
    ResourceTimestamps,
    get_current_timestamps,
    update_timestamps,
)
from statedb.db.repositories import (
    JsonRepository,
    _in_predicate,
    _json_value,
    _placeholder,
)
from statedb.db.row_parsing import coerce_datetime, row_field
from statedb.db.runtime import StateTx
from statedb.db.schema import validate_identifier


@kumo_dataclass
class JobTagsRecord:
    job_type: JobType
    job_id: JobID
    tags: JobTags


@kumo_dataclass
class NormalizedJobTagRecord:
    job_type: JobType
    job_id: JobID
    key: str
    value: str
    timestamps: ResourceTimestamps


@dataclass(frozen=True)
class _ParsedNormalizedJobTagRecord:
    job_id: JobID
    key: JobTagKey | None
    value: JobTagValue | None
    timestamps: ResourceTimestamps


@dataclass(frozen=True)
class _NormalizedFilterSQL:
    clause: str
    args: list[Any]


_EMPTY_TAG_ROW_KEY = ""
_EMPTY_TAG_ROW_VALUE = ""


def coerce_job_type(job_type: JobType | str) -> JobType:
    return job_type if isinstance(job_type, JobType) else JobType(job_type)


def job_tags_row_id(job_type: JobType | str, job_id: JobID) -> str:
    return f"{coerce_job_type(job_type).value}:{job_id}"


def job_tag_row_id(
    job_type: JobType | str,
    job_id: JobID,
    key: JobTagKey | str,
) -> str:
    return f"{job_tags_row_id(job_type, job_id)}:{JobTagKey(str(key))}"


def _json_value_from(tx: StateTx, alias: str, field_name: str) -> str:
    validate_identifier(alias)
    validate_identifier(field_name)
    if tx.backend == "postgres":
        return f"{alias}.payload->>'{field_name}'"
    return f"json_extract({alias}.payload, '$.{field_name}')"


def _json_job_tag_value(
    tx: StateTx,
    key: str,
    *,
    alias: str | None = None,
) -> str:
    JobTagKey(key)
    payload = "payload" if alias is None else f"{alias}.payload"
    if tx.backend == "postgres":
        return f"{payload} #>> '{{tags,tags,{key}}}'"
    return f"json_extract({payload}, '$.tags.tags.\"{key}\"')"


def _timestamps_from_normalized_payload(
    payload: Mapping[str, object],
    row: object,
) -> ResourceTimestamps:
    raw_timestamps = payload.get("timestamps")
    if isinstance(raw_timestamps, Mapping):
        created_at = raw_timestamps.get("created_at")
        updated_at = raw_timestamps.get("updated_at")
        if created_at is not None and updated_at is not None:
            return ResourceTimestamps(
                created_at=coerce_datetime(created_at),
                updated_at=coerce_datetime(updated_at),
            )
    return ResourceTimestamps(
        created_at=coerce_datetime(row_field(row, "created_at", 2)),
        updated_at=coerce_datetime(row_field(row, "updated_at", 3)),
    )


def _normalized_row_from_list_row(row: object) -> _ParsedNormalizedJobTagRecord:
    payload = json.loads(str(row_field(row, "payload", 1)))
    if not isinstance(payload, dict):
        raise ValueError("Normalized job tag payload must be an object")
    key = str(payload["key"])
    timestamps = _timestamps_from_normalized_payload(payload, row)
    if key == _EMPTY_TAG_ROW_KEY:
        return _ParsedNormalizedJobTagRecord(
            job_id=JobID(str(payload["job_id"])),
            key=None,
            value=None,
            timestamps=timestamps,
        )
    return _ParsedNormalizedJobTagRecord(
        job_id=JobID(str(payload["job_id"])),
        key=JobTagKey(key),
        value=JobTagValue(str(payload["value"])),
        timestamps=timestamps,
    )


def _merge_timestamps(
    left: ResourceTimestamps,
    right: ResourceTimestamps,
) -> ResourceTimestamps:
    return ResourceTimestamps(
        created_at=min(left.created_at, right.created_at),
        updated_at=max(left.updated_at, right.updated_at),
    )


class JobTagsRepository:
    def __init__(
        self,
        rows: JsonRepository[JobTagsRecord] | None = None,
    ) -> None:
        self._rows = rows or JsonRepository("job_tags", JobTagsRecord)
        self._normalized_rows = JsonRepository("job_tags",
                                               NormalizedJobTagRecord)

    async def get(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_id: JobID,
    ) -> JobTags | None:
        job_type = coerce_job_type(job_type)
        normalized_tags = await self._get_normalized(tx, job_type, job_id)
        if normalized_tags is not None:
            return normalized_tags

        row = await self._rows.get(tx, job_tags_row_id(job_type, job_id))
        if row is None:
            return None
        return deepcopy(row.payload.tags)

    async def apply(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_id: JobID,
        tags: dict[JobTagKey, JobTagValue],
    ) -> bool:
        job_type = coerce_job_type(job_type)
        current_tags = await self.get(tx, job_type, job_id)
        job_tags = deepcopy(current_tags) if current_tags is not None else (
            JobTags(timestamps=get_current_timestamps()))
        tags_updated = job_tags.update_tags(tags)
        should_normalize_legacy = (
            current_tags is not None
            and await self._get_normalized(tx, job_type, job_id) is None)
        if tags_updated:
            job_tags.timestamps = update_timestamps(job_tags.timestamps)
        if tags_updated or should_normalize_legacy:
            await self._replace_normalized(tx, job_type, job_id, job_tags)
        return tags_updated

    async def delete_tag(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_id: JobID,
        tag_key: JobTagKey,
    ) -> bool:
        job_type = coerce_job_type(job_type)
        job_tags = await self.get(tx, job_type, job_id)
        if job_tags is None or tag_key not in job_tags.tags:
            return False
        job_tags = deepcopy(job_tags)
        job_tags.tags.pop(tag_key)
        job_tags.timestamps = update_timestamps(job_tags.timestamps)
        await self._replace_normalized(tx, job_type, job_id, job_tags)
        return True

    async def delete_all(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_id: JobID,
    ) -> bool:
        job_type = coerce_job_type(job_type)
        existed = await self.get(tx, job_type, job_id) is not None
        if not existed:
            return False
        await self._delete_normalized(tx, job_type, job_id)
        await self._rows.delete(tx, job_tags_row_id(job_type, job_id))
        return True

    async def list_by_filters(
        self,
        tx: StateTx,
        job_type: JobType | str,
        filters: Mapping[str, str],
    ) -> dict[JobID, JobTags]:
        job_type = coerce_job_type(job_type)
        normalized_matches = await self._list_normalized_by_filters(
            tx, job_type, filters)
        aggregate_matches = await self._list_aggregate_by_filters(
            tx, job_type, filters)
        return aggregate_matches | normalized_matches

    async def list_by_ids(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_ids: Iterable[JobID],
    ) -> dict[JobID, JobTags]:
        job_type = coerce_job_type(job_type)
        job_id_values = [str(job_id) for job_id in dict.fromkeys(job_ids)]
        if not job_id_values:
            return {}

        args: list[object] = [job_type.value]
        job_id_clause = _in_predicate(
            tx, _json_value(tx, "job_id"), job_id_values, args
        )
        aggregate_rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._rows._table_name}
            WHERE {_json_value(tx, 'job_type')} = {_placeholder(tx, 1)}
              AND {job_id_clause}
              AND {_json_value(tx, 'key')} IS NULL
            ORDER BY id
            """,
            *args,
        )
        aggregate_tags = {
            row.payload.job_id: deepcopy(row.payload.tags)
            for row in (
                self._rows._row_from_list_row(row) for row in aggregate_rows
            )
        }

        normalized_rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._rows._table_name}
            WHERE {_json_value(tx, 'job_type')} = {_placeholder(tx, 1)}
              AND {job_id_clause}
              AND {_json_value(tx, 'key')} IS NOT NULL
            ORDER BY id
            """,
            *args,
        )
        normalized_tags = self._normalized_rows_by_job_id(normalized_rows)
        return aggregate_tags | normalized_tags

    async def delete_by_ids(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_ids: Iterable[JobID],
    ) -> None:
        job_type = coerce_job_type(job_type)
        job_id_values = [str(job_id) for job_id in dict.fromkeys(job_ids)]
        if not job_id_values:
            return
        args: list[object] = [job_type.value]
        job_id_clause = _in_predicate(
            tx, _json_value(tx, "job_id"), job_id_values, args
        )
        await tx.execute(
            f"""
            DELETE FROM {self._rows._table_name}
            WHERE {_json_value(tx, 'job_type')} = {_placeholder(tx, 1)}
              AND {job_id_clause}
            """,
            *args,
        )

    async def _list_aggregate_by_filters(
        self,
        tx: StateTx,
        job_type: JobType | str,
        filters: Mapping[str, str],
    ) -> dict[JobID, JobTags]:
        job_type = coerce_job_type(job_type)
        where_clauses = [
            f"{_json_value_from(tx, 'legacy', 'job_type')} = "
            f"{_placeholder(tx, 1)}",
            f"{_json_value_from(tx, 'legacy', 'key')} IS NULL",
        ]
        args: list[Any] = [job_type.value]
        for raw_key, raw_value in filters.items():
            tag_value_expr = _json_job_tag_value(
                tx, raw_key, alias="legacy")
            if JobTagValue.null_or_empty(raw_value):
                where_clauses.append(f"{tag_value_expr} IS NULL")
                continue
            args.append(str(raw_value))
            where_clauses.append(
                f"{tag_value_expr} = {_placeholder(tx, len(args))}")
        where_clauses.append(
            self._normalized_absence_clause(
                tx,
                outer_job_id_expr=_json_value_from(tx, 'legacy', 'job_id'),
                job_type_arg_position=len(args) + 1,
            ))
        args.append(job_type.value)
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._rows._table_name} legacy
            WHERE {" AND ".join(where_clauses)}
            ORDER BY id
            """,
            *args,
        )
        return {
            row.payload.job_id: deepcopy(row.payload.tags)
            for row in (self._rows._row_from_list_row(row) for row in rows)
        }

    async def _get_normalized(
        self,
        tx: StateTx,
        job_type: JobType | str,
        job_id: JobID,
    ) -> JobTags | None:
        job_type = coerce_job_type(job_type)
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._rows._table_name}
            WHERE {_json_value(tx, 'job_type')} = {_placeholder(tx, 1)}
              AND {_json_value(tx, 'job_id')} = {_placeholder(tx, 2)}
              AND {_json_value(tx, 'key')} IS NOT NULL
            ORDER BY id
            """,
            job_type.value,
            str(job_id),
        )
        return self._job_tags_from_normalized_rows(rows)

    async def _list_normalized_by_filters(
        self,
        tx: StateTx,
        job_type: JobType,
        filters: Mapping[str, str],
    ) -> dict[JobID, JobTags]:
        job_type = coerce_job_type(job_type)
        filter_sql = self._normalized_filter_sql(tx, filters)
        where_clauses = [
            f"{_json_value_from(tx, 'base', 'job_type')} = {_placeholder(tx, 1)}",
            f"{_json_value_from(tx, 'base', 'key')} IS NOT NULL",
        ]
        if filter_sql.clause:
            where_clauses.append(filter_sql.clause)
        rows = await tx.fetch(
            f"""
            SELECT id, payload, created_at, updated_at
            FROM {self._rows._table_name} base
            WHERE {" AND ".join(where_clauses)}
            ORDER BY id
            """,
            job_type.value,
            *filter_sql.args,
        )
        return self._normalized_rows_by_job_id(rows)

    async def _replace_normalized(
        self,
        tx: StateTx,
        job_type: JobType,
        job_id: JobID,
        job_tags: JobTags,
    ) -> None:
        await self._delete_normalized(tx, job_type, job_id)
        await self._rows.delete(tx, job_tags_row_id(job_type, job_id))
        if not job_tags.tags:
            await self._normalized_rows.upsert(
                tx,
                f"{job_tags_row_id(job_type, job_id)}:",
                NormalizedJobTagRecord(
                    job_type=job_type,
                    job_id=job_id,
                    key=_EMPTY_TAG_ROW_KEY,
                    value=_EMPTY_TAG_ROW_VALUE,
                    timestamps=job_tags.timestamps,
                ),
            )
            return
        for key, value in job_tags.tags.items():
            await self._normalized_rows.upsert(
                tx,
                job_tag_row_id(job_type, job_id, key),
                NormalizedJobTagRecord(
                    job_type=job_type,
                    job_id=job_id,
                    key=str(key),
                    value=str(value),
                    timestamps=job_tags.timestamps,
                ),
            )

    async def _delete_normalized(
        self,
        tx: StateTx,
        job_type: JobType,
        job_id: JobID,
    ) -> None:
        await tx.execute(
            f"""
            DELETE FROM {self._rows._table_name}
            WHERE {_json_value(tx, 'job_type')} = {_placeholder(tx, 1)}
              AND {_json_value(tx, 'job_id')} = {_placeholder(tx, 2)}
              AND {_json_value(tx, 'key')} IS NOT NULL
            """,
            job_type.value,
            str(job_id),
        )

    def _normalized_absence_clause(
        self,
        tx: StateTx,
        *,
        outer_job_id_expr: str,
        job_type_arg_position: int,
    ) -> str:
        return f"""
            NOT EXISTS (
                SELECT 1
                FROM {self._rows._table_name} normalized
                WHERE {_json_value_from(tx, 'normalized', 'job_type')} =
                    {_placeholder(tx, job_type_arg_position)}
                  AND {_json_value_from(tx, 'normalized', 'job_id')} =
                    {outer_job_id_expr}
                  AND {_json_value_from(tx, 'normalized', 'key')} IS NOT NULL
            )
        """

    def _normalized_filter_sql(
        self,
        tx: StateTx,
        filters: Mapping[str, str],
    ) -> _NormalizedFilterSQL:
        clauses: list[str] = []
        args: list[Any] = []
        for index, (raw_key, raw_value) in enumerate(filters.items()):
            alias = f"filter_{index}"
            key = JobTagKey(raw_key)
            args.append(str(key))
            key_placeholder = _placeholder(tx, len(args) + 1)
            if JobTagValue.null_or_empty(raw_value):
                clauses.append(
                    self._normalized_filter_exists_clause(
                        tx,
                        alias,
                        key_placeholder,
                        None,
                        negate=True,
                    ))
                continue
            value = JobTagValue(str(raw_value))
            args.append(str(value))
            value_placeholder = _placeholder(tx, len(args) + 1)
            clauses.append(
                self._normalized_filter_exists_clause(
                    tx,
                    alias,
                    key_placeholder,
                    value_placeholder,
                    negate=False,
                ))
        return _NormalizedFilterSQL(" AND ".join(clauses), args)

    def _normalized_filter_exists_clause(
        self,
        tx: StateTx,
        alias: str,
        key_placeholder: str,
        value_placeholder: str | None,
        *,
        negate: bool,
    ) -> str:
        value_clause = (
            ""
            if value_placeholder is None else
            f"AND {_json_value_from(tx, alias, 'value')} = {value_placeholder}")
        exists_sql = f"""
            EXISTS (
                SELECT 1
                FROM {self._rows._table_name} {alias}
                WHERE {_json_value_from(tx, alias, 'job_type')} =
                    {_json_value_from(tx, 'base', 'job_type')}
                  AND {_json_value_from(tx, alias, 'job_id')} =
                    {_json_value_from(tx, 'base', 'job_id')}
                  AND {_json_value_from(tx, alias, 'key')} = {key_placeholder}
                  {value_clause}
            )
        """
        return f"NOT {exists_sql}" if negate else exists_sql

    def _job_tags_from_normalized_rows(
        self,
        rows: list[object] | tuple[object, ...] | Any,
    ) -> JobTags | None:
        rows_by_job_id = self._normalized_rows_by_job_id(rows)
        if not rows_by_job_id:
            return None
        if len(rows_by_job_id) > 1:
            raise ValueError("Expected normalized rows for one job")
        return next(iter(rows_by_job_id.values()))

    def _normalized_rows_by_job_id(
        self,
        rows: list[object] | tuple[object, ...] | Any,
    ) -> dict[JobID, JobTags]:
        tags_by_job_id: dict[JobID, JobTags] = {}
        for row in rows:
            tag_row = _normalized_row_from_list_row(row)
            if tag_row.job_id not in tags_by_job_id:
                tags_by_job_id[tag_row.job_id] = JobTags(
                    timestamps=tag_row.timestamps,
                )
            job_tags = tags_by_job_id[tag_row.job_id]
            if tag_row.key is None:
                continue
            job_tags.tags[tag_row.key] = tag_row.value
            job_tags.timestamps = _merge_timestamps(
                job_tags.timestamps,
                tag_row.timestamps,
            )
        return {
            job_id: deepcopy(tags)
            for job_id, tags in tags_by_job_id.items()
        }
