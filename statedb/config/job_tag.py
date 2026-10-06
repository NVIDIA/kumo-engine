# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Define allowed pattern for custom job tag key/value.

import re
from dataclasses import field
from typing import Dict, Mapping, Optional

from kumo_core.types.schema import FPID
from pydantic import field_validator

from common_utils.string import validate_max_length, validate_non_empty
from io_utils import kumo_dataclass
from statedb.config.common import DBEntry, JobID, PredictiveQueryID, ProjectID


class JobTagKey(str):
    """Job tag key may be at most 64 characters long, and may only contain
    alphanumeric, dot, underscore and dash characters. This makes them valid
    URL query parameter name.
    """
    def __new__(cls, name: str):
        validate_non_empty(name)
        validate_max_length(name, 64)
        if not re.fullmatch(r'[a-zA-Z0-9\-\.\_\~]*', name):
            raise ValueError(
                "Job tag key must only contain unreserved characters that do "
                f"not need URL encoding, got: {name}")
        return super().__new__(cls, name)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema())


# "pquery_name" is the name of PQuery template, if a job is launched from a
# named pquery template; "pquery_id" is the FPID of frozen PQueryDefinition
# used to launch the job, should always exist unless the pquery contains SQL
# view.
TAG_KEY_PQUERY_NAME = JobTagKey('pquery_name')
TAG_KEY_PQUERY_ID = JobTagKey('pquery_id')
TAG_KEY_WARM_START_JOB_ID = JobTagKey('base_train_job_id')

TAG_KEY_MODEL_ID = JobTagKey('model_id')  # tag for BP job

# Tag key for job that's implicitly created as child of a parent job.
# For example, a generate-train|pred-table job may have this tag set to its
# parent traning|prediction job.
TAG_KEY_PARENT_JOB = JobTagKey('parent_job')

# Tag key to mark a job as archived. Value should be 'archived'
TAG_KEY_ARCHIVED = JobTagKey('archived')

_IMMUTABLE_TAG_KEYS = set([
    TAG_KEY_PQUERY_NAME,
    TAG_KEY_PQUERY_ID,
    TAG_KEY_WARM_START_JOB_ID,
    TAG_KEY_MODEL_ID,
    TAG_KEY_PARENT_JOB,
])

# Set of reserved keywords that CANNOT be used as job tag key.
RESERVED_KEYWORDS = {
    # Options for LIST request (maybe be extended in future)
    "limit",
    "job_status",
    "status",
    "skip_cache",  # common API caching param
    "cache_ttl",  # common API caching param
    TAG_KEY_ARCHIVED,
}


# Check if a custom-defined tag key is allowed, raise ValueError if it's a
# reserved keyword or conflict with a system-predefined key/attribute name.
def check_custom_tag_key(key: JobTagKey) -> None:
    if key in RESERVED_KEYWORDS:
        raise ValueError(
            f'Custom job tag key={key} must not be a reserved keyword')
    if key in _IMMUTABLE_TAG_KEYS:
        raise ValueError(
            f'Custom job tag key={key} must not be kumo predefined tag key.')


class JobTagValue(str):
    """
    Job tag value can be arbitrary non-empty text with max length of 256.
    """
    def __new__(cls, name: str):
        validate_non_empty(name)
        validate_max_length(name, 256)
        return super().__new__(cls, name)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema())

    @classmethod
    def null_or_empty(cls, v: Optional[str]) -> bool:
        if not v:
            return True
        lowered_v = v.lower()
        return lowered_v in ['null', 'none', 'na']

    @classmethod
    def of_pquery_name(cls, pquery_name: Optional[str]) -> 'JobTagValue':
        """Convert pquery name to JobTagValue to handle none/null/na etc."""
        if not pquery_name or JobTagValue.null_or_empty(pquery_name):
            return JobTagValue('null')

        if PredictiveQueryID.has_fpid(pquery_name):
            pquery_name = PredictiveQueryID.parse(pquery_name)[0]

        lowered_name = pquery_name.lower()
        if lowered_name in ['na', 'n/a', 'pquery']:
            return JobTagValue('null')
        return JobTagValue(pquery_name)

    @classmethod
    def of_pquery_id(cls, pquery_def_fpid: Optional[FPID]) -> 'JobTagValue':
        if not pquery_def_fpid:
            return JobTagValue('null')
        return JobTagValue(PredictiveQueryID.public(pquery_def_fpid))


TAG_VALUE_ARCHIVED = JobTagValue('archived')


@kumo_dataclass
class JobTags(DBEntry):
    tags: Dict[JobTagKey, JobTagValue] = field(default_factory=dict)

    @field_validator('tags', mode='before')
    @classmethod
    def _drop_empty_tags(cls, data):
        # Drop null/empty tag values BEFORE pydantic coerces each value to
        # JobTagValue. JobTagValue.__new__ runs validate_non_empty, so an
        # empty value would raise during item coercion otherwise. This keeps
        # persisted rows that carry an empty tag value loadable.
        if not isinstance(data, Mapping):
            return data
        return {
            k: v
            for k, v in data.items() if not JobTagValue.null_or_empty(v)
        }

    # ====== Predefined Attributes ======
    #
    @property
    def pquery_name(self) -> Optional[PredictiveQueryID]:
        name = self.tags.get(TAG_KEY_PQUERY_NAME)
        return PredictiveQueryID(
            name) if name and not JobTagValue.null_or_empty(name) else None

    @property
    def pquery_id(self) -> Optional[PredictiveQueryID]:
        id = self.tags.get(TAG_KEY_PQUERY_ID)
        return PredictiveQueryID(
            id) if id and not JobTagValue.null_or_empty(id) else None

    def match(self, filters: Mapping[str, str]) -> bool:
        for k, v in filters.items():
            if JobTagValue.null_or_empty(v):
                v = None
            if v != self.tags.get(k):  # type: ignore
                return False
        return True

    def update_tags(self, new_tags: Dict[JobTagKey, JobTagValue]) -> bool:
        updated = False
        for k, v in new_tags.items():
            if self.tags.get(k) == v:  # k-v already exist, noop
                continue
            if JobTagValue.null_or_empty(v) and k not in self.tags:
                continue

            if k in _IMMUTABLE_TAG_KEYS and k in self.tags:
                raise ValueError(f'Unable to mutate tag value for key {k}')

            updated = True
            if JobTagValue.null_or_empty(v):
                self.tags.pop(k)
            else:
                self.tags[k] = v

        return updated


@kumo_dataclass
class JobTagRequest():
    """
    Used when adding and deleting tags associated with a job.
    """
    job_id: JobID
    project_id: ProjectID
    key: str
    value: Optional[str] = None


@kumo_dataclass
class JobTagsMap():
    mapping: Dict[JobID, JobTags]
