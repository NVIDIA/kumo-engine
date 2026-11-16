# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import cached_property
from hashlib import md5
from typing import Optional

from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.pquery import QueryType
from pydantic import field_validator

from io_utils import kumo_dataclass
from io_utils.yaml import dict_to_yaml
from statedb.config.common import DBEntry


class QueryDialect(str, Enum):
    YAML = 'YAML'
    PQLANG = 'PQLANG'
    PQLANG_V2 = 'PQLANG_V2'


@kumo_dataclass
class FilterOverrideArgs:
    """NOTE: The temporal_entity_filter_override field is deprecated.
    Please use the entity_filter_override field instead.
    """
    entity_filter_override: Optional[str] = None
    temporal_entity_filter_override: Optional[str] = None
    target_filter_override: Optional[str] = None

    # Conversion to the YAML representation persisted with job state.
    def to_yaml(self) -> str:
        arg_dict = {
            'entity_filter_override': self.entity_filter_override,
            'temporal_entity_filter_override':
            self.temporal_entity_filter_override,
            'target_filter_override': self.target_filter_override,
        }
        yaml_repr = dict_to_yaml(arg_dict)
        assert yaml_repr is not None
        return yaml_repr


@kumo_dataclass
class PQueryRuntimeOptions:
    list_distinct_target_is_array: Optional[bool] = None
    lp_bp_layout: Optional[str] = None


def pquery_runtime_options_or_none(
    list_distinct_target_is_array: Optional[bool] = None,
    lp_bp_layout: Optional[str] = None,
) -> Optional[PQueryRuntimeOptions]:
    if list_distinct_target_is_array is None and lp_bp_layout is None:
        return None
    return PQueryRuntimeOptions(
        list_distinct_target_is_array=list_distinct_target_is_array,
        lp_bp_layout=lp_bp_layout,
    )


def compute_pquery_fpid(
    pquery_name: Optional[str],
    query_str: str,
    graph_fpid: FPID,
) -> FPID:
    hasher = md5((pquery_name or "").encode())
    hasher.update(f'query={query_str}'.encode())
    hasher.update('dialect=PQLANG_V2'.encode())
    hasher.update(f'graph_fpid={graph_fpid}'.encode())
    return FPID(hasher.hexdigest())


@kumo_dataclass
class PQueryDefinition(DBEntry):
    """
    Similar to api.PQueryResource, this represents an immutable, compact and
    minimal definition of a PQuery at the time of launching job. This is meant
    to be stored as part of DB configuration of jobs that are launched from
    PQuery.
    Note that PQueryDefinition does NOT support pqueries defined over graph
    that contains any SQL View.
    """

    # Name of the pquery
    pquery_name: Optional[str]

    # Content of the query string.
    query_str: str

    # Graph FPID serves as a key/pointer to an immutable GraphDefinition in the
    # StateDB.  Any graph with SQL View is NOT supported.
    graph_fpid: FPID

    list_distinct_target_is_array: Optional[bool] = None
    lp_bp_layout: Optional[str] = None

    # Both task and query type are currently optional because older queries
    # don't have these fields. Remove `Optional` once the DBs
    # has been re-deployed.
    task_type: Optional[TaskType] = None

    # DEPRECATED: This field is deprecated but not removing to support forwards
    # compatibility.
    # Dialect/syntax version of the query.
    query_dialect: QueryDialect = QueryDialect.PQLANG_V2

    query_type: Optional[QueryType] = None

    @field_validator('graph_fpid')
    @classmethod
    def _validate_graph_fpid(cls, v: str) -> FPID:
        return FPID(v)

    @cached_property
    def fpid(self) -> FPID:
        return compute_pquery_fpid(
            pquery_name=self.pquery_name,
            query_str=self.query_str,
            graph_fpid=self.graph_fpid,
        )
