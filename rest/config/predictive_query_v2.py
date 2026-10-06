# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Dict, Optional

import kumoapi.pquery as api
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.types import TaskType
from kumo_core.types.schema import FPID
from kumoapi.graph import GraphDefinition as apiGraphDefinition
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from io_utils import kumo_dataclass
from io_utils.dataclasses import AllowHiddenFieldConfig
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import (
    DefaultModelPlanInfo,
    PQueryRuntimeOptions,
    pquery_runtime_options_or_none,
)


@kumo_dataclass
class ModelPlanResponse:
    """
    Response object for suggesting a model plan.
    """
    suggested_model_plan: DefaultModelPlanInfo
    suggested_model_plan_yaml: Optional[str]


@kumo_dataclass
class PQueryResponse:
    """
    Predictive Query resource definition.

    Attributes:
        name (Optional[str]): The name of the predictive query. This is an
            optional attribute.
        query_string (str): The query string used for the predictive query.
        graph (GraphDefinition): The graph definition associated with the
            predictive query.
        desc (str): A description of the predictive query. Defaults to an
            empty string.
    """
    name: Optional[str]
    query_string: str
    graph: apiGraphDefinition
    desc: str = ''


@kumo_dataclass
class PQueryMetadataResponse:
    """
    Response object describing the task and query type of a predictive query.

    Attributes:
        task_type (TaskType): The task type parsed from the predictive query.
        query_type (QueryType): The query type parsed from the predictive
            query.
    """
    task_type: TaskType
    query_type: api.QueryType


@kumo_dataclass
class PQueryEncodersResponse:
    """
    Response object for the encoders used by a predictive query.

    Attributes:
        encoders (Dict[str, str]): A mapping from fully qualified table name to
            encoder representation.
    """
    encoders: Dict[str, str]


_PRIVATE_PQUERY_CONFIG = ConfigDict(AllowHiddenFieldConfig, extra='forbid')


@pydantic_dataclass(config=_PRIVATE_PQUERY_CONFIG)
class PQueryResource(api.PQueryResource):
    """Kumo-private predictive-query resource definition.

    The public API owns the strict four-field ``api.PQueryResource`` wire
    contract. Private Kumo routes use this strict superset for UI and internal
    execution metadata that must never cross the public boundary.
    """
    graph_display_name: Optional[str] = None
    # For internal use only (testing) where query string is written in YAML.
    advanced_options_yaml: Optional[str] = Field(default=None, hidden=True)
    runtime_options: Optional[PQueryRuntimeOptions] = Field(
        default=None, hidden=True)

    @property
    def list_distinct_target_is_array(self) -> Optional[bool]:
        if self.runtime_options is None:
            return None
        return self.runtime_options.list_distinct_target_is_array

    @property
    def lp_bp_layout(self) -> Optional[str]:
        if self.runtime_options is None:
            return None
        return self.runtime_options.lp_bp_layout

    @field_validator('graph')
    @classmethod
    def _coerce_graph_def(cls, v):
        if isinstance(v, GraphDefinition):
            return v
        if dataclasses.is_dataclass(v):
            return GraphDefinition(**dataclasses.asdict(v))
        if isinstance(v, dict):
            return GraphDefinition(**v)
        if isinstance(v, BaseModel):
            return GraphDefinition(**dict(v))

        raise ValueError(f'Invalid GraphDefinition type ({type(v)}),'
                         f' unable to coerce graph: {v}')


_PQUERY_RUNTIME_OPTION_FIELDS = (
    'list_distinct_target_is_array',
    'lp_bp_layout',
)


def pquery_resource_with_runtime_options(
    pquery_resource: api.PQueryResource,
    raw_body: Any,
) -> PQueryResource:
    data = dataclasses.asdict(pquery_resource)
    if isinstance(raw_body, dict):
        if 'advanced_options_yaml' in raw_body:
            data['advanced_options_yaml'] = raw_body['advanced_options_yaml']
        if 'runtime_options' in raw_body:
            data['runtime_options'] = raw_body['runtime_options']
        elif any(field_name in raw_body
                 for field_name in _PQUERY_RUNTIME_OPTION_FIELDS):
            data['runtime_options'] = pquery_runtime_options_or_none(
                list_distinct_target_is_array=raw_body.get(
                    'list_distinct_target_is_array'),
                lp_bp_layout=raw_body.get('lp_bp_layout'),
            )
    return PQueryResource(**data)


@kumo_dataclass
class PQueryValidationRequest:
    pquery_resource: PQueryResource
    model_plan: Optional[str] = ''

    # ``pquery_resource`` is required, so a before-validator covers every
    # valid construction path without Pydantic v1's ``always=True`` behavior.
    @field_validator('pquery_resource', mode='before')
    @classmethod
    def _coerce_graph_in_pquery_resource(cls, v):
        """
        This validator runs whenever pquery_resource is set, and it tries
        to coerce the `graph` attribute inside pquery_resource.
        """
        if not v:
            return v

        # If `v` is already a PQueryResource instance, just coerce inside it
        if isinstance(v, PQueryResource):
            v.graph = cls._coerce_graph_def(v.graph)
            return v

        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            data = dataclasses.asdict(v)
            data['graph'] = cls._coerce_graph_def(data['graph'])
            return PQueryResource(**data)

        # If `v` is a dict, then coerce v['graph']
        if isinstance(v, dict) and 'graph' in v:
            v['graph'] = cls._coerce_graph_def(v['graph'])
        return v

    @staticmethod
    def _coerce_graph_def(graph_val: apiGraphDefinition):
        """
        Coerce the GraphDefinition with the enhanced one from internal object
        config.
        """
        if isinstance(graph_val, GraphDefinition):
            return graph_val
        if dataclasses.is_dataclass(graph_val):
            if not isinstance(graph_val, type):
                return GraphDefinition(**dataclasses.asdict(graph_val))
        if isinstance(graph_val, dict):
            return GraphDefinition(**graph_val)
        if isinstance(graph_val, BaseModel):
            return GraphDefinition(**dict(graph_val))

        raise ValueError(
            f'Invalid GraphDefinition type ({type(graph_val)}), unable to '
            f'coerce graph: {graph_val}')


@kumo_dataclass
class PQueryValidationArgs:
    query_string: str
    graph: GraphDefinition
    name: str
    desc: Optional[str] = ""
    # Raw user-supplied model-plan YAML.
    model_plan: Optional[str] = ""
    fpid: Optional[FPID] = None
    runtime_options: Optional[PQueryRuntimeOptions] = None
    # Pre-built train-table options, for in-process callers that already hold
    # them. When set, these are used instead of parsing `model_plan`, so the
    # options do not make a pointless trip through YAML.
    train_table_options: Optional[TrainingTableGenerationPlan] = None

    @property
    def list_distinct_target_is_array(self) -> Optional[bool]:
        if self.runtime_options is None:
            return None
        return self.runtime_options.list_distinct_target_is_array

    @property
    def lp_bp_layout(self) -> Optional[str]:
        if self.runtime_options is None:
            return None
        return self.runtime_options.lp_bp_layout
