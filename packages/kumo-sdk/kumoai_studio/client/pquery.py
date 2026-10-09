# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import fields, make_dataclass
from http import HTTPStatus
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
)

from kumoapi.common import ValidationResponse
from kumoapi.json_serde import to_json_dict
from kumoapi.model_plan import (
    MissingType,
    ModelPlan,
    ModelPlanInfo,
    PlanMixin,
    SuggestModelPlanRequest,
    TrainingTableGenerationPlan,
)
from kumoapi.pquery import PQueryResource, QueryType
from kumoapi.task import TaskType
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

from kumoai_studio.client.endpoints import PQueryEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import (
    Returns,
    parse_id_response,
    parse_response,
    raise_on_error,
)

GraphID = str
PQueryID = str


class PQueryAPI:
    r"""Typed API definition for Kumo Predictive Query definition."""
    def __init__(self, client: Transport) -> None:
        self._client = client

    @staticmethod
    def _params(project_id: Optional[str]) -> Optional[dict]:
        return ({'project_id': project_id} if project_id is not None else None)

    def create(self, pquery: PQueryResource,
               project_id: Optional[str] = None) -> PQueryID:
        r"""Creates a Predictive Query resource object in Kumo."""
        resp = self._client._request(PQueryEndpoints.create,
                                     params=self._params(project_id),
                                     json=to_json_dict(pquery))
        raise_on_error(resp)
        return parse_id_response(resp)

    def get_if_exists(
        self,
        name_or_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[PQueryResource]:
        r"""Fetches a Predictive Query given its name(id)."""
        resp = self._client._request(PQueryEndpoints.get.with_id(name_or_id),
                                     params=self._params(project_id))
        if resp.status_code == HTTPStatus.NOT_FOUND:
            return None

        raise_on_error(resp)
        return parse_response(PQueryResource, resp)

    def list(self, *, name_pattern: Optional[str] = None,
             project_id: Optional[str] = None) -> List[PQueryResource]:
        params = {}
        if name_pattern:
            params['name_pattern'] = name_pattern
        if project_id is not None:
            params['project_id'] = project_id
        resp = self._client._request(PQueryEndpoints.list, params=params)
        raise_on_error(resp)
        return parse_response(List[PQueryResource], resp)

    def delete(self, name: str, project_id: Optional[str] = None) -> None:
        resp = self._client._request(PQueryEndpoints.delete.with_id(name),
                                     params=self._params(project_id))
        raise_on_error(resp)

    def infer_task_type(
        self,
        pquery_string: str,
        graph_id: GraphID,
        project_id: Optional[str] = None,
    ) -> Tuple[TaskType, str]:
        params = {'pquery_string': pquery_string, 'graph_id': graph_id}
        if project_id is not None:
            params['project_id'] = project_id
        resp = self._client._request(
            PQueryEndpoints.infer_task_type,
            params=params,
        )
        raise_on_error(resp)
        return parse_response(Returns[Tuple[TaskType, str]], resp)

    def validate(
        self,
        pquery: PQueryResource,
        project_id: Optional[str] = None,
    ) -> ValidationResponse:
        r"""Validates a Predictive Query resource object in Kumo."""
        resp = self._client._request(PQueryEndpoints.validate,
                                     params=self._params(project_id),
                                     json=to_json_dict(pquery))
        raise_on_error(resp)
        return parse_response(Returns[ValidationResponse], resp)

    def suggest_training_table_plan(
        self,
        req: SuggestModelPlanRequest,
        project_id: Optional[str] = None,
    ) -> TrainingTableGenerationPlan:
        resp = self._client._request(
            PQueryEndpoints.suggest_training_table_plan,
            params=self._params(project_id),
            json=to_json_dict(req),
        )
        raise_on_error(resp)
        return parse_response(TrainingTableGenerationPlan, resp)

    def suggest_model_plan(
        self,
        req: SuggestModelPlanRequest,
        project_id: Optional[str] = None,
    ) -> ModelPlan:
        resp = self._client._request(
            PQueryEndpoints.suggest_model_plan,
            params=self._params(project_id),
            json=to_json_dict(req),
        )
        raise_on_error(resp)
        plan_info = parse_response(ModelPlanInfo, resp)

        return filter_model_plan(
            plan_info.model_plan,
            plan_info.task_type,
            plan_info.query_type,
            plan_info.has_train_table_weight_col,
        )


def filter_model_plan(
    plan: ModelPlan,
    task_type: TaskType,
    query_type: QueryType,
    has_train_table_weight_col: bool,
) -> ModelPlan:
    r"""Filters the model plan ``plan`` down to its valid fields only, given
    ``task_type`` and ``query_type``.
    """
    new_section_fields = []
    new_sections = []
    for section_field in fields(plan):
        section = getattr(plan, section_field.name)

        new_opt_fields = []
        new_opts = []
        for field in fields(section):
            field_default = cast(Any, field.default)
            metadata = field_default.json_schema_extra
            if not section.is_valid_option(
                    name=field.name,
                    metadata=metadata,
                    task_type=task_type,
                    query_type=query_type,
                    has_train_table_weight_col=has_train_table_weight_col,
            ):
                continue

            # Remove `MissingType` from type annotation:
            hidden = metadata["hidden"]
            if not hidden and get_origin(field.type) is Union:
                _types = tuple(_type for _type in get_args(field.type)
                               if _type is not MissingType)
                assert len(_types) > 0
                _type = Union[_types] if len(_types) > 1 else _types[0]
            else:
                _type = field.type

            default = copy.copy(field.default)
            # Field() exposes Pydantic's public required-field default in both
            # v1 and v2 without importing its pydantic-core implementation.
            default.default = PydanticField().default  # type: ignore

            # Forward compatibility - Remove any newly introduced arguments not
            # returned yet by the backend:
            value = getattr(section, field.name)
            if value != MissingType.VALUE:
                new_opt_fields.append((field.name, _type, default))
                new_opts.append(value)

        Section = dataclass(
            config=dict(validate_assignment=True),
            repr=False,
        )(make_dataclass(
            type(section).__name__,
            new_opt_fields,
            bases=(PlanMixin, ),
            repr=False,
        ))
        new_section_fields.append((section_field.name, Section))
        new_sections.append(Section(*new_opts))

    Plan = dataclass(
        config=dict(validate_assignment=True),
        repr=False,
    )(make_dataclass(
        'ModelPlan',
        new_section_fields,
        bases=(ModelPlan, ),
        repr=False,
    ))
    return Plan(*new_sections)
