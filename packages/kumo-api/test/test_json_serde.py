# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import field
from typing import Annotated, Union

import pytest
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

from kumoapi.json_serde import dataclass_to_dict, from_json
from kumoapi.model_plan import ModelPlan


@dataclass
class MetadataEnvelope:
    metadata: dict


@pytest.mark.parametrize('descriptor,obj,expected', [
    (int, '"7"', 7),
    (list[int], '["1", 2]', [1, 2]),
    (Union[int, None], '"3"', 3),
    (Annotated[int, Field(gt=0)], '"4"', 4),
    (dict[str, int], {
        'value': '5'
    }, {
        'value': 5
    }),
])
def test_from_json_accepts_type_descriptors(descriptor, obj, expected):
    assert from_json(obj, descriptor) == expected


def test_deserialization_does_not_migrate_unrelated_provider_metadata():
    metadata = {
        'type': 'gcs',
        'bucket': 'display-only',
        'data_source_type': 'GCS',
        'source_table_name': 'display-only.csv',
        'pred_table_source': {
            'connector_id': 'display-only',
        },
    }
    envelope = from_json({'metadata': metadata}, MetadataEnvelope)
    assert envelope.metadata == metadata


def test_dataclass_to_dict():
    """Test dataclass_to_dict converts nested structures independently."""
    @dataclass
    class Inner:
        value: int

    @dataclass
    class Container:
        inner: Inner
        data: dict = field(default_factory=dict)
        items: list = field(default_factory=list)

    original = Container(
        inner=Inner(value=42),
        data={'key': {
            'nested': 'value'
        }},
        items=[{
            'a': 1
        }, {
            'b': 2
        }],
    )

    result = dataclass_to_dict(original)

    assert isinstance(result, dict)
    assert not dataclasses.is_dataclass(result)
    assert result['inner'] == {'value': 42}
    assert result['data'] == {'key': {'nested': 'value'}}
    assert result['items'] == [{'a': 1}, {'b': 2}]

    result['data']['key']['nested'] = 'MODIFIED'
    result['items'][0]['a'] = 999

    assert original.data == {'key': {'nested': 'value'}}
    assert original.items == [{'a': 1}, {'b': 2}]


def test_dataclass_to_dict_with_model_plan():
    """Test dataclass_to_dict with ModelPlan (complex nested dataclass)."""
    model_plan = ModelPlan()
    model_plan.training_job.metrics = ['accuracy', 'loss']
    model_plan.optimization.base_lr = [0.001, 0.01]

    original_metrics = model_plan.training_job.metrics.copy()
    original_base_lr = model_plan.optimization.base_lr.copy()

    result = dataclass_to_dict(model_plan)

    assert isinstance(result, dict)
    assert isinstance(result['training_job'], dict)
    assert isinstance(result['optimization'], dict)

    result['training_job']['metrics'].append('MODIFIED')
    result['optimization']['base_lr'].append(999.0)

    assert model_plan.training_job.metrics == original_metrics
    assert model_plan.optimization.base_lr == original_base_lr


def test_dataclass_to_dict_with_nested_basemodel():
    """Test dataclass_to_dict with nested Pydantic BaseModel."""
    class UserInfo(BaseModel):
        sub: str
        username: str
        email: str | None = None

    @dataclass
    class UserMetadata:
        user_id: str
        auth: UserInfo
        tags: list = field(default_factory=list)

    user_info = UserInfo(sub='auth0|123', username='testuser',
                         email='test@example.com')
    metadata = UserMetadata(user_id='u123', auth=user_info,
                            tags=['tag1', 'tag2'])

    result = dataclass_to_dict(metadata)

    assert isinstance(result, dict)
    assert result['user_id'] == 'u123'
    assert isinstance(result['auth'], dict)
    assert result['auth']['sub'] == 'auth0|123'
    assert result['auth']['username'] == 'testuser'
    assert result['auth']['email'] == 'test@example.com'
    assert result['tags'] == ['tag1', 'tag2']

    result['auth']['username'] = 'MODIFIED'
    assert user_info.username == 'testuser'
