# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU/memory pool configuration, independent of Kubernetes or cloud access."""

import dataclasses

import pytest
from pydantic import ValidationError

from service_framework._raw_deploy_config import K8sPoolStrategyConfig


def pool_config(resources=None):
    kwargs = {} if resources is None else {'pool_resources': resources}
    return K8sPoolStrategyConfig(
        pool_diskgraph_node_types={'test': {
            'ssd_gib': 800
        }}, **kwargs)


@pytest.mark.parametrize('group', ['activity', 'training', 'prediction'])
def test_old_ambiguous_group_names_are_rejected(group):
    with pytest.raises(ValidationError):
        pool_config({group: {'requests': {'cpu': '2'}}})


@pytest.mark.parametrize('resources',
                         [None, {}, {
                             'daemon': {},
                             'activity_defaults': {}
                         }])
def test_omitted_resources_preserve_legacy_defaults(resources):
    config = pool_config(resources).pool_resources
    daemon = config.effective_daemon()
    assert daemon.requests == {'cpu': '1', 'memory': '2Gi'}
    assert daemon.limits == {}
    for kind in (None, 'training', 'prediction'):
        activity = config.effective_activity(kind)
        assert activity.requests == {}
        assert activity.limits == {}


def test_partial_daemon_override_preserves_unspecified_defaults():
    config = pool_config({'daemon': {'limits': {'memory': '16Gi'}}})
    daemon = config.pool_resources.effective_daemon()
    assert daemon.requests == {'cpu': '1', 'memory': '2Gi'}
    assert daemon.limits == {'memory': '16Gi'}


def test_activity_overrides_merge_per_resource_without_mutating_defaults():
    config = pool_config({
        'activity_defaults': {
            'requests': {
                'cpu': '2',
                'memory': '8Gi'
            },
            'limits': {
                'cpu': '8',
                'memory': '32Gi'
            },
        },
        'training_overrides': {
            'requests': {
                'cpu': '4'
            },
            'limits': {
                'memory': '64Gi'
            }
        },
        'prediction_overrides': {
            'requests': {
                'memory': '16Gi'
            }
        },
    }).pool_resources
    before = dataclasses.asdict(config)
    training = config.effective_activity('training')
    assert training.requests == {'cpu': '4', 'memory': '8Gi'}
    assert training.limits == {'cpu': '8', 'memory': '64Gi'}
    prediction = config.effective_activity('prediction')
    assert prediction.requests == {'cpu': '2', 'memory': '16Gi'}
    assert prediction.limits == {'cpu': '8', 'memory': '32Gi'}
    assert config.effective_activity(
        None).requests == config.activity_defaults.requests
    training.requests['cpu'] = '99'
    assert dataclasses.asdict(config) == before


@pytest.mark.parametrize('cpu,memory', [
    ('500m', '512Mi'),
    ('0.5', '0.5Gi'),
    ('5e-1', '536870912'),
    (0.5, 536870912),
    ('1', '1G'),
    ('1000m', '1e9'),
])
def test_valid_quantities_and_yaml_numbers(cpu, memory):
    config = pool_config(
        {'activity_defaults': {
            'requests': {
                'cpu': cpu,
                'memory': memory
            }
        }})
    assert config.pool_resources.activity_defaults.requests == {
        'cpu': str(cpu),
        'memory': str(memory),
    }


@pytest.mark.parametrize('field,value', [
    ('cpu', '-1'),
    ('cpu', '0'),
    ('cpu', '0.0001'),
    ('cpu', ''),
    ('cpu', 'nan'),
    ('cpu', True),
    ('cpu', None),
    ('cpu', []),
    ('memory', '4GB'),
    ('memory', '2 Gi'),
    ('memory', '1K'),
    ('memory', 'Infinity'),
    ('memory', '0Gi'),
    ('memory', '1e1000000000'),
])
def test_invalid_quantities_rejected_before_pod_creation(field, value):
    with pytest.raises(ValidationError):
        pool_config({'activity_defaults': {'requests': {field: value}}})


@pytest.mark.parametrize('resources', [
    {
        'trainning': {}
    },
    {
        'activity_defaults': {
            'request': {
                'cpu': '1'
            }
        }
    },
    {
        'activity_defaults': {
            'requests': {
                'nvidia.com/gpu': '2'
            }
        }
    },
    {
        'daemon': {
            'limits': {
                'ephemeral-storage': '1Gi'
            }
        }
    },
])
def test_unknown_fields_are_not_silently_ignored(resources):
    with pytest.raises(ValidationError):
        pool_config(resources)


@pytest.mark.parametrize('resources', [
    {
        'activity_defaults': {
            'requests': {
                'cpu': '1500m'
            },
            'limits': {
                'cpu': '1'
            }
        }
    },
    {
        'activity_defaults': {
            'requests': {
                'memory': '1Gi'
            },
            'limits': {
                'memory': '1G'
            }
        }
    },
    {
        'daemon': {
            'limits': {
                'cpu': '500m'
            }
        }
    },
    {
        'daemon': {
            'limits': {
                'memory': '1024Mi'
            }
        }
    },
    {
        'activity_defaults': {
            'requests': {
                'memory': '8Gi'
            }
        },
        'training_overrides': {
            'limits': {
                'memory': '4Gi'
            }
        }
    },
    {
        'activity_defaults': {
            'limits': {
                'cpu': '2'
            }
        },
        'prediction_overrides': {
            'requests': {
                'cpu': '4'
            }
        }
    },
])
def test_effective_request_must_not_exceed_limit(resources):
    with pytest.raises(ValidationError, match='request.*limit'):
        pool_config(resources)


def test_equal_quantities_in_different_units_are_accepted():
    pool_config({
        'activity_defaults': {
            'requests': {
                'cpu': '1000m',
                'memory': '1Gi'
            },
            'limits': {
                'cpu': '1',
                'memory': '1024Mi'
            },
        }
    })


def test_limits_only_activity_does_not_invent_requests():
    config = pool_config({'activity_defaults': {'limits': {'memory': '8Gi'}}})
    assert config.pool_resources.effective_activity(None).requests == {}
    assert config.pool_resources.effective_activity(None).limits == {
        'memory': '8Gi'
    }
