#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm -> runtime JSON resource contract (stdlib + Helm; no cluster)."""

import json
import os
import subprocess
import unittest
from pathlib import Path

CHART = Path(__file__).resolve().parents[1]


def render(resources=None, extra=None):
    values = {} if resources is None else {
        'kumoConfig': {
            'diskgraph': {
                'poolResources': resources
            }
        },
    }
    if extra is not None:
        values.setdefault('kumoConfig', {})['extra'] = {
            'data_compute': {
                'k8s_pool': {
                    'pool_resources': extra
                }
            }
        }
    return subprocess.run([
        os.environ.get('HELM', 'helm'),
        'template',
        'kumo',
        str(CHART),
        '-f',
        str(CHART / 'test/manifests/values-aws-minimal.yaml'),
        '-f',
        '-',
        '--show-only',
        'templates/configmap-kumo-config.yaml',
    ], input=json.dumps(values), text=True, capture_output=True, check=False)


def runtime_config(result):
    if result.returncode:
        raise AssertionError(result.stderr)
    line, = [
        line for line in result.stdout.splitlines()
        if line.strip().startswith('DEPLOY_ENV_CONFIG_JSON:')
    ]
    return json.loads(json.loads(line.split(':', 1)[1].strip()))


class PoolResourcesContractTest(unittest.TestCase):
    def test_explicit_request_minimums(self):
        for group in ('daemon', 'activityDefaults', 'trainingOverrides',
                      'predictionOverrides'):
            for resource, value in (('cpu', '500m'), ('cpu', '999m'),
                                    ('cpu', 0.5), ('memory', '1Gi'),
                                    ('memory', '2Gi'), ('memory', '16G'),
                                    ('memory', '17179869183')):
                with self.subTest(group=group, resource=resource, value=value):
                    result = render({group: {'requests': {resource: value}}})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('request', result.stderr)

    def test_minimum_requests_and_large_limits_are_accepted(self):
        for group in ('daemon', 'activityDefaults', 'trainingOverrides',
                      'predictionOverrides'):
            for cpu, memory in (('1', '16Gi'), ('1000m', '16384Mi'),
                                (1, 17179869184)):
                with self.subTest(group=group, cpu=cpu, memory=memory):
                    runtime_config(
                        render({
                            group: {
                                'requests': {
                                    'cpu': cpu,
                                    'memory': memory
                                },
                                'limits': {
                                    'cpu': '128',
                                    'memory': '1Ti'
                                },
                            }
                        }))

    def test_extra_cannot_bypass_validation(self):
        for resources in ({
                'activity_defaults': {
                    'limits': {
                        'cpu': '.0'
                    }
                }
        }, {
                'activity_defaults': {
                    'limits': {
                        'memory': '4GB'
                    }
                }
        }, {
                'activity_defaults': {
                    'limits': {
                        'cpu': True
                    }
                }
        }, {
                'activity_defaults': {
                    'requests': {
                        'cpu': '500m'
                    }
                }
        }, {
                'activity_defaults': {
                    'requests': {
                        'memory': '1Gi'
                    }
                }
        }, {
                'activity_defaults': {
                    'requests': {
                        'cpu': '3'
                    },
                    'limits': {
                        'cpu': '2'
                    }
                }
        }, {
                'unknown': {}
        }, {
                'daemon': {
                    'requests': {
                        'nvidia.com/gpu': '1'
                    }
                }
        }, {
                'daemon': {
                    'request': {
                        'cpu': '2'
                    }
                }
        }, {
                'daemon': {
                    'requests': []
                }
        }, {
                'daemon': []
        }, [], False):
            with self.subTest(resources=resources):
                result = render(extra=resources)
                self.assertNotEqual(result.returncode, 0)

    def test_extra_merge_is_validated_without_mutating_chart_values(self):
        resources = {'activityDefaults': {'requests': {'cpu': '2'}}}
        result = render(resources,
                        extra={'training_overrides': {
                            'limits': {
                                'cpu': '1'
                            }
                        }})
        self.assertNotEqual(result.returncode, 0)
        extra = {
            'activity_defaults': {
                'requests': {
                    'cpu': '3'
                },
                'limits': {
                    'memory': '1Ti'
                }
            }
        }
        pool = runtime_config(render(resources,
                                     extra))['data_compute']['k8s_pool']
        self.assertEqual(pool['pool_resources'], extra)

    def test_default_does_not_emit_resource_overrides(self):
        for resources in (None, {}):
            with self.subTest(resources=resources):
                pool = runtime_config(
                    render(resources))['data_compute']['k8s_pool']
                self.assertNotIn('pool_resources', pool)
                self.assertIn('pool_diskgraph_node_types', pool)

    def test_partial_requests_and_limits_do_not_invent_defaults(self):
        for daemon in ({}, {
                'requests': {
                    'cpu': '1'
                }
        }, {
                'limits': {
                    'cpu': '1',
                    'memory': '2Gi'
                }
        }):
            with self.subTest(daemon=daemon):
                pool = runtime_config(
                    render({
                        'daemon': daemon,
                        'activityDefaults': {
                            'limits': {
                                'cpu': '500m',
                                'memory': '1Gi'
                            }
                        },
                    }))['data_compute']['k8s_pool']['pool_resources']
                self.assertEqual(pool['daemon'], daemon)
                self.assertEqual(pool['activity_defaults'],
                                 {'limits': {
                                     'cpu': '500m',
                                     'memory': '1Gi'
                                 }})

    def test_all_resource_groups_reach_runtime_json(self):
        resources = {
            'daemon': {
                'requests': {
                    'cpu': '2',
                    'memory': '16Gi'
                },
                'limits': {
                    'cpu': '4',
                    'memory': '16Gi'
                }
            },
            'activityDefaults': {
                'requests': {
                    'cpu': 2,
                    'memory': '16Gi'
                },
                'limits': {
                    'cpu': '8',
                    'memory': '32Gi'
                }
            },
            'trainingOverrides': {
                'requests': {
                    'cpu': '4'
                },
                'limits': {
                    'memory': '64Gi'
                }
            },
            'predictionOverrides': {
                'requests': {
                    'memory': '16Gi'
                }
            },
        }
        pool = runtime_config(render(resources))['data_compute']['k8s_pool']
        self.assertEqual(
            pool['pool_resources'], {
                'daemon': resources['daemon'],
                'activity_defaults': resources['activityDefaults'],
                'training_overrides': resources['trainingOverrides'],
                'prediction_overrides': resources['predictionOverrides'],
            })

    def test_old_ambiguous_group_names_are_rejected(self):
        for group in ('activity', 'training', 'prediction'):
            with self.subTest(group=group):
                result = render({group: {'requests': {'cpu': '2'}}})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('poolResources', result.stderr)

    def test_invalid_values_are_rejected_by_chart_schema(self):
        cases = [
            {
                'trainning': {}
            },
            {
                'activityDefaults': {
                    'request': {
                        'cpu': '1'
                    }
                }
            },
            {
                'activityDefaults': {
                    'requests': {
                        'nvidia.com/gpu': '2'
                    }
                }
            },
            {
                'activityDefaults': {
                    'requests': {
                        'memory': '4GB'
                    }
                }
            },
            {
                'daemon': {
                    'limits': {
                        'cpu': -1
                    }
                }
            },
            {
                'daemon': {
                    'limits': {
                        'cpu': True
                    }
                }
            },
            {
                'daemon': {
                    'limits': {
                        'memory': '0Gi'
                    }
                }
            },
            {
                'daemon': {
                    'requests': '2'
                }
            },
        ]
        for resources in cases:
            with self.subTest(resources=resources):
                result = render(resources)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('poolResources', result.stderr)

    def test_invalid_quantity_values_fail_during_render(self):
        cases = [
            ('cpu', '.0'),
            ('memory', '.0'),
            ('memory', '+.000Gi'),
            ('cpu', 0.0001),
            ('cpu', '0.0001'),
            ('cpu', '0.1m'),
            ('cpu', '1e-4'),
            ('cpu', '1e-1000000000'),
            ('cpu', '100u'),
            ('cpu', '100000n'),
            ('cpu', '1.0001'),
            ('cpu', '1.000000000000000000000000000001'),
        ]
        for resource in ('cpu', 'memory'):
            cases.extend(
                (resource, value)
                for value in ('9223372036854775808', '9223372036854775808e0',
                              '9223372036854775807.001',
                              '9223372036854775807.0000000001', '9Ei', '10E',
                              '1e1000000000', 9223372036854775808,
                              '1e9223372036854775807',
                              '1e-9223372036854775808'))
        for resource, value in cases:
            with self.subTest(resource=resource, value=value):
                result = render(
                    {'activityDefaults': {
                        'limits': {
                            resource: value
                        }
                    }})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('poolResources', result.stderr)

    def test_valid_quantities_preserve_units_and_have_no_sizing_cap(self):
        cases = [
            ('cpu', '1m'),
            ('cpu', 0.001),
            ('cpu', '1e-3'),
            ('cpu', '1000u'),
            ('cpu', '1000000n'),
            ('cpu', '0.0010'),
            ('cpu', '+1.0000'),
            ('cpu', '0.125'),
            ('cpu', '1.5Gi'),
            ('memory', '0.0001'),
            ('memory', '1e-1000000000'),
            ('memory', '1Ti'),
            ('memory', '0.5Gi'),
        ]
        for resource in ('cpu', 'memory'):
            cases.extend(
                (resource, value)
                for value in ('9223372036854775807', '9223372036854775807e0',
                              '9223372036854775807000m', '7Ei', '001', 2))
        for resource, value in cases:
            with self.subTest(resource=resource, value=value):
                pool = runtime_config(
                    render({'activityDefaults': {
                        'limits': {
                            resource: value
                        }
                    }}))['data_compute']['k8s_pool']
                self.assertEqual(pool['pool_resources']['activity_defaults'],
                                 {'limits': {
                                     resource: value
                                 }})

    def test_effective_request_limit_conflicts_fail_during_render(self):
        cases = [
            {
                'activityDefaults': {
                    'requests': {
                        'cpu': '1500m'
                    },
                    'limits': {
                        'cpu': '1'
                    }
                }
            },
            {
                'activityDefaults': {
                    'requests': {
                        'memory': '16Gi'
                    },
                    'limits': {
                        'memory': '16G'
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
                'activityDefaults': {
                    'requests': {
                        'memory': '16Gi'
                    }
                },
                'trainingOverrides': {
                    'limits': {
                        'memory': '4Gi'
                    }
                }
            },
            {
                'activityDefaults': {
                    'limits': {
                        'cpu': '2'
                    }
                },
                'predictionOverrides': {
                    'requests': {
                        'cpu': '4'
                    }
                }
            },
            {
                'activityDefaults': {
                    'requests': {
                        'memory': '9007199254740993'
                    },
                    'limits': {
                        'memory': '9007199254740992'
                    }
                }
            },
        ]
        for resources in cases:
            with self.subTest(resources=resources):
                result = render(resources)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('poolResources', result.stderr)
                self.assertIn('request', result.stderr)
                self.assertIn('limit', result.stderr)

    def test_equivalent_units_and_fieldwise_overrides_are_accepted(self):
        resources = {
            'activityDefaults': {
                'requests': {
                    'cpu': '1000m',
                    'memory': '32Gi'
                },
                'limits': {
                    'cpu': '1',
                    'memory': '32768Mi'
                }
            },
            'trainingOverrides': {
                'requests': {
                    'cpu': '2'
                },
                'limits': {
                    'cpu': '2000m'
                }
            },
            'predictionOverrides': {
                'requests': {
                    'memory': '16Gi'
                }
            },
        }
        pool = runtime_config(render(resources))['data_compute']['k8s_pool']
        self.assertEqual(pool['pool_resources']['activity_defaults'],
                         resources['activityDefaults'])

    def test_exa_suffix_is_not_a_scientific_exponent(self):
        for resource in ('cpu', 'memory'):
            with self.subTest(resource=resource):
                resources = {
                    'activityDefaults': {
                        'requests': {
                            resource: '1e18'
                        },
                        'limits': {
                            resource: '1E'
                        }
                    }
                }
                pool = runtime_config(
                    render(resources))['data_compute']['k8s_pool']
                self.assertEqual(pool['pool_resources']['activity_defaults'],
                                 resources['activityDefaults'])


if __name__ == '__main__':
    unittest.main()
