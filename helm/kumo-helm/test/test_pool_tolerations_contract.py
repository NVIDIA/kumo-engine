#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm -> runtime JSON extra-tolerations contract; no cluster or SDK needed."""

import json
import os
import subprocess
import unittest
from pathlib import Path
from typing import Any

CHART = Path(__file__).resolve().parents[1]
EXTRAS = [
    dict(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'),
    dict(key='maintenance', operator='Exists', effect='NoExecute',
         tolerationSeconds=60),
]


def render(values, template='templates/configmap-kumo-config.yaml'):
    command = [os.environ.get('HELM', 'helm'), 'template', 'kumo', str(CHART)]
    command += ['-f', str(CHART / 'test/manifests/values-aws-minimal.yaml')]
    command += ['-f', '-', '--show-only', template]
    return subprocess.run(command, input=json.dumps(values), text=True,
                          capture_output=True, check=False)


def rendered_json(result, key='DEPLOY_ENV_CONFIG_JSON'):
    if result.returncode:
        raise AssertionError(result.stderr)
    line, = [
        line for line in result.stdout.splitlines()
        if line.strip().startswith(key + ':')
    ]
    return json.loads(json.loads(line.split(':', 1)[1].strip()))


def values(extras, raw=False) -> dict[str, Any]:
    if raw:
        pool = {'pool_extra_tolerations': extras}
        override = {'data_compute': {'k8s_pool': pool}}
        return {'kumoConfig': {'extra': override}}
    return {'kumoConfig': {'diskgraph': {'poolExtraTolerations': extras}}}


class PoolTolerationsContractTest(unittest.TestCase):
    def test_extras_reach_runtime_config_and_conformance_probe(self):
        for raw in (False, True):
            with self.subTest(raw=raw):
                config = rendered_json(render(values(EXTRAS, raw)))
                self.assertEqual(
                    config['data_compute']['k8s_pool'].get(
                        'pool_extra_tolerations'), EXTRAS)
                probe = rendered_json(
                    render(values(EXTRAS, raw),
                           'templates/tests/probe-inputs.yaml'), 'inputs.json')
                self.assertEqual(probe.get('poolExtraTolerations'), EXTRAS)

    def test_omitted_null_and_empty_all_mean_no_extras(self):
        for override in ({}, values(None), values([]), values(None, raw=True),
                         values([], raw=True)):
            with self.subTest(values=override):
                config = rendered_json(render(override))
                self.assertFalse(config['data_compute']['k8s_pool'].get(
                    'pool_extra_tolerations'))

    def test_raw_override_uses_the_same_effective_config(self):
        override = values(EXTRAS)
        override['kumoConfig']['extra'] = values(
            [], raw=True)['kumoConfig']['extra']
        config = rendered_json(render(override))
        self.assertEqual(
            config['data_compute']['k8s_pool']['pool_extra_tolerations'], [])

    def test_invalid_entries_fail_during_render_including_raw_overrides(self):
        invalid = [
            'nvidia.com/gpu',
            {},
            [None],
            ['nvidia.com/gpu'],
            [dict(key='nvidia.com/gpu', operator='Exist')],
            [dict(key='nvidia.com/gpu', effect='NoScheduling')],
            [dict(key='gpu', operator='Exists', value='true')],
            [dict(operator='Equal')],
            [dict(key='gpu', tolerationSeconds=60)],
            [dict(key='gpu', effect='NoExecute', tolerationSeconds='60')],
            [dict(key='gpu', effect='NoExecute', tolerationSeconds=True)],
            [dict(key='gpu', effect='NoExecute', tolerationSeconds=1.5)],
            [dict(key='gpu', effect='NoSchedule', typo='value')],
        ]
        for raw in (False, True):
            for extras in invalid:
                with self.subTest(raw=raw, extras=extras):
                    result = render(values(extras, raw))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('toleration', result.stderr.lower())

    def test_standard_kubernetes_defaults_and_wildcards_are_accepted(self):
        valid = [
            [dict(key='gpu')],
            [dict(operator='Exists')],
            [dict(key='gpu', operator='', effect='')],
            [dict(key='maintenance', effect='NoExecute', tolerationSeconds=0)],
        ]
        for extras in valid:
            with self.subTest(extras=extras):
                config = rendered_json(render(values(extras)))
                self.assertEqual(
                    config['data_compute']['k8s_pool'].get(
                        'pool_extra_tolerations'), extras)

    def test_raw_seconds_must_fit_int64_after_helm_number_encoding(self):
        for seconds in (2**63 - 1, 2**63, -(2**63), 2**80, -(2**80)):
            with self.subTest(seconds=seconds):
                result = render(
                    values([
                        dict(key='maintenance', effect='NoExecute',
                             tolerationSeconds=seconds)
                    ], raw=True))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('toleration', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
