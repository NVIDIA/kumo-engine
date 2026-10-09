# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Grafana Cloud Loki URL helpers and the
``debug_info`` carrier on ``PlacementResult``.

Covers ``pool/k8s_strategy.py`` helpers:
- ``_loki_selector``
- ``_build_loki_explore_url`` (cluster allow-list + URL shape)
"""
import json
import urllib.parse

import pool.k8s_strategy as k8s_strategy
from pool.k8s_strategy import (
    _GRAFANA_LOKI_DATASOURCE,
    _LOKI_EXPLORE_CLUSTERS,
    _build_loki_explore_url,
    _loki_selector,
)


def test_loki_selector_composes_label_matchers():
    out = _loki_selector('example-cluster', ['service_name="pod-x"'])
    assert out == '{cluster="example-cluster", service_name="pod-x"}'


def test_loki_selector_with_multiple_matchers():
    out = _loki_selector(
        'example-cluster',
        ['service_name="pod-x"', 'container="activity"'],
    )
    assert out == ('{cluster="example-cluster", service_name="pod-x", '
                   'container="activity"}')


def test_loki_selector_with_no_extra_matchers():
    out = _loki_selector('example-cluster', [])
    assert out == '{cluster="example-cluster"}'


def test_build_loki_explore_url_returns_none_for_unknown_cluster():
    # Outside the allow-list — caller omits the URL from debug_info.
    assert _build_loki_explore_url(
        cluster='preprod-uw4',
        extra_matchers=['service_name="x"'],
    ) is None


def test_build_loki_explore_url_returns_none_without_configuration(
        monkeypatch):
    monkeypatch.setattr(k8s_strategy, '_GRAFANA_BASE', '')

    assert _build_loki_explore_url(
        cluster=next(iter(_LOKI_EXPLORE_CLUSTERS)),
        extra_matchers=['service_name="x"'],
    ) is None
    assert _build_loki_explore_url(
        cluster=None,
        extra_matchers=['service_name="x"'],
    ) is None


def test_build_loki_explore_url_for_allow_listed_cluster(monkeypatch):
    base_url = 'https://grafana.example.test'
    monkeypatch.setattr(k8s_strategy, '_GRAFANA_BASE', base_url)
    cluster = next(iter(_LOKI_EXPLORE_CLUSTERS))
    url = _build_loki_explore_url(
        cluster=cluster,
        extra_matchers=['service_name="pod-x"'],
    )

    assert url is not None
    assert url.startswith(f'{base_url}/explore?left=')

    # left= is URL-encoded JSON; decoding round-trips the LogQL expr
    # exactly + carries the Loki datasource so Grafana opens the
    # right query immediately.
    left = urllib.parse.unquote(url.split('left=', 1)[1])
    payload = json.loads(left)
    assert payload['datasource'] == _GRAFANA_LOKI_DATASOURCE
    assert payload['queries'][0]['expr'] == (
        f'{{cluster="{cluster}", service_name="pod-x"}}')
    # Time range covers the run + a margin, expressed in Grafana's
    # relative shorthand so the link doesn't go stale.
    assert payload['range']['from'].startswith('now-')
    assert payload['range']['to'] == 'now'


def test_build_loki_explore_url_url_encodes_special_chars(monkeypatch):
    monkeypatch.setattr(k8s_strategy, '_GRAFANA_BASE',
                        'https://grafana.example.test')
    cluster = next(iter(_LOKI_EXPLORE_CLUSTERS))
    url = _build_loki_explore_url(
        cluster=cluster,
        extra_matchers=['service_name="diskgraph-activity-abc"'],
    )
    assert url is not None
    # Curly braces and quotes must be percent-encoded so they don't
    # break the URL query-string parser.
    assert '%7B' in url
    assert '%22' in url


def test_daemon_pod_name_prefix_fits_under_k8s_truncation():
    # k8s truncates pod names to 63 chars: 57 chars of deployment
    # name + '-' + 5-char random hash. The daemon-pod regex prefix
    # must stay well under 57 chars so it matches every restart of
    # the deployment regardless of where k8s truncated the name.
    from pool.k8s_strategy import _DAEMON_POD_NAME_PREFIX_LEN
    assert _DAEMON_POD_NAME_PREFIX_LEN <= 57 - 8  # leave 8 chars of margin
