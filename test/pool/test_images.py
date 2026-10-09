# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pool.dataclasses import SnapshotActivity
from pool.images import activity_image_uri, daemon_image_uri


def test_daemon_image_uses_gpu_backend_and_current_version(monkeypatch):
    monkeypatch.delenv('KUMO_IMAGE_TAG', raising=False)
    monkeypatch.setattr('pool.images.get_kumo_version', lambda: 'sha-test')

    assert daemon_image_uri().endswith('/kumo-backend-gpu:sha-test')


def test_activity_image_uses_gpu_for_all_gpu_host_work(monkeypatch):
    monkeypatch.delenv('KUMO_IMAGE_TAG', raising=False)
    monkeypatch.setattr('pool.images.get_kumo_version', lambda: 'sha-test')

    assert activity_image_uri(SnapshotActivity.TRAIN).endswith(
        '/kumo-backend-gpu:sha-test')
    assert activity_image_uri(SnapshotActivity.PREDICT).endswith(
        '/kumo-backend-gpu:sha-test')
    assert activity_image_uri(SnapshotActivity.GEN_TT).endswith(
        '/kumo-backend-gpu:sha-test')
    assert activity_image_uri(
        SnapshotActivity.MATERIALIZE_TRAINING_TASK).endswith(
        '/kumo-backend-gpu:sha-test')


def test_daemon_image_prefers_release_tag(monkeypatch):
    monkeypatch.setenv('KUMO_IMAGE_TAG', 'configured-sha')
    monkeypatch.setenv('KUMO_IMAGE_DIGEST', 'sha256:configured-digest')
    monkeypatch.setattr('pool.images.get_kumo_version', lambda: 'baked-sha')

    assert daemon_image_uri().endswith(
        '/kumo-backend-gpu:configured-sha@sha256:configured-digest')
