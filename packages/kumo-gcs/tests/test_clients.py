# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from kumo_gcs.clients import ThreadClients


def _factory():
    created = []

    def make():
        client = mock.Mock()
        created.append(client)
        return client

    return make, created


def test_a_thread_reuses_its_client():
    make, created = _factory()
    clients = ThreadClients(make)

    assert clients.get() is clients.get()
    assert len(created) == 1
    created[0].close.assert_not_called()


def test_a_client_is_closed_when_its_thread_exits():
    make, created = _factory()
    clients = ThreadClients(make)

    thread = threading.Thread(target=clients.get)
    thread.start()
    thread.join()

    assert len(created) == 1
    created[0].close.assert_called_once_with()
    assert clients.live_count() == 0


def test_repeated_short_lived_pools_leave_no_open_clients():
    make, created = _factory()
    clients = ThreadClients(make)
    ready = threading.Barrier(4)

    def work():
        clients.get()
        ready.wait()

    for _ in range(5):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for future in [executor.submit(work) for _ in range(4)]:
                future.result()

    assert len(created) == 20
    assert all(client.close.call_count == 1 for client in created)
    assert clients.live_count() == 0


def test_close_releases_clients_held_by_live_threads():
    make, created = _factory()
    clients = ThreadClients(make)
    acquired = threading.Event()
    release = threading.Event()

    def hold():
        clients.get()
        acquired.set()
        release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    acquired.wait()
    clients.get()

    clients.close()

    assert len(created) == 2
    assert all(client.close.call_count == 1 for client in created)
    release.set()
    thread.join()
    assert all(client.close.call_count == 1 for client in created)
    assert clients.get() is not created[1]


def test_a_failing_close_does_not_propagate():
    client = mock.Mock()
    client.close.side_effect = RuntimeError('boom')
    clients = ThreadClients(lambda: client)
    clients.get()

    clients.close()

    client.close.assert_called_once_with()
