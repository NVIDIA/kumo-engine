# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from logging import WARNING

import pytest

from io_utils.retry import no_retry, retry


def test_reraise(caplog):
    def _matcher(e):
        return isinstance(e, ValueError)

    # map ValueError -> RuntimeError
    def _handler(e):
        assert isinstance(e, ValueError)
        raise RuntimeError from e

    @retry(
        exceptions=(KeyError, ValueError),
        callbacks={_matcher: _handler},
        tries=2,
        delay=0,
        jitter=0,
    )
    def _raiser(etype):
        raise etype()

    with caplog.at_level(WARNING):
        # matcher ignores KeyError, it gets through
        with pytest.raises(KeyError):
            _raiser(KeyError)
    # 2 tries -> 1 retry
    assert caplog.text.count("retrying in") == 1
    caplog.clear()

    with caplog.at_level(WARNING):
        # matcher matches ValueError, it gets re-raised as RuntimeError
        with pytest.raises(RuntimeError):
            _raiser(ValueError)
    # re-raise -> 0 retries
    assert len(caplog.text) == 0


def test_no_retry_skips_retries(caplog):
    @retry(exceptions=ValueError, tries=5, delay=0, jitter=0)
    def _fail():
        raise ValueError('x')

    with caplog.at_level(WARNING):
        with pytest.raises(ValueError):
            _fail()
    assert caplog.text.count('retrying in') == 4

    caplog.clear()
    with caplog.at_level(WARNING):
        with pytest.raises(ValueError):
            with no_retry():
                _fail()
    assert caplog.text.count('retrying in') == 0


def test_no_retry_nested(caplog):
    @retry(exceptions=ValueError, tries=5, delay=0, jitter=0)
    def _fail():
        raise ValueError('x')

    with caplog.at_level(WARNING):
        with pytest.raises(ValueError):
            with no_retry():
                with no_retry():
                    _fail()
    assert caplog.text.count('retrying in') == 0


def test_no_retry_restores_after_exit(caplog):
    @retry(exceptions=ValueError, tries=5, delay=0, jitter=0)
    def _fail():
        raise ValueError('x')

    with caplog.at_level(WARNING):
        with pytest.raises(ValueError):
            with no_retry():
                _fail()
        caplog.clear()
        with pytest.raises(ValueError):
            _fail()
    assert caplog.text.count('retrying in') == 4


def test_no_retry_applies_to_transitive_calls(caplog):
    @retry(exceptions=ValueError, tries=5, delay=0, jitter=0)
    def _fail():
        raise ValueError('x')

    def _call_fail():
        _fail()

    with caplog.at_level(WARNING):
        with pytest.raises(ValueError):
            with no_retry():
                _call_fail()
    assert caplog.text.count('retrying in') == 0


def test_no_retry_flaky_fn_single_attempt_under_context(caplog):
    attempts = []

    @retry(exceptions=ValueError, tries=3, delay=0, jitter=0)
    def _flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError()
        return 'ok'

    with pytest.raises(ValueError):
        with no_retry():
            _flaky()
    assert len(attempts) == 1

    attempts.clear()
    assert _flaky() == 'ok'
    assert len(attempts) == 2


def test_retry_log_never_interpolates_exception_text(caplog):
    provider_body = 'provider-body password=retry-secret-canary'

    @retry(exceptions=RuntimeError, tries=2, delay=0, jitter=0)
    def _fail():
        raise RuntimeError(provider_body)

    with caplog.at_level(WARNING), pytest.raises(RuntimeError):
        _fail()

    assert provider_body not in caplog.text
    assert 'retry-secret-canary' not in caplog.text
    assert 'error_type=RuntimeError' in caplog.text
    assert caplog.text.count('retrying in') == 1
