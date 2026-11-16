# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from common_utils.error import PublicException
from common_utils.error_diagnostics import (
    exception_chain_contains_restricted_detail,
    exception_message_should_be_logged_at_error,
    safe_exception_message_for_log,
)
from io_utils.errors import (
    RESTRICTED_LOG_MESSAGE_ATTRIBUTE,
    SAFE_LOG_MESSAGE_ATTRIBUTE,
    RestrictedDiagnosticError,
    mark_exception_restricted,
    restricted_diagnostic_boundary,
)


class _RestrictedMarkerError(RuntimeError):
    _kumo_restricted_log_message = True

    def __init__(self, message: str, safe_message: str = '') -> None:
        if safe_message:
            setattr(self, SAFE_LOG_MESSAGE_ATTRIBUTE, safe_message)
        super().__init__(message)


class _RestrictedPublicError(PublicException):
    _kumo_restricted_log_message = True


@pytest.mark.parametrize('module', [
    'requests.exceptions',
    'snowflake.connector.errors',
    'future_provider.errors',
    'diskgraph._c',
])
def test_module_name_does_not_restrict_an_exception(module: str) -> None:
    error_type = type('ProviderError', (Exception, ), {'__module__': module})
    error = error_type('diagnostic detail')

    assert not exception_chain_contains_restricted_detail(error)
    assert exception_message_should_be_logged_at_error(error)


@pytest.mark.parametrize('error', [
    FileNotFoundError('/private/customer/path'),
    UnicodeDecodeError('utf-8', b'private-row-value', 0, 1, 'invalid'),
    SyntaxError('private source expression'),
])
def test_builtin_exception_families_are_visible_until_marked(error) -> None:
    assert exception_message_should_be_logged_at_error(error)


def test_restricted_cause_restricts_a_local_wrapper() -> None:
    provider_error = RuntimeError('authorization=secret')
    mark_exception_restricted(provider_error)
    outer = RuntimeError(f'local wrapper copied: {provider_error}')
    outer.__cause__ = provider_error

    assert exception_chain_contains_restricted_detail(outer)
    assert not exception_message_should_be_logged_at_error(outer)


def test_restricted_cause_repr_restricts_a_local_wrapper() -> None:
    provider_error = RuntimeError('authorization=secret\nsecond-line')
    mark_exception_restricted(provider_error)
    outer = RuntimeError(f'local wrapper copied: {provider_error!r}')
    outer.__cause__ = provider_error

    assert not exception_message_should_be_logged_at_error(outer)


def test_fixed_outer_message_remains_visible_with_restricted_cause() -> None:
    provider_error = RuntimeError('authorization=secret')
    mark_exception_restricted(provider_error)
    outer = RuntimeError('fixed Kumo-authored failure')
    outer.__cause__ = provider_error

    assert exception_chain_contains_restricted_detail(outer)
    assert exception_message_should_be_logged_at_error(outer)
    assert not exception_message_should_be_logged_at_error(provider_error)


def test_suppressed_context_keeps_a_fixed_local_message_visible() -> None:
    provider_error = RuntimeError('authorization=secret')
    mark_exception_restricted(provider_error)
    outer = RuntimeError('fixed Kumo-authored failure')
    outer.__context__ = provider_error
    outer.__suppress_context__ = True

    assert exception_chain_contains_restricted_detail(outer)
    assert exception_message_should_be_logged_at_error(outer)


def test_suppressed_context_restricts_a_verbatim_copied_message() -> None:
    provider_error = RuntimeError('authorization=secret')
    mark_exception_restricted(provider_error)

    try:
        raise provider_error
    except RuntimeError as error:
        try:
            raise RuntimeError(f'provider failed: {error}') from None
        except RuntimeError as outer:
            assert outer.__cause__ is None
            assert outer.__context__ is provider_error
            assert outer.__suppress_context__
            assert exception_chain_contains_restricted_detail(outer)
            assert not exception_message_should_be_logged_at_error(outer)


def test_marker_can_supply_an_explicit_safe_error_message() -> None:
    error = _RestrictedMarkerError(
        'stderr password=secret',
        safe_message='converter failed (exit code 17)',
    )

    assert getattr(type(error), RESTRICTED_LOG_MESSAGE_ATTRIBUTE) is True
    assert not exception_message_should_be_logged_at_error(error)
    assert safe_exception_message_for_log(error) == (
        'converter failed (exit code 17)')


def test_boundary_helper_preserves_exception_identity_and_pickling() -> None:
    error = ValueError('customer-row-canary')

    marked_error = mark_exception_restricted(
        error,
        safe_message='Warehouse operation failed',
    )
    round_tripped = pickle.loads(pickle.dumps(marked_error))

    assert marked_error is error
    assert type(marked_error) is ValueError
    assert not exception_message_should_be_logged_at_error(marked_error)
    assert safe_exception_message_for_log(marked_error) == (
        'Warehouse operation failed')
    assert not exception_message_should_be_logged_at_error(round_tripped)
    assert safe_exception_message_for_log(round_tripped) == (
        'Warehouse operation failed')


def test_boundary_context_marks_and_reraises_same_exception() -> None:
    error = RuntimeError('authorization=secret')

    with pytest.raises(RuntimeError) as raised:
        with restricted_diagnostic_boundary(exception_types=RuntimeError,
                                            safe_message='HTTP request failed'):
            raise error

    assert raised.value is error
    assert safe_exception_message_for_log(error) == 'HTTP request failed'


def test_boundary_context_leaves_nonmatching_exception_visible() -> None:
    error = ValueError('ordinary Kumo failure')

    with pytest.raises(ValueError) as raised:
        with restricted_diagnostic_boundary(
                safe_message='StateDB operation failed',
                exception_types=RuntimeError):
            raise error

    assert raised.value is error
    assert exception_message_should_be_logged_at_error(error)


def test_boundary_context_uses_fixed_fallback_for_unmarkable_exception(
) -> None:
    class UnmarkableError(Exception):
        def __setattr__(self, name, value):
            raise TypeError('metadata disabled')

    with pytest.raises(RestrictedDiagnosticError) as raised:
        with restricted_diagnostic_boundary(
                exception_types=UnmarkableError,
                safe_message='Provider operation failed'):
            raise UnmarkableError('authorization=secret')

    assert str(raised.value) == 'Provider operation failed'
    assert raised.value.original_error_type == 'UnmarkableError'
    assert raised.value.__cause__ is None
    assert not exception_message_should_be_logged_at_error(raised.value)


def test_existing_public_messages_remain_visible() -> None:
    assert exception_message_should_be_logged_at_error(
        PublicException('existing public detail'))


def test_explicit_restricted_marker_overrides_public_compatibility() -> None:
    error = _RestrictedPublicError(message='restricted public detail')

    assert not exception_message_should_be_logged_at_error(error)


def test_restricted_chain_walk_is_cycle_safe() -> None:
    first = RuntimeError('first')
    second = RuntimeError('second')
    first.__cause__ = second
    second.__cause__ = first

    assert not exception_chain_contains_restricted_detail(first)
