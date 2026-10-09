# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    SourceNotFoundError,
)
from kumo_s3.errors import translated


def test_a_profile_no_config_file_defines_is_a_credential_failure():
    from botocore.exceptions import ProfileNotFound

    with pytest.raises(ReconnectRequiredError, match="not-a-profile"):
        with translated("resolving a credential"):
            raise ProfileNotFound(profile="not-a-profile")


def test_an_absent_credential_is_a_credential_failure():
    from botocore.exceptions import NoCredentialsError

    with pytest.raises(ReconnectRequiredError):
        with translated("resolving a credential"):
            raise NoCredentialsError()


def test_any_other_boto_failure_is_permanent_rather_than_unclassified():
    from botocore.exceptions import NoRegionError

    with pytest.raises(PermanentSourceError) as raised:
        with translated("resolving a credential"):
            raise NoRegionError()
    assert type(raised.value) is PermanentSourceError


def test_a_source_error_raised_inside_is_left_as_it_is():
    with pytest.raises(SourceNotFoundError):
        with translated("resolving a credential"):
            raise SourceNotFoundError("absent")


def test_the_vendor_cause_is_not_chained_onto_the_reported_error():
    """A chained cause prints the vendor text, which may carry a role ARN or a
    token endpoint."""
    from botocore.exceptions import NoCredentialsError

    with pytest.raises(ReconnectRequiredError) as raised:
        with translated("resolving a credential"):
            raise NoCredentialsError()
    assert raised.value.__cause__ is None
