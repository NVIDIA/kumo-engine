# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from temporalio import activity

from statedb.rest.connector_authorization import (
    CONNECTOR_CAPABILITY_ACTION_BY_ACTIVITY_TYPE,
    CONNECTOR_ISSUABLE_ACTIONS,
    ConnectorCapabilityAction,
    connector_capability_action_for_activity_type,
)

_F = TypeVar('_F', bound=Callable[..., Any])
_ACTION_ATTRIBUTE = '__connector_capability_action__'
_RESOLUTION_HELPER_ATTRIBUTE = '__connector_resolution_helper__'
_REGISTERED: dict[ConnectorCapabilityAction, Callable[..., Any]] = {}


def connector_authorized_activity(
    func: _F | None = None,
    *,
    name: str | None = None,
):
    """Define and register one activity under its exact closed action name.

    The Python function name is the Temporal activity type and therefore the
    capability action. Import fails if that name is absent from the closed
    registry, preventing an activity from reaching the credential broker via
    an arbitrary string.
    """
    if func is None:
        return lambda decorated: connector_authorized_activity(
            decorated, name=name)
    activity_type = name or func.__name__
    action = connector_capability_action_for_activity_type(activity_type)
    if action not in CONNECTOR_ISSUABLE_ACTIONS:
        raise RuntimeError(
            f'Connector activity {activity_type!r} has no issuance policy')
    existing = _REGISTERED.get(action)
    if existing is not None:
        raise RuntimeError(
            f'Connector activity action {action.value!r} is already bound')
    setattr(func, _ACTION_ATTRIBUTE, action)
    defined = activity.defn(name=action.value)(func)
    setattr(defined, _ACTION_ATTRIBUTE, action)
    _REGISTERED[action] = defined
    return defined  # type: ignore[return-value]


def connector_resolution_helper(func: _F) -> _F:
    """Mark a non-activity helper that directly invokes the broker."""
    setattr(func, _RESOLUTION_HELPER_ATTRIBUTE, True)
    return func


def registered_connector_activities(
) -> dict[ConnectorCapabilityAction, Callable[..., Any]]:
    return dict(_REGISTERED)


def validate_connector_activity_registration(
    activities: Iterable[Callable[..., Any]], ) -> None:
    """Fail worker startup if a closed capability activity is misdecorated."""
    for activity_func in activities:
        definition = activity._Definition.must_from_callable(activity_func)
        action = CONNECTOR_CAPABILITY_ACTION_BY_ACTIVITY_TYPE.get(
            definition.name)
        declared = getattr(activity_func, _ACTION_ATTRIBUTE, None)
        if action is None:
            if declared is not None:
                raise RuntimeError(
                    f'Activity {definition.name!r} declares unknown connector '
                    'authority')
            continue
        if declared != action:
            raise RuntimeError(
                f'Connector activity {definition.name!r} is not bound to its '
                'closed capability action')
        if action not in CONNECTOR_ISSUABLE_ACTIONS:
            raise RuntimeError(
                f'Connector activity {definition.name!r} has no issuance '
                'policy')
