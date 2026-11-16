# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure input validation and planning for coordinated devserver launches.

This module deliberately does not inspect or bind ports, acquire locks, write
manifests, or start processes.  The coordinator owns those stateful steps.
Keeping the first phase pure makes the public launch contract testable before
any local resource is touched.
"""

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from rest.devserver.cgroup import CgroupLimits
from rest.devserver.instance import DevserverInstance

_CHILD_ID_RE = re.compile(r'^[a-z][a-z0-9-]{0,47}$')
_AUTO = 'auto'
_MAX_TCP_PORT = 65535


class DevserverPlanError(ValueError):
    pass


@dataclass(frozen=True)
class PlannedDevserver:
    """One child devserver after pure validation and normalization."""

    devserver_id: str
    base_port: Optional[int]
    cgroup_limits: Optional[CgroupLimits]

    @property
    def expected_ports(self) -> Mapping[str, int]:
        """Return the complete nonlegacy port map when a port is explicit."""
        if self.base_port is None:
            return {}
        return DevserverInstance(self.base_port,
                                 child_id=self.devserver_id).ports()


@dataclass(frozen=True)
class MultiDevserverLaunchPlan:
    """Validated input to the stateful multi-devserver coordinator."""

    cgroup_mode: str
    devservers: Tuple[PlannedDevserver, ...]

    @property
    def nums(self) -> int:
        return len(self.devservers)


def _normalize_devserver_id(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DevserverPlanError(f'{field} must be a string')
    normalized = unicodedata.normalize('NFKC', value).casefold()
    if not _CHILD_ID_RE.fullmatch(normalized):
        raise DevserverPlanError(
            f'{field} must normalize to ASCII lowercase '
            '^[a-z][a-z0-9-]{0,47}$')
    return normalized


def _split_csv(value: Optional[Union[str, Sequence[Any]]], field: str
               ) -> Optional[Tuple[Any, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().casefold() == _AUTO:
            return None
        items = tuple(item.strip() for item in value.split(','))
    else:
        items = tuple(value)
    if not items or any(item == '' for item in items):
        raise DevserverPlanError(f'{field} must be a non-empty comma-separated list')
    return items


def _parse_num(value: Optional[Union[str, int]], field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DevserverPlanError(f'{field} must be a positive integer')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DevserverPlanError(f'{field} must be a positive integer') from error
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise DevserverPlanError(f'{field} must be a positive integer')
    return parsed


def _unique_ids(values: Iterable[Any], field: str) -> Tuple[str, ...]:
    ids = tuple(_normalize_devserver_id(value, field) for value in values)
    if len(ids) != len(set(ids)):
        raise DevserverPlanError(
            f'{field} contains duplicate IDs after normalization')
    return ids


def _json_object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DevserverPlanError(
                f'CGROUP_CONFIG contains duplicate JSON key {key!r}')
        result[key] = value
    return result


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevserverPlanError(f'{field} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise DevserverPlanError(f'{field} must be a finite positive number')
    return value


def _parse_custom_cgroup_config(
        value: str) -> Tuple[Tuple[str, ...], Mapping[str, CgroupLimits]]:
    try:
        parsed = json.loads(value, object_pairs_hook=_json_object_pairs)
    except json.JSONDecodeError as error:
        raise DevserverPlanError(
            'CGROUP_CONFIG must be "auto" or a JSON object keyed by '
            'devserver ID') from error
    if not isinstance(parsed, dict) or not parsed:
        raise DevserverPlanError(
            'CGROUP_CONFIG JSON must be a non-empty object keyed by devserver ID')

    raw_ids = tuple(parsed)
    ids = _unique_ids(raw_ids, 'CGROUP_CONFIG key')
    limits = {}
    for raw_id, devserver_id in zip(raw_ids, ids):
        allocation = parsed[raw_id]
        if not isinstance(allocation, dict):
            raise DevserverPlanError(
                f'CGROUP_CONFIG[{raw_id!r}] must be an object with cpus and '
                'memory_gib')
        unexpected = set(allocation) - {'cpus', 'memory_gib'}
        missing = {'cpus', 'memory_gib'} - set(allocation)
        if missing or unexpected:
            raise DevserverPlanError(
                f'CGROUP_CONFIG[{raw_id!r}] must contain exactly cpus and '
                f'memory_gib (missing={sorted(missing)}, '
                f'unexpected={sorted(unexpected)})')
        limits[devserver_id] = CgroupLimits(
            cpus=_positive_number(allocation['cpus'],
                                  f'CGROUP_CONFIG[{raw_id!r}].cpus'),
            memory_gib=_positive_number(
                allocation['memory_gib'],
                f'CGROUP_CONFIG[{raw_id!r}].memory_gib'))
    return ids, limits


def _parse_base_ports(value: Optional[Union[str, Sequence[Any]]], count: int,
                      ids: Tuple[str, ...]) -> Tuple[Optional[int], ...]:
    values = _split_csv(value, 'BASE_PORTS')
    if values is None:
        return (None,) * count
    if len(values) != count:
        raise DevserverPlanError(
            f'BASE_PORTS has {len(values)} values but launch count is {count}')

    ports = []
    port_maps = []
    for raw_port, devserver_id in zip(values, ids):
        if isinstance(raw_port, bool):
            raise DevserverPlanError('BASE_PORTS values must be integer TCP ports')
        try:
            port = int(raw_port)
        except (TypeError, ValueError) as error:
            raise DevserverPlanError(
                'BASE_PORTS values must be integer TCP ports') from error
        if str(port) != str(raw_port).strip() or port < 1:
            raise DevserverPlanError('BASE_PORTS values must be integer TCP ports')
        port_map = DevserverInstance(port, child_id=devserver_id).ports()
        overflow = sorted(name for name, mapped in port_map.items()
                          if mapped > _MAX_TCP_PORT)
        if overflow:
            raise DevserverPlanError(
                f'BASE_PORTS value {port} overflows TCP port range for '
                f'{", ".join(overflow)}')
        ports.append(port)
        port_maps.append(set(port_map.values()))

    if len(ports) != len(set(ports)):
        raise DevserverPlanError('BASE_PORTS contains duplicate base ports')
    for index, port_map in enumerate(port_maps):
        for other_index, other_port_map in enumerate(port_maps[:index]):
            overlap = sorted(port_map & other_port_map)
            if overlap:
                raise DevserverPlanError(
                    f'BASE_PORTS values {ports[other_index]} and {ports[index]} '
                    f'overlap on required port(s) {overlap}')
    return tuple(ports)


def build_multi_devserver_plan(
    cgroup_config: Optional[str] = None,
    nums: Optional[Union[str, int]] = None,
    devserver_ids: Optional[Union[str, Sequence[Any]]] = None,
    base_ports: Optional[Union[str, Sequence[Any]]] = None,
    *,
    unfenced: bool = False,
) -> MultiDevserverLaunchPlan:
    """Validate a multi-devserver request without touching local resources.

    ``CGROUP_CONFIG`` is either ``auto`` (the coordinator calculates limits
    later) or JSON keyed by stable child devserver ID.  Omitted/``auto``
    ``DEVSERVER_IDS`` generates stable ``devserver-N`` IDs only for auto
    allocation; JSON keys are the IDs for explicit allocations.  Omitted or
    ``auto`` ``BASE_PORTS`` leaves address allocation to the coordinator.
    """
    if unfenced and cgroup_config is not None and cgroup_config.strip():
        raise DevserverPlanError(
            'UNFENCED is mutually exclusive with CGROUP_CONFIG')
    if not unfenced and (not isinstance(cgroup_config, str)
                         or not cgroup_config.strip()):
        raise DevserverPlanError('CGROUP_CONFIG is required')
    parsed_nums = _parse_num(nums, 'NUMS')
    configured_ids = _split_csv(devserver_ids, 'DEVSERVER_IDS')

    if unfenced:
        if parsed_nums is None:
            raise DevserverPlanError('UNFENCED=1 requires NUMS')
        if configured_ids is None:
            ids = tuple(f'devserver-{index}'
                        for index in range(1, parsed_nums + 1))
        else:
            ids = _unique_ids(configured_ids, 'DEVSERVER_IDS')
            if len(ids) != parsed_nums:
                raise DevserverPlanError(
                    f'DEVSERVER_IDS has {len(ids)} values but NUMS is '
                    f'{parsed_nums}')
        limits_by_id = {devserver_id: None for devserver_id in ids}
        cgroup_mode = 'unfenced'
    elif cgroup_config.strip().casefold() == _AUTO:
        if parsed_nums is None:
            raise DevserverPlanError('CGROUP_CONFIG=auto requires NUMS')
        if configured_ids is None:
            ids = tuple(f'devserver-{index}'
                        for index in range(1, parsed_nums + 1))
        else:
            ids = _unique_ids(configured_ids, 'DEVSERVER_IDS')
            if len(ids) != parsed_nums:
                raise DevserverPlanError(
                    f'DEVSERVER_IDS has {len(ids)} values but NUMS is '
                    f'{parsed_nums}')
        limits_by_id = {devserver_id: None for devserver_id in ids}
        cgroup_mode = _AUTO
    else:
        config_ids, custom_limits = _parse_custom_cgroup_config(cgroup_config)
        if parsed_nums is not None and parsed_nums != len(config_ids):
            raise DevserverPlanError(
                f'NUMS is {parsed_nums} but CGROUP_CONFIG has '
                f'{len(config_ids)} allocations')
        if configured_ids is not None:
            ids = _unique_ids(configured_ids, 'DEVSERVER_IDS')
            if len(ids) != len(config_ids):
                raise DevserverPlanError(
                    f'DEVSERVER_IDS has {len(ids)} values but CGROUP_CONFIG '
                    f'has {len(config_ids)} allocations')
            if set(ids) != set(config_ids):
                raise DevserverPlanError(
                    'DEVSERVER_IDS must exactly match CGROUP_CONFIG keys '
                    'after normalization')
        else:
            ids = config_ids
        limits_by_id = custom_limits
        cgroup_mode = 'custom'

    ports = _parse_base_ports(base_ports, len(ids), ids)
    return MultiDevserverLaunchPlan(
        cgroup_mode=cgroup_mode,
        devservers=tuple(
            PlannedDevserver(devserver_id=devserver_id,
                             base_port=base_port,
                             cgroup_limits=limits_by_id[devserver_id])
            for devserver_id, base_port in zip(ids, ports)))
