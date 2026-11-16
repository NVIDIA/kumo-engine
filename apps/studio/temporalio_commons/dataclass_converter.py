# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
import json
import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Sequence, Type

from pydantic import SecretStr
from temporalio import activity, workflow
from temporalio.api.common.v1 import Payload, Payloads
from temporalio.converter import (
    BinaryNullPayloadConverter,
    BinaryPlainPayloadConverter,
    CompositePayloadConverter,
    DefaultPayloadConverter,
    EncodingPayloadConverter,
    JSONPlainPayloadConverter,
    value_to_type,
)
from temporalio.exceptions import TemporalError

from io_utils.dataclasses import ValidatedStr
from io_utils.json import from_json, to_json
from io_utils.locate import locate
from monitoring.tracing import trace

log = logging.getLogger(__name__)

LARGE_PAYLOAD_TO_WARN: int = 128 << 10  # 128KB of data


def _reject_forbidden_temporal_values(
    value: Any,
    seen: set[int] | None = None,
) -> None:
    """Fail before encoding any object marked as activity-local only."""
    seen = seen or set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, SecretStr):
        raise TypeError(
            'SecretStr values are activity-local and cannot be serialized '
            'into a Temporal payload')
    if getattr(type(value), '__temporal_serializable__', True) is False:
        raise TypeError(
            f'{type(value).__name__} is activity-local and cannot be '
            'serialized into a Temporal payload')
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _reject_forbidden_temporal_values(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_forbidden_temporal_values(key, seen)
            _reject_forbidden_temporal_values(item, seen)
    elif isinstance(value, (list, set, tuple)):
        for item in value:
            _reject_forbidden_temporal_values(item, seen)


class PassthroughPayloadDataConverter(EncodingPayloadConverter):
    """A data converter that passes through raw Payload objects."""
    @property
    def encoding(self) -> str:
        return "binary/passthrough"

    def to_payload(self, value: Any) -> Optional[Payload]:
        if isinstance(value, Payload):
            metadata = dict(value.metadata)  # shallow copy of metadata
            metadata['original_encoding'] = value.metadata["encoding"]
            metadata["encoding"] = self.encoding.encode()
            return Payload(metadata=metadata, data=value.data)
        return None

    def from_payload(
        self,
        payload: Payload,
        type_hint: Optional[Type] = None,
    ) -> Any:
        if payload.metadata.get("encoding") == self.encoding.encode():
            metadata = dict(payload.metadata)  # shallow copy of metadata
            # Restore original encoding if present
            metadata['encoding'] = metadata.pop('original_encoding')
            return Payload(metadata=metadata, data=payload.data)
        raise ValueError("Not a passthrough payload")


# Kumo's custom converter plugin to support json encoding for common Pydantic
# dataclass types.  This converter is also capable of handling container types
# including: 'set', 'list', and 'dict' by recursively applying all possible
# converters.
class DataclassJsonPayloadConverter(EncodingPayloadConverter):
    def __init__(self) -> None:
        super().__init__()
        self._encoding: str = "json/pydanticclass"

        # In order to handle container properly, we need to recursively apply
        # all converters: the "self" gets the first try and then others.
        default_converters = DefaultPayloadConverter().converters
        all_converters = {}
        all_converters["json/pydanticclass".encode()] = self
        all_converters.update(default_converters)
        self._default_converter = DefaultPayloadConverter()
        self._default_converter.converters = all_converters
        self._iterable_types = {'set': set, 'list': list, 'tuple': tuple}

    @property
    def encoding(self) -> str:
        return self._encoding

    # @trace() -- DO NOT trace to_payload() which is recursively invoked.
    def to_payload(self, value: Any) -> Optional[Payload]:
        _reject_forbidden_temporal_values(value)
        start_time = time.perf_counter()
        payload = self._to_payload(value)
        elapsed = time.perf_counter() - start_time
        if payload and (elapsed > 1.0
                        or len(payload.data) > LARGE_PAYLOAD_TO_WARN):
            if activity.in_activity():
                caller = (f'activity {activity.info().workflow_type}.'
                          f'{activity.info().activity_type}')
            else:
                caller = 'Temporal Client'
                try:
                    wf_info = workflow.info()
                    caller = (f'workflow {wf_info.workflow_type}: '
                              f'{wf_info.workflow_id}')
                except (RuntimeError, TemporalError):  # no running event loop
                    pass
            log.warning(
                'Encoded Temporal payload is too large: %d bytes, '
                'invoked by %s, value type: %s, metadata: %s, '
                'serialization took %.3f seconds.', len(payload.data), caller,
                type(value), payload.metadata, elapsed)
        return payload

    def _to_payload(self, value: Any) -> Optional[Payload]:
        """See base class."""
        cls_name = '.'.join(
            [value.__class__.__module__, value.__class__.__name__])
        if dataclasses.is_dataclass(value):
            try:
                json_dump = to_json(value)
                return Payload(
                    metadata={
                        "encoding": self._encoding.encode(),
                        "class": cls_name.encode(),
                    },
                    data=json_dump.encode(),
                )
            except TypeError as err:
                print(f'Type error: {err}')
                print(f'Trying to dump: {value}')

        if isinstance(value, ValidatedStr):
            # Special handling for string subclass types
            return Payload(
                metadata={
                    "encoding": self._encoding.encode(),
                    "class": f'str_{cls_name}'.encode(),
                },
                data=value.encode(),
            )

        if isinstance(value, (list, set, tuple)):
            type_name = type(value).__name__
            if isinstance(value, set):
                # Unfortunately set() is unordered so we need to sort it
                value = sorted(value)
            # For set, tuple, or list, we recursively apply converters to each
            # element
            payloads = self._default_converter.to_payloads(value)
            payloads = Payloads(
                payloads=self._default_converter.to_payloads(value))
            return Payload(
                metadata={
                    'encoding': self._encoding.encode(),
                    'class': f'iterable_{type_name}'.encode(),
                    'binary': ''.encode(),  # mark the data as binary
                },
                data=payloads.SerializeToString())

        if isinstance(value, dict) and len(value) > 0:
            # For dict, we serialize into a payload list of:
            # [key, ..., key, val, ..., val]
            keys = list(value.keys())
            values = list(value.values())
            key_payloads = self._default_converter.to_payloads(keys)
            value_payloads = self._default_converter.to_payloads(values)
            payloads = Payloads(payloads=key_payloads + value_payloads)
            return Payload(
                metadata={
                    'encoding': self._encoding.encode(),
                    'class': 'dict'.encode(),
                    'binary': ''.encode(),  # mark the data as binary
                },
                data=payloads.SerializeToString())

        if isinstance(value, datetime):
            return Payload(
                metadata={
                    'encoding': self._encoding.encode(),
                    'class': datetime.__name__.encode(),
                }, data=value.isoformat().encode())

        # Fall through to try other default converters.
        return None

    # @trace() -- DO NOT trace from_payload() which is recursively invoked.
    def from_payload(
        self,
        payload: Payload,
        type_hint: Optional[Type] = None,
    ) -> Any:
        start_time = time.perf_counter()
        object = self._from_payload(payload, type_hint)
        elapsed = time.perf_counter() - start_time
        if elapsed > 1.0:
            if activity.in_activity():
                caller = (f'activity {activity.info().workflow_type}.'
                          f'{activity.info().activity_type}')
            else:
                caller = 'Temporal Client'
                try:
                    wf_info = workflow.info()
                    caller = (f'workflow {wf_info.workflow_type}: '
                              f'{wf_info.workflow_id}')
                except (RuntimeError, TemporalError):  # no running event loop
                    pass
            log.warning(
                'Temporal payload is too large: %d bytes, '
                'invoked by %s, type_hint: %s, metadata: %s, '
                'deserialization took %.3f seconds.', len(payload.data), caller,
                type_hint, payload.metadata, elapsed)
        return object

    def _from_payload(
        self,
        payload: Payload,
        type_hint: Optional[Type] = None,
    ) -> Any:
        """See base class."""
        try:
            cls_name = payload.metadata['class'].decode()

            if cls_name.startswith('str_'):
                cls_name = cls_name[len('str_'):]
                cls = locate(cls_name)
                return cls(payload.data.decode())

            if cls_name == 'dict':
                payloads = Payloads()
                payloads.MergeFromString(payload.data)
                payload_list = list(payloads.payloads)
                assert len(payload_list) % 2 == 0
                dict_size = len(payload_list) // 2
                assert dict_size > 0
                keys = self._default_converter.from_payloads(
                    payload_list[:dict_size])
                values = self._default_converter.from_payloads(
                    payload_list[dict_size:])
                return {keys[i]: values[i] for i in range(dict_size)}

            if cls_name.startswith('iterable_'):
                cls_name = cls_name[len('iterable_'):]
                payloads = Payloads()
                payloads.MergeFromString(payload.data)
                values = self._default_converter.from_payloads(
                    payloads.payloads)
                return self._iterable_types[cls_name](values)

            if cls_name == datetime.__name__:
                return datetime.fromisoformat(payload.data.decode())

            if type_hint and dataclasses.is_dataclass(type_hint):
                cls = type_hint
            else:
                cls = locate(cls_name)
            return from_json(data=payload.data.decode(), cls=cls)

        except json.JSONDecodeError as err:
            raise RuntimeError("Failed parsing") from err


# pylint: disable=too-few-public-methods
class PatchedJSONPlainPayloadConverter(JSONPlainPayloadConverter):
    """Contains a patch required for the unit test test_convert_string_subtype.

    TODO: Delete me after patch is applied to upstream.
    """
    def from_payload(
        self,
        payload: Payload,
        type_hint: Optional[Type] = None,
    ) -> Any:
        try:
            obj = json.loads(payload.data, cls=self._decoder)
            if type_hint:
                if inspect.isclass(type_hint) and issubclass(type_hint, str):
                    if not isinstance(obj, str):
                        raise TypeError(f"Cannot convert to {type_hint}, value "
                                        f"not a string, value is {type(obj)}")
                    obj = type_hint(obj)
                else:
                    obj = value_to_type(type_hint, obj)
            return obj
        except json.JSONDecodeError as err:
            raise RuntimeError("Failed parsing") from err


# pylint: disable=too-few-public-methods
class KumoPayloadConverter(CompositePayloadConverter):
    def __init__(self) -> None:
        super().__init__(
            PassthroughPayloadDataConverter(),
            BinaryNullPayloadConverter(),
            DataclassJsonPayloadConverter(),
            BinaryPlainPayloadConverter(),
            PatchedJSONPlainPayloadConverter(),
        )

    @trace()
    def to_payloads(self, values: Sequence[Any]) -> List[Payload]:
        return super().to_payloads(values)

    @trace()
    def from_payloads(self, payloads: Sequence[Payload],
                      type_hints: Optional[List[Type]] = None) -> List[Any]:
        return super().from_payloads(payloads, type_hints=type_hints)
