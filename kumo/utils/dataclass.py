# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import inspect
from typing import Any, Iterator, TypeVar, cast, get_args, get_origin

import pydantic
from httpx import Response as HTTPXResponse
from pydantic import TypeAdapter
from requests import Response
from torch_geometric.config_store import to_dataclass as to_default_dataclass

from io_utils import kumo_dataclass
from io_utils.json import from_json, from_json_dict


def _field_signature(cls: Any) -> inspect.Signature:
    """Rebuild an ``__init__``-style signature from a dataclass's real fields."""
    params = [
        inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default: Any = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = f.default_factory()  # type: ignore[misc]
        else:
            default = inspect.Parameter.empty
        params.append(
            inspect.Parameter(f.name,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              default=default, annotation=f.type))
    return inspect.Signature(params)


@contextlib.contextmanager
def _real_dataclass_init_signature(cls: Any) -> Iterator[None]:
    """Temporarily expose a dataclass's real field signature on ``__init__``.

    ``torch_geometric.config_store.to_dataclass`` enumerates the schema fields
    from ``inspect.signature(cls.__init__)``. A pydantic v2 dataclass synthesizes
    a generic validating ``__init__(__dataclass_self__, *args, **kwargs)``, so PyG
    only sees ``__dataclass_self__`` and produces a broken config dataclass (no
    real fields, and "not fully defined" when later validated). There is no
    upstream PyG fix yet, so this shim is required: while ``cls`` is being
    converted we set its real field signature on ``__init__`` so PyG sees the
    actual fields, then restore the prior state on exit. It becomes deletable once
    PyG's ``config_store`` enumerates dataclass fields directly.
    """
    if not dataclasses.is_dataclass(cls):
        yield
        return

    init = cls.__init__
    sentinel = object()
    previous = getattr(init, '__signature__', sentinel)
    try:
        init.__signature__ = _field_signature(cls)  # type: ignore
        yield
    finally:
        if previous is sentinel:
            with contextlib.suppress(AttributeError):
                del init.__signature__  # type: ignore
        else:
            init.__signature__ = previous  # type: ignore


# patch to_dataclass to convert returned Python dataclasses to Pydantic
# dataclasses by default
def to_dataclass(cls: Any, *args,
                 **kwargs) -> 'pydantic.dataclasses.DataclassClassOrWrapper':
    with _real_dataclass_init_signature(cls):
        return kumo_dataclass(to_default_dataclass(cls, *args, **kwargs))


ResponseType = TypeVar('ResponseType')


def parse_dataclass(response_type: type[ResponseType],
                    response: Response | HTTPXResponse) -> ResponseType:
    """Parse the HTTP response body into a Pydantic data structure.

    Args:
        response_type (Type[ResponseType]): a class definition for a
            Pydantic data class, or a list of Pydantic data classes.
            For example: ExampleDataClass, List[ExampleDataClass]
        response (Response): a requests.Response object containing
            json data in the body

    Returns:
        ResponseType: the parsed response
    """
    # Case 0: no parsing needed
    if response_type == str:
        return response.text  # type: ignore
    if response_type == bool:
        return response.text.lower() == 'true'  # type: ignore

    if response_type == dict:
        k_type, v_type = get_args(response_type)
        assert issubclass(k_type, str)
        data = response.json()
        assert isinstance(data, dict)
        return {
            k: from_json_dict(v, v_type)
            for k, v in data.items()
        }  # type: ignore

    # Case 1: response is a list of pydantic data classes
    if get_origin(response_type) is list:
        list_inner_type = get_args(response_type)[0]
        result = _parse_dataclass_list(list_inner_type, response)
        return cast(ResponseType, result)

    # Case 2: response is pydantic data classes
    result = from_json(response.text, response_type)
    return result


_DataclassType = TypeVar('_DataclassType')


def _parse_dataclass_list(
        data_class: type[_DataclassType],
        response: Response | HTTPXResponse) -> list[_DataclassType]:
    json_response = response.json()
    assert isinstance(json_response, list)

    result: list[_DataclassType] = []
    for item in json_response:
        result.append(TypeAdapter(data_class).validate_python(item))

    return result
