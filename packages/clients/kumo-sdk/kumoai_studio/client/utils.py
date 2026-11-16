# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import requests
from kumoapi.json_serde import from_json

from kumoai_studio.exceptions import HTTPException

logger = logging.getLogger(__name__)

_DataclassT = TypeVar('_DataclassT')
ResponseT = TypeVar('ResponseT')


def _non_json_response_error(response: requests.Response) -> HTTPException:
    r"""Build an actionable error for a response whose body is not JSON.

    Names the request URL, status code, and content-type, and adds a hint
    that a misrouted URL (e.g. an extra ``/api`` segment pointing at the web
    UI) may have returned an HTML page instead of JSON.
    """
    url = getattr(response, 'url', None) or ''
    status_code = response.status_code
    content_type = response.headers.get('content-type', '') or ''
    body = response.text or ''
    looks_html = 'html' in content_type.lower() or body.lstrip()[:1] == '<'
    if looks_html:
        hint = (" The server returned an HTML page, not JSON - your API URL "
                "may be pointing at the web UI instead of the API; check for "
                "an extra '/api' path segment.")
    else:
        hint = " The server did not return a JSON response."
    snippet = ' '.join(body[:200].split())
    detail = (f"Expected a JSON response from {url!r}, but the body could not "
              f"be parsed as JSON (HTTP {status_code}, content-type "
              f"{content_type!r}).{hint}")
    if snippet:
        detail += f" Response began: {snippet!r}"
    # Keep the upstream status for diagnostics. A successful status is
    # deliberate: this reports a response-contract failure, not an HTTP-status
    # failure.
    return HTTPException(status_code, detail, dict(response.headers))


def response_json(response: requests.Response) -> Any:
    r"""Parse a response body as JSON, raising an actionable error if the body
    is not valid JSON.

    Never rejects on content-type alone: valid JSON served with a non-JSON
    content-type (e.g. ``text/html``) still parses, and only a body that
    genuinely fails to parse raises the actionable error.
    """
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise _non_json_response_error(response) from exc


# See https://github.com/python/mypy/issues/9003
class Returns(Generic[ResponseT]):
    r"""Use this class if you'd like to use Python types as annotations for
    mypy, where `Type` is expected. See `parse_response` as an example.
    """
    if not TYPE_CHECKING:

        def __class_getitem__(cls, item: object) -> object:
            """Called when Returns is used as an Annotation at runtime.
            We just return the type we're given
            """
            return item


def _parse_dataclass_list(
    data_class: Type[_DataclassT],
    response: requests.Response,
) -> List[_DataclassT]:
    json_response = response_json(response)
    assert isinstance(json_response, list)

    def _parse_elem(v: Any) -> _DataclassT:
        if data_class in (str, int):
            # `str` and `int` arrive as the element type, and calling one is
            # what parses the element -- which `Type[_DataclassT]`, whose
            # variable is unbounded, cannot say is callable with an argument.
            return cast(Callable[[object], _DataclassT], data_class)(v)
        if data_class == bool:
            return cast(_DataclassT, str(v).lower() == 'true')
        return from_json(v, data_class)

    return [_parse_elem(item) for item in json_response]


@overload
def parse_response(response_type: Type[Returns[ResponseT]],
                   response: requests.Response) -> ResponseT:
    ...


@overload
def parse_response(response_type: Type[ResponseT],
                   response: requests.Response) -> ResponseT:
    ...


def parse_response(response_type: Union[Type[ResponseT],
                                        Type[Returns[ResponseT]]],
                   response: requests.Response) -> ResponseT:
    """Parse the HTTP response body into a Pydantic data structure.

    Args:
        response_type: a class definition for a Pydantic dataclass, or a
            :class:`list` of Pydantic dataclasses.
        response: a requests.Response object containing
            json data in the body

    Returns:
        The parsed response.
    """
    # Case 0: no parsing needed
    if response_type == str:
        return cast(ResponseT, response.text)
    if response_type == bool:
        return cast(ResponseT, response.text.lower() == 'true')

    # Case 1: dict[str, V] type
    if response_type == dict:
        k_type, v_type = get_args(response_type)
        assert issubclass(k_type, str)
        data = response.json()
        assert isinstance(data, dict)
        data = {k: from_json(v, v_type) for k, v in data.items()}
        return cast(ResponseT, data)

    # Case 2: response is a list of pydantic data classes
    if get_origin(response_type) is list:
        list_inner_type = get_args(response_type)[0]
        result = _parse_dataclass_list(list_inner_type, response)
        return cast(ResponseT, result)

    # Case 3: response is pydantic data classes
    try:
        return cast(ResponseT, from_json(response.text, response_type))
    except Exception:
        # Deliberately parse a second time only on this failure path. If this
        # succeeds, bare ``raise`` preserves the original schema error; if it
        # fails, ``response_json`` raises the actionable non-JSON error.
        # Pre-decoding before ``from_json`` would change how it handles
        # top-level JSON strings.
        response_json(response)
        logger.error(f"Unable to parse response {response.text}")
        raise


def parse_id_response(response: requests.Response) -> str:
    return parse_response(Dict[str, str], response)['id']


def parse_patch_response(response: requests.Response) -> bool:
    """Parse PATCH response to indicate whether the resource was modified."""
    return parse_response(Dict[str, bool], response)['resource_updated']


def raise_on_error(response: requests.Response) -> None:
    r"""Raises an :class:`~kumoai_studio.exceptions.HTTPException` if a response does
    not return with an OK status code.
    """
    if not response.ok:
        assert response.status_code is not None
        detail: Any = response.text
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict) and 'detail' in payload:
                detail = payload['detail']
            elif isinstance(payload, (dict, list, str)):
                detail = payload
        raise HTTPException(response.status_code, detail,
                            dict(response.headers))
