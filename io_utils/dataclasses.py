# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import typing

import pydantic
import pydantic.dataclasses
from pydantic import ConfigDict
from pydantic_core import core_schema

# Present the Pydantic decorator as the standard-library dataclass decorator to
# static type checkers, while retaining Pydantic validation at runtime.
if typing.TYPE_CHECKING:
    from dataclasses import dataclass as pydantic_dataclass  # noqa
else:
    from pydantic.dataclasses import dataclass as pydantic_dataclass


# NOTE: pydantic v1 required monkeypatching ``pydantic.main.is_valid_field`` to
# allow underscore-prefixed dataclass fields (e.g. ``__root__``) and to skip the
# OpenAI ``_options``/``_model``/``_client`` BaseModel fields. Pydantic v2 no
# longer exposes ``is_valid_field`` and treats underscore-prefixed names as
# private attributes rather than rejecting them, so the monkeypatch is both
# impossible (the symbol is gone) and unnecessary. It has been removed.


# In pydantic v2 ``arbitrary_types_allowed`` (and all model/dataclass config)
# is expressed via ``ConfigDict`` rather than a config class. We keep the
# historical names as ``ConfigDict`` instances so existing
# ``config=AllowArbitraryTypesConfig`` call sites keep working unchanged.
AllowArbitraryTypesConfig = ConfigDict(arbitrary_types_allowed=True)
"""Use this config for dataclasses with members that don't need type
checking.  For e.g. "no validator found for <class
'pandas.core.frame.DataFrame'>"
"""


class ArbitraryTypesBaseModel(pydantic.BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def before_validator_dict(data: typing.Any) -> dict:
    """Normalize the input of a ``mode='before'`` model validator to a dict.

    pydantic v1 ``@root_validator(pre=True)`` always received a dict of field
    values. In v2 a ``mode='before'`` model validator on a *dataclass* receives
    either a dict (when validated from a mapping, e.g. ``validate_python({...})``
    during deserialization) or a ``pydantic_core.ArgsKwargs`` (when constructed
    positionally/by keyword via ``Cls(...)``). This collapses both forms back to
    a plain dict so the migrated validator bodies can keep using dict access.
    """
    if isinstance(data, dict):
        # Return a shallow copy so a downstream validator that mutates the dict
        # (e.g. ``values.pop('_type', None)`` in the credentials validators)
        # does not mutate the caller's input mapping. Re-validating an
        # already-deserialized dict would otherwise strip keys in place.
        return dict(data)
    args = getattr(data, 'args', None)
    kwargs = getattr(data, 'kwargs', None)
    if args is None and kwargs is None:
        # An already-constructed model/dataclass instance can reach a
        # ``mode='before'`` validator during a credentials-union round-trip /
        # re-validation. The migrated validator bodies expect dict access
        # (e.g. ``values.pop('_type', None)``), so coerce the instance back to
        # a plain field dict here rather than returning it unchanged.
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return typing.cast(dict, dataclasses.asdict(data))
        if isinstance(data, pydantic.BaseModel):
            # A pydantic v2 model exposes its fields via iteration as
            # ``(name, value)`` pairs; ``dict(model)`` yields a shallow field
            # dict with nested model values left as objects (matching how the
            # dict-input path above preserves nested values), which is what the
            # consuming validators expect for dict access.
            return dict(data)
        # The consuming ``mode='before'`` validators always perform dict access
        # (``values.pop(...)`` / ``values.items()``), so a non-mapping that is
        # neither a dataclass instance nor a BaseModel cannot be normalized and
        # would otherwise crash later with an obscure attribute error. Surface
        # it here and keep the return type provably a dict.
        raise TypeError(
            f'before_validator_dict cannot normalize input of type '
            f'{type(data).__name__!r} to a dict')
    # These config dataclasses are always constructed by keyword / from a
    # mapping. Positional args cannot be mapped back to field names here, so
    # silently dropping them (the previous behaviour) would turn ``K(512)``
    # into ``K()`` with all-default fields. Surface it as a clear error rather
    # than discarding the value.
    if args:
        raise TypeError(
            'positional arguments are not supported when constructing '
            'this config; pass fields by keyword (got '
            f'{len(args)} positional argument(s))')
    return dict(kwargs or {})


def _hide_marked_fields(schema: dict, *_: typing.Any) -> None:
    """Strip fields marked with ``hidden=True`` from the generated JSON schema.

    This is particularly useful for the public REST API to hide certain
    internal dataclass fields from the public OpenAPI schema shown in swagger
    docs. In pydantic v2 this is wired through ``json_schema_extra`` (a callable
    that mutates the generated schema). Pydantic may invoke it with either one
    argument (the schema) or two (schema, model class), so we accept extras.
    """
    props = {}
    for k, v in schema.get('properties', {}).items():
        if isinstance(v, dict) and v.get('hidden', False):
            continue
        props[k] = v
    schema["properties"] = props


# Config applied by ``kumo_dataclass``. In pydantic v2 this is a ``ConfigDict``
# whose ``json_schema_extra`` callback performs the hidden-field stripping that
# v1 expressed via ``Config.schema_extra``. ``arbitrary_types_allowed`` matches
# the v1 behavior where ``kumo_dataclass`` fields could be typed with classes
# pydantic cannot build a validator for (e.g. PyG ``NumNeighbors``); v2 raises
# ``PydanticSchemaGenerationError`` for those unless this is set.
AllowHiddenFieldConfig = ConfigDict(json_schema_extra=_hide_marked_fields,
                                    arbitrary_types_allowed=True)


if typing.TYPE_CHECKING:
    from dataclasses import dataclass as kumo_dataclass  # noqa # type: ignore
else:

    def kumo_dataclass(cls=None, /, *, init=True, repr=True, eq=True,
                       order=False, unsafe_hash=False, frozen=False):
        """This is the recommended dataclass to use at Kumo, for any data that
        is serialized or shared across services. It uses pydantic dataclasses,
        to support json serialization and deserialization.

        For data that is not serialized anywhere, feel free to use the standard
        library dataclass.

        To ensure proper forward and backwards compatibility, all engineers
        must follow these rules when making changes to existing dataclasses:

            - All newly added fields must have default values (does not apply
              for new dataclasses)
            - Renaming field and data type change is forbidden (including
              adding options to a Union). Please add a new field instead.
            - Deprecating a field is a multi-step process:
                - First delete all of the code that uses the field (but keep
                  the field in the dataclass definition, but leave a TODO
                  comment saying "this field should be removed after all prod
                  releases are above version x.y.z".
                - Wait until this is deployed to 100% of production. This
                  includes waiting for all customers to retrain their active
                  models with the new version of code.
                - Then delete the field from the dataclass.

        For more details, see https://github.com/kumo-ai/kumo/issues/4144

        In situations where kumo_dataclass does not meet your need, feel free
        to use pydantic_dataclass. Just note that you may need to add # type:
        ignore as our current version of python is not compatible with pydantic
        type hints.

        TODO: Add automated tests & validations to enforce the above rules
        """
        # Pydantic v2's ``dataclass`` only accepts ``init=False`` -- it always
        # synthesizes its own validating ``__init__`` and does not forward an
        # ``init=True`` flag (that was a v1 concept). We therefore drop the
        # ``init`` argument when delegating; the ``init`` parameter is kept on
        # this shim's signature purely for backwards-compatible call sites.
        # NOTE: call sites that previously relied on ``init=False`` to provide
        # their OWN ``__init__`` no longer get that behavior in v2 (pydantic
        # generates the init regardless). Those sites must be migrated to
        # ``@model_validator(mode='before')`` / default-factory patterns.
        del init
        return pydantic_dataclass(cls, repr=repr, eq=eq,
                                  order=order, unsafe_hash=unsafe_hash,
                                  frozen=frozen, config=AllowHiddenFieldConfig)


class ValidatedStr(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: typing.Any,
                                     handler: typing.Any):
        return core_schema.no_info_after_validator_function(
            cls._validate, core_schema.str_schema())

    @classmethod
    def _validate(cls, value: str):
        return cls(value)
