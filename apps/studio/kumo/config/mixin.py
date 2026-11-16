# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import datetime
import inspect
import marshal
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import IO, Any, Optional

import numpy as np
import pandas as pd
from torch_geometric import is_torch_instance
from torch_geometric.config_mixin import ConfigMixin as PyGConfigMixin
from torch_geometric.config_store import (
    class_from_dataclass,
    dataclass_from_class,
)

from io_utils import kumo_dataclass, logging
from io_utils.locate import locate
from kumo.utils.cast import to_python_scalar

_KUMO_INTERNAL_TIMESTAMP_PREFIX = 'kumo_timestamp_'
_PANDAS_DATE_OFFSET_PREFIX = 'pandas_date_offset_'
_CONFIG_TYPE_KEY = '__kumo_config_type__'
_MARSHALLED_FUNCTION_TYPE = 'marshalled_function_v1'
_MARSHALLED_FUNCTION_CODE_KEY = 'code'
logger = logging.getLogger(__name__)


def _resolve_config_target(target: Any) -> Any:
    if not isinstance(target, str) or not target:
        raise ValueError("config target must be a non-empty string")
    try:
        owner = locate(target)
    except (AttributeError, ImportError, ValueError) as error:
        raise ValueError(f"unsupported config target: {target}") from error
    if owner is None:
        raise ValueError(f"unsupported config target: {target}")
    return owner


class ConfigMixin:
    r"""Enables a class to serialize/deserialize itself to a dataclass.

    Serialization is done through reflection of the class's `__init__`
    function, as well as the fields on the dataclass. Fields on the
    dataclass are automatically passed as function arguments with the
    same name, and vice versa.
    """
    def config(self, data_cls: Any | None = None) -> Any:
        r"""Creates a serializable configuration of the class.

        * Instances of :class:`ConfigMixin` in the configuration will also be
            converted to their corresponding dataclasses by this method calling
            `.config()` recursively.
        * Instances of a dataclass (that are not of `ConfigMixin`) are not
            converted to any other dataclasses, but are left as is.
        """
        data_cls = data_cls or dataclass_from_class(self.__class__)
        if data_cls is None:
            raise ValueError(f"Could not find the configuration class that "
                             f"belongs to '{self.__class__.__name__}'. Please "
                             f"register it in the global configuration store.")

        kwargs: dict[str, Any] = {}
        for field in fields(data_cls):
            key = field.name
            if not hasattr(self, key):
                continue
            kwargs[key] = _recursive_config(getattr(self, key))
        # TODO: customize_config should be before recursive
        return data_cls(**self.customize_config(kwargs))

    @classmethod
    def from_config(cls, cfg: Any, *args, **kwargs) -> Any:
        r"""Instantiates the class from a serializable configuration."""
        if cfg is None:
            return None
        if getattr(cfg, '_target_', None):
            cls = _resolve_config_target(cfg._target_)
        elif isinstance(cfg, dict) and '_target_' in cfg:
            cls = _resolve_config_target(cfg['_target_'])

        data_cls = cfg.__class__
        if not is_dataclass(data_cls):
            data_cls = dataclass_from_class(cls)
            if data_cls is None:
                raise ValueError(
                    f"Could not find the configuration class that "
                    f"belongs to '{cls.__name__}'. Please "
                    f"register it in the global configuration store.")
        field_names = {field.name for field in fields(data_cls)}
        if isinstance(cfg, dict):
            unknown_keys = set(cfg.keys()) - field_names
            if len(unknown_keys) > 0:
                warnings.warn(
                    f"'{cls.__name__}' is being initialized from a "
                    f"configuration that includes the unknown attribute(s) "
                    f"{unknown_keys}, which will be ignored")
            params = {k: v for k, v in cfg.items() if k in field_names}
            # Convert the data class into a `kumo_dataclass` in case it is not
            # one yet. We check for the existence of `pydantic` attributes here
            # since there seems to exist no cleaner way to do it. In pydantic
            # v2 a pydantic dataclass is marked with `__is_pydantic_dataclass__`
            # (v1 used `__pydantic_model__`, which no longer exists).
            if '__is_pydantic_dataclass__' not in data_cls.__dict__:
                data_cls = kumo_dataclass(data_cls)
            cfg = data_cls(**params)

        # A pydantic v2 dataclass synthesizes
        # ``__init__(__dataclass_self__, *args, **kwargs)``, so plainly
        # reflecting ``cls.__init__`` would surface only ``__dataclass_self__``
        # and none of the real fields would be matched below (leaving
        # ``cls(**kwargs)`` unpopulated and failing pydantic validation).
        # ``_real_dataclass_init_signature`` temporarily restores the true
        # field signature for dataclasses; for non-dataclass classes it is a
        # no-op and the plain reflection is used.
        from kumo.utils.dataclass import _real_dataclass_init_signature
        with _real_dataclass_init_signature(cls):
            params = dict(inspect.signature(cls.__init__).parameters)
        params.pop('args', None)
        params.pop('kwargs', None)

        # Transform the values according to the customization function
        field_values: dict[str, Any] = {
            key: getattr(cfg, key)
            for key in field_names
        }
        field_values = cls.customize_from_config(field_values)
        field_names = set(field_values.keys())

        for name, arg in zip(params.keys(), args):
            assert name not in kwargs
            kwargs[name] = arg

        # Iterate over all input arguments of `cls`:
        for key in field_names & set(params.keys()):
            if key not in kwargs:
                kwargs[key] = _recursive_from_config(field_values[key])

        out = cls(**kwargs)  # type: ignore

        # Iterate over all remaining properties of `cls`:
        for key in field_names - set(params.keys()):
            if key == '_target_':
                continue
            setattr(out, key, _recursive_from_config(field_values[key]))

        return out

    def customize_config(self, values: dict[str, Any]) -> dict[str, Any]:
        r"""Customizes the output of :meth:`config`.

            Args:
                values: A dictionary containing a key for each field in the
                    config dataclass, and the value of the field on the
                    :class:`ConfigMixin` subclass

            Returns:
                A dictionary with the same shape as the function argument, but
                    containing modified values for each field, which will be
                    passed to the constructor of the dataclass.
        """
        return values

    @classmethod
    def customize_from_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        r"""Customizes the output of :meth:`from_config`.

            Args:
                values: A dictionary containing a key for each field in the
                    config dataclass, and the value of the field on the config
                    dataclass.

            Returns:
                A dictionary with the same shape as the function argument, but
                    containing modified values for each field, which will be
                    used to construct the original class.
        """
        return values

    def diff_config(self, other: Any, file: Optional[IO] = None):
        r"""Diffs the configurations of two configurations, printing the
        resulting diff output or indicating that no differences were found.

        Args:
            file: destination stream for the diff output. Defaults to the
                process stdout.
        """
        import difflib
        import pprint
        file = file if file is not None else sys.stdout
        cfg = asdict(self.config())
        other_cfg = asdict(other.config())
        if cfg == other_cfg:
            print('No differences found.', file=file)
        else:
            print('\n' + '\n'.join(
                difflib.unified_diff(
                    pprint.pformat(cfg).splitlines(),
                    pprint.pformat(other_cfg).splitlines(), n=6)),
                  file=file)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.config() == other.config()

    def __hash__(self) -> int:
        # Do not overwrite the default `__hash__` method by defining a custom
        # `__eq__` function:
        return super().__hash__()


###############################################################################

ConfigMixins = (ConfigMixin, PyGConfigMixin)


def _recursive_config(value: Any) -> Any:
    if isinstance(value, ConfigMixins):
        return value.config()
    elif is_torch_instance(value, ConfigMixins):
        return value.config()
    elif isinstance(value, (tuple, list)):
        return [_recursive_config(v) for v in value]
    elif isinstance(value, dict):
        return {k: _recursive_config(v) for k, v in value.items()}
    elif isinstance(value, pd.DateOffset):
        return _PANDAS_DATE_OFFSET_PREFIX + str(value.kwds)
    elif isinstance(value, Callable):
        return {
            _CONFIG_TYPE_KEY: _MARSHALLED_FUNCTION_TYPE,
            _MARSHALLED_FUNCTION_CODE_KEY: marshal.dumps(value.__code__),
        }
    elif isinstance(value, np.generic):
        return to_python_scalar(value)
    elif isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
        return _KUMO_INTERNAL_TIMESTAMP_PREFIX + str(value)
    else:
        return value


def _recursive_from_config(value: Any) -> Any:
    cls: Any = None
    if is_dataclass(value):
        if getattr(value, '_target_', None):
            cls = _resolve_config_target(value._target_)
        else:
            cls = class_from_dataclass(value.__class__)
    if isinstance(value, dict) and '_target_' in value:
        cls = _resolve_config_target(value['_target_'])

    if cls is not None and issubclass(cls, ConfigMixins):
        return cls.from_config(value)
    elif cls is not None and is_dataclass(cls):
        # In some cases, `value` is passed as a dictionary:
        # >>> {'_target_': 'kumoapi.encoder.Null', 'na_strategy': 'ZERO'}
        # In other cases, `value` is passed as a dataclass:
        # >>> Null(na_strategy=NAStrategy.ZERO)
        if isinstance(value, cls):
            return value
        return cls(**value)
    elif isinstance(value, dict) and _CONFIG_TYPE_KEY in value:
        if (set(value) != {
                _CONFIG_TYPE_KEY, _MARSHALLED_FUNCTION_CODE_KEY
        } or value[_CONFIG_TYPE_KEY] != _MARSHALLED_FUNCTION_TYPE):
            raise ValueError('Invalid tagged Kumo config payload')
        code = value[_MARSHALLED_FUNCTION_CODE_KEY]
        if not isinstance(code, (bytes, bytearray)):
            raise TypeError('Marshalled function code must be bytes')
        try:
            code_object = marshal.loads(code)
        except (EOFError, TypeError, ValueError) as error:
            raise ValueError('Invalid marshalled function code') from error
        if not isinstance(code_object, types.CodeType):
            raise TypeError('Marshalled function payload must contain code')
        return types.FunctionType(code_object, globals())
    elif isinstance(value, (tuple, list)):
        return [_recursive_from_config(v) for v in value]
    elif isinstance(value, dict):
        return {k: _recursive_from_config(v) for k, v in value.items()}
    elif isinstance(value, (bytes, bytearray)):
        return value
    elif (isinstance(value, str)
          and value.startswith(_KUMO_INTERNAL_TIMESTAMP_PREFIX)):
        time_value = value[len(_KUMO_INTERNAL_TIMESTAMP_PREFIX):]
        if time_value is pd.NaT or time_value is None:
            return pd.NaT
        try:
            # A bare-integer epoch string (e.g. "1659304801388") makes a plain
            # ``pd.Timestamp(time_value)`` parse it as a calendar string and
            # raise ``OutOfBoundsDatetime`` (a ``ValueError`` subclass, not
            # ``TypeError``). ``normalize_timestamp_like_value`` magnitude-infers
            # the unit for a bare epoch and clamps to ``pd.Timestamp.max/min`` on
            # overflow, while formatted date strings still parse normally.
            # The import is function-local on purpose: ``kumo.diskgraph.__init__``
            # transitively re-enters ``kumo.config`` while this low-level
            # ``mixin`` module is still loading, so a top-level import here
            # creates a circular import.
            #
            # yapf and isort disagree about the closing paren below and no
            # formatting satisfies both: isort wants it on its own line (as
            # written), yapf wants it trailing the name. The file is
            # isort-clean and yapf-dirty, so running root yapf here trades
            # one failure for the other. Local only -- root CI runs flake8
            # and isort and no yapf, and pre-commit runs in CI only against
            # the two `packages/` configs -- so this reformats nothing in CI
            # and turns no check red. Leave it as it is.
            from kumo.diskgraph.diskgraph_native_tt_artifacts import (
                normalize_timestamp_like_value,
            )

            # Pass the already-stripped value: for non-numeric strings the
            # helper returns its input unchanged, so it must not still carry the
            # ``kumo_timestamp_`` prefix or ``pd.Timestamp`` would fail to parse.
            normalized = normalize_timestamp_like_value(time_value)
            if normalized is None:
                return pd.NaT
            return pd.Timestamp(normalized)
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError,
                TypeError):
            logger.warning(
                f"Time value {time_value} in config is invalid, using pd.NaT")
            return pd.NaT

    elif (isinstance(value, str)
          and value.startswith(_PANDAS_DATE_OFFSET_PREFIX)):
        date_offset_value = value[len(_PANDAS_DATE_OFFSET_PREFIX):]
        return pd.DateOffset(**ast.literal_eval(date_offset_value))
    else:
        return value


def construct_object_string_from_config(obj: Any) -> str:
    """
    This method takes a config object and returns a string that represents the
    object as if it were in code.

    Currently, it doesn't support nested objects as its input.

    Example:
        >>> # config as a dataclass
        >>> config = Index(na_strategy='separate', min_occ=1)
        >>> construct_object_string_from_config(config)
        "Index(na_strategy='separate', min_occ=1)"
        >>> # config as a dictionary
        >>> config = {
        ...     _target_: 'EncoderClass',
        ...     parameter_1: 5,
        ...     parameter2: 'Stuff',
        ...     p3: [1, 2, 3],
        ... }
        >>> construct_object_string_from_config(config)
        "EncoderClass(parameter_1=5, parameter2='Stuff', p3=[1,2,3])"
        >>> # config as a ConfigMixin object
        >>> config = RandomSplit(ratio=[0.6, 0.2, 0.2])
        >>> construct_object_string_from_config(config)
        'RandomSplit(ratio=[0.6, 0.2, 0.2])'

    This is meant to be used to show complex objects to the user in a nicer
    format that can then be parsed by :meth:`io_utils.yaml.repr_to_object`.
    """
    parameters: list[str] = []

    if isinstance(obj, dict):
        obj_fields = obj
        obj_cls_name = _resolve_config_target(obj['_target_']).__name__
    elif is_dataclass(obj):
        obj_fields = asdict(obj)
        for field in fields(obj):
            if not field.repr:
                del obj_fields[field.name]
        obj_cls_name = obj.__class__.__name__
    else:
        obj_fields = asdict(obj.config())
        obj_cls_name = obj.__class__.__name__

    for key, value in obj_fields.items():
        if key.startswith('_'):
            continue
        if (hasattr(obj, '_external_args')
                and key not in obj._external_args):  # type: ignore
            continue

        if isinstance(value, Enum):
            value = value.value

        if isinstance(value, str):
            parameters.append(f"{key}='{value}'")
        else:
            parameters.append(f'{key}={value}')

    return f"{obj_cls_name}({', '.join(parameters)})"
