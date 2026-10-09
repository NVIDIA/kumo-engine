# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repo-wide guard against the pydantic-v2 bare-arbitrary-type field bug family.

Under pydantic v2 (2.13 / Python 3.14) a model/dataclass with
``arbitrary_types_allowed=True`` -- which is the case for *every*
``@kumo_dataclass`` -- compiles an ``is-instance`` validator for a field typed
with a bare, non-pydantic-native class such as ``pd.Timestamp`` or a numpy
scalar. Such a field serializes fine (``pd.Timestamp`` -> ISO string via the
default encoder; ``np.float64`` -> float), but when the model is *re-validated*
from its serialized form over the AutoRPC ``json/pydanticclass`` codec or from a
job-dir JSON, the raw value is a ``str``/``float`` and the strict
``is_instance_of`` check raises ``ValidationError`` (pydantic v1 silently
coerced; v2 does not). This crashed train-table progress polling and is a likely
cause of "kumo-auto subprocess exited non-zero" worker deaths.

The fix is to annotate such fields with a validator (e.g. ``PandasTimestamp =
Annotated[pd.Timestamp, PlainValidator(...)]``). This test compiles every
pydantic model's core schema and fails if any field still carries a raw
``is-instance`` validator on one of these "serializes-but-fails-strict" types,
so the bug family cannot be silently re-introduced.

Types that CANNOT be serialized by ``io_utils.json.to_json`` at all (e.g.
``pd.DataFrame``, ``np.ndarray``, ``np.int64``) are intentionally *not* flagged:
they crash at encode time (a different, loud failure) and are carried by models
that round-trip through parquet rather than the JSON codec (``EvalOutput`` /
``PredOutput``).
"""
import importlib
import inspect
import pkgutil
import warnings
from typing import List, Tuple

import pydantic
from pydantic import TypeAdapter

# Packages whose models flow over the AutoRPC codec / job-dir JSON.
_TARGET_PACKAGES = [
    'kumo',
    'rest',
    'temporalio_commons',
    'statedb',
    'io_utils',
]

# Fully-qualified class names of the (module, name) form whose ``is-instance``
# schema is acceptable because the value cannot be serialized through the JSON
# codec anyway (they round-trip via parquet, and encode-time crashes loudly).
_ALLOWED_INSTANCE_TYPES = {
    'pandas.core.frame.DataFrame',
    'numpy.ndarray',
}


def _is_pydantic_model(obj: object) -> bool:
    try:
        if inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel):
            return True
    except TypeError:
        # issubclass() raises TypeError on non-class / odd introspection
        # objects; intentionally ignore and fall through to the
        # ``__pydantic_fields__`` attribute check below.
        pass
    return hasattr(obj, '__pydantic_fields__')


def _instance_type_fqn(cls: object) -> str:
    return f'{getattr(cls, "__module__", "")}.{getattr(cls, "__name__", cls)}'


def _walk_schema(schema: object, path: str,
                 found: List[Tuple[str, str]]) -> None:
    if not isinstance(schema, dict):
        return
    if schema.get('type') == 'is-instance':
        cls = schema.get('cls')
        fqn = _instance_type_fqn(cls)
        # Only flag types that pydantic's default encoder WILL serialize but
        # then strict-reject on re-validation: datetime subclasses (pandas
        # Timestamp) and float subclasses (numpy float scalars).
        try:
            import datetime as _dt

            import numpy as _np
            serializable_but_strict = (
                isinstance(cls, type)
                and issubclass(cls, (_dt.datetime, _dt.date, _np.floating)))
        except Exception:
            serializable_but_strict = False
        if serializable_but_strict and fqn not in _ALLOWED_INSTANCE_TYPES:
            found.append((path, fqn))
    for key, value in schema.items():
        if isinstance(value, dict):
            _walk_schema(value, f'{path}.{key}', found)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _walk_schema(item, f'{path}.{key}[{i}]', found)


def _collect_models() -> List[type]:
    warnings.filterwarnings('ignore')
    seen: set = set()
    models: List[type] = []

    def walk_module(modname: str) -> None:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            # Some modules require env (PROD auth, cloud creds) or optional
            # deps to import; their models are defined elsewhere and reached
            # through other modules, so skipping them does not reduce coverage.
            return
        for name in dir(mod):
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            if not inspect.isclass(obj):
                continue
            if getattr(obj, '__module__', None) != modname:
                continue
            if not _is_pydantic_model(obj):
                continue
            key = f'{modname}.{name}'
            if key in seen:
                continue
            seen.add(key)
            models.append(obj)

    for pkg_name in _TARGET_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:
            continue
        walk_module(pkg_name)
        if hasattr(pkg, '__path__'):
            for mod_info in pkgutil.walk_packages(pkg.__path__,
                                                  prefix=pkg_name + '.'):
                walk_module(mod_info.name)
    return models


def test_no_serializable_bare_arbitrary_type_fields() -> None:
    """No pydantic model may carry an unannotated ``pd.Timestamp`` / numpy-float
    field that serializes but fails strict re-validation over the JSON codec."""
    models = _collect_models()
    # Sanity check: the sweep actually imported a large body of models. If this
    # collapses (e.g. an import regression), the guard would silently pass.
    assert len(models) > 100, (
        f'expected to inspect many pydantic models, got {len(models)}; '
        'model discovery likely broke')

    offenders: List[Tuple[str, str, str]] = []
    for model in models:
        qualname = f'{model.__module__}.{model.__name__}'
        try:
            schema = TypeAdapter(model).core_schema
        except Exception:
            continue
        found: List[Tuple[str, str]] = []
        _walk_schema(schema, qualname, found)
        for path, fqn in found:
            offenders.append((qualname, path, fqn))

    assert not offenders, (
        'pydantic-v2 models with bare arbitrary-type fields that serialize but '
        'fail strict re-validation (annotate with a PlainValidator, e.g. '
        '`PandasTimestamp` from kumo/stats/utils.py):\n' +
        '\n'.join(f'  {m}: {p} -> {t}' for m, p, t in offenders))
