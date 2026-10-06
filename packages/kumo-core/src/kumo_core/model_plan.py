# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import itertools
import logging
import os
from dataclasses import field, is_dataclass
from typing import (
    Any,
    ClassVar,
    Optional,
    Sequence,
    TypeVar,
    get_args,
    get_type_hints,
)

import kumoapi.encoder as encoder_api
import kumoapi.model_plan as api
from kumo_core.errors import TrainingValidationError as ValidationException
from kumo_core.types import TaskType
from kumoapi.common import ValidationWarning
from kumoapi.encoder import (
    Datetime,
    Encoder,
    EncoderType,
    Index,
    MultiCategorical,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.model_plan import (
    ActivationType,
    AggregationType,
    DistanceMeasureType,
    EarlyStoppingConfig,
    FocalLossConfig,
    HopConfig,
    HuberLossConfig,
    InferredType,
    IntervalType,
    LinkPredOutputType,
    LossType,
    LRSchedulerConfig,
    Metadata,
    MissingType,
    NAStrategy,
    NormalizationType,
    NumNeighborsConfig,
    OptimizerConfig,
    OptimizerType,
    OrdinalLogLossConfig,
    PastEncoderType,
    PositionalEncodingType,
    PruningConfig,
    QuantileLossConfig,
    RHSEmbeddingMode,
    RunMode,
    WeightMode,
    compat_field,
)
from kumoapi.pquery import QueryType
from kumoapi.typing import Stype, TimeUnit
from pydantic import BaseModel, ValidationError, conlist, field_validator
from pydantic.dataclasses import dataclass
from pydantic_core import InitErrorDetails, PydanticCustomError
from typing_extensions import Self

GraphConfig = Any
ColumnConfig = Any


def _parse_ast_node(node: ast.AST, classes: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                    (ast.UAdd, ast.USub)):
        value = _parse_ast_node(node.operand, classes)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Tuple):
        return tuple(_parse_ast_node(value, classes) for value in node.elts)
    if isinstance(node, ast.List):
        return [_parse_ast_node(value, classes) for value in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_parse_ast_node(value, classes) for value in node.args]
        kwargs = {
            keyword.arg: _parse_ast_node(keyword.value, classes)
            for keyword in node.keywords
        }
        return classes[node.func.id](*args, **kwargs)
    raise ValueError(f'Unsupported configuration expression: {ast.dump(node)}')


def repr_to_object(object_str: str, classes: dict[str, Any]) -> Any:
    tree = ast.parse(object_str)
    return _parse_ast_node(tree.body[0].value,
                           classes)  # type: ignore[attr-defined]


# Under pydantic v2, kumoapi builds NumNeighborsConfig by wrapping a raw
# ``_NumNeighborsConfig`` with ``pydantic.dataclasses.dataclass(...)``, which
# returns a NEW class whose ``__qualname__`` still says ``_NumNeighborsConfig``.
# Pickle-by-reference then resolves ``type(obj).__qualname__`` to the module's
# raw ``_NumNeighborsConfig`` attribute, which is NOT the instance's class, so
# pickling fails when a Temporal activity ships the model plan into a spawn-mode
# ProcessPoolExecutor. Rebind the shadowed attribute to the wrapper so the
# resolved class matches the instance. Idempotent; accepted stopgap.
# Proper fix is upstream in kumoapi (widen/rebind ``_NumNeighborsConfig``
# there); this monkeypatch can be removed once that lands.
api._NumNeighborsConfig = api.NumNeighborsConfig

MIN_INFERRED_NUM_NEIGHBORS = 64
MAX_INFERRED_NUM_NEIGHBORS = 512
RHS_NUM_EMB_SIZE_LIMIT = (5_000_000, 64)
SAMPLE_SIZE_WARN = int(os.getenv('SAMPLE_SIZE_WARN', str(6000 * 1024)))
T = TypeVar("T")

logger = logging.getLogger(__name__)


def _use_decoupled_splits() -> bool:
    return os.getenv('USE_DECOUPLED_SPLITS',
                     '').lower() in {'1', 'true', 'yes', 'on'}


def _raw_time_split_error(split_repr: Any) -> ValueError:
    r"""Error raised when a user supplies a raw, internal-only ``TimeSplit``.

    ``TimeSplit`` is an internal representation; users should use a public
    split family such as ``TemporalSplit(ratio=[0.8, 0.1, 0.1])`` instead.
    """
    return ValueError(
        f"The selected split '{split_repr}' uses `TimeSplit`, which is an "
        "internal representation and is not supported as a user-supplied "
        "split. Use a public split family such as "
        "`TemporalSplit(ratio=[0.8, 0.1, 0.1])`, `RandomSplit`, "
        "`TimeRangeSplit`, or `DateOffsetSplit`.")


# A pydantic v2 "line error": the dict shape consumed by
# `pydantic_core.ValidationError.from_exception_data` and produced by
# `ValidationError.errors()`. The v1 `pydantic.error_wrappers.ErrorList`
# (`ErrorWrapper` / nested lists) was removed in pydantic v2; we now collect
# these dicts directly.
LineError = InitErrorDetails

# The set of enum member values that represent the "missing"/unset sentinel
# (e.g. the `'???'` placeholder). Used to suppress spurious enum-validation
# errors emitted by the `MissingType` arm of a `MissingType | X` union.
_MISSING_ENUM_VALUES = frozenset(member.value for member in MissingType)

_FIELD_UNSET = object()


def _is_missing(value: Any) -> bool:
    r"""Whether `value` represents the "missing"/unset sentinel.

    Under pydantic v2 smart-union, a ``MissingType | X`` field may deserialize
    the sentinel (``'???'``) to a plain ``str`` rather than the ``MissingType``
    enum member, so an ``isinstance(value, MissingType)`` check alone misses it.
    Treat the raw sentinel value(s) as missing too.
    """
    if isinstance(value, MissingType):
        return True
    # `_MISSING_ENUM_VALUES` only holds `str` sentinels, so the membership test
    # is meaningful only for `str` values; gating on `str` also keeps unhashable
    # plan values (e.g. list-typed fields) from raising in the `in` check.
    return isinstance(value, str) and value in _MISSING_ENUM_VALUES


def _is_inferred(value: Any) -> bool:
    r"""Whether `value` represents the "inferred" sentinel.

    Under pydantic v2 smart-union, an ``InferredType | X`` field with the
    ``InferredType.VALUE`` default (``'inferred'``) deserializes to the plain
    ``str`` arm rather than the ``InferredType`` enum member, so an
    ``isinstance(value, InferredType)`` check alone misses it. Treat the raw
    sentinel value as inferred too. The ``str`` guard runs before the ``==``
    so unhashable/list-typed plan values cannot raise.
    """
    if isinstance(value, InferredType):
        return True
    return isinstance(value, str) and value == InferredType.VALUE.value


def _field_default(obj: Any, key: str) -> Any:
    r"""Returns the default value of pydantic-dataclass field `key`.

    Resolves `default_factory` when present so the result matches the value a
    freshly-constructed instance carries. v1's `compat_field` stored a concrete
    `default` (it called the factory eagerly), so the JSON schema exposed it; v2
    keeps the factory and omits the schema `default`. `to_dict` needs the live
    default to decide whether a hidden field is still unset.
    """
    fields = getattr(type(obj), '__pydantic_fields__', None)
    if fields is None or key not in fields:
        return _FIELD_UNSET
    field_info = fields[key]
    if field_info.default_factory is not None:
        return field_info.default_factory()
    return field_info.default


def ExtraFieldsError(valid_top_sections: list[str]) -> PydanticCustomError:
    r"""Builds the custom error raised for a disallowed top-level field.

    Returns a :class:`pydantic_core.PydanticCustomError` (the pydantic v2
    replacement for the v1 `pydantic.errors.PydanticValueError` subclass). The
    error ``type`` is ``'extra_fields'`` and ``valid_top_sections`` is carried
    in the error ``ctx`` (it is interpolated into the message via the
    ``{valid_top_sections}`` placeholder, mirroring the v1 message text).
    """
    return PydanticCustomError(
        'extra_fields',
        'This top-level field is not allowed. '
        'The only allowed fields with no indentation '
        'are {valid_top_sections}. If you are trying to set '
        'a field in a nested section, please add the right amount of '
        'indentation.',
        {'valid_top_sections': valid_top_sections},
    )


def _is_missing_enum_error(error: LineError) -> bool:
    r"""Whether a v2 line error is the spurious enum error contributed by the
    `MissingType` arm of a union (permitted values are only the missing
    sentinel)."""
    if error.get('type') != 'enum':
        return False
    expected = (error.get('ctx') or {}).get('expected')
    if not isinstance(expected, str):
        return False
    # v2 renders the permitted values as e.g. "'???'" (or "'a' or 'b'"). The
    # `MissingType` arm of a union is always a single-member enum, so its
    # rendering is exactly the quoted sentinel value.
    return expected in {repr(value) for value in _MISSING_ENUM_VALUES}


def filter_missing_error(errors: Sequence[LineError], ) -> list[LineError]:
    r"""Filters any validation errors related to `MissingType`.

    In pydantic v2, assigning a bad value to a ``MissingType | X`` union field
    emits one sub-error per union arm, including a spurious ``enum`` error from
    the ``MissingType`` arm ("Input should be '???'"). This drops those, exactly
    as the v1 implementation dropped the corresponding ``EnumMemberError``
    whose permitted ``enum_values`` were all ``MissingType`` members.
    """
    return [error for error in errors if not _is_missing_enum_error(error)]


_INVALID_SYNTAX_PREFIX = "Invalid syntax in model plan:"
_WEIGHT_COL_FORMAT_ERROR = "Weight column must be in table.column form."


def _parse_hop_key(key: str) -> tuple[str, str, str, str]:
    r"""Splits the key of format 'table1.col1->table2.col2' into a tuple
    (table1,col1,table2,col2)."""

    arrow_count = key.count('->')
    if arrow_count == 0:
        raise ModelPlanSyntaxError(
            f"This hop key is missing the '->' separator.\n"
            f"Fix by adding '->' between source and destination:\n"
            f"  {repr(key.replace('.', '.X->Y.', 1) if '.' in key else 'table1.col1->table2.col2')}"
        )
    elif arrow_count > 1:
        fixed_key = '->'.join([part.strip() for part in key.split('->')[:2]])
        raise ModelPlanSyntaxError(
            f"This hop has too many '->'. Only one '->' is allowed.\n"
            f"Fix by removing the extra arrow(s):\n"
            f"  {repr(fixed_key)}")

    parts = key.split('->')
    src = parts[0].strip()
    dst = parts[1].strip()

    if not src:
        raise ModelPlanSyntaxError(
            f"This hop key has an empty source (before '->').\n"
            f"Fix by adding a table.column before '->':\n"
            f"  table1.col1->{repr(dst) if dst else 'table2.col2'}")

    if not dst:
        raise ModelPlanSyntaxError(
            f"This hop key has an empty destination (after '->').\n"
            f"Fix by adding a table.column after '->':\n"
            f"  {repr(src)}->table2.col2")

    if '.' not in src:
        raise ModelPlanSyntaxError(
            f"This hop key source is missing the dot separator.\n"
            f"Fix by adding a dot between table and column:\n"
            f"  table.{repr(src)}->{repr(dst)}")

    if '.' not in dst:
        raise ModelPlanSyntaxError(
            f"This hop key destination is missing the dot separator.\n"
            f"Fix by adding a dot between table and column:\n"
            f"  {repr(src)}->table.{repr(dst)}")

    try:
        src_table_name, src_col_name = src.rsplit('.', maxsplit=1)
        dst_table_name, dst_col_name = dst.rsplit('.', maxsplit=1)
    except ValueError:
        raise ModelPlanSyntaxError(
            "This hop key cannot be parsed into table.column pairs.\n"
            "Fix by ensuring both sides have exactly one dot:\n"
            "  table1.col1->table2.col2")

    if not all([src_table_name, src_col_name, dst_table_name, dst_col_name]):
        empty_parts = []
        if not src_table_name:
            empty_parts.append("source table")
        if not src_col_name:
            empty_parts.append("source column")
        if not dst_table_name:
            empty_parts.append("destination table")
        if not dst_col_name:
            empty_parts.append("destination column")
        raise ModelPlanSyntaxError(
            f"This hop key has empty {', '.join(empty_parts)}.\n"
            f"Fix by ensuring all parts are non-empty:\n"
            f"  table1.col1->table2.col2")

    return (src_table_name, src_col_name, dst_table_name, dst_col_name)


def _parse_weight_col(weight_col: str) -> tuple[str, str]:
    r"""Parse `weight_col` in table.column format."""
    normalized_weight_col = weight_col.strip()
    if not normalized_weight_col:
        raise ValidationException(message="Weight column cannot be empty.")
    if '(' in normalized_weight_col or ')' in normalized_weight_col:
        raise ValidationException(message=_WEIGHT_COL_FORMAT_ERROR)

    parts = normalized_weight_col.split('.')
    if len(parts) != 2:
        raise ValidationException(message=_WEIGHT_COL_FORMAT_ERROR)

    table_name, col_name = parts
    if not table_name or not col_name:
        raise ValidationException(message=_WEIGHT_COL_FORMAT_ERROR)

    return table_name, col_name


def _primary_target_to_entity_paths(
    target_to_entity_paths: list[list[tuple[str, str, str]]] | None,
) -> list[list[tuple[str, str, str]]] | None:
    """Return the target path model-plan policy applies to."""
    if not target_to_entity_paths:
        return target_to_entity_paths
    return target_to_entity_paths[:1]


def _resolve_target_table_name(
    graph: GraphConfig,
    entity_table_names: list[str] | None,
    target_to_entity_paths: list[list[tuple[str, str, str]]] | None,
) -> str | None:
    r"""Resolve the target table name from model plan validation context."""
    entity_table_name = None
    if entity_table_names:
        try:
            entity_table_name = graph[entity_table_names[0]].table_name
        except KeyError:
            entity_table_name = entity_table_names[0]

    primary_paths = _primary_target_to_entity_paths(target_to_entity_paths)
    if primary_paths and len(primary_paths[0]) > 0:
        table_with_fkey_name, _, referenced_table_name = (primary_paths[0][0])
        try:
            table_with_fkey = graph[table_with_fkey_name]
            referenced_table = graph[referenced_table_name]
        except KeyError:
            return entity_table_name

        if entity_table_name is not None:
            if (table_with_fkey.table_name == entity_table_name
                    and referenced_table.table_name != entity_table_name):
                return referenced_table.table_name
            if (referenced_table.table_name == entity_table_name
                    and table_with_fkey.table_name != entity_table_name):
                return table_with_fkey.table_name

        return table_with_fkey.table_name

    return entity_table_name


def _validate_weight_col(
    weight_col: str,
    graph: GraphConfig | None,
    entity_table_names: list[str] | None,
    target_to_entity_paths: list[list[tuple[str, str, str]]] | None,
) -> None:
    weight_table_name, weight_col_name = _parse_weight_col(weight_col)
    if graph is None:
        return

    if weight_table_name not in graph:
        raise ValidationException(
            f"Weight column '{weight_col}' references unknown table "
            f"'{weight_table_name}'.")

    weight_table = graph[weight_table_name]
    weight_column = weight_table.get_col(weight_col_name)
    if weight_column is None:
        raise ValidationException(
            f"Weight column '{weight_col}' references unknown column "
            f"'{weight_col_name}' on table '{weight_table.table_name}'.")

    target_table_name = _resolve_target_table_name(
        graph,
        entity_table_names,
        target_to_entity_paths,
    )
    if (target_table_name is not None
            and weight_table.table_name != target_table_name):
        raise ValidationException(
            f"Weight column '{weight_col}' must reference target table "
            f"'{target_table_name}'.")

    if weight_column.dtype is None or not weight_column.dtype.is_numerical():
        raise ValidationException(
            f"Weight column '{weight_col}' must reference a numerical "
            f"column (got dtype={weight_column.dtype}).")


class ModelPlanValidationError(ValueError):
    r"""Raised when a model plan fails structural validation.

    In pydantic v1 this subclassed ``pydantic.ValidationError`` and was raised
    with a list of ``ErrorWrapper``s. pydantic v2's ``ValidationError`` is a
    Rust type that cannot be hand-constructed via ``__init__`` (only via
    ``from_exception_data``) and is not meaningfully subclassable, so this is
    now a plain ``ValueError`` that holds the collected v2 line errors and
    reproduces the v1 user-facing string. The ``errors()`` accessor returns the
    v2 error-dict shape (``{'type', 'loc', 'msg', 'input', 'ctx', ...}``).
    """
    def __init__(
        self,
        errors: Sequence[LineError],
        model: type | None = None,
    ) -> None:
        self._model = model
        # Normalize the raw line-error dicts to the canonical v2 `errors()`
        # shape (fills in `msg` from the custom-error template, etc.) via a
        # throwaway `ValidationError`.
        title = getattr(model, '__name__', None) or 'ModelPlan'
        self._validation_error = ValidationError.from_exception_data(
            title, list(errors))
        super().__init__(str(self))

    def errors(self) -> list[dict[str, Any]]:
        return self._validation_error.errors()

    def __str__(self) -> str:
        errors = self.errors()
        reprs = '\n'.join([f"'{e['loc'][0]}': {e['msg']}" for e in errors])
        return f"{_INVALID_SYNTAX_PREFIX}\n{reprs}"


# TODO: Combine this with `ModelPlanValidationError`, e.g., by collecting all
# input `Exception`s as line-error dicts in advance.
class ModelPlanValueError(ValueError):
    def __init__(self, excs: Sequence[Exception]) -> None:
        reprs = '\n'.join([str(e) for e in excs])
        message = f"{_INVALID_SYNTAX_PREFIX}\n{reprs}"
        super().__init__(message)


class ModelPlanSyntaxError(ValueError):
    """Single error for model plan syntax validation issues."""
    def __init__(self, message: str) -> None:
        if not message.startswith(_INVALID_SYNTAX_PREFIX):
            message = f"{_INVALID_SYNTAX_PREFIX}\n\n{message}"
        super().__init__(message)


class PlanMixin(api.PlanMixin):
    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:
        r"""Creates a default plan, according to a task type, a query type, and
        a run mode."""
        raise NotImplementedError

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        r"""Validates that all attributes for a given :obj:`task_type` and
        `query_type` are correctly set, *i.e.* there do not exist required but
        missing attributes. `has_train_table_weight_col` is used to validate
        the `weight_mode`, which should not be present if
        `has_train_table_weight_col` is False.
        """
        default_plan = self.__class__.from_default(  # type: ignore
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )
        errors: list[Exception] = []
        warnings: list[ValidationWarning] = []

        for key, value, metadata in self.items():
            # Set missing value based on default value if not specified:
            is_missing = _is_missing(value)
            is_missing &= self.is_valid_option(
                key,
                metadata,
                task_type,
                query_type,
                has_train_table_weight_col,
            )
            default_value = getattr(default_plan, key)
            if is_missing:
                value = default_value
                setattr(self, key, value)

            is_custom_value = not _is_missing(value) and value != default_value

            # Throw error if an invalid option was set:
            if (not self.is_valid_option(
                    key,
                    metadata,
                    task_type,
                    query_type,
                    has_train_table_weight_col,
            ) and is_custom_value):
                task_type_repr = task_type.value.replace('_', ' ')
                message_suffix = ''
                if (key == 'weight_col' and query_type == QueryType.TEMPORAL
                        and task_type.is_node_pred):
                    task_type_repr = 'temporal node prediction and forecasting'
                    message_suffix = '.'
                elif not task_type.is_link_pred:
                    task_type_repr = f'{query_type.value} {task_type_repr}'
                if (key in {'start_time', 'end_time'}
                        and query_type == QueryType.TEMPORAL
                        and not _use_decoupled_splits()):
                    message_suffix = (
                        ". Use 'train_start_offset' and 'train_end_offset' "
                        "instead.")
                message = (f"The '{key}' field is not supported for "
                           f"{task_type_repr} tasks{message_suffix}")
                if key == 'weight_col':
                    errors.append(ValidationException(message=message))
                else:
                    errors.append(ValueError(message))

        return errors, warnings

    def to_dict(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_weight_col: bool = False,
    ) -> dict[str, Any]:

        options: dict[str, Any] = {}
        # The hidden-field filter below skips a field only when its value still
        # equals its default. v1 read that default from the JSON schema, where
        # `compat_field` stored a concrete `default` (it eagerly called the
        # factory). v2 keeps the factory and emits no schema `default` for such
        # fields, so a schema lookup returns None and every MissingType-defaulted
        # hidden field would leak into the output. Resolve the default from the
        # live field instead so the v1 behaviour is preserved.
        for key, value, metadata in self.items():
            default_value = _field_default(self, key)

            # Filter options according to task/query/PRPR/Hidden:
            if not self.is_valid_option(key, metadata, task_type, query_type,
                                        has_train_table_weight_col):
                continue
            elif metadata.hidden and value == default_value:
                continue

            options[key] = value

        return options

    def update(
        self,
        options_dict: dict[str, Any],
        raise_on_error: bool = True,
    ) -> list[LineError]:
        if not isinstance(options_dict, dict):
            return []

        errors: list[LineError] = []
        options_dict = copy.copy(options_dict)
        for key, *_ in self.items():
            if key in options_dict.keys():
                try:
                    setattr(self, key, options_dict.pop(key))
                except ValidationError as error:
                    if raise_on_error:
                        raise error
                    # v2: `errors()` returns line-error dicts (v1 used the
                    # removed `raw_errors` list of `ErrorWrapper`s).
                    errors.extend(filter_missing_error(error.errors()))

        for name in options_dict.keys():
            valid_top_sections = [
                field.name for field in self.__dataclass_fields__.values()
            ]
            error: LineError = {
                'type': ExtraFieldsError(valid_top_sections=valid_top_sections),
                'loc': (name, ),
                'input': options_dict[name],
            }
            if raise_on_error:
                # Pydantic v2's ``ValidationError`` cannot be hand-constructed.
                # ``ModelPlanValidationError`` preserves the user-facing
                # message and is the exception consumed by the external caller.
                raise ModelPlanValidationError(
                    [error],
                    model=self.__class__,  # type: ignore
                )
            errors.append(error)

        return errors


@dataclass
class TrainingTableGenerationPlan(api.TrainingTableGenerationPlan, PlanMixin):
    @dataclass
    class SplitDescriptor:
        name: str

        #: The families that derive their stage assignment from the training
        #: table's time column. Enumerated, not negated: "not random" is not
        #: the same question as "derived from time". `ManualSplit` is
        #: neither -- it reads the stage out of a column the user filled in
        #: -- and a negated list silently answers "temporal" for it.
        TIME_DERIVED_SPLITS: ClassVar[frozenset[str]] = frozenset({
            'ApproxDateOffsetSplit',
            'DateOffsetSplit',
            'TemporalSplit',
            'TimeRangeSplit',
            'TimeSplit',
        })

        #: The families that carry the caller's own per-row stage
        #: assignment. A gate that demands a temporal split wants an
        #: assurance that training rows do not come from the future of the
        #: rows it is evaluated on. A time-derived split supplies that
        #: structurally; `ManualSplit` supplies it by assertion -- the
        #: caller placed every row in a stage themselves, which is the
        #: stronger statement and the whole point of supplying a split
        #: column. Refusing it would block the case a supplied training
        #: table exists to serve.
        ROW_ASSIGNED_SPLITS: ClassVar[frozenset[str]] = frozenset({
            'ManualSplit',
        })

        @property
        def is_temporal_split(self) -> bool:
            r"""Whether stage assignment is derived from `table.time_col`."""
            return self.name in self.TIME_DERIVED_SPLITS

        @property
        def assigns_stages_explicitly(self) -> bool:
            r"""Whether the caller assigned every row to a stage themselves."""
            return self.name in self.ROW_ASSIGNED_SPLITS

    @staticmethod
    def get_split(
        value: str | None
    ) -> Optional['TrainingTableGenerationPlan.SplitDescriptor']:
        if value is None:
            return None
        try:
            expression = ast.parse(value, mode='eval').body
        except (SyntaxError, ValueError) as error:
            raise ValueError(f'"{value}" is not a valid split. Error: {error}')
        if not isinstance(expression, ast.Call) or not isinstance(
                expression.func, ast.Name):
            raise ValueError(f'"{value}" is not a valid split.')
        valid_names = {
            'ApproxDateOffsetSplit',
            'DateOffsetSplit',
            'ManualSplit',
            'RandomLinkSplit',
            'RandomSplit',
            'TemporalSplit',
            'TimeRangeSplit',
            'TimeSplit',
        }
        if expression.func.id not in valid_names:
            raise ValueError(f'"{value}" is not a valid split.')
        if expression.func.id == 'TimeSplit':
            TrainingTableGenerationPlan._validate_time_split(expression)
        return TrainingTableGenerationPlan.SplitDescriptor(expression.func.id)

    @staticmethod
    def _validate_time_split(expression: ast.Call) -> None:
        arguments = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in expression.keywords if keyword.arg is not None
        }
        unit = arguments.get('unit')
        if unit is not None:
            TimeUnit(unit.lower() if isinstance(unit, str) else unit)

        boundaries = arguments.get('timestamp_boundaries')
        if boundaries is None:
            return
        for previous, current in itertools.pairwise(boundaries):
            if current[0] < previous[1]:
                raise ValueError(
                    f'The end of each timestamp boundary must be strictly '
                    f'smaller than the start of the next boundary: '
                    f'{previous!r}, {current!r}')

    @field_validator('split')
    @classmethod
    def is_valid_split(cls, value: InferredType | str) -> InferredType | str:
        if value != InferredType.VALUE:
            cls.get_split(value)
        return value

    def is_valid_option(
        self,
        name: str,
        metadata: Metadata,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_weight_col: bool = False,
    ) -> bool:
        r"""Check if a given option is valid for the current task and
        query type.

        Args:
            name (str): Name of the option to validate.
            metadata (Metadata): Metadata about the option.
            task_type (TaskType): Type of ML task\
                (e.g. regression, classification).
            query_type (QueryType): Type of query (static vs temporal).
            has_train_table_weight_col: Used to validate `weight_mode`.
                `weight_mode` is only valid if this is `True`.

        Returns:
            bool: True if the option is valid for this configuration,
                False otherwise.
        """
        # First check parent class validation
        if not super().is_valid_option(name, metadata, task_type, query_type,
                                       has_train_table_weight_col):
            return False

        # Check split-related options based on decoupled splits setting
        # When using decoupled splits, only allow start/end time
        # When using legacy splits, only allow train start/end offset
        if name in ['start_time', 'end_time']:
            return _use_decoupled_splits()
        if name in ['train_start_offset', 'train_end_offset']:
            return not _use_decoupled_splits()

        return True

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        weight_col = self.weight_col
        if weight_col is not None:
            errors.append(
                ValidationException(
                    title="Unsupported feature",
                    message=("Weighted training-table generation is not yet "
                             "supported on DiskGraph. Remove "
                             "'training_table_generation.weight_col' from the "
                             "model plan.")))

        if self.split != InferredType.VALUE:
            try:
                split = self.get_split(self.split)
            except ValueError as e:
                errors.append(e)
                return errors, warnings

            assert split is not None

            if (not has_train_table_time_col and split.is_temporal_split):
                # Require non-temporal split for static prediction without
                # timestamp columns:
                errors.append(
                    ValueError(
                        f"The selected split '{self.split}' is temporal, but "
                        "the target table doesn't have a 'Create Date'. Please "
                        "add a 'Create Date' column to the target table, or "
                        "choose 'RandomSplit' for this task."))
            if has_train_table_time_col and query_type == QueryType.STATIC:
                # For static prediction with timestamp columns,
                # `TemporalSplit`, `TimeRangeSplit` are the recommended
                # choices. `RandomSplit` is allowed but could cause leakage
                # problems.
                if split.name == 'RandomSplit':
                    warnings.append(
                        ValidationWarning(
                            title="Possible Data Leakage: Random Split",
                            message=(
                                "`RandomSplit` is being used for a query "
                                "with a target table timestamp, which could "
                                "lead to data leakage. Consider using "
                                "`TemporalSplit` or `TimeRangeSplit` instead "
                                "to guarantee no data leakage in your data "
                                "split."),
                        ))
                else:
                    # Raw `TimeSplit` is an internal representation and must
                    # never be accepted from a user-supplied plan: with
                    # `USE_DECOUPLED_SPLITS=False` the DiskGraph producer
                    # assigns the split and the user's `TimeSplit` is silently
                    # dropped. Reject it uniformly for all query types.
                    if split.name == 'TimeSplit':
                        errors.append(_raw_time_split_error(self.split))
                    acceptable_splits = [
                        'RandomSplit',
                        'TemporalSplit',
                        'TimeRangeSplit',
                    ]
                    # A user who assigned every row to a stage has already
                    # answered the leakage question this list exists to ask,
                    # and answered it more precisely than any family name
                    # can. Accepting it here keeps the static branch
                    # consistent with the temporal-link-prediction gate
                    # below, which faces the strictly harder version of the
                    # same question.
                    if (split.name not in acceptable_splits
                            and not split.assigns_stages_explicitly):
                        errors.append(
                            ValueError(
                                f"For queries without target aggregation, but "
                                f"contains a timestamp '{self.split}' is "
                                f"selected, but only {acceptable_splits} are "
                                f"allowed."))

            # Reject raw `TimeSplit` for temporal queries and temporal link
            # prediction. Unlike the static branch above, these paths only
            # ever warned (on `TemporalSplit` leakage) and let a raw
            # `TimeSplit` fall through. `TimeSplit` is not a `TemporalSplit`
            # subclass, so it also slips past the `is_temporal_split` TLP
            # check below while being silently dropped by the DG producer.
            if (split.name == 'TimeSplit'
                    and (query_type == QueryType.TEMPORAL
                         or task_type == TaskType.TEMPORAL_LINK_PREDICTION)):
                errors.append(_raw_time_split_error(self.split))

            if (query_type == QueryType.TEMPORAL
                    and split.name == 'TemporalSplit'):
                # Warn on `TemporalSplit` for temporal problems
                warnings.append(
                    ValidationWarning(
                        title="Possible Data Leakage: Temporal Split",
                        message=(
                            "`TemporalSplit` is being used for a query with "
                            "target aggregation, which could lead to data "
                            "leakage due to overlapping timestamps across "
                            "splits. Consider using `ApproxDateOffsetSplit` "
                            "instead to guarantee no data leakage in your "
                            "data split."),
                    ))
            # Temporal link prediction must not evaluate on links it was
            # trained over. A time-derived split guarantees that from the
            # time column; a `ManualSplit` is the user making the same
            # assertion row by row, which is the stronger statement. A
            # random split makes no such assertion.
            # TODO: support `RandomSplit` for temporal link prediction
            # problem. (P2)
            if (task_type == TaskType.TEMPORAL_LINK_PREDICTION
                    and not (split.is_temporal_split
                             or split.assigns_stages_explicitly)):
                errors.append(
                    ValueError("Temporal link prediction requires a "
                               "temporal split"))

            if task_type == TaskType.FORECASTING:
                # Forecasting only accepts offset/range temporal families; raw
                # `TimeSplit` is rejected uniformly (no flag gate) and
                # timestamp-fraction is intentionally not offered here.
                if split.name == 'TimeSplit':
                    errors.append(_raw_time_split_error(self.split))
                acceptable_splits = ['DateOffsetSplit', 'TimeRangeSplit']
                if split.name not in acceptable_splits:
                    errors.append(
                        ValueError(
                            f"Forecasting requires a {acceptable_splits} "
                            f"split. Please choose one of these splits or "
                            f"set `split` to `inferred`."))
        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:
        out = cls()

        # There exists three cases where we can select the split in
        # advance. In all other cases, we keep `split` as `inferred`
        # since whether to choose `DateOffsetSplit` or `ApproxDateOffsetSplit`
        # depends on training table stats/number of training windows.
        if task_type == TaskType.STATIC_LINK_PREDICTION:
            # 1) Static link prediction:
            out.split = 'RandomSplit([0.8, 0.1, 0.1])'
        elif query_type == QueryType.STATIC:
            if has_train_table_time_col:
                # 2) Static node prediction with a time column:
                out.split = 'TemporalSplit([0.8, 0.1, 0.1])'
            else:
                # 3) Static node prediction without a time column:
                out.split = 'RandomSplit([0.8, 0.1, 0.1])'

        if query_type == QueryType.TEMPORAL:
            # For temporal queries, set either:
            # - start_time/end_time (if using decoupled splits)
            # - train_start_offset/train_end_offset (if using legacy splits)
            # This controls the time range used for training data
            if _use_decoupled_splits():
                out.start_time = InferredType.VALUE
                out.end_time = InferredType.VALUE
            else:
                out.train_start_offset = InferredType.VALUE
                out.train_end_offset = InferredType.VALUE
            out.timeframe_step = InferredType.VALUE

        if (query_type == QueryType.TEMPORAL
                and task_type in {TaskType.REGRESSION, TaskType.FORECASTING}):
            if task_type != TaskType.FORECASTING:
                out.forecast_length = 1
                out.lag_timesteps = 0
            else:
                out.lag_timesteps = 1
            out.year_over_year = False

        return out


@dataclass
class TrainingJobPlan(api.TrainingJobPlan, PlanMixin):
    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        # Check that `metrics` are valid for a given task type:
        from kumo_core.evaluation.suite import MetricStore
        try:
            assert isinstance(self.metrics, list)

            stype_dict: dict[str, dict[str, Stype]] | None = None
            if graph is not None:
                stype_dict = {
                    table.table_name: {
                        col.name: col.stype
                        for col in table.cols
                    }
                    for table in graph.tables
                }

            MetricStore.validate_metrics(
                task_type,
                self.metrics,
                stype_dict,
                entity_table_names,
            )
        except Exception as e:
            errors.append(e)
            return errors, warnings

        # Check that `tune_metric` is also present in validated `metrics`:
        metrics = self.metrics
        if not task_type.is_link_pred:
            metrics = metrics + ['loss']
        if self.tune_metric not in metrics:
            errors.append(
                ValueError(f"The tune metric '{self.tune_metric}' is not "
                           f"present in the list of metrics "
                           f"{self.metrics}"))

        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:

        out = cls()

        # Set default for `num_experiments`:
        if run_mode == RunMode.DEBUG:
            out.num_experiments = 2
        elif run_mode == RunMode.FAST:
            out.num_experiments = 4
        elif run_mode == RunMode.NORMAL:
            out.num_experiments = 8
        elif run_mode == RunMode.BEST:
            out.num_experiments = 16

        if task_type.is_classification:
            assert isinstance(out.num_experiments, int)
            out.num_experiments = max(out.num_experiments // 2, 2)

        # Set defaults for `metrics` and `tune_metric`:
        if task_type == TaskType.BINARY_CLASSIFICATION:
            out.metrics = ['auroc', 'ap', 'auprc']
            out.tune_metric = 'auroc'
        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            out.metrics = ['acc']
            out.tune_metric = 'acc'
        elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
            out.metrics = [
                f'{metric}_{average}' for metric, average in itertools.product(
                    ['ap', 'auroc', 'auprc'],
                    ['macro', 'micro', 'per_label'],
                )
            ]
            out.tune_metric = 'ap_macro'
        elif task_type == TaskType.REGRESSION:
            out.metrics = ['mae', 'mse', 'rmse']
            out.tune_metric = 'mae'
        elif task_type == TaskType.FORECASTING:
            out.metrics = ['mae', 'mse', 'rmse']
            out.tune_metric = 'mae'
        elif task_type == TaskType.MULTILABEL_RANKING:
            out.metrics = [
                f'{metric}@{k}' for metric, k in itertools.product(
                    ['map', 'precision', 'recall', 'f1'],
                    [1, 10, 100],
                )
            ]
            out.tune_metric = 'map@10'

        elif task_type.is_link_pred:
            if has_train_table_weight_col:
                metric_names = ['weighted_ndcg', 'weighted_recall', 'precision']
            else:
                metric_names = ['map', 'precision', 'recall', 'f1']

            out.metrics = [
                f'{metric}@{k}' for metric, k in itertools.product(
                    metric_names,
                    [1, 10, 100],
                )
            ]
            out.tune_metric = f'{metric_names[0]}@10'

        out.pruning = PruningConfig(
            min_epochs=2,
            k=3,
            min_delta=0.0,
            patience=1,
        )

        if run_mode in [RunMode.DEBUG, RunMode.FAST]:
            out.refit_trainval = False

        return out


def _get_encoder_overrides(
        values: dict[str, EncoderType | str] | None,  #
) -> dict[str, EncoderType]:

    if values is None:
        return {}

    encoder_classes = {
        name: cls
        for name, cls in vars(encoder_api).items()
        if isinstance(cls, type) and issubclass(cls, Encoder) and cls != Encoder
    }

    out_dict: dict[str, EncoderType] = {}
    failed_col_names = []
    for key, value in values.items():
        if isinstance(value, str) and value.startswith('OneHot'):
            raise ValueError("'OneHot()' has been removed in favor of "
                             "the equivalent 'Index()'")

        try:
            if isinstance(value, str):
                obj = repr_to_object(value, encoder_classes)
            else:
                obj = value
            assert isinstance(obj, (Encoder, BaseModel))
            out_dict[key] = obj
        except Exception:
            failed_col_names.append(key)
            continue

    if len(failed_col_names) == 1:
        col_name = failed_col_names[0]
        raise ValueError(f"value for '{col_name}' is not a valid encoder")
    elif len(failed_col_names) > 1:
        col_names = ', '.join(f"'{col}'" for col in failed_col_names[:-1])
        col_names += f" and '{failed_col_names[-1]}'"
        raise ValueError(f"values for {col_names} are not valid encoders")

    return out_dict


@dataclass
class ColumnProcessingPlan(api.ColumnProcessingPlan, PlanMixin):
    @staticmethod
    def get_encoder_overrides(
        values: dict[str, EncoderType | str] | None,
    ) -> dict[str, EncoderType]:  # type: ignore
        return _get_encoder_overrides(values)

    @field_validator('encoder_overrides')
    @classmethod
    def is_valid_encoder(
        cls,
        values: dict[str, EncoderType | str] | None,
    ) -> dict[str, EncoderType | str] | None:
        # Confirm that encoders can be resolved:
        _get_encoder_overrides(values)
        return values

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        encoder_classes = {
            name: cls
            for name, cls in vars(encoder_api).items() if isinstance(cls, type)
            and issubclass(cls, Encoder) and cls != Encoder
        }

        failed_names = []
        failed_encoders = []
        text_models = set()
        for key, value in (self.encoder_overrides or {}).items():
            if '.' not in key:
                failed_names.append(key)
                continue

            if graph is None:
                continue

            if key.endswith('train_table.TARGET'):
                continue

            table_name, col_name = key.rsplit('.', maxsplit=1)

            if table_name not in graph:
                failed_names.append(key)
                continue

            col: ColumnConfig | None = None
            for c in graph[table_name].cols:
                if c.name == col_name:
                    col = c

            if col is None:
                failed_names.append(key)
                continue

            if col.stype is None:
                continue

            if isinstance(value, str):
                encoder = repr_to_object(value, encoder_classes)
            else:
                encoder = value
            assert isinstance(encoder, Encoder)
            if col.stype not in encoder.supported_stypes:
                failed_encoders.append((key, col.stype, value))

            if isinstance(encoder, TextEmbedding):
                text_models.add(encoder.model)

        error_message = ''
        if len(failed_names) == 1:
            name = failed_names[0]
            error_message += f"'{name}' is not a valid column name. "
        elif len(failed_names) > 1:
            names = ', '.join(f"'{name}'" for name in failed_names[:-1])
            names += f" and '{failed_names[-1]}'"
            error_message += f"{names} are not valid column names. "

        for name, stype, value in failed_encoders:
            error_message += (f"'{name}' has an incompatible encoder '{value}'"
                              f" for the column type '{stype.value}'. ")

        if len(error_message) > 0:
            error_message += ("Please check the column names and encoders "
                              "within the model planner 'encoder_overrides' "
                              "field.")
            errors.append(ValueError(error_message.rstrip()))

        if len(text_models) > 1:
            warning = ValidationWarning(
                title="GPU Memory: Multiple Text Embedding Models",
                message=f"We recommend using only a single text embedding "
                f"model across all columns to avoid a possible GPU "
                f"out-of-memory error due to high GPU memory usage. "
                f"(got {sorted(text_models)}).",
            )
            warnings.append(warning)

        invalid_strategies: list[tuple[Stype, NAStrategy]] = []
        for key, value in (self.na_strategy or {}).items():
            if key == Stype.numerical:
                cls = Numerical
            elif key in {Stype.categorical, Stype.ID}:
                cls = Index
            elif key == Stype.multicategorical:
                cls = MultiCategorical
            elif key == Stype.text:
                cls = TextEmbedding
            elif key == Stype.timestamp:
                cls = Datetime
            elif key == Stype.sequence:
                cls = NumericalList
            else:
                continue
            valid_na_strategies = get_args(get_type_hints(cls)['na_strategy'])
            if value not in valid_na_strategies:
                invalid_strategies.append((key, value))

        error_message = ''
        if len(invalid_strategies) == 1:
            key, value = invalid_strategies[0]
            error_message = (f"'{value}' is not a supported N/A imputation "
                             f"strategy for semantic type '{key}'")
        elif len(invalid_strategies) > 1:
            stypes = ', '.join(f"'{key}'" for key, _ in invalid_strategies)

            error_message = (f"The semantic types {stypes} hold unsupported "
                             f"N/A imputation strategies")

        if len(error_message) > 0:
            errors.append(ValueError(error_message))

        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:
        return cls()


@dataclass
class NeighborSamplingPlan(api.NeighborSamplingPlan, PlanMixin):
    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        target_to_entity_paths = _primary_target_to_entity_paths(
            target_to_entity_paths)
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        assert isinstance(self.num_neighbors, list)

        max_hops = 0
        for num_neighbors in self.num_neighbors:
            if isinstance(num_neighbors, list):
                max_hops = max(max_hops, len(num_neighbors))
                continue

            assert isinstance(num_neighbors, NumNeighborsConfig)

            # 1) Check that hop definition is consecutive:
            num_neighbors._validate_consecutive_hops()
            max_hops = max(max_hops, num_neighbors.num_hops())

            # 2) Check that all edge types exist in the graph:
            missing_tables = []
            missing_keys = []
            invalid_key_formats = []
            if graph is not None:
                for hop_cfg in num_neighbors.to_dict().values():
                    for key in hop_cfg.__pydantic_extra__.keys():
                        try:
                            (src_table_name, src_col_name, dst_table_name,
                             dst_col_name) = self._split_hop_key(key)
                        except ValueError as e:
                            invalid_key_formats.append(
                                f"  {repr(key)}\n\n{str(e)}")
                            continue

                        if src_table_name not in graph:
                            missing_tables.append(src_table_name)
                            continue

                        if dst_table_name not in graph:
                            missing_tables.append(dst_table_name)
                            continue

                        src_table = graph[src_table_name]
                        dst_table = graph[dst_table_name]

                        all_src_fkeys = [
                            (
                                f.name,
                                graph[f.dst_table].table_name,  # type: ignore
                                f.dst_key,
                            ) for f in src_table.fkeys
                        ]
                        all_dst_fkeys = [
                            (
                                f.name,
                                graph[f.dst_table].table_name,  # type: ignore
                                f.dst_key,
                            ) for f in dst_table.fkeys
                        ]

                        if ((
                                src_col_name,
                                dst_table.table_name,
                                dst_col_name,
                        ) not in all_src_fkeys and (
                                dst_col_name,
                                src_table.table_name,
                                src_col_name,
                        ) not in all_dst_fkeys):
                            missing_keys.append(key)
                            continue

            if len(missing_tables) > 0:
                errors.append(
                    ModelPlanSyntaxError(
                        f"The specified table(s) {missing_tables} "
                        f"in the 'num_neighbors' option do not exist "
                        f"in the graph"))

            if len(missing_keys) > 0:
                errors.append(
                    ModelPlanSyntaxError(
                        f"The specified connection(s) {missing_keys} "
                        f"in the 'num_neighbors' option do not exist "
                        f"in the graph"))

            if len(invalid_key_formats) > 0:
                errors.append(
                    ModelPlanSyntaxError(
                        "Invalid hop key in 'num_neighbors':\n\n" +
                        "\n".join(invalid_key_formats)))

            # 3) Check that 'inferred' option is only passed in the
            # first hop for the target->entity connection(s):
            if graph is not None and target_to_entity_paths is not None:
                valid_inferred_keys = []
                for target_to_entity_path in target_to_entity_paths:
                    assert len(target_to_entity_path) == 1
                    (
                        dst_table_name,
                        fkey_name,
                        src_table_name,
                    ) = target_to_entity_path[0]
                    src_table = graph[src_table_name]
                    dst_table = graph[dst_table_name]
                    assert src_table.pkey is not None
                    valid_inferred_keys.append((
                        src_table,
                        src_table.pkey.name,
                        dst_table,
                        fkey_name,
                    ))

                for i, hop_cfg in num_neighbors.to_dict().items():
                    for key, value in hop_cfg.__pydantic_extra__.items():
                        try:
                            (src_table_name, src_col_name, dst_table_name,
                             dst_col_name) = self._split_hop_key(key)
                        except ValueError:
                            continue

                        if src_table_name not in graph:
                            continue
                        if dst_table_name not in graph:
                            continue

                        src_table = graph[src_table_name]
                        dst_table = graph[dst_table_name]

                        is_target_to_entity_key = (
                            src_table, src_col_name, dst_table,
                            dst_col_name) in valid_inferred_keys

                        if (isinstance(value, InferredType)
                                and (i != 0 or not is_target_to_entity_key)):
                            errors.append(
                                ValueError(
                                    f"The 'inferred' option in 'num_neighbors'"
                                    f" is only allowed in the first hop of "
                                    f"the target to entity table connection "
                                    f"(got {key} in hop {i + 1})"))
        if not self.adaptive_sampling and max_hops > 3:
            warning = ValidationWarning(
                title="GPU Memory: Too Many Hops",
                message=f"'num_neighbors' exceeds the recommended number of "
                f"three hops, which can potentially cause a GPU "
                f"out-of-memory error (got {max_hops} hops).")
            warnings.append(warning)
        if self.adaptive_sampling:
            flat_num_neighbor_value: list[int] = []
            for num_neighbors in self.num_neighbors:
                if isinstance(num_neighbors, list):
                    flat_num_neighbor_value.extend(num_neighbors)
                else:
                    assert isinstance(num_neighbors, NumNeighborsConfig)
                    for hop_cfg in num_neighbors.to_dict().values():
                        flat_num_neighbor_value.append(hop_cfg.default)
                        for value in hop_cfg.__pydantic_extra__.values():
                            if isinstance(value, int):
                                flat_num_neighbor_value.append(value)
            if any(n < 0 for n in flat_num_neighbor_value):
                errors.append(
                    ValueError('Negative neighbor numbers are not '
                               'compatible with `adaptive_samping=True`.'))
        return errors, warnings

    def _split_hop_key(self, key: str) -> tuple[str, str, str, str]:
        return _parse_hop_key(key)

    def estimate_max_adaptive_sample_size(
        self,
        graph: GraphConfig,
        entity_table_names: list[str],
        batch_size: int,
    ) -> int:
        all_edge_types = []
        for table in graph.tables:
            for fkey in table.fkeys:
                all_edge_types.append(
                    (table.table_name, fkey.name, fkey.dst_table, fkey.dst_key))
                all_edge_types.append(
                    (fkey.dst_table, fkey.dst_key, table.table_name, fkey.name))
        max_size = 0
        for num_neighbors in self.num_neighbors:
            max_size = max(
                max_size,
                self._estimate_max_adaptive_sample_size(graph,
                                                        entity_table_names,
                                                        batch_size,
                                                        all_edge_types,
                                                        num_neighbors))
        return max_size

    def _estimate_max_adaptive_sample_size(
        self,
        graph: GraphConfig,
        entity_table_names: list[str],
        batch_size: int,
        all_edge_types: list[tuple[str, str, str, str]],
        num_neighbors: "list[int] | NumNeighborsConfig",
    ) -> int:
        # In LP, only the first entity table is the LHS entity
        node_count_per_hop = [{entity_table_names[0]: batch_size}]
        edges_traversed = [{}]
        for hop in range(len(num_neighbors)):
            last_count = node_count_per_hop[hop]
            last_traversed_edges = edges_traversed[hop]
            next_count = {}
            next_traversed_edges = {}
            for (src_table_name, src_col_name, dst_table_name,
                 dst_col_name) in all_edge_types:
                if src_table_name not in last_count:
                    continue
                # Determine the number of neighbors for this edge type
                # Due to adaptive sampling, we assume that they will all get
                # sampled.
                # One case needs to be handled separately: fkey to a node
                # that we just visited (e.g. user_id in transaction if we just
                # got there from a user table). This node is guaranteed to not
                # get visited, discarding a whole subtree of samples.
                num_n = 0
                if isinstance(num_neighbors, list):
                    num_n = num_neighbors[hop]
                else:
                    assert isinstance(num_neighbors, NumNeighborsConfig)
                    hop_cfg = num_neighbors.to_dict()[hop]
                    num_n = hop_cfg.default
                    config_key = (f'{src_table_name}.{src_col_name}->'
                                  f'{dst_table_name}.{dst_col_name}')
                    if config_key in hop_cfg.__pydantic_extra__:
                        value = hop_cfg.__pydantic_extra__[config_key]
                        if value == InferredType.VALUE:
                            # In absence of stats, we assume the upper bound
                            # TODO (vid): this is overly conservative
                            num_n = MAX_INFERRED_NUM_NEIGHBORS
                        elif isinstance(value, int):
                            num_n = value
                node_count = num_n * last_count[src_table_name]
                # If the edge points back to a node that has already been
                # visited, we do not count it
                reversed_edge = (dst_table_name, dst_col_name, src_table_name,
                                 src_col_name)
                is_1_to_1 = src_col_name in [
                    fk.name for fk in graph[src_table_name].fkeys
                ]
                if reversed_edge in last_traversed_edges and is_1_to_1:
                    node_count -= last_traversed_edges[reversed_edge] * num_n
                if node_count > 0:
                    next_traversed_edges[(src_table_name, src_col_name,
                                          dst_table_name,
                                          dst_col_name)] = node_count
                    if dst_table_name not in next_count:
                        next_count[dst_table_name] = 0
                    next_count[dst_table_name] += node_count
            node_count_per_hop.append(next_count)
            edges_traversed.append(next_traversed_edges)
        return sum(
            sum(node_count.values()) for node_count in node_count_per_hop[1:])

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:

        out = cls()
        target_to_entity_paths = _primary_target_to_entity_paths(
            target_to_entity_paths)

        if (query_type == QueryType.STATIC
                and task_type in TaskType.get_node_pred_tasks()):
            out.sample_from_entity_table = True

        # Set default for `num_neighbors`:
        if run_mode == RunMode.DEBUG:
            default = 8
        elif run_mode == RunMode.FAST:
            default = 12
        elif run_mode == RunMode.NORMAL:
            default = 16
        else:
            assert run_mode == RunMode.BEST
            default = 24

        num_neighbors = NumNeighborsConfig(
            hop1=HopConfig(default=default),
            hop2=HopConfig(default=default),
        )

        if graph is not None and target_to_entity_paths is not None:
            for target_to_entity_path in target_to_entity_paths:
                assert len(target_to_entity_path) == 1
                (
                    dst_table_name,
                    fkey_name,
                    src_table_name,
                ) = target_to_entity_path[0]
                src_table = graph[src_table_name]
                dst_table = graph[dst_table_name]
                assert src_table.pkey is not None

                key = (f'{src_table.table_name}.{src_table.pkey.name}->'
                       f'{dst_table.table_name}.{fkey_name}')
                num_neighbors.hop1[key] = InferredType.VALUE  # type: ignore

        out.num_neighbors = [num_neighbors]
        return out


@dataclass
class OptimizationPlan(api.OptimizationPlan, PlanMixin):
    @field_validator('lr_scheduler')
    @classmethod
    def is_valid_lr_scheduler(
        cls,
        values: list[LRSchedulerConfig | None] | MissingType,
    ) -> list[LRSchedulerConfig | None] | MissingType:
        if isinstance(values, MissingType):
            return values

        import torch
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from torch_geometric.nn.resolver import lr_scheduler_resolver

        optimizer = torch.optim.Adam(torch.nn.Linear(1, 1).parameters())

        failed_names = []
        for value in [value for value in values if value is not None]:
            try:
                lr_scheduler = lr_scheduler_resolver(
                    value.name,
                    optimizer=optimizer,
                    num_training_steps=1000,
                    **value.kwargs,
                )
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    raise NotImplementedError
            except Exception:
                failed_names.append(value.name)

        if len(failed_names) == 1:
            name = failed_names[0]
            raise ValueError(f"configuration for '{name}' is not a valid "
                             f"learning rate scheduler")
        elif len(failed_names) > 1:
            names = ', '.join(f"'{name}'" for name in failed_names[:-1])
            names += f" and '{failed_names[-1]}'"
            raise ValueError(f"configurations for {names} are not valid "
                             f"learning rate schedulers")

        return values

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        if task_type in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            valid_losses = {LossType.BINARY_CROSS_ENTROPY, LossType.FOCAL_LOSS}
            valid_loss_configs = {FocalLossConfig}

        elif task_type.is_link_pred:
            valid_losses = {
                LossType.CROSS_ENTROPY,
                LossType.BINARY_CROSS_ENTROPY,
            }
            valid_loss_configs = set()

        elif task_type in {
                TaskType.MULTILABEL_RANKING,
                TaskType.MULTICLASS_CLASSIFICATION,
        }:
            valid_losses = {LossType.CROSS_ENTROPY, LossType.ORDINAL_LOG}
            valid_loss_configs = {OrdinalLogLossConfig}

        elif task_type in {
                TaskType.REGRESSION,
                TaskType.FORECASTING,
        }:
            valid_losses = {
                LossType.MAE,
                LossType.MSE,
                LossType.HUBER,
                LossType.WAPE,
                LossType.NORMAL_DISTRIBUTION,
                LossType.NEGATIVE_BINOMIAL,
                LossType.LOG_NORMAL_DISTRIBUTION,
                LossType.MULTI_QUANTILE,
            }
            valid_loss_configs = {
                HuberLossConfig,
                QuantileLossConfig,
            }

        else:
            raise NotImplementedError

        assert isinstance(self.loss, list)
        invalid_losses: set[str] = set()
        for loss in self.loss:
            if is_dataclass(loss) and loss.__class__ not in valid_loss_configs:
                invalid_losses.add(loss.name)
            elif isinstance(loss, str) and loss not in valid_losses:
                invalid_losses.add(loss)

        if len(invalid_losses) > 0:
            errors.append(
                ValueError(f"The loss functions "
                           f"{sorted(list(invalid_losses))} "
                           f"are not valid for '{task_type.value}' tasks"))

        if has_train_table_weight_col and task_type.is_link_pred:
            valid_weight_modes = {
                WeightMode.WEIGHTED_LOSS,
                None,
            }
        elif has_train_table_weight_col:
            valid_weight_modes = {
                WeightMode.WEIGHTED_LOSS,
                WeightMode.SAMPLE,
                None,
            }
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            valid_weight_modes = {
                WeightMode.WEIGHTED_LOSS,
                WeightMode.SAMPLE,
                None,
            }
        else:
            valid_weight_modes = {None}

        if (isinstance(self.weight_mode, list)
                and not set(self.weight_mode).issubset(valid_weight_modes)):
            errors.append(
                ValueError(f"Valid weight modes for '{task_type.value}' tasks "
                           f"are {list(valid_weight_modes)} "
                           f"(got {self.weight_mode})"))

        if (isinstance(self.majority_sampling_ratio, list)
                and any([v is not None for v in self.majority_sampling_ratio])
                and isinstance(self.weight_mode, list)
                and any([v is None for v in self.weight_mode])):
            errors.append(
                ValueError("Specified majority sampling ratio without a "
                           "proper weight mode"))

        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:

        out = cls()

        # Set defaults for `max_epochs`, `max_steps_per_epoch`,
        # `max_val_steps`, `max_test_steps` and `batch_size`:
        if run_mode == RunMode.DEBUG:
            out.max_epochs = 2
            out.min_steps_per_epoch = 1
            out.max_steps_per_epoch = 2
            out.max_val_steps = 2
            out.max_test_steps = 2
            out.batch_size = [16]
        elif run_mode == RunMode.FAST:
            out.max_epochs = 8
            out.min_steps_per_epoch = 30
            out.max_steps_per_epoch = 1000
            out.max_val_steps = 1000
            out.max_test_steps = 2000
            out.batch_size = [512, 1024]
        elif run_mode == RunMode.NORMAL:
            out.max_epochs = 12
            out.min_steps_per_epoch = 30
            out.max_steps_per_epoch = 2000
            out.max_val_steps = 2000
            out.max_test_steps = 4000
            out.batch_size = [512, 1024]
        elif run_mode == RunMode.BEST:
            out.max_epochs = 16
            out.min_steps_per_epoch = 30
            out.max_steps_per_epoch = 2000
            out.max_val_steps = 2000
            out.max_test_steps = 4000
            out.batch_size = [512, 1024]

        out.optimizer = [OptimizerType.ADAM]

        if task_type.is_link_pred:
            out.lookup_optimization = [InferredType.VALUE]

        # Set default for `loss`:
        if task_type in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            out.loss = [LossType.BINARY_CROSS_ENTROPY]

        elif (task_type.is_ranking
              or task_type == TaskType.MULTICLASS_CLASSIFICATION):
            out.loss = [LossType.CROSS_ENTROPY]

        elif task_type in {
                TaskType.REGRESSION,
                TaskType.FORECASTING,
        }:
            out.loss = [LossType.HUBER]

        # Set defaults for `base_lr` and `weight_decay`:
        out.base_lr = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        out.weight_decay = [0.0, 5e-8, 5e-7, 5e-6]

        # Set defaults for `early_stopping` and `lr_scheduler`:
        out.early_stopping = [
            EarlyStoppingConfig(min_delta=0.0, patience=3),
        ]
        out.lr_scheduler = [
            LRSchedulerConfig(
                name='cosine_with_warmup_restarts',
                interval=IntervalType.STEP,
                kwargs=dict(warmup_ratio_or_steps=0.1),
            ),
            LRSchedulerConfig(
                name='constant_with_warmup',
                interval=IntervalType.STEP,
                kwargs=dict(warmup_ratio_or_steps=0.1),
            ),
            LRSchedulerConfig(
                name='linear_with_warmup',
                interval=IntervalType.STEP,
                kwargs=dict(warmup_ratio_or_steps=0.1),
            ),
            LRSchedulerConfig(
                name='cosine_with_warmup',
                interval=IntervalType.STEP,
                kwargs=dict(warmup_ratio_or_steps=0.1),
            ),
        ]

        if has_train_table_weight_col:
            out.weight_mode = [WeightMode.WEIGHTED_LOSS]
            out.majority_sampling_ratio = [None]
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            out.majority_sampling_ratio = [None]
            out.weight_mode = [None]

        if task_type == TaskType.STATIC_LINK_PREDICTION:
            out.target_fraction = [1.0]
            out.overlap_ratio = [1.0]

        return out


@dataclass
class GNNModelPlan(api.GNNModelPlan, PlanMixin):
    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )
        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:
        out = cls()
        out.channels = [64, 128, 256]
        out.aggregation = [
            [
                AggregationType.SUM,
                AggregationType.MEAN,
                AggregationType.MAX,
            ],
            [
                AggregationType.SUM,
                AggregationType.MEAN,
                AggregationType.MIN,
                AggregationType.MAX,
                AggregationType.STD,
            ],
        ]
        out.dropout = [0.0]
        return out


@dataclass
class GraphTransformerModelPlan(api.GraphTransformerModelPlan, PlanMixin):
    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        for num_head in self.num_heads:
            for channels in self.channels:
                if channels % num_head != 0:
                    errors.append(
                        ValueError(f"'channels' must be divisible by "
                                   f"'num_heads' (got {channels} and "
                                   f"{num_head})"))

        if task_type.is_link_pred:
            errors.append(
                ValueError("GraphTransformer models are not supported for "
                           "Link Prediction tasks."))
        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:
        out = cls()
        out.channels = [64, 128]
        out.num_layers = [4, 6]
        out.num_heads = [8]
        out.dropout = [0.2]
        out.positional_encodings = [
            [PositionalEncodingType.HOP, PositionalEncodingType.RANDOM_GNN],
        ]
        return out


@dataclass
class ModelArchitecturePlan(api.ModelArchitecturePlan, PlanMixin):

    # Overrides model so the classes defined in this file are used for
    # serialization.
    # Use ``compat_field`` (the kumoapi v1/v2 bridge) rather than a raw
    # ``Field(metadata=...)``: pydantic v2 does not recognize a ``metadata``
    # kwarg and would stuff it into ``json_schema_extra`` as
    # ``{'metadata': Metadata(...)}``. ``PlanMixin.items()`` reads
    # ``json_schema_extra`` as the flat ``Metadata`` field dict
    # (``Metadata(**json_schema_extra)``), so the nested shape raises
    # ``TypeError: unexpected keyword argument 'metadata'``. ``compat_field``
    # sets ``json_schema_extra = dataclasses.asdict(metadata)`` (the flat shape)
    # under v2.
    model: conlist(
        GNNModelPlan | GraphTransformerModelPlan,
        min_length=1,
    ) | MissingType = compat_field(  # type: ignore
        default=MissingType.VALUE,
        metadata=Metadata(tunable=True),
    )

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:

        # Backward compatibility:
        if task_type == TaskType.STATIC_LINK_PREDICTION:
            self.ranking_embedding_loss_coeff = MissingType.VALUE

        errors, warnings = super().validate(
            task_type,
            query_type,
            has_train_table_time_col,
            run_mode,
            graph,
            entity_table_names,
            target_to_entity_paths,
            has_train_table_weight_col,
        )

        aggregations = self.aggregation.copy() if isinstance(
            self.aggregation, list) else []
        channels = self.channels.copy() if isinstance(self.channels,
                                                      list) else []
        if isinstance(self.model, list):
            aggregations += [
                a for m in self.model if isinstance(m, api.GNNModelPlan)
                for a in m.aggregation
            ]
            channels += [
                c for m in self.model if isinstance(m, api.GNNModelPlan)
                for c in m.channels
            ]

        aggregation_to_warn = [[a.value for a in aggrs]
                               for aggrs in aggregations if len(aggrs) > 3]
        if len(aggregation_to_warn) > 0:
            channels_to_warn = [c for c in channels if c > 256]
            if len(channels_to_warn) > 0:
                warning = ValidationWarning(
                    title="GPU Memory: High Channels with Many Aggregations",
                    message=f"The combination of high 'channels' values and "
                    f"the large number of aggregation operators can "
                    f"potentially cause a GPU out-of-memory error. We "
                    f"recommend to use not more than 256 channels "
                    f"and/or not more than 3 aggregation operators "
                    f"(got aggregation={aggregation_to_warn}, "
                    f"channels={channels_to_warn}).",
                )
                warnings.append(warning)

        if isinstance(self.model, list):
            for idx, m in enumerate(self.model):
                model_errors, model_warnings = m.validate(
                    task_type,
                    query_type,
                    has_train_table_time_col,
                    run_mode,
                    graph,
                    entity_table_names,
                    target_to_entity_paths,
                    has_train_table_weight_col,
                )
                errors.extend(model_errors)
                warnings.extend(model_warnings)

                if isinstance(self.channels, list):
                    warning = ValidationWarning(
                        title="Deprecated Option: channels",
                        message=f"The options "
                        f"'model_architecture.model[{idx}].channels' and "
                        f"deprecated 'model_architecture.channels' are both "
                        f"set. Please remove 'model_architecture.channels' in "
                        f"favor of "
                        f"'model_architecture.model[{idx}].channels', or we "
                        f"will use 'model_architecture.channels' and ignore "
                        f"'model_architecture.model[{idx}].channels'.",
                    )
                    warnings.append(warning)

                if (isinstance(m, GNNModelPlan)
                        and isinstance(self.aggregation, list)):
                    warning = ValidationWarning(
                        title="Deprecated Option: aggregation",
                        message=f"The options "
                        f"'model_architecture.model[{idx}].aggregation' and "
                        f"deprecated 'model_architecture.aggregation' are "
                        f"both set. Please remove "
                        f"'model_architecture.aggregation' in favor of "
                        f"'model_architecture.model[{idx}].aggregation', or "
                        f"we will use 'model_architecture.aggregation' and "
                        f"ignore "
                        f"'model_architecture.model[{idx}].aggregation'.",
                    )
                    warnings.append(warning)

        if task_type.is_link_pred:
            # `CONTEXT` and `PAIRWISE` modules are only supported for temporal
            # link prediction (see asserts in `kumo/train/trainer.py`):
            if (task_type == TaskType.STATIC_LINK_PREDICTION
                    and self.module in (LinkPredOutputType.CONTEXT,
                                        LinkPredOutputType.PAIRWISE)):
                errors.append(
                    ValueError(
                        f"The '{self.module.value}' link-prediction module "
                        f"requires temporal link prediction, but the task "
                        f"type is static link prediction. Use 'shallow_rhs' "
                        f"or 'two_tower' instead."))

            # `ranking_embedding_loss_coeff` is only consumed by
            # `HybridLinkGNNLightningModule`.
            if (task_type == TaskType.TEMPORAL_LINK_PREDICTION
                    and self.module not in {
                        LinkPredOutputType.CONTEXT, LinkPredOutputType.RANKING
                    }):
                assert isinstance(self.ranking_embedding_loss_coeff, list)
                if any(c > 0 for c in self.ranking_embedding_loss_coeff):
                    warnings.append(
                        ValidationWarning(
                            title="Ignored Option: "
                            "ranking_embedding_loss_coeff",
                            message=f"The option "
                            f"'ranking_embedding_loss_coeff' will be ignored "
                            f"since it is only valid for ContextGNN on "
                            f"temporal link prediction tasks "
                            f"(got {self.ranking_embedding_loss_coeff}).",
                        ))

            rhs_emb_modules = (
                LinkPredOutputType.SHALLOW_RHS,
                LinkPredOutputType.CONTEXT,
                LinkPredOutputType.RANKING,
                LinkPredOutputType.EMBEDDING,
            )
            assert isinstance(self.target_embedding_mode, list)
            if (self.module in rhs_emb_modules
                    and self.handle_new_target_entities and
                (RHSEmbeddingMode.LOOKUP in self.target_embedding_mode
                 or RHSEmbeddingMode.FUSION in self.target_embedding_mode)):
                modes = set(mode.value for mode in self.target_embedding_mode)
                errors.append(
                    ValueError(
                        f"'handle_new_target_entities' is set to 'True' which "
                        f"requires 'feature' target embedding mode only, but "
                        f"'target_embedding_mode' is set to {modes}."))

        return errors, warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
    ) -> Self:

        out = cls()

        # Set task-agnostic model options:
        out.model = [
            GNNModelPlan.from_default(
                task_type=task_type,
                query_type=query_type,
                has_train_table_time_col=has_train_table_time_col,
                run_mode=run_mode,
                graph=graph,
                entity_table_names=entity_table_names,
                target_to_entity_paths=target_to_entity_paths,
                has_train_table_weight_col=has_train_table_weight_col,
            )
        ]

        out.num_pre_message_passing_layers = [0, 1, 2]
        out.num_post_message_passing_layers = [1, 2]
        out.activation = [
            ActivationType.RELU,
            ActivationType.LEAKY_RELU,
            ActivationType.ELU,
            ActivationType.GELU,
        ]
        out.normalization = [NormalizationType.LAYER_NORM]
        out.use_seq_id = [False]
        if (query_type == QueryType.TEMPORAL
                and task_type != TaskType.TEMPORAL_LINK_PREDICTION):
            out.prediction_time_encodings = [False]

        # Set link prediction-specific model options:
        if task_type.is_link_pred:
            if task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                out.module = LinkPredOutputType.CONTEXT
            else:
                out.module = LinkPredOutputType.SHALLOW_RHS
            out.handle_new_target_entities = False
            out.target_embedding_mode = [RHSEmbeddingMode.LOOKUP]
            out.output_embedding_dim = [32]
            out.embedding_norm_reg_coeff = [0.0]
        if task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            out.ranking_embedding_loss_coeff = [0.0]

        # Set forecasting-specific model options:
        if (task_type in {TaskType.REGRESSION, TaskType.FORECASTING}
                and query_type == QueryType.TEMPORAL):
            out.past_encoder = [PastEncoderType.DECOMPOSED]
            out.handle_new_entities = True

        if task_type.is_link_pred:
            out.distance_measure = [DistanceMeasureType.DOT_PRODUCT]

        return out

    def unify(self) -> 'ModelArchitecturePlan':
        """Unifies the model Architecture plan to a single source of truth.

        The model can be defined by:
        - Old API: using the ModelArchitecturePlan attributes like `channels`,
        `aggregation`, `num_pre_message_passing_layers`, etc.
            Example:
            ```python
            cfg = ModelArchitecturePlan()
            cfg.channels = [64, 128]
            cfg.aggregation = [[AggregationType.SUM, AggregationType.MEAN]]
            cfg.num_pre_message_passing_layers = [0, 1, 2]
            ```
        - New API: Setting the attribute `model` to list of
        `MessagePassingModelPlan` objects.
            Example:
            ```python
            cfg = ModelArchitecturePlan()
            cfg.model = [
                GNNModelPlan(
                    channels=[128, 256],
                    aggregation = [[AggregationType.MEAN]],
                    ...
                ),
                GraphTransformerModelPlan(
                    channels=[64, 128],
                    num_layers=[2, 4],
                    ...
                ),
            ]
            ```

        This function converts the ModelArchitecturePlan attributes from the
        old API to the new API, setting the attribute `model` to a list of
        `MessagePassingModelPlan` objects.
        """
        out = copy.deepcopy(self)

        if not isinstance(out.model, list):
            out.model = [
                GNNModelPlan(
                    channels=[64, 128, 256],
                    aggregation=[
                        [
                            AggregationType.SUM,
                            AggregationType.MEAN,
                            AggregationType.MAX,
                        ],
                        [
                            AggregationType.SUM,
                            AggregationType.MEAN,
                            AggregationType.MIN,
                            AggregationType.MAX,
                            AggregationType.STD,
                        ],
                    ],
                    dropout=[0.0],
                )
            ]

        if isinstance(out.channels, list):
            for model in out.model:
                model.channels = out.channels
            out.channels = MissingType.VALUE

        if isinstance(out.aggregation, list):
            for model in out.model:
                if isinstance(model, GNNModelPlan):
                    model.aggregation = out.aggregation
            out.aggregation = MissingType.VALUE

        return out


@dataclass
class PredictiveQueryAdvancedOptions:
    """Portable model-plan overrides for a predictive-query config."""

    timeframe_step: int | None = None
    forecast_length: int | None = None
    lag_timesteps: int | None = None
    train_start_offset: int | None = None
    train_end_offset: int | None = None
    split: Any | None = None
    year_over_year: bool | None = False
    start_time: int | str | None = None
    end_time: int | str | None = None
    weight_col: str | None = None

    def apply_overrides(self, pq_config: Any) -> Any:
        for field_name in (
                'timeframe_step',
                'forecast_length',
                'lag_timesteps',
                'train_start_offset',
                'train_end_offset',
                'start_time',
                'end_time',
                'year_over_year',
                'weight_col',
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(pq_config, field_name, value)
        if self.split is not None:
            if hasattr(self.split, 'flexible_boundaries'):
                self.split.flexible_boundaries = False
            pq_config.split = self.split
        return pq_config


@dataclass(config=dict(extra='forbid'))  # type: ignore
class ModelPlan:
    training_table_generation: TrainingTableGenerationPlan = field(
        default_factory=TrainingTableGenerationPlan)

    training_job: TrainingJobPlan = field(default_factory=TrainingJobPlan)

    column_processing: ColumnProcessingPlan = field(
        default_factory=ColumnProcessingPlan)

    neighbor_sampling: NeighborSamplingPlan = field(
        default_factory=NeighborSamplingPlan)

    optimization: OptimizationPlan = field(default_factory=OptimizationPlan)

    model_architecture: ModelArchitecturePlan = field(
        default_factory=ModelArchitecturePlan)

    # =========================================================================

    def validate(
        self,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        rhs_table_name_rows: tuple[str, int] | None = None,
        raise_on_error: bool = True,
        has_train_table_weight_col: bool = False,
    ) -> tuple[list[Exception], list[ValidationWarning]]:
        r"""Validates that all attributes for a given :obj:`task_type` and
        `query_type` are correctly set, *i.e.* there do not exist required but
        missing attributes.
        """
        all_errors: list[Exception] = []
        all_warnings: list[ValidationWarning] = []

        for key in self.__dataclass_fields__:  # type: ignore
            section = getattr(self, key)
            errors, warnings = section.validate(
                task_type,
                query_type,
                has_train_table_time_col,
                run_mode,
                graph,
                entity_table_names,
                target_to_entity_paths,
                has_train_table_weight_col,
            )
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        with_lookup_optimization = any(
            isinstance(lookup_optimization, OptimizerConfig)
            for lookup_optimization in self.optimization.lookup_optimization)
        if task_type.is_link_pred and with_lookup_optimization:
            # `TWO_TOWER` and `PAIRWISE` never use lookup embeddings
            # (regardless of task type). The legacy `EMBEDDING` alias resolves
            # to `TWO_TOWER` only for static LP (for temporal LP it maps to
            # `SHALLOW_RHS`, which does use lookup).
            module = self.model_architecture.module
            if (module in (LinkPredOutputType.TWO_TOWER,
                           LinkPredOutputType.PAIRWISE)
                    or (task_type == TaskType.STATIC_LINK_PREDICTION
                        and module == LinkPredOutputType.EMBEDDING)):
                warning = ValidationWarning(
                    title="Ignored Option: lookup_optimization",
                    message="The option 'lookup_optimization' will be "
                    "ignored since no lookup embeddings will be optimized "
                    "for this link-prediction module.",
                )
                all_warnings.append(warning)
            elif not any(emb_mode.use_rhs_lookup for emb_mode in
                         self.model_architecture.target_embedding_mode):
                warning = ValidationWarning(
                    title="Ignored Option: lookup_optimization",
                    message="The option 'lookup_optimization' will be "
                    "ignored since no lookup embeddings will be optimized "
                    "for feature-based target embedding modes.",
                )
                all_warnings.append(warning)

        rhs_table_name, num_rhs_rows = rhs_table_name_rows or ('', 0)
        rhs_n_limit, rhs_dim_limit = RHS_NUM_EMB_SIZE_LIMIT
        if task_type.is_link_pred and num_rhs_rows > rhs_n_limit:
            dim = self.model_architecture.output_embedding_dim
            if (isinstance(dim, list)
                    and (dim_to_warn := [d for d in dim if d > rhs_dim_limit])):
                warning = ValidationWarning(
                    title="GPU Memory: High Embedding Dimension",
                    message=(
                        f"Table '{rhs_table_name}' contains {rhs_table_name} "
                        f"rows, and 'output_embedding_dim' is set to "
                        f"{dim_to_warn}. This combination may cause a GPU "
                        f"out-of-memory error. We recommend setting "
                        f"'output_embedding_dim' to {rhs_dim_limit} or "
                        f"lower when the "
                        f"table has over 5 million rows."),
                )
                all_warnings.append(warning)

        if (self.neighbor_sampling.adaptive_sampling
                and entity_table_names is not None
                and len(entity_table_names) > 0 and graph is not None):
            total_node_count = (
                self.neighbor_sampling.estimate_max_adaptive_sample_size(
                    graph, entity_table_names,
                    max(self.optimization.batch_size)))
            if total_node_count > SAMPLE_SIZE_WARN:
                all_warnings.append(
                    ValidationWarning(
                        title="GPU Memory: Large Sample Size",
                        message=f"'num_neighbors' config is expected to "
                        f"produce graph samples "
                        f"of size {total_node_count}, which exceeds the "
                        f"recommended maximum of {SAMPLE_SIZE_WARN}. "
                        f"This is likely to slow down the training or cause a "
                        f"GPU out-of-memory error. Consider reducing the "
                        f"number of hops or neighbors in your "
                        f"'num_neighbors' configuration."))

        # `target_fraction` and `overlap_ratio` are only consumed by
        # `ShallowRHSLinkGNNLightningModule` on static LP. Warn the user when
        # the configured module does not route to that code path.
        if (task_type == TaskType.STATIC_LINK_PREDICTION
                and self.model_architecture.module
                in (LinkPredOutputType.EMBEDDING,
                    LinkPredOutputType.TWO_TOWER)):
            if any([v != 1.0 for v in self.optimization.target_fraction]):
                all_warnings.append(
                    ValidationWarning(
                        title="Ignored option 'target_fraction'",
                        message="The option 'target_fraction' is only "
                        "supported for the 'shallow_rhs' module on static "
                        "link prediction tasks. It will be ignored for the "
                        f"'{self.model_architecture.module.value}' module."))

            if any([v != 1.0 for v in self.optimization.overlap_ratio]):
                all_warnings.append(
                    ValidationWarning(
                        title="Ignored option 'overlap_ratio'",
                        message="The option 'overlap_ratio' is only "
                        "supported for the 'shallow_rhs' module on static "
                        "link prediction tasks. It will be ignored for the "
                        f"'{self.model_architecture.module.value}' module."))

        for warning in all_warnings:
            logger.warning(f"{warning.title}: {warning.message}")

        if raise_on_error and len(all_errors) > 0:
            raise ModelPlanValueError(all_errors)

        return all_errors, all_warnings

    @classmethod
    def from_default(
        cls,
        task_type: TaskType,
        query_type: QueryType,
        has_train_table_time_col: bool,
        run_mode: RunMode,
        graph: GraphConfig | None = None,
        entity_table_names: list[str] | None = None,
        target_to_entity_paths: list[list[tuple[str, str, str]]] | None = None,
        has_train_table_weight_col: bool = False,
        *,
        split: str | None = None,
        forecast_length: int | None = None,
        top_k: int | None = None,
    ) -> Self:
        kwargs = dict(
            task_type=task_type,
            query_type=query_type,
            has_train_table_time_col=has_train_table_time_col,
            run_mode=run_mode,
            graph=graph,
            entity_table_names=entity_table_names,
            target_to_entity_paths=target_to_entity_paths,
            has_train_table_weight_col=has_train_table_weight_col,
        )
        plan = cls(
            TrainingTableGenerationPlan.from_default(
                **kwargs,  # type: ignore
            ),
            TrainingJobPlan.from_default(
                **kwargs,  # type: ignore
            ),
            ColumnProcessingPlan.from_default(
                **kwargs,  # type: ignore
            ),
            NeighborSamplingPlan.from_default(
                **kwargs,  # type: ignore
            ),
            OptimizationPlan.from_default(
                **kwargs,  # type: ignore
            ),
            ModelArchitecturePlan.from_default(
                **kwargs,  # type: ignore
            ),
        )
        if split is not None:
            plan.training_table_generation.split = split
        if forecast_length is not None:
            plan.training_table_generation.forecast_length = forecast_length
        if task_type.is_ranking and top_k is not None:
            metrics = plan.training_job.metrics
            assert isinstance(metrics, list)
            metrics += [
                f'map@{top_k}', f'precision@{top_k}', f'recall@{top_k}',
                f'f1@{top_k}'
            ]
            plan.training_job.metrics = list(set(metrics))
        return plan

    def unify(self) -> 'ModelPlan':
        out = copy.deepcopy(self)
        out.model_architecture = out.model_architecture.unify()
        return out


# Helpers =====================================================================


def inferred_or_missing_type_to_none(
        val: InferredType | MissingType | T) -> T | None:
    return None if _is_inferred(val) or _is_missing(val) else val


def inferred_or_missing_type_to_false(
        val: InferredType | MissingType | bool) -> bool:
    return False if _is_inferred(val) or _is_missing(val) else val
