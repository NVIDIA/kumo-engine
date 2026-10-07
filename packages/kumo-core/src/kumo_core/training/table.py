# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable DiskGraph train-table generation for node and link tasks."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import diskgraph
from kumoapi.typing import ProblemType

from ..model_plan import repr_to_object
from .splits import PLAN_SPLITS, SplitArguments, uses_pseudo_time
from .validation import GenerationValidationError, validate_target_classes

_LINK_TASK_TYPES = frozenset({
    "static_link_prediction",
    "temporal_link_prediction",
    "link_prediction",
})
#: Only these splits append ``exclude_target_keys``, so only these can have
#: it resolved.
_PER_LINK_SPLIT_FAMILIES = frozenset({
    "RandomLinkSplit",
    "CustomLinkSplit",
})
_NODE_TASK_TYPES = frozenset({
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
    "regression",
})
_SPLIT_FIELDS = frozenset({
    "ratio",
    "rnd_seed",
    "offsets",
    "ranges",
    "backtesting_timestamp",
    "split_column",
    "manual_train_values",
    "manual_val_values",
    "manual_test_values",
    "timeframe",
    "unit",
    "offset_unit",
    "anchor_start_offset",
    "flexible_boundaries",
    "timeframe_step",
})


def split_from_repr(value: str) -> SplitArguments:
    """Resolve a model-plan constructor into its split descriptor."""

    try:
        split = repr_to_object(value, PLAN_SPLITS)
    except SyntaxError as error:
        raise ValueError(f"split {value!r} is not a valid constructor "
                         f"expression: {error}") from error
    except KeyError as error:
        raise ValueError(
            f"split {value!r} is not supported by portable training; use one "
            f"of {sorted(PLAN_SPLITS)}") from error
    except TypeError as error:
        raise ValueError(f"split {value!r} does not match its constructor: "
                         f"{error}") from error
    except ValueError:
        raise
    except Exception as error:  # The constructor rejected its arguments.
        raise ValueError(f"split {value!r} cannot be resolved: "
                         f"{type(error).__name__}: {error}") from error
    if not isinstance(split, SplitArguments):
        raise ValueError(f"split {value!r} must call one of "
                         f"{sorted(PLAN_SPLITS)}")
    return split


def split_config_from_repr(value: str) -> tuple[dict[str, Any], str]:
    """Resolve a split constructor without query-specific normalization."""
    return split_config_from_descriptor(split_from_repr(value))


def split_config_from_descriptor(
    split: SplitArguments,
) -> tuple[dict[str, Any], str]:
    config = {
        key: item
        for key, item in split.to_split_config().items() if item is not None
    }
    unknown = set(config) - _SPLIT_FIELDS
    assert not unknown, f"portable split emitted unknown fields: {unknown!r}"
    return config, type(split).__name__


@dataclass(frozen=True)
class TrainTableRequest:
    query: Any
    timestamps: tuple[int, ...]
    task_type: str
    forecast_length: int
    lag_timesteps: int | None
    list_distinct_target_is_array: bool
    include_frequency: bool
    entity_time_col: str | None
    index_path: str
    output_path: str
    tt_id: str
    split_config: Any
    split_family: str | None
    input_timestamp_unit: Any
    target_stype: str
    entity_node_type: str
    entity_primary_key: str
    rhs_node_type: str | None = None
    rhs_is_temporal: bool = False
    downsampling: Mapping[str, Any] | None = None
    compiler_options: Mapping[str, Any] | None = None
    index_path_verified: bool = False


@dataclass(frozen=True)
class GeneratedTrainTableResult:
    train_table_path: str
    manifest_path: str
    staging_dir: str


@dataclass(frozen=True)
class CustomTrainingTable:
    """Warehouse table containing user-supplied training rows."""

    source_table_name: str
    weight_column: str | None = None
    time_column: str = "TIMESTAMP"

    def __post_init__(self) -> None:
        if not self.source_table_name:
            raise ValueError("custom training table requires a source name")
        if self.weight_column is not None and not self.weight_column:
            raise ValueError("weight_column must be non-empty when provided")
        if not self.time_column:
            raise ValueError("time_column must be non-empty")


@dataclass(frozen=True)
class PreparedCustomTrainingTable:
    """Connector-staged custom training table ready for DiskGraph."""

    parquet_path: str
    weight_column: str | None = None
    time_column: str = "TIMESTAMP"

    def __post_init__(self) -> None:
        if not self.parquet_path:
            raise ValueError("prepared custom training table requires a path")


@dataclass(frozen=True)
class TrainTableIngestResult:
    extension_path: str
    manifest_path: str


def _coerce_split_config(split_config: Any) -> dict[str, Any]:
    if split_config is None:
        return {}
    if not isinstance(split_config, Mapping):
        raise RuntimeError("split_config must be a resolved mapping")
    return dict(split_config)


def _canonicalize_split(config: Mapping[str, Any],
                        family: str) -> dict[str, Any]:
    result = dict(config)
    del family
    for legacy, canonical in {
            "_target_": "split_family",
            "timestamp": "backtesting_timestamp",
            "train_value": "manual_train_values",
            "val_value": "manual_val_values",
            "test_value": "manual_test_values",
            "col": "split_column",
    }.items():
        if legacy in result:
            raise RuntimeError(
                f"split_config.{legacy} is a legacy alias; provide exact "
                f"{canonical} through the compatibility adapter")
    return result


def _epoch_seconds(value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError("timestamp value must not be bool")
    if isinstance(value, Integral):
        return int(value)
    raise RuntimeError(
        f"timestamp value must be resolved integer epoch seconds; got {value!r}"
    )


def build_split_spec(
    split_config: Any,
    split_family_hint: str | None,
    split_spec_class: type,
) -> Any | None:
    config = _coerce_split_config(split_config)
    if not config and not split_family_hint:
        return None
    if not isinstance(split_family_hint, str) or not split_family_hint:
        raise RuntimeError("split_family must be resolved explicitly")
    family = split_family_hint
    if "." in family:
        raise RuntimeError(
            "split_family must be an exact unqualified portable name")
    config = _canonicalize_split(config, family)
    unknown_fields = set(config) - _SPLIT_FIELDS
    if unknown_fields:
        raise RuntimeError("split_config contains unsupported fields: "
                           f"{sorted(unknown_fields)!r}")

    def string_list(value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or not all(
                isinstance(item, str) for item in value):
            raise RuntimeError(
                "manual split values must be resolved lists of strings")
        return list(value)

    ranges = config.get("ranges")
    parsed_ranges = None
    if ranges is not None:
        if not isinstance(ranges, (list, tuple)) or not all(
                isinstance(pair, (list, tuple)) and len(pair) == 2
                for pair in ranges):
            raise RuntimeError(
                "split_config.ranges must contain exact start/end pairs")
        parsed_ranges = [(_epoch_seconds(pair[0]), _epoch_seconds(pair[1]))
                         for pair in ranges] or None
    column = config.get("split_column")
    if column is not None and (not isinstance(column, str) or not column):
        raise RuntimeError("split_column must be a non-empty string")

    def integer(field_name: str) -> int | None:
        value = config.get(field_name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RuntimeError(f"split_config.{field_name} must be an integer")
        return int(value)

    def integer_list(field_name: str) -> list[int] | None:
        value = config.get(field_name)
        if value is None:
            return None
        if (not isinstance(value, (list, tuple)) or not value or any(
                isinstance(item, bool) or not isinstance(item, Integral)
                for item in value)):
            raise RuntimeError(
                f"split_config.{field_name} must be a non-empty list of "
                "integers")
        return [int(item) for item in value]

    ratio = config.get("ratio")
    parsed_ratio = None
    if ratio is not None:
        if (not isinstance(ratio, (list, tuple)) or not ratio or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in ratio)):
            raise RuntimeError(
                "split_config.ratio must be a non-empty list of numbers")
        parsed_ratio = [float(item) for item in ratio]
    unit = config.get("unit")
    if unit is not None and (not isinstance(unit, str) or not unit):
        raise RuntimeError("split_config.unit must be a non-empty string")
    offset_unit = config.get("offset_unit")
    if offset_unit is not None and (not isinstance(offset_unit, str)
                                    or not offset_unit):
        raise RuntimeError(
            "split_config.offset_unit must be a non-empty string")
    flexible_boundaries = config.get("flexible_boundaries")
    if flexible_boundaries is not None and not isinstance(
            flexible_boundaries, bool):
        raise RuntimeError("split_config.flexible_boundaries must be a bool")
    return split_spec_class(
        ratio=parsed_ratio,
        kind=family,
        rnd_seed=integer("rnd_seed"),
        is_temporal=family in {
            "TemporalSplit",
            "ApproxDateOffsetSplit",
            "DateOffsetSplit",
            "TimeRangeSplit",
            "BacktestingSplit",
        },
        offsets=integer_list("offsets"),
        ranges=parsed_ranges,
        backtesting_timestamp=_epoch_seconds(config["backtesting_timestamp"])
        if config.get("backtesting_timestamp") is not None else None,
        split_column=column if isinstance(column, str) and column else None,
        manual_train_values=string_list(config.get("manual_train_values")),
        manual_val_values=string_list(config.get("manual_val_values")),
        manual_test_values=string_list(config.get("manual_test_values")),
        timeframe=integer("timeframe"),
        unit=unit,
        offset_unit=offset_unit,
        anchor_start_offset=integer("anchor_start_offset"),
        flexible_boundaries=flexible_boundaries,
        timeframe_step=integer("timeframe_step"),
    )


def apply_reference_resolution(
    plan_json: str,
    *,
    entity_node_type: str,
    rhs_node_type: str | None,
    resolve_link_targets: bool,
    is_static_link: bool,
    rhs_is_temporal: bool,
) -> str:
    plan = json.loads(plan_json)
    if not isinstance(entity_node_type, str) or not entity_node_type:
        raise RuntimeError("entity_node_type must be resolved before "
                           "train-table generation")
    if "." in entity_node_type:
        raise RuntimeError(
            "entity_node_type must be an exact unqualified graph node type")
    if rhs_node_type is not None and (not isinstance(rhs_node_type, str)
                                      or not rhs_node_type
                                      or "." in rhs_node_type):
        raise RuntimeError(
            "rhs_node_type must be an exact unqualified graph node type")
    resolution: list[dict[str, Any]] = [{
        "src_col": "ENTITY1",
        "dst_table": entity_node_type,
        "kind": "scalar",
        "node_id_col": "entity_node_id",
    }]
    if resolve_link_targets:
        if not isinstance(rhs_node_type, str) or not rhs_node_type:
            raise RuntimeError(
                "rhs_node_type must be resolved before link prediction")
        resolution.append({
            "src_col": "TARGET1",
            "dst_table": rhs_node_type,
            "kind": "list",
            "node_id_col": "target_node_ids",
        })
        if is_static_link:
            resolution.append({
                "src_col": "exclude_target_keys",
                "dst_table": rhs_node_type,
                "kind": "list",
                "node_id_col": "exclude_node_ids",
            })
    plan["lp_resolution"] = resolution
    if resolve_link_targets:
        plan["lp_rhs_is_temporal"] = bool(rhs_is_temporal)
    return json.dumps(plan)


def _discard_invalid_staging_dir(staging_dir: str,
                                 request: TrainTableRequest) -> None:
    """Delete only a generated staging directory beneath this request's paths."""

    candidate = Path(os.path.abspath(staging_dir))
    allowed_parents = (
        Path(os.path.abspath(request.index_path)) / "train_tables",
        Path(os.path.abspath(request.output_path)).parent,
    )
    if (candidate.name != f".{request.tt_id}.staging"
            or candidate.parent not in allowed_parents):
        raise RuntimeError(
            f"refusing to remove staging directory outside the index or "
            f"output staging path: {staging_dir!r}")
    if any(path.is_symlink() for path in (candidate, *candidate.parents)):
        raise RuntimeError(
            f"refusing to remove staging directory through a symlink: "
            f"{staging_dir!r}")
    shutil.rmtree(candidate)


def _observed_target_classes(staging_dir: str, target_stype: str) -> list[Any]:
    """Read the exact target vocabulary from DiskGraph's staging bundle."""

    stats_path = Path(staging_dir) / "train_table_col_stats.json"
    with stats_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (payload.get("version"), payload.get("kind"),
            payload.get("stats_format")) != (1, "tt_train_table_col_stats",
                                             "kumo_col_stats_v1"):
        raise RuntimeError("generated train-table target stats are invalid")
    stats = payload.get("stats")
    kind = ("MULTI_CATEGORY_COUNTS"
            if target_stype == "multicategorical" else "CATEGORY_COUNTS")
    # Native category statistics canonicalize ID and boolean to categorical.
    stats_stype = ("categorical"
                   if target_stype in {"ID", "boolean"} else target_stype)
    entry = (stats.get(f"TARGET1__kumo.{stats_stype}__{kind}__False")
             if isinstance(stats, dict) else None)
    if (not isinstance(entry, dict) or entry.get("approx") is not False
            or not isinstance(entry.get("value"), list)):
        raise RuntimeError(
            "generated train table is missing exact target-class counts")
    return [value for value, count in entry["value"] if count > 0]


def _observed_target_class_count(staging_dir: str) -> int:
    return len(_observed_target_classes(staging_dir, "multicategorical"))


def _validate_classification_targets(request: TrainTableRequest,
                                     staging_dir: str) -> None:
    if request.task_type.lower() not in {
            "binary_classification", "multiclass_classification",
            "multilabel_classification"
    }:
        return
    classes = _observed_target_classes(staging_dir, request.target_stype)
    errors = validate_target_classes(
        max(1, len(classes)), classes,
        classify=getattr(request.query, "problem_type",
                         None) == ProblemType.CLASSIFY)
    if errors:
        raise GenerationValidationError(errors[0])


def _fact_pseudo_time(request: TrainTableRequest) -> dict[str, str] | None:
    from ..pquery import static_link_fact
    if not uses_pseudo_time(request.task_type):
        return None
    return static_link_fact(request.query, request.rhs_node_type)


def generate_train_table(
    request: TrainTableRequest,
    *,
    runtime: Any | None = None,
) -> GeneratedTrainTableResult:
    """Compile and execute the real DiskGraph TT-NG node/link producer."""

    task_type = request.task_type.lower()
    if task_type not in _NODE_TASK_TYPES | _LINK_TASK_TYPES | {"forecasting"}:
        raise RuntimeError("native train-table generation does not support "
                           f"prediction tasks; got {request.task_type!r}")
    if (task_type in _LINK_TASK_TYPES
            and not request.list_distinct_target_is_array):
        raise RuntimeError(
            "native link train-table generation requires array targets")
    if request.split_config is None:
        raise RuntimeError("train-table generation requires split_config")
    if (not request.index_path or (not request.index_path_verified
                                   and not Path(request.index_path).exists())):
        raise RuntimeError(
            f"diskgraph index path does not exist: {request.index_path!r}")
    if (not request.tt_id or request.tt_id in {".", ".."}
            or any(char in request.tt_id for char in ("/", "\\", "\0"))):
        raise RuntimeError(
            f"tt_id must be filename-safe; got {request.tt_id!r}")

    if runtime is None:
        runtime = diskgraph

    split = build_split_spec(
        request.split_config,
        request.split_family,
        runtime.SplitSpec,
    )
    if split is None:
        raise RuntimeError("split_config has no resolvable split family")
    compiler = runtime.DiskGraphPlanCompiler()
    compile_options = dict(request.compiler_options or {})
    plan_json = compiler.compile(
        query=request.query,
        timestamps=list(request.timestamps),
        task_type=request.task_type,
        forecast_length=request.forecast_length,
        lag_timesteps=request.lag_timesteps,
        list_distinct_target_is_array=request.list_distinct_target_is_array,
        include_frequency=request.include_frequency,
        entity_time_col=request.entity_time_col,
        input_timestamp_unit=request.input_timestamp_unit,
        target_stype=request.target_stype,
        split=split,
        lp_rhs_entity_table=request.rhs_node_type,
        **compile_options,
    )
    is_link_array = task_type in _LINK_TASK_TYPES
    resolve_past_visits = (task_type == "static_link_prediction"
                           and split.kind in _PER_LINK_SPLIT_FAMILIES)
    plan_json = apply_reference_resolution(
        plan_json,
        entity_node_type=request.entity_node_type,
        rhs_node_type=request.rhs_node_type,
        resolve_link_targets=is_link_array,
        is_static_link=resolve_past_visits,
        rhs_is_temporal=request.rhs_is_temporal,
    )
    fact_pseudo_time = _fact_pseudo_time(request)
    if fact_pseudo_time is not None:
        plan_json = json.dumps({
            **json.loads(plan_json),
            "fact_pseudo_time": fact_pseudo_time,
        })
    os.makedirs(
        os.path.dirname(request.output_path) or request.output_path,
        exist_ok=True)
    result = runtime.execute_tt_gen(plan_json, request.index_path,
                                    request.tt_id)
    staging_dir = result.get("staging_dir")
    table_path = result.get("tt_parquet_path")
    if not isinstance(staging_dir, str) or not staging_dir:
        raise RuntimeError("execute_tt_gen result missing staging_dir")
    if not isinstance(table_path, str) or not table_path:
        raise RuntimeError("execute_tt_gen result missing tt_parquet_path")
    try:
        _validate_classification_targets(request, staging_dir)
    except ValueError:
        _discard_invalid_staging_dir(staging_dir, request)
        raise
    top_k = getattr(request.query, "top_k", None)
    if (task_type in {"multilabel_classification", "multilabel_ranking"}
            and top_k is not None):
        class_count = _observed_target_class_count(staging_dir)
        if top_k > class_count:
            # The generator's staging bundle is private until publication.
            # Discard it so callers cannot mistake invalid rows for a table.
            _discard_invalid_staging_dir(staging_dir, request)
            raise ValueError(
                f"TOP K {top_k} exceeds the {class_count} observed target "
                f"classes; supported range is 1..{class_count}.")
    return GeneratedTrainTableResult(
        train_table_path=table_path,
        manifest_path=os.path.join(staging_dir, "manifest.json"),
        staging_dir=staging_dir,
    )


def generate_custom_train_table(
    request: TrainTableRequest,
    custom_table: PreparedCustomTrainingTable,
    *,
    runtime: Any | None = None,
) -> GeneratedTrainTableResult:
    """Build the standard TT-NG bundle from an existing parquet table."""

    if not isinstance(custom_table, PreparedCustomTrainingTable):
        raise TypeError("custom_table must be PreparedCustomTrainingTable")
    if not Path(custom_table.parquet_path).exists():
        raise RuntimeError("custom training table parquet does not exist: "
                           f"{custom_table.parquet_path!r}")
    if runtime is None:
        runtime = diskgraph
    split = build_split_spec(
        request.split_config,
        request.split_family,
        runtime.SplitSpec,
    )
    if split is None:
        raise RuntimeError("custom training table requires a resolved split")
    result = runtime.build_train_table_artifacts_for_existing_table(
        output_path=custom_table.parquet_path,
        split_config=split,
        tt_id=request.tt_id,
        split_family_hint=split.kind,
        split_stats_config={
            "entity_table": request.entity_node_type,
            "entity_pk_column": request.entity_primary_key,
        },
        train_table_col_stats_config={
            "target_stype":
            request.target_stype,
            "time_col":
            custom_table.time_column,
            **({
                "weight_col": custom_table.weight_column
            } if custom_table.weight_column else {}),
        },
        index_path=request.index_path,
        input_timestamp_unit=request.input_timestamp_unit,
        target_entity_table=request.rhs_node_type,
        task_type=request.task_type,
        lp_rhs_is_temporal=request.rhs_is_temporal,
        fact_pseudo_time=_fact_pseudo_time(request),
    )
    if not isinstance(result, Mapping):
        raise RuntimeError(
            "custom train-table builder did not return a mapping")
    staging_dir = result.get("staging_dir")
    if not isinstance(staging_dir, str) or not staging_dir:
        raise RuntimeError(
            "custom train-table builder result missing staging_dir")
    _validate_classification_targets(request, staging_dir)
    return GeneratedTrainTableResult(
        train_table_path=custom_table.parquet_path,
        manifest_path=os.path.join(staging_dir, "manifest.json"),
        staging_dir=staging_dir,
    )


def ingest_train_table(
    *,
    staging_dir: str,
    index_path: str,
    tt_id: str,
    runtime: Any | None = None,
) -> TrainTableIngestResult:
    """Publish a generated TT-NG bundle into its graph-index extension."""

    if not staging_dir or not os.path.isdir(staging_dir):
        raise RuntimeError(
            f"train-table staging directory does not exist: {staging_dir!r}")
    if not index_path or not os.path.isdir(index_path):
        raise RuntimeError(
            f"diskgraph index path does not exist: {index_path!r}")
    if runtime is None:
        try:
            import diskgraph as runtime
        except ImportError as error:
            raise RuntimeError(
                "the diskgraph runtime is required for train-table ingest"
            ) from error
    result = runtime.execute_tt_ingest(
        staging_dir=staging_dir,
        index_path=index_path,
        tt_id=tt_id,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("execute_tt_ingest did not return a mapping")
    expected = os.path.abspath(os.path.join(index_path, "train_tables", tt_id))
    published = result.get("tt_extension_path", expected)
    if not isinstance(published, str) or os.path.abspath(published) != expected:
        raise RuntimeError(
            "execute_tt_ingest published the extension at an unexpected path: "
            f"{published!r}; expected {expected!r}")
    manifest_path = os.path.join(expected, "manifest.json")
    diskgraph_manifest = os.path.join(expected, "diskgraph_manifest.json")
    if not os.path.isfile(manifest_path) or not os.path.isfile(
            diskgraph_manifest):
        raise RuntimeError(
            "execute_tt_ingest did not publish the required train-table manifests"
        )
    return TrainTableIngestResult(expected, manifest_path)
