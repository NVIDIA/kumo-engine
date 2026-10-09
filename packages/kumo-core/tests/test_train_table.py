# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.training.splits import (
    PLAN_SPLITS,
    TrainTableTime,
    train_table_time,
)
from kumo_core.training.table import (
    PreparedCustomTrainingTable,
    TrainTableRequest,
    _discard_invalid_staging_dir,
    _observed_target_class_count,
    apply_reference_resolution,
    build_split_spec,
    generate_custom_train_table,
    generate_train_table,
    split_config_from_repr,
)
from kumo_core.training.validation import GenerationValidationError
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType


class FakeSplitSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCompiler:
    def __init__(self, calls):
        self.calls = calls

    def compile(self, **kwargs):
        self.calls.append(("compile", kwargs))
        return json.dumps({"entity": {"table": "users"}})


class FakeRuntime:
    SplitSpec = FakeSplitSpec

    def __init__(self, staging_dir):
        self.calls = []
        self.DiskGraphPlanCompiler = lambda: FakeCompiler(self.calls)
        self.staging_dir = staging_dir

    def execute_tt_gen(self, plan, index_path, tt_id):
        self.calls.append(("execute", json.loads(plan), index_path, tt_id))
        write_valid_class_stats(Path(self.staging_dir))
        return {
            "staging_dir": self.staging_dir,
            "tt_parquet_path": f"{self.staging_dir}/tt.parquet",
        }


def request(tmp_path, **overrides):
    values = dict(
        query=object(),
        timestamps=(1, 2),
        task_type="binary_classification",
        forecast_length=1,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        index_path=str(tmp_path / "index"),
        output_path=str(tmp_path / "output" / "train.parquet"),
        tt_id="tt-1",
        split_config={"ratio": [0.8, 0.1, 0.1]},
        split_family="RandomSplit",
        input_timestamp_unit="nanoseconds",
        target_stype="categorical",
        entity_node_type="users",
        entity_primary_key="user_id",
    )
    values.update(overrides)
    (tmp_path / "index").mkdir(exist_ok=True)
    return TrainTableRequest(**values)


def static_link_query():
    from kumo_core.pquery import normalize_graph, parse_and_validate_query

    from tests.test_pquery import STATIC_LINK_PQL, graph_config
    return parse_and_validate_query(STATIC_LINK_PQL,
                                    normalize_graph(graph_config())).query


def write_target_stats(staging, entry):
    stats_path = staging / 'train_table_col_stats.json'
    stats_path.write_text(
        json.dumps({
            'version': 1,
            'kind': 'tt_train_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'stats': {
                'TARGET1__kumo.multicategorical__MULTI_CATEGORY_COUNTS__False':
                entry
            },
        }))


def write_valid_class_stats(staging, classes=None):
    staging.mkdir(parents=True, exist_ok=True)
    stats_path = staging / 'train_table_col_stats.json'
    if stats_path.exists():
        return
    stats_path.write_text(
        json.dumps({
            'version': 1,
            'kind': 'tt_train_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'stats': {
                'TARGET1__kumo.categorical__CATEGORY_COUNTS__False': {
                    'approx':
                    False,
                    'value': ([[False, 400], [True, 400]]
                              if classes is None else classes),
                },
            },
        }))


@pytest.mark.parametrize(
    'task_type,target_stype,classes,message',
    [('binary_classification', 'categorical', [[False, 800]
                                               ], 'Too few classes'),
     ('multiclass_classification', 'categorical', [['only', 800]
                                                   ], 'Too few classes'),
     ('multiclass_classification', 'ID', [[10, 800]], 'Too few classes'),
     ('binary_classification', 'boolean', [[False, 800]], 'Too few classes'),
     ('multilabel_classification', 'multicategorical', [],
      'No positive samples')],
)
def test_generated_degenerate_labels_are_rejected_before_publication(
        tmp_path, task_type, target_stype, classes, message):
    staging = tmp_path / 'index' / 'train_tables' / '.tt-1.staging'
    staging.mkdir(parents=True)
    stats_stype = ('categorical'
                   if target_stype in {'ID', 'boolean'} else target_stype)
    kind = ('MULTI_CATEGORY_COUNTS'
            if target_stype == 'multicategorical' else 'CATEGORY_COUNTS')
    (staging / 'train_table_col_stats.json').write_text(
        json.dumps({
            'version': 1,
            'kind': 'tt_train_table_col_stats',
            'stats_format': 'kumo_col_stats_v1',
            'stats': {
                f'TARGET1__kumo.{stats_stype}__{kind}__False': {
                    'approx': False,
                    'value': classes,
                },
            },
        }))
    with pytest.raises(GenerationValidationError, match=message):
        generate_train_table(
            request(tmp_path, task_type=task_type, target_stype=target_stype),
            runtime=FakeRuntime(str(staging)))
    assert not staging.exists()
    assert not (staging.parent / 'tt-1').exists()


@pytest.mark.parametrize('target_stype', ['categorical', 'ID', 'boolean'])
def test_generate_node_table_calls_real_compiler_and_executor(
        tmp_path, target_stype) -> None:
    runtime = FakeRuntime(str(tmp_path / "staging"))

    if target_stype == 'ID':
        write_valid_class_stats(Path(runtime.staging_dir),
                                [[10, 400], [20, 400]])

    result = generate_train_table(request(tmp_path, target_stype=target_stype),
                                  runtime=runtime)

    assert result.train_table_path.endswith("staging/tt.parquet")
    assert result.manifest_path.endswith("staging/manifest.json")
    assert runtime.calls[0][0] == "compile"
    assert runtime.calls[1][0] == "execute"
    assert runtime.calls[1][1]["lp_resolution"] == [{
        "src_col":
        "ENTITY1",
        "dst_table":
        "users",
        "kind":
        "scalar",
        "node_id_col":
        "entity_node_id",
    }]


@pytest.mark.parametrize('top_k', [3, 4])
@pytest.mark.parametrize('task_type',
                         ['multilabel_classification', 'multilabel_ranking'])
def test_multilabel_top_k_checked_against_generated_classes(
        tmp_path, top_k, task_type) -> None:
    staging = tmp_path / 'index' / 'train_tables' / '.tt-1.staging'
    staging.mkdir(parents=True)
    write_target_stats(staging, {
        'approx': False,
        'value': [['a', 2], ['b', 2], ['c', 2]],
    })
    runtime = FakeRuntime(str(staging))
    query = SimpleNamespace(top_k=top_k)
    train_request = request(tmp_path, query=query, task_type=task_type,
                            target_stype='multicategorical')

    if top_k == 3:
        assert generate_train_table(train_request,
                                    runtime=runtime).staging_dir == str(staging)
    else:
        with pytest.raises(ValueError, match='TOP K 4.*1\\.\\.3'):
            generate_train_table(train_request, runtime=runtime)
        assert not staging.exists()
        assert not (tmp_path / 'index' / 'train_tables' / 'tt-1').exists()


@pytest.mark.parametrize('entry', [None, {'approx': True, 'value': 3}])
def test_target_class_count_requires_exact_stats(tmp_path, entry) -> None:
    write_target_stats(tmp_path, entry)

    with pytest.raises(RuntimeError, match='missing exact target-class'):
        _observed_target_class_count(str(tmp_path))


def test_target_class_count_uses_staging_bundle(tmp_path) -> None:
    write_target_stats(tmp_path,
                       {'approx': False, 'value': [['a', 2], ['b', 2]]})

    assert _observed_target_class_count(str(tmp_path)) == 2


def test_invalid_staging_cleanup_rejects_outside_directory(tmp_path) -> None:
    train_request = request(tmp_path)
    outside = tmp_path / '.tt-1.staging'
    outside.mkdir()
    marker = outside / 'keep'
    marker.touch()

    with pytest.raises(RuntimeError, match='outside the index or output'):
        _discard_invalid_staging_dir(str(outside), train_request)
    assert marker.exists()


def test_invalid_staging_cleanup_rejects_other_index_directories(
        tmp_path) -> None:
    train_request = request(tmp_path)
    unrelated = tmp_path / 'index' / 'train_tables' / 'published'
    unrelated.mkdir(parents=True)
    marker = unrelated / 'keep'
    marker.touch()

    with pytest.raises(RuntimeError, match='outside the index or output'):
        _discard_invalid_staging_dir(str(unrelated), train_request)
    assert marker.exists()


def test_invalid_staging_cleanup_allows_output_directory(tmp_path) -> None:
    train_request = request(tmp_path)
    staging = tmp_path / 'output' / '.tt-1.staging'
    staging.mkdir(parents=True)

    _discard_invalid_staging_dir(str(staging), train_request)
    assert not staging.exists()


@pytest.mark.parametrize('symlink_target', ['staging', 'parent'])
def test_invalid_staging_cleanup_rejects_symlinks(tmp_path,
                                                  symlink_target) -> None:
    train_request = request(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'keep'
    marker.touch()
    index = tmp_path / 'index' / 'train_tables'
    index.mkdir()
    if symlink_target == 'staging':
        staging = index / '.tt-1.staging'
        staging.symlink_to(outside, target_is_directory=True)
    else:
        parent = tmp_path / 'index'
        index.rmdir()
        parent.rmdir()
        parent.symlink_to(outside, target_is_directory=True)
        (outside / 'train_tables').mkdir()
        staging = parent / 'train_tables' / '.tt-1.staging'
        staging.mkdir()

    with pytest.raises(RuntimeError, match='through a symlink'):
        _discard_invalid_staging_dir(str(staging), train_request)
    assert marker.exists()
    assert staging.exists()


def test_generate_link_table_adds_fk_free_resolution(tmp_path) -> None:
    runtime = FakeRuntime(str(tmp_path / "staging"))

    generate_train_table(
        request(
            tmp_path,
            query=static_link_query(),
            task_type="static_link_prediction",
            list_distinct_target_is_array=True,
            rhs_node_type="products",
            entity_node_type="users",
            rhs_is_temporal=True,
        ),
        runtime=runtime,
    )

    plan = runtime.calls[1][1]
    assert [item["node_id_col"] for item in plan["lp_resolution"]] == [
        "entity_node_id",
        "target_node_ids",
    ]
    assert plan["lp_rhs_is_temporal"] is True
    assert plan["fact_pseudo_time"] == {
        "table": "orders",
        "entity_col": "user_id",
        "target_col": "product_id",
    }


def test_static_link_under_a_per_link_split_resolves_past_visits(
        tmp_path) -> None:
    """The exclude column is resolved only where the producer appends it."""
    runtime = FakeRuntime(str(tmp_path / "staging"))

    generate_train_table(
        request(
            tmp_path,
            query=static_link_query(),
            task_type="static_link_prediction",
            list_distinct_target_is_array=True,
            rhs_node_type="products",
            entity_node_type="users",
            split_family="RandomLinkSplit",
        ),
        runtime=runtime,
    )

    plan = runtime.calls[1][1]
    assert [item["node_id_col"] for item in plan["lp_resolution"]] == [
        "entity_node_id",
        "target_node_ids",
        "exclude_node_ids",
    ]


def test_temporal_link_never_resolves_past_visits(tmp_path) -> None:
    runtime = FakeRuntime(str(tmp_path / "staging"))

    generate_train_table(
        request(
            tmp_path,
            task_type="temporal_link_prediction",
            list_distinct_target_is_array=True,
            rhs_node_type="products",
            entity_node_type="users",
            split_family="RandomLinkSplit",
        ),
        runtime=runtime,
    )

    plan = runtime.calls[1][1]
    assert [item["node_id_col"] for item in plan["lp_resolution"]] == [
        "entity_node_id",
        "target_node_ids",
    ]


@pytest.mark.parametrize('target_stype', ['categorical', 'ID', 'boolean'])
@pytest.mark.parametrize('ingest_on_error', ['abort', 'skip_row_on_error'])
@pytest.mark.parametrize('task_type', ['binary_classification', 'temporal_link_prediction'])
def test_generate_custom_table_builds_standard_artifact_bundle(
        tmp_path, target_stype, ingest_on_error, task_type) -> None:
    parquet = tmp_path / "custom.parquet"
    parquet.mkdir()
    staging = tmp_path / "custom.parquet.artifacts"
    calls = []

    class Runtime:
        SplitSpec = FakeSplitSpec

        @staticmethod
        def build_train_table_artifacts_for_existing_table(**kwargs):
            calls.append(kwargs)
            write_valid_class_stats(
                staging,
                [[10, 400], [20, 400]] if target_stype == 'ID' else None)
            return {"staging_dir": str(staging),
                    "tt_parquet_path": str(staging / "tt.parquet")}

    result = generate_custom_train_table(
        request(tmp_path, target_stype=target_stype, task_type=task_type),
        PreparedCustomTrainingTable(
            str(parquet),
            weight_column="WEIGHT",
            ingest_on_error=ingest_on_error,
        ),
        runtime=Runtime,
    )

    assert calls[0]["ingest_on_error"] == ingest_on_error
    assert calls[0]["allow_negative_weight"] == (
        task_type == "temporal_link_prediction")
    assert result.train_table_path == str(staging / "tt.parquet")
    assert result.staging_dir == str(staging)
    assert calls[0]["output_path"] == str(parquet)
    assert calls[0]["split_stats_config"] == {
        "entity_table": "users",
        "entity_pk_column": "user_id",
    }
    assert calls[0]["train_table_col_stats_config"] == {
        "target_stype": target_stype,
        "time_col": "TIMESTAMP",
        "weight_col": "WEIGHT",
    }


def test_generate_link_table_rejects_non_array_targets(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="requires array targets"):
        generate_train_table(
            request(
                tmp_path,
                task_type="static_link_prediction",
                list_distinct_target_is_array=False,
                rhs_node_type="products",
            ),
            runtime=SimpleNamespace(),
        )


def test_split_contract_accepts_only_canonical_manual_values() -> None:
    split = build_split_spec(
        {
            "manual_train_values": ["train"],
            "manual_val_values": ["val-a", "val-b"],
        },
        "ManualSplit",
        FakeSplitSpec,
    )

    assert split.manual_train_values == ["train"]
    assert split.manual_val_values == ["val-a", "val-b"]


@pytest.mark.parametrize(
    "legacy",
    ["_target_", "timestamp", "train_value", "val_value", "test_value", "col"],
)
def test_split_contract_rejects_legacy_aliases(legacy) -> None:
    with pytest.raises(RuntimeError, match=rf"split_config\.{legacy}"):
        build_split_spec({legacy: "legacy"}, "ManualSplit", FakeSplitSpec)


def test_split_contract_rejects_unresolved_timestamp_shapes() -> None:
    with pytest.raises(RuntimeError, match="resolved integer epoch seconds"):
        build_split_spec({"backtesting_timestamp": "2025-01-01"},
                         "BacktestingSplit", FakeSplitSpec)

    with pytest.raises(RuntimeError,
                       match="split_family must be resolved explicitly"):
        build_split_spec({"ratio": [0.8, 0.1, 0.1]}, None, FakeSplitSpec)

    with pytest.raises(RuntimeError, match="exact unqualified"):
        build_split_spec({"ratio": [0.8, 0.1, 0.1]},
                         "kumo.task.split.RandomSplit", FakeSplitSpec)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({
            "unknown": 1
        }, "unsupported fields"),
        ({
            "ratio": "0.8,0.2"
        }, "non-empty list of numbers"),
        ({
            "rnd_seed": "7"
        }, "rnd_seed must be an integer"),
        ({
            "offsets": [1, "2"]
        }, "non-empty list of integers"),
        ({
            "flexible_boundaries": 1
        }, "must be a bool"),
    ],
)
def test_split_contract_rejects_silent_type_fallbacks(config, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        build_split_spec(config, "RandomSplit", FakeSplitSpec)


def test_apply_reference_resolution_for_node_prediction() -> None:
    raw = '{"entity": {"table": "users"}}'
    plan = json.loads(
        apply_reference_resolution(
            raw,
            entity_node_type="users",
            rhs_node_type=None,
            resolve_link_targets=False,
            is_static_link=False,
            rhs_is_temporal=False,
        ))
    assert plan["lp_resolution"][0]["node_id_col"] == "entity_node_id"
    assert "lp_rhs_is_temporal" not in plan


def test_apply_link_resolution_rejects_unresolved_entity_type() -> None:
    with pytest.raises(RuntimeError, match="entity_node_type must be resolved"):
        apply_reference_resolution(
            '{"entity": {"table": "users"}}',
            entity_node_type=None,
            rhs_node_type="products",
            resolve_link_targets=True,
            is_static_link=False,
            rhs_is_temporal=False,
        )

    with pytest.raises(RuntimeError, match="exact unqualified"):
        apply_reference_resolution(
            '{}',
            entity_node_type="main.features.users",
            rhs_node_type="products",
            resolve_link_targets=True,
            is_static_link=False,
            rhs_is_temporal=False,
        )


def test_generate_rejects_unknown_task_type(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="does not support"):
        generate_train_table(
            request(tmp_path, task_type="unsupported"),
            runtime=SimpleNamespace(),
        )


def test_ingest_train_table_publishes_expected_extension(tmp_path) -> None:
    from kumo_core.training.table import ingest_train_table

    index = tmp_path / "index"
    staging = tmp_path / "staging"
    extension = index / "train_tables" / "tt-1"
    index.mkdir()
    staging.mkdir()

    class Runtime:
        @staticmethod
        def execute_tt_ingest(**kwargs):
            assert kwargs["staging_dir"] == str(staging)
            extension.mkdir(parents=True)
            (extension / "manifest.json").write_text("{}")
            (extension / "diskgraph_manifest.json").write_text("{}")
            return {"tt_extension_path": str(extension)}

    result = ingest_train_table(
        staging_dir=str(staging),
        index_path=str(index),
        tt_id="tt-1",
        runtime=Runtime,
    )

    assert result.extension_path == str(extension.resolve())


def test_split_repr_resolves_constructor_defaults_and_derived_values():
    config, family = split_config_from_repr("DateOffsetSplit([-2, 0])")

    assert family == "DateOffsetSplit"
    assert config == {
        "offsets": [-2, 0],
        # `timeframe` is derived from the offsets by the constructor, and the
        # rest are its declared defaults. None of them is typed by the caller.
        "timeframe": 2,
        "offset_unit": "days",
        "unit": "days",
        "anchor_start_offset": 0,
        "flexible_boundaries": True,
    }


def test_split_repr_reads_the_quote_stripped_platform_repr():
    # `TimeRangeSplit.__repr__` strips quotes, so `unit` arrives as a bare
    # name and `timeframe` as a bare None.
    config, family = split_config_from_repr(
        "TimeRangeSplit([[1704067200000000000, 1706745600000000000], "
        "[1706745600000000000, 1709251200000000000]], unit=days, "
        "timeframe=None, anchor_start_offset=0, timeframe_step=None)")

    assert family == "TimeRangeSplit"
    assert config["unit"] == "days"
    # A bare integer boundary is a pandas timestamp, so nanoseconds, exactly
    # as the platform constructor reads it. The split config takes seconds.
    assert config["ranges"] == [[1704067200, 1706745600],
                                [1706745600, 1709251200]]
    assert "timeframe" not in config
    assert "timeframe_step" not in config


def test_manual_split_repr_carries_a_column_split_through_a_plan():
    # The only family that splits on a column the table already carries, and
    # so the only way a plan can express a supplied table's own split. The
    # values are strings because `ColumnDeciderManual` compares them as
    # strings (`split_decider.rs:1715`).
    config, family = split_config_from_repr(
        "ManualSplit(col='SPLIT', train_value=0, val_value=1, test_value=2)")

    assert family == "ManualSplit"
    assert config == {
        "split_column": "SPLIT",
        "manual_train_values": ["0"],
        "manual_val_values": ["1"],
        "manual_test_values": ["2"],
    }

    spec = build_split_spec(config, family, FakeSplitSpec)
    assert spec.kind == "ManualSplit"
    # No output name is carried, so DiskGraph names the projection after the
    # split column (`ttgen_resolve.rs:689`) and finds it by that name in the
    # customer's own parquet.
    assert spec.split_column == "SPLIT"
    assert spec.is_temporal is False


def test_manual_split_takes_several_values_per_stage():
    config, _ = split_config_from_repr(
        "ManualSplit(col='fold', train_value=[0, 1], val_value=2, "
        "test_value=3)")

    assert config["manual_train_values"] == ["0", "1"]
    assert config["manual_val_values"] == ["2"]


def test_split_repr_accepts_keyword_and_positional_forms():
    keyword, _ = split_config_from_repr("RandomSplit(ratio=[0.8, 0.2], seed=3)")
    positional, _ = split_config_from_repr("RandomSplit([0.8, 0.2], 3)")

    assert keyword == positional == {"ratio": [0.8, 0.2], "rnd_seed": 3}


def test_split_repr_resolves_date_literal_range_boundaries():
    config, _ = split_config_from_repr(
        "TimeRangeSplit(ranges=[['2024-01-01', '2024-02-01'], "
        "['2024-02-01', '2024-03-01']])")

    assert config["ranges"] == [[1_704_067_200, 1_706_745_600],
                                [1_706_745_600, 1_709_251_200]]


def test_split_repr_output_is_accepted_by_the_split_contract():
    config, family = split_config_from_repr("DateOffsetSplit([-2, 0])")
    spec = build_split_spec(config, family, FakeSplitSpec)

    assert spec.kind == "DateOffsetSplit"
    assert spec.is_temporal is True
    assert spec.offsets == [-2, 0]
    assert spec.timeframe == 2
    assert spec.flexible_boundaries is True


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("TimeSplit(fractions=[0.8, 0.1, 0.1])", "not supported"),
        ("RandomSplit", "must call one of"),
        ("RandomSplit([0.8], nope=1)", "does not match its constructor"),
        ("TimeRangeSplit(ranges=[['not-a-date', '2024-02-01'], "
         "['2024-02-01', '2024-03-01']])", "unable to parse"),
        ("TimeRangeSplit(ranges=[[0, 1]])",
         "At least two training splits are required"),
        ("TimeRangeSplit(ranges=[[0, 2], [1, 3]])", "not overlap"),
        ("DateOffsetSplit(offsets=[])", "at least one element"),
        ("DateOffsetSplit(offsets=[1, 2])", "need to be negative"),
    ),
)
def test_split_repr_rejects_what_portable_training_cannot_run(value, message):
    with pytest.raises(ValueError, match=message):
        split_config_from_repr(value)


def test_every_portable_split_family_is_a_valid_model_plan_split():
    # The other drift direction: a family portable training claims to support
    # must be one a model plan will actually accept into its `split` field.
    for family in PLAN_SPLITS:
        descriptor = TrainingTableGenerationPlan.get_split(f"{family}()")
        assert descriptor is not None and descriptor.name == family


@pytest.mark.parametrize(
    ("family", "args", "expected"),
    (
        ("RandomSplit", ([0.8, 0.2], ), 2),
        ("RandomSplit", ([0.8, 0.2, 0.0], ), 3),
        ("RandomLinkSplit", ([0.8, 0.2], ), 2),
        ("RandomLinkSplit", ([0.8, 0.2, 0.0], ), 3),
        ("TemporalSplit", ([0.8, 0.2], ), 2),
        ("TemporalSplit", ([0.8, 0.1, 0.1], ), 3),
        ("ApproxDateOffsetSplit", ([0.8, 0.2], ), 2),
        ("ApproxDateOffsetSplit", ([0.8, 0.1, 0.1], ), 3),
        ("DateOffsetSplit", ([-2], ), 2),
        ("DateOffsetSplit", ([-2, 0], ), 3),
        ("TimeRangeSplit", ([[0, 1], [1, 2]], ), 2),
        ("TimeRangeSplit", ([[0, 1], [1, 2], [2, 3]], ), 3),
        ("ManualSplit", ("SPLIT", ), 2),
        ("ManualSplit", ("SPLIT", 0, 1, 2), 3),
    ),
)
def test_plan_split_reports_its_configured_stage_count(
    family: str,
    args: tuple,
    expected: int,
) -> None:
    split = PLAN_SPLITS[family](*args)

    # A configured third ratio still counts as a stage when its fraction is
    # zero. Stage count does not promise that materialization produces rows.
    assert split.num_splits == expected


def _hosted_get_time_col(task_type, query_type, is_array, entity_has_time):
    if (task_type == TaskType.STATIC_LINK_PREDICTION and not is_array):
        return TrainTableTime.PSEUDO
    if query_type == QueryType.TEMPORAL or entity_has_time:
        return TrainTableTime.REAL
    return TrainTableTime.NONE


@pytest.mark.parametrize(
    ("task_type", "query_type", "is_array", "entity_has_time"),
    itertools.product(TaskType, QueryType, (False, True), (False, True)),
)
def test_train_table_time_is_the_hosted_rule(task_type, query_type, is_array,
                                             entity_has_time):
    assert train_table_time(
        task_type,
        query_type,
        list_distinct_target_is_array=is_array,
        entity_table_has_time_col=entity_has_time,
    ) == _hosted_get_time_col(task_type, query_type, is_array,
                              entity_has_time)


@pytest.mark.parametrize(
    ("task_type", "query_type", "is_array", "entity_has_time", "expected"),
    (
        (TaskType.STATIC_LINK_PREDICTION, QueryType.STATIC, False, True,
         TrainTableTime.PSEUDO),
        (TaskType.STATIC_LINK_PREDICTION, QueryType.STATIC, True, False,
         TrainTableTime.NONE),
        (TaskType.STATIC_LINK_PREDICTION, QueryType.STATIC, True, True,
         TrainTableTime.REAL),
        (TaskType.TEMPORAL_LINK_PREDICTION, QueryType.TEMPORAL, True, False,
         TrainTableTime.REAL),
        (TaskType.BINARY_CLASSIFICATION, QueryType.STATIC, False, True,
         TrainTableTime.REAL),
        (TaskType.BINARY_CLASSIFICATION, QueryType.STATIC, False, False,
         TrainTableTime.NONE),
        (TaskType.REGRESSION, QueryType.TEMPORAL, False, False,
         TrainTableTime.REAL),
    ),
)
def test_train_table_time_branches(task_type, query_type, is_array,
                                   entity_has_time, expected):
    assert train_table_time(
        task_type,
        query_type,
        list_distinct_target_is_array=is_array,
        entity_table_has_time_col=entity_has_time,
    ) == expected


@pytest.mark.parametrize('target_stype', ['categorical', 'ID', 'boolean'])
def test_rejected_custom_labels_preserve_customer_parquet(
        tmp_path, target_stype):
    parquet = tmp_path / 'custom.parquet'
    parquet.write_bytes(b'customer-owned-parquet')
    staging = tmp_path / 'custom.parquet.artifacts'

    class Runtime:
        SplitSpec = FakeSplitSpec

        @staticmethod
        def build_train_table_artifacts_for_existing_table(**kwargs):
            write_valid_class_stats(staging)
            stats_path = staging / 'train_table_col_stats.json'
            stats = json.loads(stats_path.read_text())
            stats['stats']['TARGET1__kumo.categorical__CATEGORY_COUNTS__False'][
                'value'] = [[False, 800]]
            stats_path.write_text(json.dumps(stats))
            return {'staging_dir': str(staging),
                    'tt_parquet_path': str(staging / 'tt.parquet')}

    with pytest.raises(GenerationValidationError, match='Too few classes'):
        generate_custom_train_table(
            request(tmp_path, target_stype=target_stype),
            PreparedCustomTrainingTable(str(parquet)), runtime=Runtime)
    assert parquet.read_bytes() == b'customer-owned-parquet'
    assert (staging / 'train_table_col_stats.json').exists()
