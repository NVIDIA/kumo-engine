import pytest
from diskgraph._c import DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

import json
from kumoapi.pquery.validated_predictive_query import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    Column,
    Condition,
    Constant,
    Filter,
    Join,
    LogicalOperation,
    DateOffsetRange,
)
from kumoapi.typing import AggregationType, BoolOp, MemberOp, RelOp, StrOp
from diskgraph import DiskgraphConfigError, DiskGraphPlanCompiler, SplitSpec, TimeUnit
from diskgraph.compiler import (
    AUTOREGRESSION_OUTPUT_ALIAS,
    FUTURE_LABELS_OUTPUT_ALIAS,
    YOY_OUTPUT_ALIAS,
)

def test_compiler_basic_aggregation():
    # Construct a Mock Validated AST
    # PREDICT SUM(transactions.amount, -30, 0, days) FOR users.user_id

    # 1. Target: SUM(transactions.amount)
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"), # Valid initial state
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start=-30, end=0, unit="days")
    )
    # Mutate to bypass pydantic __init__ type check which doesn't allow Join
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.amount"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )

    # 2. Entity: users.user_id
    entity_ast = Column(fqn="users.user_id")

    query = ValidatedPredictiveQuery(
        entity_ast=entity_ast,
        target_ast=target_ast,
    )
    # mock query type as temporal since we added time range
    query._query_type = "TEMPORAL"

    compiler = DiskGraphPlanCompiler()
    timestamps = [1609459200000000, 1609545600000000] # dummy microseconds

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Microseconds)

    plan = json.loads(plan_json)

    # Assertions
    assert plan["entity"]["table"] == "users"
    assert plan["entity"]["column"] == "user_id"

    # Normalized to seconds by default
    assert plan["temporal"]["timestamps"] == [1609459200, 1609545600]
    assert plan["temporal"]["forecast_length"] == 1
    assert plan["temporal"]["lag_timesteps"] == 30

    assert "node" in plan["prediction_type"]

    # Should have 1 stage with 1 op
    assert len(plan["stages"]) == 1
    stage = plan["stages"][0]
    assert len(stage["ops"]) == 1
    op = stage["ops"][0]["aggregate"]

    assert op["func"] == "sum"
    assert op["target_col"] == "amount"
    assert op["window"] == [-30, 0]
    assert op["window_unit"] == "days"

    # Path should have 1 hop
    path = op["path"]
    assert len(path) == 1
    assert path[0]["src_table"] == "users"
    assert path[0]["src_col"] == "user_id"
    assert path[0]["dst_table"] == "transactions"
    assert path[0]["dst_col"] == "user_id"


def test_compile_signature_prefers_split_spec_param():
    import inspect

    sig = inspect.signature(DiskGraphPlanCompiler.compile)
    assert 'split' in sig.parameters
    assert 'split_ratio' not in sig.parameters


def _make_temporal_query() -> ValidatedPredictiveQuery:
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start=-30, end=0, unit="days")
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.amount"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "TEMPORAL"
    return query


def test_compiler_accepts_split_spec_object():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        split=SplitSpec(ratio=[0.8, 0.1, 0.1], kind="FixedSplit"),
    )
    plan = json.loads(plan_json)
    assert plan["split"]["ratio"] == [0.8, 0.1, 0.1]
    assert plan["split"]["kind"] == "FixedSplit"


def test_compiler_accepts_date_offset_split_without_ratio():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="temporal_link_prediction", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        split=SplitSpec(kind="DateOffsetSplit", offsets=[-30, -7], offset_unit="days"),
    )

    plan = json.loads(plan_json)
    assert plan["split"]["kind"] == "DateOffsetSplit"
    # Offsets are converted from days to seconds by the compiler
    assert plan["split"]["offsets"] == [-30 * 86400, -7 * 86400]
    assert plan["split"]["ratio"] is None


def test_compiler_preserves_date_offset_window_semantics_and_normalizes_units():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="temporal_link_prediction", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        split=SplitSpec(
            kind="DateOffsetSplit",
            offsets=[-6, 0],
            offset_unit="h",
            unit="h",
            timeframe=2,
            anchor_start_offset=2,
        ),
    )

    split = json.loads(plan_json)["split"]
    assert split["timeframe"] == 2
    assert split["unit"] == "hours"
    assert split["offset_unit"] == "hours"
    assert split["anchor_start_offset"] == 2


def test_compiler_rejects_date_offset_calendar_units():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    with pytest.raises(DiskgraphConfigError, match="DateOffsetSplit.*months"):
        compiler.compile(
            query=query,
            timestamps=[1609459200000000],
            forecast_length=1,
            lag_timesteps=30,
            task_type="temporal_link_prediction", target_stype="categorical",
            input_timestamp_unit=TimeUnit.Microseconds,
            split=SplitSpec(
                kind="DateOffsetSplit",
                offsets=[-6, 0],
                offset_unit="days",
                unit="months",
                timeframe=2,
                anchor_start_offset=2,
            ),
        )


def test_compiler_accepts_backtesting_split_without_ratio():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        split=SplitSpec(kind="BacktestingSplit", backtesting_timestamp=1704067200),
    )

    plan = json.loads(plan_json)
    assert plan["split"]["kind"] == "BacktestingSplit"
    assert plan["split"]["backtesting_timestamp"] == 1704067200
    assert plan["split"]["ratio"] is None


def test_compiler_rejects_legacy_split_dict_payload():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    with pytest.raises(DiskgraphConfigError, match='requires split=SplitSpec'):
        compiler.compile(
            query=query,
            timestamps=[1609459200000000],
            forecast_length=1,
            lag_timesteps=30,
            task_type="binary_classification", target_stype="categorical",
            input_timestamp_unit=TimeUnit.Microseconds,
            split={"split_kind": "FixedSplit", "ratio": [0.8, 0.1, 0.1]},
        )


def test_compiler_rejects_fixed_split_without_ratio():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    with pytest.raises(DiskgraphConfigError, match='FixedSplit requires ratio'):
        compiler.compile(
            query=query,
            timestamps=[1609459200000000],
            forecast_length=1,
            lag_timesteps=30,
            task_type="binary_classification", target_stype="categorical",
            input_timestamp_unit=TimeUnit.Microseconds,
            split=SplitSpec(kind="FixedSplit"),
        )


def test_compiler_accepts_random_link_split_with_ratio():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="temporal_link_prediction", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        split=SplitSpec(kind="RandomLinkSplit", ratio=[0.8, 0.1, 0.1], rnd_seed=11),
    )
    plan = json.loads(plan_json)
    assert plan.get("split", {}).get("kind") == "RandomLinkSplit"
    assert plan.get("split", {}).get("ratio") == [0.8, 0.1, 0.1]
    assert plan.get("split", {}).get("rnd_seed") == 11


def test_compiler_with_row_filter():
    # PREDICT COUNT(transactions.transaction_id WHERE transactions.status = 'paid') FOR users.user_id

    target_ast = Aggregation(
        target=Filter(
            target=Column(fqn="transactions.transaction_id"), # valid initial state
            condition=Condition(
                target=Column(fqn="transactions.status"),
                op=RelOp.EQ,
                value=Constant.from_value("paid")
            )
        ),
        aggr=AggregationType.COUNT
    )
    # Mutate to bypass pydantic
    target_ast.target.target = Join(
        rhs_target=Column(fqn="transactions.transaction_id"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    op = plan["stages"][0]["ops"][0]["aggregate"]
    assert op["func"] == "count"
    assert op["target_col"] == "transaction_id"

    row_filter = op["row_filter"]["compare"]
    # The filter column (transactions.status) is on the destination table of the
    # aggregation, so it resolves as a local column, not a remote one.
    assert row_filter["lhs"]["column"] == "status"
    assert row_filter["op"] == "eq"
    assert row_filter["rhs"]["constant"]["str"] == "paid"


@pytest.mark.parametrize("member_op", [MemberOp.IN, MemberOp.IS_IN])
def test_compiler_with_in_row_filter(member_op):
    target_ast = Aggregation(
        target=Filter(
            target=Column(fqn="transactions.transaction_id"),
            condition=Condition(
                target=Column(fqn="transactions.status"),
                op=member_op,
                value=Constant.from_value(["paid", "pending"]),
            ),
        ),
        aggr=AggregationType.COUNT,
    )
    target_ast.target.target = Join(
        rhs_target=Column(fqn="transactions.transaction_id"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    plan = json.loads(
        DiskGraphPlanCompiler().compile(
            query=query,
            timestamps=[],
            task_type="binary_classification",
            target_stype="categorical",
            input_timestamp_unit=TimeUnit.Seconds,
        )
    )

    row_filter = plan["stages"][0]["ops"][0]["aggregate"]["row_filter"]["compare"]
    assert row_filter["op"] == "in"
    assert row_filter["rhs"]["constant"]["list"] == [
        {"str": "paid"},
        {"str": "pending"},
    ]


def test_compiler_with_starts_with_row_filter():
    target_ast = Aggregation(
        target=Filter(
            target=Column(fqn="transactions.account_type"),
            condition=Condition(
                target=Column(fqn="transactions.account_type"),
                op=StrOp.STARTS_WITH,
                value=Constant.from_value("investment"),
            ),
        ),
        aggr=AggregationType.LIST_DISTINCT,
    )
    target_ast.target.target = Join(
        rhs_target=Column(fqn="transactions.account_type"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[],
        forecast_length=1,
        lag_timesteps=None,
        task_type="multilabel_classification",
        target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
    )
    plan = json.loads(plan_json)

    row_filter = plan["stages"][0]["ops"][0]["aggregate"]["row_filter"]["compare"]
    assert row_filter["lhs"]["column"] == "account_type"
    assert row_filter["op"] == "starts_with"
    assert row_filter["rhs"]["constant"]["str"] == "investment"


@pytest.mark.parametrize(
    ("task_type", "expected_filter"),
    [
        ("multilabel_classification", None),
        ("multilabel_ranking", {"non_empty_list": {"alias": "agg_1"}}),
    ],
)
def test_compiler_encodes_task_specific_empty_list_target_admission(task_type, expected_filter):
    target_ast = Aggregation(
        target=Column(fqn="transactions.account_type"),
        aggr=AggregationType.LIST_DISTINCT,
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.account_type"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    plan = json.loads(
        DiskGraphPlanCompiler().compile(
            query=query,
            timestamps=[],
            task_type=task_type,
            target_stype="multicategorical",
            input_timestamp_unit=TimeUnit.Seconds,
        )
    )

    assert plan["stages"][0]["filter"] == expected_filter


def test_compiler_rejects_ranking_without_multicategorical_target_stype():
    query = _make_temporal_query()

    with pytest.raises(DiskgraphConfigError, match="requires target_stype='multicategorical'"):
        DiskGraphPlanCompiler().compile(
            query=query,
            timestamps=[1609459200],
            task_type="multilabel_ranking",
            target_stype="categorical",
            input_timestamp_unit=TimeUnit.Seconds,
        )


def test_compiler_stage_filter():
    # PREDICT SUM(transactions.amount) FOR users.user_id WHERE COUNT(transactions.transaction_id) > 5

    # 1. Entity Filter: COUNT(transactions.transaction_id) > 5
    agg = Aggregation(
        target=Column(fqn="transactions.transaction_id"),
        aggr=AggregationType.COUNT
    )
    agg.target = Join(
        rhs_target=Column(fqn="transactions.transaction_id"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )

    entity_ast = Filter(
        target=Column(fqn="users.user_id"),
        condition=Condition(
            target=agg,
            op=RelOp.GT,
            value=Constant.from_value(5)
        )
    )

    # 2. Target: SUM(transactions.amount)
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.amount"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )

    query = ValidatedPredictiveQuery(
        entity_ast=entity_ast,
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    # Should have 2 stages because of the filter on aggregation
    assert len(plan["stages"]) == 2

    # Stage 1: The COUNT aggregation + the filter
    stage1 = plan["stages"][0]
    assert stage1["ops"][0]["aggregate"]["func"] == "count"
    assert stage1["filter"]["compare"]["alias"].startswith("agg_")
    assert stage1["filter"]["compare"]["op"] == "gt"
    assert stage1["filter"]["compare"]["rhs"]["constant"]["int"] == 5

    # Stage 2: The final SUM target
    stage2 = plan["stages"][1]
    assert stage2["ops"][0]["aggregate"]["func"] == "sum"


def test_compiler_with_in_stage_filter():
    agg = Aggregation(
        target=Column(fqn="transactions.transaction_id"),
        aggr=AggregationType.COUNT,
    )
    agg.target = Join(
        rhs_target=Column(fqn="transactions.transaction_id"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    entity_ast = Filter(
        target=Column(fqn="users.user_id"),
        condition=Condition(
            target=agg,
            op=MemberOp.IN,
            value=Constant.from_value([1, 3]),
        ),
    )
    query = ValidatedPredictiveQuery(
        entity_ast=entity_ast,
        target_ast=Column(fqn="users.status"),
    )
    query._query_type = "STATIC"

    plan = json.loads(
        DiskGraphPlanCompiler().compile(
            query=query,
            timestamps=[],
            task_type="binary_classification",
            target_stype="categorical",
            input_timestamp_unit=TimeUnit.Seconds,
        )
    )

    stage_filter = plan["stages"][0]["filter"]["compare"]
    assert stage_filter["op"] == "in"
    assert stage_filter["rhs"]["constant"]["list"] == [{"int": 1}, {"int": 3}]


def test_compiler_with_in_target_condition():
    agg = Aggregation(
        target=Column(fqn="transactions.transaction_id"),
        aggr=AggregationType.COUNT,
    )
    agg.target = Join(
        rhs_target=Column(fqn="transactions.transaction_id"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=Condition(
            target=agg,
            op=MemberOp.IN,
            value=Constant.from_value([1, 3]),
        ),
    )
    query._query_type = "STATIC"

    plan = json.loads(
        DiskGraphPlanCompiler().compile(
            query=query,
            timestamps=[],
            task_type="binary_classification",
            target_stype="categorical",
            input_timestamp_unit=TimeUnit.Seconds,
        )
    )

    compare = plan["stages"][0]["ops"][1]["compare"]
    assert compare["op"] == "in"
    assert compare["rhs"]["list"] == [{"int": 1}, {"int": 3}]

def test_compiler_pql_e2e():
    kumopql = pytest.importorskip("kumopql", reason="kumopql not installed")
    from kumopql.parser import PQLParser

    pql = "PREDICT SUM(transactions.amount, -30, 0, days) FOR users.user_id WHERE users.country = 'US'"

    # 1. Parse raw string
    parsed = PQLParser().to_parsed_predictive_query(pql)

    # 2. Wrap in ValidatedPredictiveQuery (compiler only needs the AST structure)
    query = ValidatedPredictiveQuery(
        entity_ast=parsed.entity_ast,
        target_ast=parsed.target_ast,
        problem_type=parsed.problem_type,
        top_k=parsed.top_k
    )
    query._query_type = "TEMPORAL"

    compiler = DiskGraphPlanCompiler()
    timestamps = [1700000000000000] # dummy

    plan_json = compiler.compile(
        query=query,
        timestamps=timestamps,
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Microseconds)

    plan = json.loads(plan_json)

    # Verify entity filter was handled
    assert plan["entity"]["table"] == "users"
    # The WHERE users.country = 'US' should be an entity row filter
    # For entity table, it is a direct column filter
    assert plan["entity"]["filter"]["compare"]["lhs"]["column"] == "country"
    assert plan["entity"]["filter"]["compare"]["op"] == "eq"
    assert plan["entity"]["filter"]["compare"]["rhs"]["constant"]["str"] == "US"

    # Verify target aggregation
    stage = plan["stages"][0]
    op = stage["ops"][0]["aggregate"]
    assert op["func"] == "sum"
    assert op["target_col"] == "amount"
    assert op["window"] == [-30, 0]
    # Normalized
    assert plan["temporal"]["timestamps"] == [1700000000]

def test_compiler_not_target():
    # PREDICT NOT transactions.status='paid' FOR users.user_id
    inner_cond = Condition(
        target=Column(fqn="transactions.status"),
        op=RelOp.EQ,
        value=Constant.from_value("paid"),
    )
    inner_cond.target = Join(
        rhs_target=Column(fqn="transactions.status"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    target_ast = LogicalOperation(
        left=inner_cond,
        bool_op=BoolOp.NOT,
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    stage = plan["stages"][0]
    ops = stage["ops"]
    last_op = ops[-1]
    assert "logical_op" in last_op, f"Expected logical_op, got {list(last_op.keys())}"
    lop = last_op["logical_op"]
    assert lop["op"] == "not"
    assert len(lop["operands"]) == 1


def test_compiler_and_target():
    # PREDICT (transactions.status='paid' AND transactions.status='shipped') FOR users.user_id
    cond1 = Condition(
        target=Column(fqn="transactions.status"),
        op=RelOp.EQ,
        value=Constant.from_value("paid"),
    )
    cond1.target = Join(
        rhs_target=Column(fqn="transactions.status"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    cond2 = Condition(
        target=Column(fqn="transactions.status"),
        op=RelOp.EQ,
        value=Constant.from_value("shipped"),
    )
    cond2.target = Join(
        rhs_target=Column(fqn="transactions.status"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    target_ast = LogicalOperation(
        left=cond1,
        right=cond2,
        bool_op=BoolOp.AND,
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    stage = plan["stages"][0]
    ops = stage["ops"]
    last_op = ops[-1]
    assert "logical_op" in last_op, f"Expected logical_op, got {list(last_op.keys())}"
    lop = last_op["logical_op"]
    assert lop["op"] == "and"
    assert len(lop["operands"]) == 2


def test_compiler_or_target():
    # PREDICT (transactions.status='paid' OR transactions.status='shipped') FOR users.user_id
    cond1 = Condition(
        target=Column(fqn="transactions.status"),
        op=RelOp.EQ,
        value=Constant.from_value("paid"),
    )
    cond1.target = Join(
        rhs_target=Column(fqn="transactions.status"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    cond2 = Condition(
        target=Column(fqn="transactions.status"),
        op=RelOp.EQ,
        value=Constant.from_value("shipped"),
    )
    cond2.target = Join(
        rhs_target=Column(fqn="transactions.status"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id",
    )
    target_ast = LogicalOperation(
        left=cond1,
        right=cond2,
        bool_op=BoolOp.OR,
    )

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    stage = plan["stages"][0]
    ops = stage["ops"]
    last_op = ops[-1]
    assert "logical_op" in last_op, f"Expected logical_op, got {list(last_op.keys())}"
    lop = last_op["logical_op"]
    assert lop["op"] == "or"
    assert len(lop["operands"]) == 2


def test_compiler_static_node_prediction_is_null():
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.amount"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"

    compiler = DiskGraphPlanCompiler()
    # is_prediction=True, task_type="binary_classification", target_stype="categorical", timestamps=[] triggers static node prediction IS NULL injection
    plan_json = compiler.compile(query=query, timestamps=[], is_prediction=True, task_type="binary_classification", target_stype="categorical", input_timestamp_unit=TimeUnit.Seconds)
    plan = json.loads(plan_json)

    # Node BP emits entities only — PassThrough, no target column.
    assert plan["prediction_type"] == "pass_through"

    # The static IS-NULL restriction must still be applied as a stage filter.
    assert len(plan["stages"]) >= 1

    # Look for the has_edges op and the == False stage filter
    found_has_edges = False
    found_filter = False
    for stage in plan["stages"]:
        for op in stage.get("ops", []):
            if "has_edges" in op:
                found_has_edges = True
        if "filter" in stage and stage["filter"]:
            if "compare" in stage["filter"]:
                cmp = stage["filter"]["compare"]
                if cmp["op"] == "eq" and cmp["rhs"] == {"constant": {"bool": False}}:
                    found_filter = True

    assert found_has_edges, f"Expected an 'has_edges' operation to be injected. Plan: {json.dumps(plan, indent=2)}"
    assert found_filter, f"Expected a '== False' stage filter to be injected. Plan: {json.dumps(plan, indent=2)}"

def test_compiler_link_prediction_legacy_pseudo_time():
    target_ast = Aggregation(
        target=Column(fqn="products.product_id"),
        aggr=AggregationType.LIST_DISTINCT
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="products.product_id"),
        lhs_key="users.user_id",
        rhs_key="orders.user_id"
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"
    query.task_type = "static_link_prediction"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[],
        is_prediction=True, task_type="link_prediction", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        list_distinct_target_is_array=False # Legacy HStack mode
    )
    plan = json.loads(plan_json)

    assert "link" in plan["prediction_type"]
    assert plan["entity"]["output_time_col_name"] == "PSEUDO_TIMESTAMP"
    assert plan["entity"]["default_time_value"] == 100_000_000_000_000

def _make_static_link_prediction_query():
    target_ast = Aggregation(
        target=Column(fqn="products.product_id"),
        aggr=AggregationType.LIST_DISTINCT
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="products.product_id"),
        lhs_key="users.user_id",
        rhs_key="orders.user_id"
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "STATIC"
    query.task_type = "static_link_prediction"
    return query

def test_compiler_lp_bp_layout_controls_static_prediction_time_column():
    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=_make_static_link_prediction_query(),
        timestamps=[],
        is_prediction=True,
        task_type="link_prediction",
        target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        list_distinct_target_is_array=False,
        lp_bp_layout="separate_lhs_rhs",
        lp_prediction_mode="LHS",
    )
    plan = json.loads(plan_json)

    assert plan["prediction_type"] == "pass_through"
    assert plan["entity"]["output_time_col_name"] == "PSEUDO_TIMESTAMP"
    assert plan["entity"]["default_time_value"] == 100_000_000_000_000

def test_compiler_lp_bp_layout_can_request_hstack_with_legacy_array_flag():
    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=_make_static_link_prediction_query(),
        timestamps=[],
        is_prediction=True,
        task_type="link_prediction",
        target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        list_distinct_target_is_array=True,
        lp_bp_layout="hstack",
        lp_prediction_mode="LHS",
    )
    plan = json.loads(plan_json)

    assert plan["prediction_type"] == "pass_through"
    assert plan["entity"]["output_time_col_name"] == "PSEUDO_TIMESTAMP"
    assert plan["entity"]["default_time_value"] == 100_000_000_000_000

def test_compiler_training_table_uses_pseudo_timestamp_when_non_array_target_with_separate_lhs_rhs():
    # lp_bp_layout="separate_lhs_rhs" is a BP-only knob; TT format is governed by
    # list_distinct_target_is_array. With non-array targets the TT still needs
    # PSEUDO_TIMESTAMP regardless of BP layout.
    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=_make_static_link_prediction_query(),
        timestamps=[],
        is_prediction=False,
        task_type="link_prediction",
        target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        list_distinct_target_is_array=False,
        lp_bp_layout="separate_lhs_rhs",
    )
    plan = json.loads(plan_json)

    assert plan["entity"]["output_time_col_name"] == "PSEUDO_TIMESTAMP"
    assert plan["entity"]["default_time_value"] == 100_000_000_000_000

def test_compiler_lp_layout_rejects_invalid_values():
    compiler = DiskGraphPlanCompiler()
    with pytest.raises(DiskgraphConfigError, match="lp_bp_layout"):
        compiler.compile(
            query=_make_static_link_prediction_query(),
            timestamps=[],
            is_prediction=True,
            task_type="link_prediction",
            target_stype="categorical",
            input_timestamp_unit=TimeUnit.Seconds,
            lp_bp_layout="standalone",
        )

def test_compiler_link_prediction_rhs_pass_through():
    # Target has a filter: products.category = 'shoes'
    filtered_target = Filter(
        target=Column(fqn="products.product_id"),
        condition=Condition(
            target=Column(fqn="products.category"),
            op=RelOp.EQ,
            value=Constant.from_value("shoes")
        )
    )
    target_ast = Aggregation(
        target=filtered_target,
        aggr=AggregationType.LIST_DISTINCT
    )
    # Bypass Pydantic validation by assigning after initialization
    j = Join(
        rhs_target=Column(fqn="products.product_id"), # valid initially
        lhs_key="users.user_id",
        rhs_key="orders.user_id"
    )
    # The Filter wraps the Join in Kumo AST
    target_ast.target.target = j

    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1700000000000000],
        is_prediction=True, task_type="link_prediction", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        lp_prediction_mode='RHS', lp_rhs_entity_table="products", lp_rhs_entity_column="product_id"
    )
    plan = json.loads(plan_json)

    assert plan["prediction_type"] == "pass_through"
    # Should extract RHS entity
    assert plan["entity"]["table"] == "products"
    assert plan["entity"]["column"] == "product_id"

    # Should extract RHS filter
    assert plan["entity"]["filter"]["compare"]["lhs"]["column"] == "category"
    assert plan["entity"]["filter"]["compare"]["op"] == "eq"
    assert plan["entity"]["filter"]["compare"]["rhs"]["constant"]["str"] == "shoes"

def test_compiler_mixed_entity_filter_column_and_aggregation():
    """Entity WHERE with both a column condition and an aggregation.

    PQL: PREDICT COUNT(msgs.*, 0, 30) > 10 FOR EACH users.user_id
         WHERE users.is_verified = False AND COUNT(friends.*, -30, 0) > 3

    The column condition (is_verified = False) must emit a column-fetch op
    so the Rust engine can resolve the alias in the stage filter.
    """
    kumopql = pytest.importorskip("kumopql", reason="kumopql not installed")
    from kumopql.parser import PQLParser

    pql = (
        "PREDICT COUNT(message_requests.*, 0, 30) > 10 "
        "FOR EACH user_entities.user_id "
        "WHERE user_entities.is_verified = False "
        "AND COUNT(friend_requests.*, -30, 0) > 3"
    )
    parsed = PQLParser().to_parsed_predictive_query(pql)
    query = ValidatedPredictiveQuery(
        entity_ast=parsed.entity_ast,
        target_ast=parsed.target_ast,
        problem_type=parsed.problem_type,
        top_k=parsed.top_k,
    )
    query._query_type = "TEMPORAL"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1700000000],
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        forecast_length=1,
        entity_time_col="signup_date",
    )
    plan = json.loads(plan_json)

    # Stage 0 should have the entity filter (column + aggregation)
    stage0 = plan["stages"][0]
    ops = stage0["ops"]

    # Must have a column-fetch op for is_verified
    col_ops = [op for op in ops if "column" in op]
    assert len(col_ops) >= 1, (
        f"Expected a column-fetch op for is_verified, got ops: {ops}"
    )
    col_op = col_ops[0]["column"]
    assert col_op["column"] == "is_verified"
    col_alias = col_op["alias"]

    # Must have an aggregate op for COUNT(friend_requests.*)
    agg_ops = [op for op in ops if "aggregate" in op]
    assert len(agg_ops) >= 1
    agg_alias = agg_ops[0]["aggregate"]["alias"]

    # The stage filter must reference both aliases (not raw column names)
    filt = stage0["filter"]
    assert "and" in filt
    aliases_in_filter = [
        branch["compare"]["alias"] for branch in filt["and"]
    ]
    assert col_alias in aliases_in_filter, (
        f"Expected column alias '{col_alias}' in filter, got {aliases_in_filter}"
    )
    assert agg_alias in aliases_in_filter, (
        f"Expected agg alias '{agg_alias}' in filter, got {aliases_in_filter}"
    )


def test_compiler_forecasting():
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM
    )
    target_ast.target = Join(
        rhs_target=Column(fqn="transactions.amount"),
        lhs_key="users.user_id",
        rhs_key="transactions.user_id"
    )
    query = ValidatedPredictiveQuery(
        entity_ast=Column(fqn="users.user_id"),
        target_ast=target_ast,
    )
    query._query_type = "TEMPORAL"

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1700000000000000],
        is_prediction=True, task_type="forecasting", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
        lag_offsets_secs=[-86400, -172800],
        yoy_offsets_secs=[-31536000]
    )
    plan = json.loads(plan_json)

    assert plan["prediction_type"] == "forecasting"

    # Should find two Autoregression ops in the final stage
    stage = plan["stages"][0]
    auto_ops = [op["autoregression"] for op in stage["ops"] if "autoregression" in op]

    assert len(auto_ops) == 2
    assert auto_ops[0]["alias"] == AUTOREGRESSION_OUTPUT_ALIAS
    assert auto_ops[0]["lag_offsets"] == [-86400, -172800]

    assert auto_ops[1]["alias"] == YOY_OUTPUT_ALIAS
    assert auto_ops[1]["lag_offsets"] == [-31536000]


def test_compiler_forecasting_training_future_labels():
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    plan_json = compiler.compile(
        query=query,
        timestamps=[1700000000000000],
        is_prediction=False,
        task_type="forecasting",
        target_stype="numerical",
        input_timestamp_unit=TimeUnit.Microseconds,
        lag_offsets_secs=[-86400],
        future_offsets_secs=[0, 86400, 172800],
    )
    plan = json.loads(plan_json)
    stage = plan["stages"][0]

    future_ops = [op["future_labels"] for op in stage["ops"] if "future_labels" in op]
    assert len(future_ops) == 1
    assert future_ops[0]["alias"] == FUTURE_LABELS_OUTPUT_ALIAS
    assert future_ops[0]["future_offsets"] == [0, 86400, 172800]


def test_autoregression_lag_offset_unit_days():
    """Autoregression builder converts lag offsets from declared unit to seconds."""
    query = _make_temporal_query()

    compiler = DiskGraphPlanCompiler()
    # Pass lag offsets as seconds (-86400 = 1 day, -604800 = 7 days)
    # and verify they appear in the plan unchanged
    plan_json = compiler.compile(
        query=query,
        timestamps=[1700000000],
        is_prediction=True,
        task_type="forecasting", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
        lag_offsets_secs=[-86400, -604800],
    )
    plan = json.loads(plan_json)
    assert plan["prediction_type"] == "forecasting"
    stage = plan["stages"][0]
    auto_ops = [op["autoregression"] for op in stage["ops"] if "autoregression" in op]
    assert len(auto_ops) == 1
    # Offsets passed as seconds should appear as-is
    assert auto_ops[0]["lag_offsets"] == [-86400, -604800]


def test_autoregression_lag_offset_unit_days_via_builder():
    """PyO3 builder converts lag offsets from days to seconds."""
    from diskgraph import DiskGraphTTBuilder, EdgeRef
    builder = DiskGraphTTBuilder()
    builder.temporal([1700000000], forecast_length=1, unit="seconds")
    builder.forecasting_prediction()
    builder.for_each("users", "user_id")
    # Aggregate op to get target_op_json for autoregression
    edge = EdgeRef("users", "user_id", "transactions", "user_id")
    builder.aggregate("x", edge, "sum", target_col="amount")
    target_op_json = builder.pop_last_op_json()
    builder.autoregression("LAG", target_op_json, [-1, -7], lag_offset_unit="days")
    plan = json.loads(builder.build())
    auto_ops = [op["autoregression"] for op in plan["stages"][0]["ops"] if "autoregression" in op]
    assert len(auto_ops) == 1
    # -1 day = -86400 seconds, -7 days = -604800 seconds
    assert auto_ops[0]["lag_offsets"] == [-86400, -604800]


def test_nanosecond_timestamps_converted_to_seconds():
    """Regression: _infer_unit misclassified 15-digit nanos as micros.

    Verify that nanosecond timestamps passed with input_timestamp_unit=TimeUnit.Nanoseconds
    are correctly converted to seconds in the emitted JSON plan.
    """
    query = _make_temporal_query()
    compiler = DiskGraphPlanCompiler()

    # 1609459200 seconds = 2021-01-01T00:00:00Z
    nanos = [1609459200000000000, 1609545600000000000]
    plan_json = compiler.compile(
        query=query,
        timestamps=nanos,
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Nanoseconds,
    )
    plan = json.loads(plan_json)
    assert plan["temporal"]["timestamps"] == [1609459200, 1609545600]


def test_microsecond_timestamps_converted_to_seconds():
    """Verify microsecond timestamps are correctly converted to seconds."""
    query = _make_temporal_query()
    compiler = DiskGraphPlanCompiler()

    micros = [1609459200000000, 1609545600000000]
    plan_json = compiler.compile(
        query=query,
        timestamps=micros,
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Microseconds,
    )
    plan = json.loads(plan_json)
    assert plan["temporal"]["timestamps"] == [1609459200, 1609545600]


def test_second_timestamps_unchanged():
    """Verify second timestamps pass through unchanged."""
    query = _make_temporal_query()
    compiler = DiskGraphPlanCompiler()

    secs = [1609459200, 1609545600]
    plan_json = compiler.compile(
        query=query,
        timestamps=secs,
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification", target_stype="categorical",
        input_timestamp_unit=TimeUnit.Seconds,
    )
    plan = json.loads(plan_json)
    assert plan["temporal"]["timestamps"] == [1609459200, 1609545600]


# ─── Multicategorical pass-through tests removed ──────────────────────────────
#
# The compiler no longer propagates `column_metadata` / `target_semantic_type`
# into plan JSON; the Rust resolver reads kumo stype from the persisted
# `diskgraph_config.json` at the index root instead. Coverage for the
# end-to-end multicat dispatch lives in `ttgen_integration.rs` (which exercises
# the on-disk-config path via fixtures built through real ingest).
