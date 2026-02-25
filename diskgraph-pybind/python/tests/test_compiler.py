import pytest
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
from kumoapi.typing import AggregationType, RelOp, TimeUnit
from diskgraph import DiskGraphPlanCompiler

def test_compiler_basic_aggregation():
    # Construct a Mock Validated AST
    # PREDICT SUM(transactions.amount, -30, 0, days) FOR users.user_id

    # 1. Target: SUM(transactions.amount)
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"), # Valid initial state
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start=-30, end=0, unit=TimeUnit.DAYS)
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
        lag_timesteps=30
    )

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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None)
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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None)
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
        lag_timesteps=30
    )

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

def test_timestamp_normalization():
    from diskgraph.compiler import DiskGraphPlanCompiler
    compiler = DiskGraphPlanCompiler()

    # 1. Micros to Seconds (standard Kumo case)
    us_ts = [1700000000000000] # Nov 2023 in micros
    norm_s = compiler._normalize_timestamps(us_ts, "seconds")
    assert norm_s[0] == 1700000000

    # 2. Seconds to Seconds (No change)
    s_ts = [1700000000]
    norm_s = compiler._normalize_timestamps(s_ts, "seconds")
    assert norm_s[0] == 1700000000

    # 3. Nanos to Seconds
    ns_ts = [1700000000000000000]
    norm_s = compiler._normalize_timestamps(ns_ts, "seconds")
    assert norm_s[0] == 1700000000
