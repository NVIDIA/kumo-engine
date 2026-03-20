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
from kumoapi.typing import AggregationType, BoolOp, RelOp, TimeUnit
from diskgraph import DiskGraphPlanCompiler, SplitSpec

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
        lag_timesteps=30,
        task_type="binary_classification")

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


def test_compiler_accepts_split_spec_object():
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start=-30, end=0, unit=TimeUnit.DAYS)
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
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification",
        split=SplitSpec(ratio=[0.8, 0.1, 0.1], kind="FixedSplit"),
    )
    plan = json.loads(plan_json)
    assert plan["split"]["ratio"] == [0.8, 0.1, 0.1]
    assert plan["split"]["kind"] == "FixedSplit"


def test_compiler_accepts_legacy_split_dict_payload():
    target_ast = Aggregation(
        target=Column(fqn="transactions.amount"),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start=-30, end=0, unit=TimeUnit.DAYS)
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
        timestamps=[1609459200000000],
        forecast_length=1,
        lag_timesteps=30,
        task_type="binary_classification",
        split={"split_kind": "FixedSplit", "ratio": [0.8, 0.1, 0.1]},
    )
    plan = json.loads(plan_json)
    assert plan["split"]["kind"] == "FixedSplit"
    assert plan["split"]["ratio"] == [0.8, 0.1, 0.1]


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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification")
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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification")
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
        lag_timesteps=30,
        task_type="binary_classification")

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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification")
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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification")
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
    plan_json = compiler.compile(query=query, timestamps=[], forecast_length=1, lag_timesteps=None, task_type="binary_classification")
    plan = json.loads(plan_json)

    stage = plan["stages"][0]
    ops = stage["ops"]
    last_op = ops[-1]
    assert "logical_op" in last_op, f"Expected logical_op, got {list(last_op.keys())}"
    lop = last_op["logical_op"]
    assert lop["op"] == "or"
    assert len(lop["operands"]) == 2


def test_timestamp_normalization():
    from diskgraph.compiler import DiskGraphPlanCompiler
    from diskgraph import TimeUnit
    compiler = DiskGraphPlanCompiler()

    # 1. Micros to Seconds (standard Kumo case)
    us_ts = [1700000000000000] # Nov 2023 in micros
    norm_s = compiler._normalize_timestamps(us_ts, TimeUnit.Seconds)
    assert norm_s[0] == 1700000000

    # 2. Seconds to Seconds (No change)
    s_ts = [1700000000]
    norm_s = compiler._normalize_timestamps(s_ts, TimeUnit.Seconds)
    assert norm_s[0] == 1700000000

    # 3. Nanos to Seconds
    ns_ts = [1700000000000000000]
    norm_s = compiler._normalize_timestamps(ns_ts, TimeUnit.Seconds)
    assert norm_s[0] == 1700000000

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
    # is_prediction=True, task_type="binary_classification", timestamps=[] triggers static node prediction IS NULL injection
    plan_json = compiler.compile(query=query, timestamps=[], is_prediction=True, task_type="binary_classification")
    plan = json.loads(plan_json)

    assert "node" in plan["prediction_type"]

    # We should have the target traversed to get the alias, and a stage filter applied
    # Also, we injected a dummy column for the entity_id_target
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
        is_prediction=True, task_type="link_prediction",
        list_distinct_target_is_array=False # Legacy HStack mode
    )
    plan = json.loads(plan_json)

    assert "link" in plan["prediction_type"]
    assert plan["entity"]["output_time_col_name"] == "PSEUDO_TIMESTAMP"
    assert plan["entity"]["default_time_value"] == 100_000_000_000_000

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
        is_prediction=True, task_type="link_prediction",
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
        task_type="binary_classification",
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
        is_prediction=True, task_type="binary_classification",
        lag_offsets_secs=[-86400, -172800],
        yoy_offsets_secs=[-31536000]
    )
    plan = json.loads(plan_json)

    assert plan["prediction_type"] == "forecasting"

    # Should find two Autoregression ops in the final stage
    stage = plan["stages"][0]
    auto_ops = [op["autoregression"] for op in stage["ops"] if "autoregression" in op]

    assert len(auto_ops) == 2
    assert auto_ops[0]["alias"] == "AUTOREGRESSION_COL"
    assert auto_ops[0]["lag_offsets"] == [-86400, -172800]

    assert auto_ops[1]["alias"] == "YOY_COL"
    assert auto_ops[1]["lag_offsets"] == [-31536000]
