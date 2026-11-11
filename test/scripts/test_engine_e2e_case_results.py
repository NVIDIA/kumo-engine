# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import runpy
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
common = NS(**runpy.run_path(str(ROOT / "packages/kumo-engine/test/cases/common.py")))


def compare(requested, frames, keys):
    return common.check_predictions(*frames, .005, requested=requested,
                                    coordinates=keys)


@pytest.mark.parametrize("metric,val,test", [
    ("auroc", "val_auroc", "test_auroc"),
    ("recall@50", "val_recall@50", "test_recall@50"),
])
def test_required_metrics(metric, val, test):
    metrics = {val: .6, test: .7, "train_loss": .2}
    common.check_metrics(metrics, [metric])
    for broken in ({}, {val: .6, "test_loss": .2}, {test: .7},
                   metrics | {"train_loss": float("nan")}):
        with pytest.raises(ValueError):
            common.check_metrics(broken, [metric])
    with pytest.raises(ValueError, match="reference metrics"):
        common.check_metrics(metrics, [metric, "mae"], "reference")


def test_requested_population_and_counts():
    requested = pd.DataFrame({"ENTITY": range(20), "ANCHOR_TIMESTAMP":
                              [pd.Timestamp("2025-01-01")] * 20})
    keys = list(requested)
    frame = requested.assign(True_PROB=.5)
    counts = dict(requested_rows=20, scored_rows=20, skipped_rows=0)
    assert compare(requested, [frame, frame.iloc[::-1], NS(frame=frame)], keys)["reload"] == counts
    for broken in (frame.iloc[:19], frame.assign(ENTITY=range(1, 21))):
        with pytest.raises(ValueError, match="reload prediction population differs"):
            compare(requested, [frame, frame, broken], keys)
    common.check_population(counts, 20)
    with pytest.raises(ValueError):
        common.check_population(dict(requested_rows=20, scored_rows=19, skipped_rows=1), 20)
    static = pd.DataFrame({"ENTITY": [1]})
    topk = pd.DataFrame({"ENTITY": [1, 1], "CLASS": ["a", "b"], "SCORE": [.8, .2]})
    compare(static, [topk] * 3, ["ENTITY", "CLASS"])


def test_forecast_coordinates_and_timestamp_units():
    requested = pd.DataFrame({"ENTITY": [1, 1], "ANCHOR_TIMESTAMP":
                              pd.to_datetime(["2025-01-01", "2025-01-08"]).as_unit("ns")})
    frame = requested.loc[requested.index.repeat(2)].reset_index(drop=True)
    frame = frame.assign(signup_time=pd.to_datetime([
        "2025-01-08", "2025-01-15", "2025-01-15", "2025-01-22"]), TARGET_PRED=.5)
    keys = ["ENTITY", "ANCHOR_TIMESTAMP", "signup_time"]
    microseconds = frame.assign(**{key: frame[key].dt.as_unit("us") for key in keys[1:]})
    compare(requested, [microseconds, microseconds.iloc[::-1], microseconds], keys)
    changed = frame.copy()
    changed["signup_time"] = changed["signup_time"].dt.as_unit("ns")
    changed.loc[0, "signup_time"] += pd.Timedelta(nanoseconds=1)
    with pytest.raises(ValueError):
        compare(requested, [frame, changed, frame], keys)
    for broken in (frame.iloc[:2], pd.concat([frame, frame.iloc[:1]]),
                   frame.drop(columns="signup_time"), frame.assign(TARGET_PRED=float("inf"))):
        with pytest.raises(ValueError):
            compare(requested, [broken] * 3, keys)


def test_prediction_coordinates(monkeypatch):
    anchored = ["ENTITY", "ANCHOR_TIMESTAMP"]
    for task, keys in [("binary", anchored), ("temporal_link", anchored + ["CLASS"]),
                       ("static_link", ["ENTITY", "CLASS"])]:
        assert common.prediction_coordinates(task, None, None) == keys
    module = ModuleType("kumo_core.pquery")
    module.normalize_graph = lambda config: NS(table=lambda name: NS(time_column=config[name]))
    module.parse_and_validate_query = lambda query, graph: NS(entity_table="users")
    monkeypatch.setitem(sys.modules, "kumo_core.pquery", module)
    for clock, expected in [("signup_time", "signup_time"), (None, "TIMESTAMP")]:
        snapshot = NS(graph_config={"orders": "order_time", "users": clock})
        assert common.prediction_coordinates("forecasting", NS(query="query"), snapshot) == [
            "ENTITY", "ANCHOR_TIMESTAMP", expected]
