# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import time

import pytest
from prometheus_client import Summary

from io_utils import fsutil, logging
from kumo.metrics import Metrics, MetricsClient
from kumo.metrics.metrics_client import DEFAULT_METRIC_LABEL
from kumo.utils.profiling import Timer

_SLEEP_TIME = 0.5
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), ".test_files")
LOG_FILE = os.path.join(TEST_OUTPUT_DIR, logging.LOGGING_FILE_NAME)


def setup_function():
    fsutil.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    logging.initialize_logging(logging.LoggingMode.DEBUG, TEST_OUTPUT_DIR,
                               log_to_console=False)


def teardown_function():
    import shutil
    shutil.rmtree(TEST_OUTPUT_DIR)


def _filter_timer_logs(records):
    return [r.message for r in records if "profiling" in r.name]


# Extract title, real_time, process_time from log message.
def _extract_entities(record: str):
    title = record.split("took")[0].strip()
    (real_time, process_time) = re.findall(r'\d+\.\d+', record)
    return (title, float(real_time), float(process_time))


@Timer()
def function_with_decorator():
    time.sleep(_SLEEP_TIME)


@Timer(logging_prefix="custom title")
def function_with_decorator_title():
    time.sleep(_SLEEP_TIME)


def test_decorator(caplog):
    function_with_decorator()
    logs = _filter_timer_logs(caplog.records)
    title, real_time, process_time = _extract_entities(logs[0])
    assert title == function_with_decorator.__name__
    assert real_time > _SLEEP_TIME
    assert process_time < _SLEEP_TIME


def test_decorator_with_title(caplog):
    function_with_decorator_title()
    logs = _filter_timer_logs(caplog.records)
    title, real_time, process_time = _extract_entities(logs[0])
    assert title == "custom title"
    assert real_time > _SLEEP_TIME
    assert process_time < _SLEEP_TIME


def test_code_block(caplog):
    with Timer(logging_prefix="code block"):
        time.sleep(_SLEEP_TIME)

    logs = _filter_timer_logs(caplog.records)
    title, real_time, process_time = _extract_entities(logs[0])
    assert title == "code block"
    assert real_time > _SLEEP_TIME
    assert process_time < _SLEEP_TIME


def test_metric_usage():
    # Metric_label without metric should throw error.
    with pytest.raises(ValueError):
        Timer(metric_labels={"table_name": "table1"})

    # Clear metric before use
    MetricsClient().get_metric(Metrics.TABLE_READ_TIME).clear()
    with Timer(metric_name=Metrics.TABLE_READ_TIME,
               metric_labels={"table_name": "table1"}):
        time.sleep(_SLEEP_TIME)

    metric: Summary = MetricsClient().get_metric(
        Metrics.TABLE_READ_TIME, labels={"table_name": "table1"})
    # Check the values as (sum, count, avg)
    values = MetricsClient.get_metric_values(metric)[DEFAULT_METRIC_LABEL]
    assert values[0] > _SLEEP_TIME
    assert values[1] == 1
    assert values[2] == values[0]
