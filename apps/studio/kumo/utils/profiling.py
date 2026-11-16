# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import ContextDecorator
from typing import Dict, Optional

from prometheus_client import Histogram, Summary

from io_utils import logging
from kumo.metrics.metrics_client import MetricsClient
from kumo.metrics.typing import Metrics

logger = logging.getLogger(__name__)
metrics_client = MetricsClient()


class Timer(ContextDecorator):
    r"""
    Utility to facilitate timing a function, track it in a metric, and log it.
    This can be used both as a decorator for a function, or ContextManager
    for a block of code.

    Example usage - Metric tracking:
        For metric tracking, we only support the metrics listed in the MetricsClient.
        # Summary metrics without label.
        Metrics.TABLE_READ_TIME = Summary("table_read_time", "")
        @Timer(metric_name=Metrics.TABLE_READ_TIME):
        def read_table()

        # For metrics with labels.
        Metrics.TABLE_READ_TIME = Summary("table_read_time", "", labelnames=["table_name"]) # noqa: E501
        @Timer(metric_name=Metrics.TABLE_READ_TIME, metric_labels={"table_name":"table1"}): # noqa: E501
        def read_table()

        # Use Timer as ContextManager.
        class Table:
            def read_method(self):
                with Timer(metric_name=Metrics.TABLE_READ_TIME, metric_labels={"table_name":self.name}): # noqa: E501
                    # Read table.

    Example usage - Debugging only (no metric tracking):
        with Timer(logging_prefix="my_code_block"):
            code_block()

        @Timer(logging_prefix="Complicated logging_prefix")
        my_function():

        # This uses function name as logging_prefix.
        @Timer()
        my_function2():
    """
    def __init__(self, logging_prefix: Optional[str] = None,
                 metric_name: Optional[Metrics] = None,
                 metric_labels: Optional[Dict[str, str]] = None):
        r"""
        Args:
            logging_prefix (str, optional): Prefix used for logging the
                running time of the function. Only affects the name used
                in logging. If set to :obj:`None` and used as a function
                decorator, the function name is used as the logging_prefix.
                (default :obj:`None`)
            metric_name (Metrics, optional): Name of the metric in
                metrics/typing. Only `Summary` and `Histogram` objects are
                supported.
            metric_labels (Dict[str, str], optional): Dictionary of label name
                to label values to use for the Prometheus metric. This is
                mandatory for metrics which have labels.
                E.g. {"table_name": "table1"}
                Refer https://github.com/prometheus/client_python#labels
        """
        self.logging_prefix = logging_prefix
        metric_labels = metric_labels or {}
        # Validation of metric_name and metric_labels.
        if metric_name is None and len(metric_labels):
            raise ValueError(
                "Cannot set metric_labels to %s without a metric_name.",
                str(metric_labels))
        self.metric = None
        if metric_name is not None:
            self.metric = metrics_client.get_metric(metric_name,
                                                    labels=metric_labels)
            if not isinstance(self.metric, (Summary, Histogram)):
                raise NotImplementedError(
                    "Timer only suppports Summary or Histogram metrics."
                    " Got type %s.", type(self.metric))

    def __call__(self, func):
        # If used as a function decorator, use the function name as
        # logging_prefix (if not set explicitly).
        if self.logging_prefix is None and func is not None:
            self.logging_prefix = func.__name__

        return super().__call__(func)

    def __enter__(self):
        # TODO: Use something more sophisticated than perf_counter() here.
        self.start_real_time = time.perf_counter()
        self.start_process_time = time.process_time()
        return self

    def __exit__(self, *exc):
        self.end_real_time = time.perf_counter()
        self.end_process_time = time.process_time()
        self.elapsed_real_time = self.end_real_time - self.start_real_time
        self.elapsed_process_time = (self.end_process_time -
                                     self.start_process_time)

        # Record the running time in the metric.
        if self.metric is not None:
            metrics_client._set_metric_value(self.metric,
                                             self.elapsed_real_time)
        # Only log if there is a logging_prefix
        if self.logging_prefix:
            logger.debug(
                f"{self.logging_prefix} took {self.elapsed_real_time:.6f}s in"
                f" real_time, {self.elapsed_process_time:.6f}s in process_time")
