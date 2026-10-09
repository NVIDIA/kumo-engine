# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Metrics collection is configured by the following environment variables:
- KUMO_PUSH_GATEWAY_ADDRESS: The address of the push gateway.
    Default: localhost:9091
- KUMO_ENABLE_METRICS_COLLECTION: Whether to push the collected metrics to
    the push gateway. Default: False.
"""
import os
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    push_to_gateway,
)
from tabulate import tabulate

from io_utils import logging
from kumo.metrics.typing import Metrics, MetricType
from kumo.utils.singleton import Singleton

DEFAULT_METRIC_LABEL = "_DEFAULT_METRIC_LABEL"
DEFAULT_PUSH_GATEWAY_ADDRESS = "localhost:9091"

logger = logging.getLogger(__name__)
MetricValues = Dict[str, List[Any]]


# TODO: Add more functionality like clearing metrics etc.
class MetricsClient(metaclass=Singleton):
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.job_name = "kumo"
        self.push_gateway_address = os.environ.get(
            "KUMO_PUSH_GATEWAY_ADDRESS", DEFAULT_PUSH_GATEWAY_ADDRESS)
        # How frequently to push metrics to the push_gateway.
        # Prometheus currently scrapes the metrics from the push_gateway every
        # ~10 seconds. We don't gain any benefit from pushing metrics faster
        # than this time period. Hence, we set the push interval to be
        # 5 seconds to maintain freshness.
        self.push_interval_seconds = 5

        # Use this registry for all the main-process metrics. Use this in case
        # of uncertain of subprocess or main-process.
        self.registry = CollectorRegistry()
        # Use this registry for all the subprocess metrics - this is typically
        # used for all the metrics called in DataLoaders.
        self.subprocess_registry = CollectorRegistry()
        # Define metrics which should be reported to Grafana.
        # Follow https://prometheus.io/docs/practices/naming/ for best
        # practices on naming metrics and labels.
        self._metrics: Dict[Metrics, MetricType] = {
            # Graph-level metrics #############################################

            # Table-level metrics #############################################
            Metrics.TABLE_MATERIALIZATION_TIME:
            Summary("table_materialization_time_seconds",
                    "Time spent materializing a table",
                    labelnames=["table_name"], registry=self.registry),
            Metrics.TABLE_ENCODING_TIME:
            Summary("table_encoding_time_seconds",
                    "Time spent encoding data of thea table",
                    labelnames=["table_name"], registry=self.registry),
            Metrics.TABLE_READ_TIME:
            Summary("table_read_time_seconds",
                    "Time spent reading a table using connector",
                    labelnames=["table_name"], registry=self.registry),

            # Column-level metrics ############################################
            Metrics.COLUMN_ENCODING_TIME:
            Summary("column_encoding_time_seconds",
                    "Time spent encoding a column", labelnames=["column_name"],
                    registry=self.registry),

            # graph-engine prom names (referenced by Grafana materialization_metrics) #
            Metrics.GRAPH_ENGINE_DF_TO_SST_TIME:
            Summary("graph_engine_df_to_sst_time_seconds",
                    "Time spent converting a dataframe to an SST file",
                    labelnames=["table_name"], registry=self.registry),
            Metrics.GRAPH_ENGINE_SST_LOAD_TIME:
            Summary("graph_engine_sst_load_time_seconds",
                    "Time spent loading an SST file into the graph engine",
                    labelnames=["table_name"], registry=self.registry),
            Metrics.FEATURE_STORE_GET_TIME:
            Summary("feature_store_get_time_seconds",
                    "Time spent on getting features from feature store",
                    registry=self.subprocess_registry),
            Metrics.FEATURE_STORE_BATCH_SIZE:
            Gauge("feature_store_batch_size",
                  "Number of nodes in a FeatureStore batch",
                  registry=self.subprocess_registry),
            Metrics.FEATURE_STORE_BATCH_ENCODING_TIME:
            Summary("feature_store_batch_encoding_time_seconds",
                    "Time spent on encoding a batch of features",
                    registry=self.subprocess_registry),
        }

        self.start_push_gateway_thread(is_subprocess=False)

    def push_to_gateway(self, registry: CollectorRegistry) -> None:
        r"""Push metrics to the push_gateway."""
        # Use the current process id as the grouping_key when pushing to
        # gateway. Each push to gateway will override the metric values of the
        # same name and grouping_key.
        # This is a workaround since Prometheus client in Python doesn't
        # support multiprocess mode with a push_gateway.
        # TODO: Update this once we have a better solution.
        try:
            push_to_gateway(self.push_gateway_address, job=self.job_name,
                            registry=registry,
                            grouping_key={"pid": os.getpid()})
        except URLError:
            logger.warning(f"Failed to push metrics to push_gateway: "
                           f"{self.push_gateway_address}. Check if the "
                           f"push_gateway is up and running.")

    def start_push_gateway_thread(self, is_subprocess: bool) -> None:
        # If metrics collection is not enabled, don't start any thread.
        push_enabled = os.environ.get("KUMO_ENABLE_METRICS_COLLECTION", False)
        if not push_enabled:
            return

        def _push_to_gateway(registry: CollectorRegistry):
            logger.info("Starting pushgateway background thread for pid: %d",
                        self.pid)
            while True:
                self.push_to_gateway(registry)
                time.sleep(self.push_interval_seconds)

        registry = self.subprocess_registry if is_subprocess else self.registry
        self.push_thread = threading.Thread(target=_push_to_gateway,
                                            args=(registry, ))
        self.push_thread.daemon = True
        self.push_thread.start()

    def check_push_gateway_thread(self) -> None:
        r"""
        If the push_gateway_thread is not running, start it.
        This should start a new thread only when called from a subprocess.
        """
        pid = os.getpid()
        # When a process is forked, Python(Linux) uses CopyOnWrite to replicate
        # (only) the execution context of the parent process. This means, the
        # constructor doesn't get called again and there won't be any
        # background thread running in the forked process (as per Linux spec).
        # Hence, we start the background thread here and set the
        # self.pid to the forked process's pid to avoid starting more
        # background threads (idempotent).
        if self.pid != pid:
            self.pid = pid
            self.start_push_gateway_thread(is_subprocess=True)

    def get_metric(self, name: Metrics,
                   labels: Optional[Dict[str, str]] = None) -> MetricType:
        r"""
        Get a metric by name and labels.
        Args:
            name(Metrics): Metric name. E.g. Metrics.TABLE_READ_TIME
            labels(Dict[str, str], Optional): Dictionary of label name to
                label value. E.g. {"table_name": "table1"}
        """
        metric = self._metrics.get(name, None)
        labels = labels or {}
        if metric is None:
            raise KeyError("Metric %s is not defined in MetricsClient",
                           name.value)
        if len(labels):
            metric = metric.labels(**labels)
        return metric

    def set_metric_value(self, name: Metrics, value: Any,
                         labels: Optional[Dict[str, str]] = None):
        metric = self.get_metric(name, labels)
        self._set_metric_value(metric, value)

    def _set_metric_value(self, metric: MetricType, value: Any):
        # Call this to ensure that a push_gateway_thread is running.
        self.check_push_gateway_thread()
        if isinstance(metric, Summary) or isinstance(metric, Histogram):
            metric.observe(value)
        elif isinstance(metric, Counter):
            metric.inc(value)
        elif isinstance(metric, Gauge):
            metric.set(value)
        else:
            raise ValueError("Unsupported metric type %s", type(metric))

    # Metric report utils ####################################################
    def print_report(self):
        for metric_name in Metrics:
            metric = self.get_metric(metric_name)
            MetricsClient._print_metric_report(metric)

    @staticmethod
    def _get_summary_metric_values(metric: Summary) -> MetricValues:
        r"""
        For a metric with labels, will return a dictionary
        {label: (sum, count, avg)}
        For a metric without labels, will return
        {`DEFAULT_METRIC_LABEL`: (sum, count, avg)}.
        """
        if len(metric._labelnames) != len(metric._labelvalues):
            return {
                ",".join(label):
                MetricsClient._get_summary_metric_values(metric).get(
                    DEFAULT_METRIC_LABEL)
                for label, metric in metric._metrics.items()
            }
        else:
            # Don't return anything if metric._count == 0
            if metric._count.get() == 0:
                return {DEFAULT_METRIC_LABEL: (0, 0, 0)}
            return {
                DEFAULT_METRIC_LABEL: (
                    metric._sum.get(),
                    metric._count.get(),
                    float(metric._sum.get() / metric._count.get()),
                )
            }

    @staticmethod
    def get_metric_values(metric: MetricType) -> MetricValues:
        if isinstance(metric, Summary):
            return MetricsClient._get_summary_metric_values(metric)
        elif isinstance(metric, Gauge):
            return {DEFAULT_METRIC_LABEL: (metric._value.get(), )}
        else:
            raise NotImplementedError("Metric type: %s not supported yet.",
                                      type(metric))

    @staticmethod
    def _print_metric_report(metric: MetricType) -> None:
        data = MetricsClient.get_metric_values(metric)
        data = [(label, *values) for label, values in data.items()]
        print(metric._name)
        headers = ["Label"]
        if isinstance(metric, Summary):
            headers += ["Sum", "Count", "Mean"]
        elif isinstance(metric, Gauge):
            headers += ["Value"]
        else:
            raise NotImplementedError("Metric type: %s not supported yet.",
                                      type(metric))
        print(tabulate(data, headers, tablefmt="psql"))
