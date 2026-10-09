# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for rest/utils/ui_to_sdk_utils.py."""
import ast
import dataclasses
import sys
import types

import pytest
from kumo_core.types import TaskType
from kumoapi.jobs import (
    ADLSPredictionOutput,
    GCSPredictionOutput,
    PredictionArtifactType,
    S3PredictionOutput,
)
from kumoapi.model_plan import HopConfig, NumNeighborsConfig


def _load_ui_to_sdk_utils():
    """Load ui_to_sdk_utils symbols while mocking heavy deps.

    The target module (rest.utils.ui_to_sdk_utils) has deep transitive imports
    (io_utils, service_framework, kumo.connector, etc.) that require the full
    Kumo environment. We mock them out since the function under test only uses
    stdlib ``dataclasses`` and ``getattr``.
    """
    heavy_modules = [
        "io_utils",
        "io_utils.logging",
        "rest.config",
        "rest.config.data_backend",
        "rest.config.jobs",
        "rest.config.table_v2",
    ]
    saved = {}
    sentinels = {}
    for mod_name in heavy_modules:
        saved[mod_name] = sys.modules.get(mod_name)
        sentinels[mod_name] = types.ModuleType(mod_name)
        sys.modules[mod_name] = sentinels[mod_name]

    # io_utils.logging needs getLogger
    sentinels["io_utils.logging"].getLogger = lambda name: __import__(
        "logging").getLogger(name)

    class FakeConnectorSpec:
        pass

    class FakeDataSourceType:
        S3 = "S3"
        GCS = "GCS"
        ADLS = "ADLS"
        SNOWFLAKE = "SNOWFLAKE"
        DATABRICKS = "DATABRICKS"
        BIGQUERY = "BIGQUERY"

    sentinels["rest.config.data_backend"].ConnectorSpec = FakeConnectorSpec
    sentinels["rest.config.jobs"].PredictionOutputConfig = object
    sentinels["rest.config.table_v2"].DataSourceType = FakeDataSourceType
    sentinels["rest.config.table_v2"].TableReferenceMetadata = object

    try:
        # Force reimport of the target module
        sys.modules.pop("rest.utils.ui_to_sdk_utils", None)
        from rest.utils.ui_to_sdk_utils import (
            bp_job_ouput_conf,
            generate_conn_graph_or_table_sdk_script,
            generate_header,
            generate_unsupported_diskgraph_forecasting_bp_sdk_script,
            is_diskgraph_forecasting_batch_prediction_sdk_unsupported,
            recursively_convert_dataclass_to_dict,
        )
        return (
            recursively_convert_dataclass_to_dict,
            generate_unsupported_diskgraph_forecasting_bp_sdk_script,
            is_diskgraph_forecasting_batch_prediction_sdk_unsupported,
            generate_header,
            generate_conn_graph_or_table_sdk_script,
            bp_job_ouput_conf,
        )
    finally:
        for mod_name in heavy_modules:
            if saved[mod_name] is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = saved[mod_name]


(
    recursively_convert_dataclass_to_dict,
    generate_unsupported_diskgraph_forecasting_bp_sdk_script,
    is_diskgraph_forecasting_batch_prediction_sdk_unsupported,
    generate_header,
    generate_conn_graph_or_table_sdk_script,
    bp_job_ouput_conf,
) = _load_ui_to_sdk_utils()


def test_generated_header_initializes_exact_project():
    project_id = "project_0123456789abcdef0123456789abcdef"
    script = generate_header(project_id)

    tree = ast.parse(script)
    init_call = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "init")
    project_keyword = next(keyword for keyword in init_call.keywords
                           if keyword.arg == "project")
    assert ast.literal_eval(project_keyword.value) == project_id


def test_generated_connectors_are_safe_named_lookups_for_all_six_types():
    connector_types = {
        "S3": "S3Connector",
        "GCS": "GCSConnector",
        "ADLS": "ADLSConnector",
        "SNOWFLAKE": "SnowflakeConnector",
        "DATABRICKS": "DatabricksConnector",
        "BIGQUERY": "BigQueryConnector",
    }
    configs = [{
        "connector_id": f"connector_{index}",
        "lookup_name": f"Shared connector {index}",
        "data_source_type": connector_type,
    } for index, connector_type in enumerate(connector_types)]

    script = generate_conn_graph_or_table_sdk_script(
        configs,
        project_id="project_0123456789abcdef0123456789abcdef",
    )

    compile(script, "<generated-sdk>", "exec")
    assert "root_dir=" not in script
    for index, connector_class in enumerate(connector_types.values()):
        assert (
            f"connector_{index}_connector = kumo.{connector_class}.get_by_name("
            f"\"Shared connector {index}\")" in script)


def test_generated_project_upload_uses_project_scoped_file_connector():
    connector_id = "connector_0123456789abcdef0123456789abcdef"
    table_name = "uploaded_orders"
    table_config = {
        "connector_id": connector_id,
        "table_name": table_name,
        "file_type": "PARQUET",
        "cols": [],
        "pkey": None,
        "time_col": None,
        "end_time_col": None,
    }

    script = generate_conn_graph_or_table_sdk_script(
        [{
            "connector_id": connector_id,
            "data_source_type": "S3",
            "is_project_upload": True,
        }],
        [
            types.SimpleNamespace(
                source_table_name=table_name,
                table_config=table_config,
            )
        ],
        project_id="project_0123456789abcdef0123456789abcdef",
    )

    compile(script, "<generated-upload-sdk>", "exec")
    assert "kumo.FileUploadConnector(file_type=" in script
    assert 'uploaded_orders_config["file_type"].lower()' in script
    assert "S3Connector.get_by_name" not in script
    assert "root_dir=" not in script


def test_file_prediction_output_uses_persisted_connector_reference():
    connector_id = "connector_0123456789abcdef0123456789abcdef"
    output_config, connector_config = bp_job_ouput_conf([
        S3PredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            file_path="predictions/results.parquet",
            connector_id=connector_id,
        )
    ])

    assert connector_config == {
        "connector_id": connector_id,
        "data_source_type": "S3",
    }
    assert output_config["output_connector"] == (
        f"\\{connector_id}_connector\\")
    assert output_config["output_table_name"] == "results_sdk"


@pytest.mark.parametrize(('output_type', 'uri', 'connector_type'), [
    (S3PredictionOutput, 's3://bucket/deep/results.parquet', 'S3'),
    (GCSPredictionOutput, 'gs://bucket/deep/results.parquet', 'GCS'),
    (ADLSPredictionOutput,
     'abfss://data@account.dfs.core.windows.net/deep/results.parquet', 'ADLS'),
])
def test_rootless_file_prediction_output_replay_preserves_uri_parent(
        output_type, uri, connector_type):
    connector_id = "connector_0123456789abcdef0123456789abcdef"
    output_config, connector_config = bp_job_ouput_conf([
        output_type(
            artifact_type=PredictionArtifactType.PREDICTIONS,
            file_path=uri,
            connector_id=connector_id,
        )
    ])

    assert connector_config["data_source_type"] == connector_type
    assert output_config["output_table_name"] == uri.rsplit(
        '/', maxsplit=1)[0] + '/results_sdk'


def test_prediction_sdk_replay_rejects_multiple_output_connectors():
    with pytest.raises(ValueError, match="only one output connector"):
        bp_job_ouput_conf([
            S3PredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS,
                file_path="predictions/results.parquet",
                connector_id="connector_first",
            ),
            GCSPredictionOutput(
                artifact_type=PredictionArtifactType.EMBEDDINGS,
                file_path="embeddings/results.parquet",
                connector_id="connector_second",
            ),
        ])


def test_diskgraph_forecasting_bp_unsupported_detector_is_scoped():
    detector = is_diskgraph_forecasting_batch_prediction_sdk_unsupported

    assert detector(True, "forecasting")
    assert detector(True, TaskType.FORECASTING)
    assert detector(True, "regression", forecast_length=30)
    # Boundary: forecast_length=2 is the minimum multi-step horizon that
    # marks a regression plan as a forecasting batch prediction.
    assert detector(True, "regression", forecast_length=2)
    assert not detector(False, "forecasting")
    assert not detector(False, TaskType.FORECASTING)
    assert not detector(True, "regression")
    assert not detector(True, "regression", forecast_length=1)
    assert not detector(True, "regression", forecast_length=0)


def test_unsupported_diskgraph_forecasting_bp_sdk_script_exits_before_work():
    script = generate_unsupported_diskgraph_forecasting_bp_sdk_script()

    assert script.startswith("import sys")
    assert ("SKIPPED_UNSUPPORTED: DiskGraph forecasting batch prediction"
            in script)
    assert "sys.exit(0)" in script
    assert "generate_training_table" not in script
    assert "generate_prediction_table" not in script


class TestRecursivelyConvertDataclassToDict:
    """Tests for recursively_convert_dataclass_to_dict function."""
    def test_simple_dataclass(self):
        """Test conversion of a plain dataclass without pydantic extras."""
        @dataclasses.dataclass
        class SimpleConfig:
            name: str
            value: int

        result = recursively_convert_dataclass_to_dict(
            SimpleConfig(name="test", value=42))
        assert result == {"name": "test", "value": 42}

    def test_hop_config_default_only(self):
        """Test HopConfig with only a default value (no overrides)."""
        hop = HopConfig(default=8)
        result = recursively_convert_dataclass_to_dict(hop)
        assert result == {"default": 8}

    def test_hop_config_with_edge_type_overrides(self):
        """Test HopConfig preserves per-edge-type neighbor count overrides.

        This is the core bug fix: HopConfig stores per-edge-type overrides
        in __pydantic_extra__, which dataclasses.fields() does not see.
        """
        hop = HopConfig(
            default=0,
            **{
                "ip_address_dim.ip_address->trans_fact.ip_address": 12,
                "customer_dim.customer_id->trans_fact.customer_id": 5,
            },
        )
        result = recursively_convert_dataclass_to_dict(hop)
        assert result == {
            "default": 0,
            "ip_address_dim.ip_address->trans_fact.ip_address": 12,
            "customer_dim.customer_id->trans_fact.customer_id": 5,
        }

    def test_num_neighbors_config_preserves_hop_overrides(self):
        """Test that NumNeighborsConfig conversion preserves nested
        HopConfig per-edge-type overrides end-to-end."""
        num_neighbors = NumNeighborsConfig(
            hop1=HopConfig(default=8),
            hop2=HopConfig(
                default=0,
                **{
                    "ip_address_dim.ip_address->trans_fact.ip_address": 12,
                },
            ),
        )
        result = recursively_convert_dataclass_to_dict(num_neighbors)
        assert result["hop1"] == {"default": 8}
        assert result["hop2"] == {
            "default": 0,
            "ip_address_dim.ip_address->trans_fact.ip_address": 12,
        }

    def test_dataclass_in_list(self):
        """Test conversion of dataclasses inside a list."""
        hop = HopConfig(
            default=0,
            **{"edge.a->edge.b": 10},
        )
        result = recursively_convert_dataclass_to_dict([hop])
        assert result == [{"default": 0, "edge.a->edge.b": 10}]

    def test_primitives_passthrough(self):
        """Test that primitive values are returned unchanged."""
        assert recursively_convert_dataclass_to_dict(42) == 42
        assert recursively_convert_dataclass_to_dict("hello") == "hello"
        assert recursively_convert_dataclass_to_dict(None) is None

    def test_dict_passthrough(self):
        """Test that plain dicts are recursively converted."""
        result = recursively_convert_dataclass_to_dict({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_nested_dataclass(self):
        """Test conversion of nested dataclasses."""
        @dataclasses.dataclass
        class Inner:
            x: int

        @dataclasses.dataclass
        class Outer:
            inner: Inner
            label: str

        result = recursively_convert_dataclass_to_dict(
            Outer(inner=Inner(x=5), label="test"))
        assert result == {"inner": {"x": 5}, "label": "test"}
