# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.connector import get_handlers
from kumoai_studio.connector import (
    ADLSConnector,
    BigQueryConnector,
    FileUploadConnector,
    S3Connector,
)


class TestS3ConnectorHandler:
    """Test project connector lookup code generation."""
    def test_s3_connector_with_root_dir(self):
        """Test root_dir pattern for S3Connector."""
        handlers = get_handlers()
        s3_handler = handlers[S3Connector]

        mock_connector = Mock(spec=S3Connector)
        mock_connector.name = "test-connector"
        mock_connector.root_dir = "s3://kumo-public-datasets/test"

        codegen_ctx = CodegenContext()
        lines = s3_handler.emit_lines(mock_connector, "s3connector_1", {},
                                      codegen_ctx)

        expected = ("s3connector_1 = "
                    "S3Connector.get_by_name('test-connector')")
        assert lines == [expected]

    def test_s3_connector_without_root_dir(self):
        """Test get_by_name pattern for S3Connector."""
        handlers = get_handlers()
        s3_handler = handlers[S3Connector]

        mock_connector = Mock(spec=S3Connector)
        mock_connector.name = "named-connector"
        mock_connector.root_dir = None

        codegen_ctx = CodegenContext()
        lines = s3_handler.emit_lines(mock_connector, "s3connector_1", {},
                                      codegen_ctx)

        expected = (
            "s3connector_1 = S3Connector.get_by_name('named-connector')")
        assert lines == [expected]


class TestADLSConnectorHandler:
    """Test ADLSConnector special handling patterns."""
    def test_adls_connector_with_root_dir(self):
        """Test root_dir pattern for ADLSConnector."""
        handlers = get_handlers()
        adls_handler = handlers[ADLSConnector]

        mock_connector = Mock(spec=ADLSConnector)
        mock_connector.name = "test-connector"
        mock_connector.root_dir = (
            "abfss://container@account.dfs.core.windows.net/test")

        codegen_ctx = CodegenContext()
        lines = adls_handler.emit_lines(mock_connector, "adlsconnector_1", {},
                                        codegen_ctx)

        expected = ("adlsconnector_1 = "
                    "ADLSConnector.get_by_name('test-connector')")
        assert lines == [expected]

    def test_adls_connector_without_root_dir(self):
        """Test get_by_name pattern for ADLSConnector."""
        handlers = get_handlers()
        adls_handler = handlers[ADLSConnector]

        mock_connector = Mock(spec=ADLSConnector)
        mock_connector.name = "named-connector"
        mock_connector.root_dir = None

        codegen_ctx = CodegenContext()
        lines = adls_handler.emit_lines(mock_connector, "adlsconnector_1", {},
                                        codegen_ctx)

        expected = ("adlsconnector_1 = "
                    "ADLSConnector.get_by_name('named-connector')")
        assert lines == [expected]


class TestFactoryGeneratedHandlers:
    """Test factory-generated handlers (BigQuery, Snowflake, Databricks)."""
    def test_get_by_name_pattern(self):
        """Test that factory handlers use get_by_name pattern."""
        handlers = get_handlers()
        bq_handler = handlers[BigQueryConnector]

        mock_connector = Mock(spec=BigQueryConnector)
        mock_connector.name = "test-connector"

        codegen_ctx = CodegenContext()
        lines = bq_handler.emit_lines(mock_connector, "bq_1", {}, codegen_ctx)
        imports = bq_handler.required_imports(mock_connector)
        parents = bq_handler.parents(mock_connector, codegen_ctx)

        assert lines == [
            "bq_1 = BigQueryConnector.get_by_name('test-connector')"
        ]
        assert imports == ["from kumoai_studio import BigQueryConnector"]
        assert parents == []
        assert bq_handler.detect_edits is None


class TestFileUploadConnectorHandler:
    """Test FileUploadConnector constructor pattern."""
    def test_file_upload_connector_basic(self):
        """Test FileUploadConnector basic functionality."""
        # Use real connector instead of mock to avoid type detection issues
        connector = object.__new__(FileUploadConnector)
        connector._file_type = 'parquet'

        handlers = get_handlers()
        fu_handler = handlers[FileUploadConnector]

        codegen_ctx = CodegenContext()
        lines = fu_handler.emit_lines(connector, "fu_1", {}, codegen_ctx)
        imports = fu_handler.required_imports(connector)
        parents = fu_handler.parents(connector, codegen_ctx)

        # Note: FileUploadConnector normalizes file_type to uppercase
        assert lines == ["fu_1 = FileUploadConnector(file_type='PARQUET')"]
        assert imports == ["from kumoai_studio import FileUploadConnector"]
        assert parents == []
        assert fu_handler.detect_edits is None


class TestConnectorRegistry:
    """Test connector handler registry basics."""
    def test_all_connector_types_registered(self):
        """Test that connector types are registered."""
        handlers = get_handlers()

        # Test just a few key ones instead of all
        key_connectors = [
            S3Connector,
            ADLSConnector,
            BigQueryConnector,
            FileUploadConnector,
        ]

        for connector_cls in key_connectors:
            assert connector_cls in handlers
            assert handlers[connector_cls] is not None

    def test_file_connectors_vs_factory_behavior(self):
        """Test that file connectors have special handling."""
        handlers = get_handlers()
        s3_handler = handlers[S3Connector]
        adls_handler = handlers[ADLSConnector]
        bq_handler = handlers[BigQueryConnector]

        mock_s3_with_root = Mock(spec=S3Connector)
        mock_s3_with_root.root_dir = "s3://bucket/path"

        mock_s3_without_root = Mock(spec=S3Connector)
        mock_s3_without_root.name = "test"
        mock_s3_without_root.root_dir = None

        mock_adls_with_root = Mock(spec=ADLSConnector)
        mock_adls_with_root.root_dir = (
            "abfss://container@account.dfs.core.windows.net/path")

        mock_adls_without_root = Mock(spec=ADLSConnector)
        mock_adls_without_root.name = "test"
        mock_adls_without_root.root_dir = None

        mock_bq = Mock(spec=BigQueryConnector)
        mock_bq.name = "test"

        codegen_ctx = CodegenContext()
        s3_with_root_lines = s3_handler.emit_lines(mock_s3_with_root, "s3_1",
                                                   {}, codegen_ctx)
        s3_without_root_lines = s3_handler.emit_lines(mock_s3_without_root,
                                                      "s3_2", {}, codegen_ctx)
        adls_with_root_lines = adls_handler.emit_lines(mock_adls_with_root,
                                                       "adls_1", {},
                                                       codegen_ctx)
        adls_without_root_lines = adls_handler.emit_lines(
            mock_adls_without_root, "adls_2", {}, codegen_ctx)
        bq_lines = bq_handler.emit_lines(mock_bq, "bq_1", {}, codegen_ctx)

        assert "S3Connector.get_by_name(" in s3_with_root_lines[0]
        assert "S3Connector.get_by_name(" in s3_without_root_lines[0]
        assert "ADLSConnector.get_by_name(" in adls_with_root_lines[0]
        assert "ADLSConnector.get_by_name(" in adls_without_root_lines[0]
        # BigQuery always uses get_by_name
        assert "BigQueryConnector.get_by_name(" in bq_lines[0]
