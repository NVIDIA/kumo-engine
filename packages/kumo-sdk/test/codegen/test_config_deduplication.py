# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for configuration-based deduplication in codegen."""

from kumoai_studio.codegen.identity import get_config_id


class TestConfigurationDeduplication:
    """Test configuration-based identity system."""
    def test_s3_connector_config_id_same_config(self):
        """Test that S3Connectors with same config have same config_id."""
        from kumoai_studio.connector import S3Connector

        connector1 = object.__new__(S3Connector)
        connector1._id = 'connector-one'
        connector2 = object.__new__(S3Connector)
        connector2._id = 'connector-one'

        config_id1 = get_config_id(connector1)
        config_id2 = get_config_id(connector2)

        assert config_id1 == config_id2

    def test_s3_connector_config_id_different_config(self):
        """Test that S3Connectors with different
        config have different config_id.
        """
        from kumoai_studio.connector import S3Connector

        connector1 = object.__new__(S3Connector)
        connector1._id = 'connector-one'
        connector2 = object.__new__(S3Connector)
        connector2._id = 'connector-two'

        config_id1 = get_config_id(connector1)
        config_id2 = get_config_id(connector2)

        assert config_id1 != config_id2

    def test_fallback_to_memory_id(self):
        """Test that unsupported objects fall back to memory ID."""
        class UnsupportedObject:
            pass

        obj1 = UnsupportedObject()
        obj2 = UnsupportedObject()

        config_id1 = get_config_id(obj1)
        config_id2 = get_config_id(obj2)

        # Should be different because they're different objects
        assert config_id1 != config_id2
