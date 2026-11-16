# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.graph import get_handlers
from kumoai_studio.codegen.identity import get_config_id
from kumoai_studio.graph import Graph


class TestGraphHandler:
    """Test Graph handler patterns."""
    def test_graph_with_tables_no_edges(self):
        """Test graph with tables but no edges."""
        handlers = get_handlers()
        graph_handler = handlers[Graph]

        # Create mock tables
        mock_table1 = Mock()
        mock_table2 = Mock()
        mock_table3 = Mock()

        # Create mock graph with tables but no edges
        mock_graph = Mock(spec=Graph)
        mock_graph.tables = {
            "customers": mock_table1,
            "orders": mock_table2,
            "products": mock_table3
        }
        mock_graph.edges = []

        # Mock parent_map to simulate how table variables are named
        parent_map = {
            get_config_id(mock_table1): "customers_table",
            get_config_id(mock_table2): "orders_table",
            get_config_id(mock_table3): "products_table"
        }

        context = {"target_id": "test_graph_id"}

        codegen_ctx = CodegenContext()
        codegen_ctx.object_to_var = parent_map
        lines = graph_handler.emit_lines(mock_graph, "test_graph", context,
                                         codegen_ctx)

        # Verify the generated code structure
        assert len(lines) == 6
        expected_lines = [
            "# Note: This could also be done with Graph.load("
            "'test_graph_id') for simpler code",
            "test_graph_tables = {'customers': customers_table, "
            "'orders': orders_table, 'products': products_table}",
            "test_graph_edges = []",
            "test_graph = Graph(tables=test_graph_tables, "
            "edges=test_graph_edges)", "test_graph.validate()",
            "# Optionally, you can save the graph to backend using "
            "test_graph.save(test_graph_id)"
            " and then load it"
        ]
        assert lines == expected_lines

    def test_graph_with_complex_edges(self):
        """Test graph with complex edge relationships."""
        handlers = get_handlers()
        graph_handler = handlers[Graph]

        # Create mock tables
        mock_table1 = Mock()
        mock_table2 = Mock()
        mock_table3 = Mock()

        # Create mock graph with complex edges
        mock_graph = Mock(spec=Graph)
        mock_graph.tables = {
            "customers": mock_table1,
            "orders": mock_table2,
            "products": mock_table3
        }
        mock_graph.edges = [{
            "src_table": "orders",
            "fkey": "customer_id",
            "dst_table": "customers"
        }, {
            "src_table": "orders",
            "fkey": "product_id",
            "dst_table": "products"
        }, {
            "src_table": "customers",
            "fkey": "preferred_product_id",
            "dst_table": "products"
        }]

        # Mock parent_map
        parent_map = {
            get_config_id(mock_table1): "customers_table",
            get_config_id(mock_table2): "orders_table",
            get_config_id(mock_table3): "products_table"
        }

        context = {"target_id": "complex_graph_id"}

        codegen_ctx = CodegenContext()
        codegen_ctx.object_to_var = parent_map
        lines = graph_handler.emit_lines(mock_graph, "complex_graph", context,
                                         codegen_ctx)

        # Verify the generated code structure
        assert len(lines) == 6
        expected_lines = [
            "# Note: This could also be done with Graph.load("
            "'complex_graph_id') for simpler code",
            "complex_graph_tables = {'customers': customers_table, "
            "'orders': orders_table, 'products': products_table}",
            "complex_graph_edges = [{'src_table': 'orders', 'fkey': "
            "'customer_id', 'dst_table': 'customers'}, "
            "{'src_table': 'orders', "
            "'fkey': 'product_id', 'dst_table': 'products'}, "
            "{'src_table': "
            "'customers', 'fkey': 'preferred_product_id', 'dst_table': "
            "'products'}]",
            "complex_graph = Graph(tables=complex_graph_tables, "
            "edges=complex_graph_edges)", "complex_graph.validate()",
            "# Optionally, you can save the graph to backend using "
            "complex_graph.save(complex_graph_id)"
            " and then load it"
        ]
        assert lines == expected_lines
