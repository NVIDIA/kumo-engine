# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.handlers.pquery import get_handlers
from kumoai_studio.codegen.identity import get_config_id
from kumoai_studio.pquery import PredictiveQuery


class TestPredictiveQueryHandler:
    """Test PredictiveQuery handler patterns."""
    def test_multi_line_query_with_preserved_formatting(self):
        """Test multi-line query with preserved formatting."""
        handlers = get_handlers()
        pquery_handler = handlers[PredictiveQuery]

        # Create mock graph
        mock_graph = Mock()

        # Create mock PredictiveQuery with multi-line query
        query_string = """PREDICT COUNT(transactions.* WHERE transactions.price > 0.1, 0, 5, days)
                            FOR EACH customers.customer_id"""  # noqa

        mock_pquery = Mock(spec=PredictiveQuery)
        mock_pquery.graph = mock_graph
        mock_pquery.query = query_string

        # Mock parent_map
        parent_map = {get_config_id(mock_graph): "test_graph"}

        context = {"original_id": "test_pquery_id"}

        # Test parents function
        codegen_ctx = CodegenContext()
        codegen_ctx.object_to_var = parent_map
        parents = pquery_handler.parents(mock_pquery, codegen_ctx)
        assert parents == [mock_graph]

        # Test imports function
        imports = pquery_handler.required_imports(mock_pquery)
        assert imports == ["from kumoai_studio import PredictiveQuery"]

        # Test emit_lines function
        lines = pquery_handler.emit_lines(mock_pquery, "test_pquery", context,
                                          codegen_ctx)

        # Verify the generated code structure
        expected_lines = [
            "test_pquery = PredictiveQuery(graph=test_graph,"
            " query=\"\"\"PREDICT "
            "COUNT(transactions.* WHERE transactions.price > 0.1, 0, 5, days)"
            "\n                            "
            "FOR EACH customers.customer_id\"\"\",)", "test_pquery.validate()"
        ]

        assert lines == expected_lines

        # Verify detect_edits is None
        assert pquery_handler.detect_edits is None
