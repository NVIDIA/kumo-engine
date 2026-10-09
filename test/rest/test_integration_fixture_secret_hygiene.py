# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_public_api_fixture_does_not_emit_api_key_value():
    """Keep generated integration API keys process-local and out of logs."""
    fixture_path = (Path(__file__).resolve().parents[2] / 'test' / 'rest' /
                    'integ' / 'conftest.py')
    tree = ast.parse(fixture_path.read_text())
    fixture = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == 'public_api_client')

    output_calls = {
        'print', 'debug', 'info', 'warning', 'error', 'exception', 'critical',
        'log'
    }
    for call in (node for node in ast.walk(fixture)
                 if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Name):
            call_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            call_name = call.func.attr
        else:
            continue
        if call_name in output_calls:
            rendered_call = ast.unparse(call)
            assert not ('api_key' in rendered_call and
                        'value' in rendered_call), rendered_call
