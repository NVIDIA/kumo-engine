# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from kumoai_studio.codegen.context import CodegenContext
from kumoai_studio.codegen.exceptions import (
    CyclicDependencyError,
    UnsupportedEntityError,
)
from kumoai_studio.codegen.generate import _generate, generate_code
from kumoai_studio.codegen.naming import NameManager
from kumoai_studio.testing.codegen import MockSDKObject, assert_code_equal


def test_name_manager_basic_assignment():
    """Test basic name assignment functionality."""
    nm = NameManager()

    class AnotherMockObject(MockSDKObject):
        pass

    obj1 = MockSDKObject(name="A")
    obj2 = AnotherMockObject(name="B")
    obj3 = MockSDKObject(name="C")

    assert nm.assign_entity_variable(obj1) == "a_mocksdkobject_1"
    assert nm.assign_entity_variable(obj2) == "b_anothermockobject_1"
    assert nm.assign_entity_variable(
        obj1) == "a_mocksdkobject_1"  # Same object
    assert nm.assign_entity_variable(obj3) == "c_mocksdkobject_1"


def test_name_manager_no_name_attribute():
    """Test name manager with objects that don't have name attribute."""
    nm = NameManager()

    class NoNameObject:
        pass

    obj = NoNameObject()
    assert nm.assign_entity_variable(obj) == "nonameobject_1"


def test_name_manager_sanitization():
    """Test name sanitization for invalid Python identifiers."""
    nm = NameManager()

    # Test with invalid characters
    obj1 = MockSDKObject(name="my-table")
    assert nm.assign_entity_variable(obj1) == "my_table_mocksdkobject_1"

    # Test with number start
    obj2 = MockSDKObject(name="123table")
    assert nm.assign_entity_variable(obj2) == "_123table_mocksdkobject_1"

    # Test with keyword
    obj3 = MockSDKObject(name="class")
    assert nm.assign_entity_variable(obj3) == "class__mocksdkobject_1"


def test_name_manager_temp_variables():
    """Test temporary variable assignment."""
    nm = NameManager()

    # Test with path-based naming
    temp1 = nm.assign_temp_variable("columns", [1, 2, 3])
    assert temp1 == "columns_1"

    # Test with nested path
    temp2 = nm.assign_temp_variable("user.profile.name", "Alice")
    assert temp2 == "name_1"

    # Test with collection type
    temp3 = nm.assign_temp_variable("", [1, 2, 3])
    assert temp3 == "list_1"

    # Test with object type
    class CustomObject:
        pass

    temp4 = nm.assign_temp_variable("", CustomObject())
    assert temp4 == "custom_object_1"


def test_name_manager_type_name_matching():
    """Tests that var name does not include
    type and name of object if they match
    """
    nm = NameManager()

    # Type name matching
    class Table:
        def __init__(self, name):
            self.name = name

    obj = Table("table")
    name = nm.assign_entity_variable(obj)
    assert name == "table_1"  # Should not be "table_table_1"


def test_name_manager_counter_increment():
    """Test that counters increment correctly."""
    nm = NameManager()

    obj1 = MockSDKObject(name="test")
    obj2 = MockSDKObject(name="test")

    name1 = nm.assign_entity_variable(obj1)
    name2 = nm.assign_entity_variable(obj2)

    assert name1 == "test_mocksdkobject_1"
    assert name2 == "test_mocksdkobject_2"


@patch("kumoai_studio.codegen.generate.execute_in_env")
@patch("kumoai_studio.codegen.generate.init_execution_env")
def test_generate_single_entity(mock_init_env, mock_execute_in_env):
    """Test generating code for a single entity."""
    e = MockSDKObject(name="A", imports=["import a"])
    mock_execute_in_env.return_value = None
    mock_init_env.return_value = None
    codegen_ctx = CodegenContext()
    imports, lines = _generate(e, NameManager(), {}, set(),
                               codegen_ctx=codegen_ctx)
    assert "import a" in imports
    functional_lines = [line for line in lines if not line.startswith("#")]
    assert functional_lines == ["a_mocksdkobject_1 = 'A'"]


@patch("kumoai_studio.codegen.generate.execute_in_env")
@patch("kumoai_studio.codegen.generate.init_execution_env")
def test_generate_with_parents(mock_init_env, mock_execute_in_env):
    """Test generating code with parent dependencies."""
    b = MockSDKObject(name="B")
    a = MockSDKObject(name="A", parents=[b])
    mock_execute_in_env.return_value = None
    mock_init_env.return_value = None
    codegen_ctx = CodegenContext()
    imports, lines = _generate(a, NameManager(), {}, set(),
                               codegen_ctx=codegen_ctx)
    functional_lines = [line for line in lines if not line.startswith("#")]
    assert functional_lines == [
        "b_mocksdkobject_1 = 'B'",
        "a_mocksdkobject_1 = 'A'",
    ]


@patch("kumoai_studio.codegen.generate.execute_in_env")
@patch("kumoai_studio.codegen.generate.init_execution_env")
def test_generate_shared_dependency(mock_init_env, mock_execute_in_env):
    """Test that shared dependencies are only generated once."""
    c = MockSDKObject(name="C")
    b = MockSDKObject(name="B", parents=[c])
    a = MockSDKObject(name="A", parents=[c])
    root = MockSDKObject(name="ROOT", parents=[a, b])

    mock_execute_in_env.return_value = None
    mock_init_env.return_value = None
    codegen_ctx = CodegenContext()
    imports, lines = _generate(root, NameManager(), {}, set(),
                               codegen_ctx=codegen_ctx)

    functional_lines = [line for line in lines if not line.startswith("#")]
    c_line = "c_mocksdkobject_1 = 'C'"
    a_line = "a_mocksdkobject_1 = 'A'"
    b_line = "b_mocksdkobject_1 = 'B'"

    # C should appear exactly once and before both A and B
    assert functional_lines.count(c_line) == 1
    assert functional_lines.index(c_line) < functional_lines.index(a_line)
    assert functional_lines.index(c_line) < functional_lines.index(b_line)


def test_generate_cyclic_dependency():
    """Test that cyclic dependencies are detected and raise an error."""
    a = MockSDKObject(name="A")
    b = MockSDKObject(name="B", parents=[a])
    a.parents = [b]  # Create cycle

    codegen_ctx = CodegenContext()
    with pytest.raises(CyclicDependencyError):
        _generate(a, NameManager(), {}, set(), codegen_ctx=codegen_ctx)


def test_unsupported_entity():
    """Test that unsupported entity types raise an error."""
    class UnsupportedTestObject:
        pass

    codegen_ctx = CodegenContext()
    with pytest.raises(UnsupportedEntityError):
        _generate(UnsupportedTestObject(), NameManager(), {}, set(),
                  codegen_ctx=codegen_ctx)


def test_name_manager_attribute_priority():
    """Test name manager attribute priority: name > source_name."""
    nm = NameManager()

    class ObjectWithMultipleNames:
        def __init__(self):
            self.name = "name_attr"
            self.source_name = "source_attr"

    # Test name takes priority
    obj = ObjectWithMultipleNames()
    assert nm.assign_entity_variable(
        obj) == "name_attr_objectwithmultiplenames_1"


@patch("kumoai_studio.codegen.generate.execute_in_env")
@patch("kumoai_studio.codegen.generate.init_execution_env")
def test_generate_complex_hierarchy(mock_init_env, mock_execute_in_env):
    """Test generating code for a complex object hierarchy."""
    # Create a diamond dependency pattern
    d = MockSDKObject(name="D")
    c = MockSDKObject(name="C", parents=[d])
    b = MockSDKObject(name="B", parents=[d])
    a = MockSDKObject(name="A", parents=[b, c])

    mock_execute_in_env.return_value = None
    mock_init_env.return_value = None
    codegen_ctx = CodegenContext()
    imports, lines = _generate(a, NameManager(), {}, set(),
                               codegen_ctx=codegen_ctx)
    functional_lines = [line for line in lines if not line.startswith("#")]

    # D should come first, then B and C, then A
    d_idx = functional_lines.index("d_mocksdkobject_1 = 'D'")
    b_idx = functional_lines.index("b_mocksdkobject_1 = 'B'")
    c_idx = functional_lines.index("c_mocksdkobject_1 = 'C'")
    a_idx = functional_lines.index("a_mocksdkobject_1 = 'A'")

    assert d_idx < b_idx
    assert d_idx < c_idx
    assert b_idx < a_idx
    assert c_idx < a_idx


@patch("kumoai_studio.codegen.generate.load_from_id")
@patch("kumoai_studio.codegen.generate.execute_in_env")
def test_generate_code_output(mock_execute_in_env, mock_load_from_id):
    """Test the complete generate_code function output."""
    c = MockSDKObject(name="C", imports=["import c", "import shared"])
    a = MockSDKObject(name="A", imports=["import a", "import shared"],
                      parents=[c])
    leaf_obj = MockSDKObject(name="ROOT", imports=["import root"], parents=[a])

    mock_load_from_id.return_value = leaf_obj
    # Mock execute_in_env to avoid importing fake modules
    mock_execute_in_env.return_value = None

    code = generate_code(input_spec={"id": "mock-123"})

    # Check that the code contains the expected elements
    assert "# Generated with Kumo SDK version:" in code
    expected_script = """import kumoai_studio as kumo
import os
kumo.init(url=os.getenv("KUMO_API_ENDPOINT"),
api_key=os.getenv("KUMO_API_KEY"))
import c
import shared
import a
import root
c_mocksdkobject_1 = 'C'
a_mocksdkobject_1 = 'A'
root_mocksdkobject_1 = 'ROOT'
"""
    assert_code_equal(code, expected_script)


@patch("kumoai_studio.codegen.generate.execute_in_env")
def test_generate_code_with_object_input(mock_execute_in_env):
    """Test generate_code with direct object input."""
    obj = MockSDKObject(name="DirectObject", imports=["import direct"])

    # Mock execute_in_env to avoid importing fake modules
    mock_execute_in_env.return_value = None

    code = generate_code(input_spec={"object": obj})

    assert "# Generated with Kumo SDK version:" in code
    assert "directobject_mocksdkobject_1 = 'DirectObject'" in code
    assert "import direct" in code


@patch("kumoai_studio.codegen.generate.execute_in_env")
@patch("kumoai_studio.codegen.generate.init_execution_env")
def test_import_edge_cases(mock_init_env, mock_execute_in_env):
    """Test various edge cases in generation."""
    # Empty imports
    obj = MockSDKObject(name="Simple")
    nm = NameManager()
    mock_execute_in_env.return_value = None
    mock_init_env.return_value = None
    codegen_ctx = CodegenContext()
    imports, lines = _generate(obj, nm, {}, set(), codegen_ctx=codegen_ctx)
    assert imports == []
    functional_lines = [line for line in lines if not line.startswith("#")]
    assert functional_lines == ["simple_mocksdkobject_1 = 'Simple'"]

    # Duplicate imports get deduplicated in final code
    obj1 = MockSDKObject(name="Obj1", imports=["import shared"])
    obj2 = MockSDKObject(name="Obj2", imports=["import shared"],
                         parents=[obj1])
    with patch("kumoai_studio.codegen.generate.execute_in_env"):
        code = generate_code(input_spec={"object": obj2})
    import_lines = [
        line for line in code.split('\n') if line.startswith('import shared')
    ]
    assert len(import_lines) == 1
