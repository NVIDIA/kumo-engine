# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for object edit generation system.

Tests the edit generation with variable naming, edit modes,
constructor handling, enum handling, and edge cases.

As the edit system is more general and
should apply to almost any object, it is useful to test
it with a variety of objects, not necessarily
just the ones in Kumo, to ensure it is robust.
"""

import logging
from typing import Optional

import pytest

from kumoai_studio.codegen.edits import (
    UniversalReplacementEdit,
    detect_edits_recursive,
    get_editable_attributes,
)
from kumoai_studio.codegen.naming import NameManager

logger = logging.getLogger(__name__)


class MockColumn:
    """Mock column."""
    def __init__(self, name: str = "", dtype: str = "", stype: str = ""):
        self.name = name
        self.dtype = dtype
        self.stype = stype

    def __eq__(self, other):
        return (isinstance(other, MockColumn) and self.name == other.name
                and self.dtype == other.dtype and self.stype == other.stype)


class MockTable:
    """Mock table."""
    def __init__(
        self,
        primary_key: Optional[str] = None,
        time_column: Optional[str] = None,
    ):
        self.primary_key = primary_key
        self.time_column = time_column
        self._columns = {}  # Private attribute exposed via columns property

    @property
    def columns(self):
        """Exposes _columns"""
        return list(self._columns.values())


class MockEnum:
    """Mock enum."""
    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value


class Address:
    """Mock address."""
    def __init__(self, street: str, city: str):
        self.street = street
        self.city = city


class Person:
    """Mock person."""
    def __init__(self, name: str = "", address: Optional[Address] = None):
        self.name = name
        self.address = address


class MockNested:
    """Mock nested object."""
    def __init__(self, level: int):
        self.level = level
        self.nested = None if level <= 0 else MockNested(level - 1)


class TestBasicEditing:
    """Test basic editing functionality."""
    def test_private_attribute_detection(self):
        """Test private attribute detection."""
        table = MockTable()
        table._columns["test_col"] = MockColumn("test_col", "string", "ID")

        attrs = get_editable_attributes(table)

        assert "primary_key" in attrs
        assert "time_column" in attrs
        assert "_columns" in attrs

    def test_private_collection_editing(self):
        """Test private collection editing.

        This test checks that the edit system can access private attributes
        of objects, such as the _columns attribute of a table.
        This requires the edit system to use the complex object codepath and
        the collection codepath.
        """
        nm = NameManager()

        target = MockTable()
        target._columns = {"col1": MockColumn("col1", "int64", "ID")}

        baseline = MockTable()
        baseline._columns = {"col1": MockColumn("col1", "string", "ID")}

        result = detect_edits_recursive(target, baseline, "", nm)

        assert len(result.edits) >= 1

        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("table"))

        # Should access private _columns attribute somewhere in the edits
        has_private_access = any("_columns" in line for line in all_lines)
        assert has_private_access

    def test_variable_naming_and_construction(self):
        """Test variable naming and construction."""
        nm = NameManager()

        # Test primitive assignment - now uses direct assignment
        edit = UniversalReplacementEdit("name", "Alice", nm)
        lines = edit.emit_lines("person")

        assert lines == ["person.name = 'Alice'"]

        # Test collection assignment - still uses temp variables
        edit = UniversalReplacementEdit("items", [1, 2, 3], nm)
        lines = edit.emit_lines("container")

        assert lines == [
            "items_1 = [1, 2, 3]",
            "container.items = items_1",
        ]


class TestEditModes:
    """Test different edit modes and assignment patterns."""
    def test_primitive_vs_collection_vs_object_assignment(self):
        """Test that different value types use
        appropriate assignment patterns.
        """
        nm = NameManager()

        # Test primitive - direct assignment
        primitive_edit = UniversalReplacementEdit("name", "Alice", nm)
        primitive_lines = primitive_edit.emit_lines("person")
        assert primitive_lines == ["person.name = 'Alice'"]

        # Test collection - temp variable assignment
        collection_edit = UniversalReplacementEdit("items", [1, 2, 3], nm)
        collection_lines = collection_edit.emit_lines("container")
        assert collection_lines == [
            "items_1 = [1, 2, 3]",
            "container.items = items_1",
        ]

        # Test complex object - constructor + assignment
        address = Address("123 Main St", "NYC")
        object_edit = UniversalReplacementEdit("address", address, nm)
        object_lines = object_edit.emit_lines("person")

        has_constructor = any("Address(" in line for line in object_lines)
        has_assignment = any("person.address =" in line
                             for line in object_lines)
        assert has_constructor and has_assignment

    def test_collection_element_changes(self):
        """Test changes within collections."""
        nm = NameManager()

        # Test list element change
        target = {"items": [1, 2, "changed"]}
        baseline = {"items": [1, 2, "original"]}

        result = detect_edits_recursive(target, baseline, "", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("container")
        assert lines == ["container['items'][2] = 'changed'"]

        # Test dict key change
        target_dict = {"config": {"timeout": 60, "retries": 3}}
        baseline_dict = {"config": {"timeout": 30, "retries": 3}}

        result = detect_edits_recursive(target_dict, baseline_dict, "", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("settings")
        assert lines == ["settings['config']['timeout'] = 60"]

    def test_nested_object_changes(self):
        """Test changes in nested object structures."""
        nm = NameManager()

        # Create nested structure
        target = Person("John", Address("456 Oak St", "Boston"))
        baseline = Person("John", Address("123 Main St", "NYC"))

        result = detect_edits_recursive(target, baseline, "", nm)

        # Should have edits for both address fields
        assert len(result.edits) == 2

        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("person"))

        # Should have direct assignments for nested primitive changes
        assert any("person.address.street = '456 Oak St'" in line
                   for line in all_lines)
        assert any("person.address.city = 'Boston'" in line
                   for line in all_lines)


class TestConstructor:
    """Test constructor handling and import handling."""
    def test_constructor_with_actual_values(self):
        """Test that constructors use actual
        object values instead of defaults.
        """
        nm = NameManager()

        # Create an address with specific values
        address = Address("123 Main St", "NYC")
        edit = UniversalReplacementEdit("address", address, nm)
        lines = edit.emit_lines("person")

        # Should construct with actual values from the object
        constructor_line = next(line for line in lines if "Address(" in line)
        assert "123 Main St" in constructor_line
        assert "NYC" in constructor_line

        # Should have assignment line
        has_assignment = any("person.address =" in line for line in lines)
        assert has_assignment

    def test_imports_and_complex_objects(self):
        """Test import collection and complex object handling."""
        nm = NameManager()

        # Test with MockTable which should generate imports
        target = MockTable(primary_key="new_id")
        baseline = MockTable(primary_key="old_id")

        result = detect_edits_recursive(target, baseline, "", nm)

        # Should have imports attribute
        assert hasattr(result, "imports")
        assert isinstance(result.imports, list)

        # Test with complex object that has required constructor params
        target_person = Person("Alice", Address("456 Oak St", "Boston"))
        baseline_person = Person("Bob", Address("123 Main St", "NYC"))

        result = detect_edits_recursive(target_person, baseline_person, "", nm)

        # Should handle both simple and complex changes
        # At least name and address changes
        assert len(result.edits) >= 2

        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("person"))

        # Should have direct assignment for name change
        has_name_change = any("person.name = 'Alice'" in line
                              for line in all_lines)
        assert has_name_change


class TestEnumHandling:
    """Test enum handling."""
    def test_enum_to_string_conversion(self):
        """Test enum to string conversion."""
        nm = NameManager()

        enum_value = MockEnum("ACTIVE")
        edit = UniversalReplacementEdit("status", enum_value, nm)
        lines = edit.emit_lines("user")

        string_lines = "\n".join(lines)
        has_string_or_enum = ("'ACTIVE'" in string_lines
                              or "MockEnum(" in string_lines)
        assert has_string_or_enum


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    def test_special_values_and_types(self):
        """Test handling of None, empty collections, and type mismatches."""
        nm = NameManager()

        # Test None assignment
        target = MockTable(time_column=None)
        baseline = MockTable(time_column="timestamp")
        result = detect_edits_recursive(target, baseline, "", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("table")
        assert lines == ["table.time_column = None"]

        # Test empty collection replacement
        target_empty = {"items": []}
        baseline_empty = {"items": [1, 2, 3]}
        result = detect_edits_recursive(target_empty, baseline_empty, "", nm)

        assert len(result.edits) == 3  # 3 elements become None

        # Test type mismatch (string to int)
        target_type = {"value": 42}
        baseline_type = {"value": "string"}
        result = detect_edits_recursive(target_type, baseline_type, "", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("obj")
        assert lines == ["obj['value'] = 42"]

    def test_complex_nested_structures(self):
        """Test complex nested data structures."""
        nm = NameManager()

        # Test nested object with multiple levels
        level3 = MockNested(3)
        level2 = MockNested(2)
        result = detect_edits_recursive(level3, level2, "", nm)
        assert isinstance(result.edits, list)

        # Test mixed collection and object nesting
        target_mixed = {
            "users": [
                {
                    "name": "Alice",
                    "active": True
                },
                {
                    "name": "Bob",
                    "active": False
                },
            ],
            "settings": {
                "timeout": 30
            },
        }
        baseline_mixed = {
            "users": [
                {
                    "name": "Alice",
                    "active": False
                },
                {
                    "name": "Bob",
                    "active": False
                },
            ],
            "settings": {
                "timeout": 60
            },
        }

        result = detect_edits_recursive(target_mixed, baseline_mixed, "", nm)
        # At least timeout and Alice's active status
        assert len(result.edits) >= 2

        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("config"))

        # Should handle both nested dict changes and collection element changes
        has_timeout_change = any("timeout" in line for line in all_lines)
        has_user_change = any("users" in line for line in all_lines)
        assert has_timeout_change and has_user_change


class TestDictEdits:
    """Test dictionary editing behavior."""
    def test_dict_primitive_edit(self):
        """Test editing primitive values in dictionaries."""
        nm = NameManager()

        target = {"key1": "value1", "key2": "updated_value"}
        baseline = {"key1": "value1", "key2": "original_value"}

        result = detect_edits_recursive(target, baseline, "obj", nm)

        logger.debug("\n=== PRIMITIVE DICT EDIT ===")
        for i, edit in enumerate(result.edits):
            lines = edit.emit_lines("test_var")
            logger.debug(f"Edit {i+1}:")
            for line in lines:
                logger.debug(f"  {line}")

        # Should generate direct assignment for primitives
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("test_var")
        assert lines == ["test_var.obj['key2'] = 'updated_value'"]

    def test_dict_complex_object_edit(self):
        """Test editing complex objects in dictionaries."""
        nm = NameManager()

        target_columns = {
            "article_id": MockColumn(name="article_id", dtype="int64",
                                     stype="ID"),
            "price": MockColumn(name="price", dtype="float64",
                                stype="NUMERICAL"),
        }

        baseline_columns = {
            "article_id":
            MockColumn(name="article_id", dtype="string", stype="CATEGORICAL"),
            "price":
            MockColumn(name="price", dtype="float64", stype="NUMERICAL"),
        }

        target_table = MockTable()
        target_table._columns = target_columns

        baseline_table = MockTable()
        baseline_table._columns = baseline_columns

        result = detect_edits_recursive(target_table, baseline_table, "", nm)

        logger.debug("\n=== COMPLEX OBJECT DICT EDIT ===")
        for i, edit in enumerate(result.edits):
            lines = edit.emit_lines("table")
            logger.debug(f"Edit {i+1}:")
            for line in lines:
                logger.debug(f"  {line}")

        # Should generate recursive edits for complex objects
        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("table"))

        # Should contain direct path to dictionary key property
        assert any("table._columns['article_id'].dtype" in line
                   for line in all_lines)
        assert any("table._columns['article_id'].stype" in line
                   for line in all_lines)

    def test_dict_collection_edit(self):
        """Test editing collections in dictionaries."""
        nm = NameManager()

        target = {
            "tags": ["tag1", "tag2", "tag3"],
            "settings": {
                "timeout": 30,
                "retries": 5
            },
        }

        baseline = {
            "tags": ["tag1", "tag2"],
            "settings": {
                "timeout": 60,
                "retries": 3
            },
        }

        result = detect_edits_recursive(target, baseline, "config", nm)

        logger.debug("\n=== COLLECTION DICT EDIT ===")
        for i, edit in enumerate(result.edits):
            lines = edit.emit_lines("obj")
            logger.debug(f"Edit {i+1}:")
            for line in lines:
                logger.debug(f"  {line}")

        # Should handle nested collections properly
        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("obj"))

        # Should have edits for both collections
        assert any("tags" in line for line in all_lines)
        assert any("settings" in line for line in all_lines)

    def test_dict_add_remove_keys(self):
        """Test adding and removing keys from dictionaries."""
        nm = NameManager()

        target = {
            "keep": "value",
            "new_key": "new_value",
            # "removed_key" is missing
        }

        baseline = {
            "keep": "value",
            "removed_key": "old_value",
            # "new_key" is missing
        }

        result = detect_edits_recursive(target, baseline, "mapping", nm)

        logger.debug("\n=== ADD/REMOVE KEYS DICT EDIT ===")
        for i, edit in enumerate(result.edits):
            lines = edit.emit_lines("obj")
            logger.debug(f"Edit {i+1}:")
            for line in lines:
                logger.debug(f"  {line}")

        # Should handle both adding and removing keys
        all_lines = []
        for edit in result.edits:
            all_lines.extend(edit.emit_lines("obj"))

        # Should have edits for both new and removed keys
        assert any("new_key" in line for line in all_lines)
        assert any("removed_key" in line for line in all_lines)


class TestNameManagerIntegration:
    """Test name manager integration with edits system."""
    def test_temp_variable_naming_consistency(self):
        """Test that temp variables are named consistently."""
        nm = NameManager()

        # Create multiple edits with similar patterns
        edit1 = UniversalReplacementEdit("columns", [1, 2, 3], nm)
        edit2 = UniversalReplacementEdit("rows", [4, 5, 6], nm)
        edit3 = UniversalReplacementEdit("columns", [7, 8, 9],
                                         nm)  # Same base name

        lines1 = edit1.emit_lines("table")
        lines2 = edit2.emit_lines("table")
        lines3 = edit3.emit_lines("table")

        # Should use incremental naming
        assert "columns_1 = [1, 2, 3]" in lines1
        assert "rows_1 = [4, 5, 6]" in lines2
        assert "columns_2 = [7, 8, 9]" in lines3


class TestEditStrategies:
    """Test edit strategy selection."""
    def test_strategy_selection_basics(self):
        """Test that different types use appropriate strategies."""
        nm = NameManager()

        # Primitives - direct assignment
        result = detect_edits_recursive("new", "old", "field", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("obj")
        assert lines == ["obj.field = 'new'"]

        # Collections - element replacement
        target = [1, 2, 3]
        baseline = [1, 2, 4]
        result = detect_edits_recursive(target, baseline, "items", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("container")
        assert lines == ["container.items[2] = 3"]

        # Objects - property recursion
        target = Person("Alice", None)
        baseline = Person("Bob", None)
        result = detect_edits_recursive(target, baseline, "", nm)
        assert len(result.edits) == 1
        lines = result.edits[0].emit_lines("person")
        assert lines == ["person.name = 'Alice'"]


class TestUnhappyPaths:
    """Test unhappy paths, error handling, and invalid inputs."""
    def test_circular_references_are_handled(self):
        """Test that circular references are
        detected and return empty edits.
        """
        nm = NameManager()
        obj1 = MockTable()
        obj2 = MockTable()
        setattr(obj1, 'circular_ref', obj2)
        setattr(obj2, 'circular_ref', obj1)
        baseline = MockTable()

        result = detect_edits_recursive(obj1, baseline, "", nm)

        assert isinstance(result.edits, list)
        # Most edits should be skipped due to circular reference detection
        assert len(result.edits) <= 2  # Only non-circular attributes

    def test_attribute_access_errors_are_caught(self):
        """Test that attribute access errors are caught and handled."""
        nm = NameManager()

        class BrokenObject:
            def __init__(self):
                self.safe_attr = "safe"

            @property
            def broken_property(self):
                raise ValueError("This property is broken")

            def __getattribute__(self, name):
                if name == "broken_attr":
                    raise AttributeError("Broken attribute")
                return super().__getattribute__(name)

        target = BrokenObject()
        baseline = BrokenObject()

        result = detect_edits_recursive(target, baseline, "", nm)
        assert isinstance(result.edits, list)
        assert len(result.edits) == 0

    def test_constructor_failure_falls_back_to_repr(self):
        """Test that constructor failures fallback to repr."""
        nm = NameManager()

        class UnconstructableObject:
            def __init__(self, required_param):
                if required_param is None:
                    raise ValueError("Required param cannot be None")
                self.value = required_param

        obj = UnconstructableObject("test")
        edit = UniversalReplacementEdit("field", obj, nm)
        lines = edit.emit_lines("container")

        assert any("UnconstructableObject" in line for line in lines)
        assert len(lines) >= 2

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise appropriate errors."""
        nm = NameManager()

        # A None path should raise an error because it cannot be processed.
        with pytest.raises(TypeError):
            UniversalReplacementEdit(None, "value", nm)  # type: ignore

        # An empty string path is technically valid, and should not crash.
        edit = UniversalReplacementEdit("", "value", nm)
        lines = edit.emit_lines("obj")
        assert lines == ["obj = 'value'"]

    def test_unsupported_types_are_handled(self):
        """Test that unsupported object types are handled gracefully."""
        nm = NameManager()

        def test_func():
            pass

        # Functions should be handled, likely by falling back to repr.
        edit = UniversalReplacementEdit("func", test_func, nm)
        lines = edit.emit_lines("obj")
        assert len(lines) > 0
        assert "test_func" in "".join(lines)

    def test_no_edits_for_identical_objects(self):
        """Test that identical objects produce no edits."""
        nm = NameManager()

        # Identical primitives
        result = detect_edits_recursive("same", "same", "field", nm)
        assert len(result.edits) == 0

        # Identical collections
        result = detect_edits_recursive([1, 2, 3], [1, 2, 3], "items", nm)
        assert len(result.edits) == 0

        # Identical simple objects
        addr1 = Address("123 Main", "NYC")
        addr2 = Address("123 Main", "NYC")
        result = detect_edits_recursive(addr1, addr2, "address", nm)
        assert len(result.edits) == 0

    def test_constructor_analysis_with_non_standard_objects(self):
        """Test constructor analysis with non-standard objects."""
        from kumoai_studio.codegen.edits import _get_constructor_requirements

        # Test with a built-in type that has a non-standard constructor
        requirements = _get_constructor_requirements(int)
        assert requirements == {}

    def test_emit_lines_fallback_on_unexpected_error(self):
        """Test that emit_lines falls back to repr on unexpected errors."""
        nm = NameManager()

        class ProblematicObject:
            def __init__(self):
                self.value = "test"

            def __getattr__(self, name):
                raise RuntimeError("An unexpected error")

        obj = ProblematicObject()
        edit = UniversalReplacementEdit("field", obj, nm)
        lines = edit.emit_lines("container")

        assert any("ProblematicObject" in line for line in lines)
        assert len(lines) >= 2

    def test_value_representation_with_broken_enum(self):
        """Test _get_value_repr with a broken enum-like object."""
        from kumoai_studio.codegen.edits import _get_value_repr

        class BrokenEnum:
            def __init__(self, value):
                self.value = value

            @property
            def name(self):
                raise ValueError("Broken enum")

        obj = BrokenEnum("test")
        with pytest.raises(ValueError):
            _get_value_repr(obj)
