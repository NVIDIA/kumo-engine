# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YAML rendering and parsing for :class:`kumo_core.model_plan.ModelPlan`.

YAML is a UI concern, not a core one. The Kumo web app's plan editor is a
YAML text area: the server renders a suggested plan for the editor to seed
itself with, and parses whatever the user types back. Nothing else in the
product speaks YAML -- the public API and SDK exchange plans as JSON
(:mod:`kumo_core.model_plan_conversion`), jobs persist them as JSON, and the
``kumo-core`` and ``kumo-engine`` build them in Python.

So this lives here rather than on ``ModelPlan``: the shipped core package
keeps a plain plan dataclass with no YAML surface and no PyYAML import, and
the editor's format stays next to the layer that serves the editor.

These are thin adapters over the section-level API that does the real work:
:meth:`~kumo_core.model_plan.PlanMixin.to_dict` applies the task/query-type
filter, and :meth:`~kumo_core.model_plan.PlanMixin.update` applies a sparse
overlay and collects line errors. What is added here is the format itself
plus the diagnostics a hand-authored format needs -- typo repair, line-anchored
parser errors, and hop-key pre-validation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import yaml
from kumo_core.model_plan import (
    _INVALID_SYNTAX_PREFIX,
    ExtraFieldsError,
    LineError,
    ModelPlan,
    ModelPlanSyntaxError,
    ModelPlanValidationError,
)
from kumo_core.types import TaskType
from kumoapi.encoder import Encoder
from kumoapi.model_plan import NumNeighborsConfig
from kumoapi.pquery import QueryType
from pydantic_core import PydanticCustomError
from yaml.parser import ParserError
from yaml.scanner import ScannerError


def model_plan_to_yaml(
    plan: ModelPlan,
    task_type: TaskType,
    query_type: QueryType,
    has_train_table_weight_col: bool = False,
) -> str:
    sections_dict: dict[str, dict[str, Any]] = {}
    for key in plan.__dataclass_fields__:  # type: ignore
        section = getattr(plan, key)
        sections_dict[key] = section.to_dict(
            task_type,
            query_type,
            has_train_table_weight_col,
        )

    out = yaml.dump(sections_dict, sort_keys=False, Dumper=_PrettyDumper)

    def remove_newlines_in_parentheses(text):
        def replacer(match):
            return match.group(0).replace("\n    ", " ")

        return re.sub(r'\(.*?\)', replacer, text, flags=re.DOTALL)

    # Remove newlines within parentheses to support splits with literals
    # with colons eg "2024-07-09 23:53:03":
    out = remove_newlines_in_parentheses(out)

    out = out[:-1] if out[-1] == '\n' else out  # Remove final line break.

    # Validate that this will be parsed correctly:
    model_plan_from_yaml(out)

    return out


def model_plan_from_yaml(yaml_repr: str) -> ModelPlan:
    r"""Builds a :class:`ModelPlan` from YAML, over an all-defaults plan."""
    return update_model_plan_from_yaml(ModelPlan(), yaml_repr)


def update_model_plan_from_yaml(
    plan: ModelPlan,
    yaml_repr: str,
) -> ModelPlan:
    r"""Overlays a YAML string onto ``plan``, in place, and returns it.

    Provides enhanced error handling with user-friendly messages:
    1. Auto-fixes common typos (missing spaces after colons/dashes)
    2. Translates YAML errors to show line context
    3. Pre-validates hop keys with clear error messages
    """
    yaml_repr = re.sub(r'^(\s*[\w_]+:)(?=\S)', r'\1 ', yaml_repr,
                       flags=re.MULTILINE)
    yaml_repr = re.sub(r'^(\s+-)(?=\S)', r'\1 ', yaml_repr, flags=re.MULTILINE)

    try:
        sections_dict = yaml.safe_load(yaml_repr)
    except (ScannerError, ParserError) as e:
        friendly_error = _translate_yaml_error(e, yaml_repr)
        error_message = friendly_error.replace(_INVALID_SYNTAX_PREFIX,
                                               "").strip()
        raise ModelPlanSyntaxError(error_message) from e

    if sections_dict is None:
        return plan

    if not isinstance(sections_dict, dict):
        raise ModelPlanSyntaxError(
            "Could not parse the Model Plan as valid YAML.")

    all_errors: list[LineError] = []
    for key in plan.__dataclass_fields__:  # type: ignore
        section = getattr(plan, key)

        if key in sections_dict.keys():
            options_dict = sections_dict.pop(key)

            if key == 'neighbor_sampling' and isinstance(options_dict, dict):
                neighbor_sampling_errors = _prevalidate_neighbor_sampling_keys(
                    options_dict)
                if neighbor_sampling_errors:
                    all_errors.extend(neighbor_sampling_errors)
                    continue

            errors = section.update(options_dict, raise_on_error=False)
            all_errors.extend(errors)

    for name in sections_dict.keys():
        valid_top_sections = [
            field.name
            for field in plan.__dataclass_fields__.values()  # type: ignore
        ]
        all_errors.append({
            'type':
            ExtraFieldsError(valid_top_sections=valid_top_sections),
            'loc': (name, ),
            'input':
            sections_dict[name],
        })

    if len(all_errors) > 0:
        raise ModelPlanValidationError(
            all_errors,
            model=type(plan),  # type: ignore
        )

    return plan


def _translate_yaml_error(error: Exception, yaml_repr: str) -> str:
    r"""Translates YAML parser errors into user-friendly messages with
    line context."""
    error_msg = str(error)
    lines = yaml_repr.split('\n')

    line_col_match = re.search(r'(\d+):(\d+)', error_msg)
    if line_col_match:
        line_num = int(line_col_match.group(1))
        col_num = int(line_col_match.group(2))
    else:
        line_match = re.search(r'line (\d+)', error_msg)
        line_num = int(line_match.group(1)) if line_match else None
        col_match = re.search(r'column (\d+)', error_msg)
        col_num = int(col_match.group(1)) if col_match else None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('->') or stripped.startswith('- >'):
            return (
                f"{_INVALID_SYNTAX_PREFIX}\n\n"
                f"  {repr(stripped.split(':')[0])}: empty source (before '->').\n"
                f"  Fix by adding a table.column before '->'.")

    if line_num and 1 <= line_num <= len(lines):
        problem_line = lines[line_num - 1]
        pointer = (f"{' ' * (len(f'Line {line_num}: ') + col_num - 1)}^\n"
                   if col_num else "")
        return (f"{_INVALID_SYNTAX_PREFIX}\n\n"
                f"Line {line_num}: {problem_line}\n{pointer}{error_msg}")

    return f"{_INVALID_SYNTAX_PREFIX} {error_msg}"


def _prevalidate_neighbor_sampling_keys(options_dict: dict) -> list[LineError]:
    r"""Pre-validates hop key format before Pydantic processes them.

    Only catches obvious structural errors (missing/extra arrows, missing
    dots). Empty table/column parts are caught later during validate().
    """
    errors: list[LineError] = []
    invalid_keys: list[str] = []

    num_neighbors = options_dict.get('num_neighbors')
    if not isinstance(num_neighbors, list):
        return errors

    for item in num_neighbors:
        if not isinstance(item, dict):
            continue
        for hop_config in item.values():
            if not isinstance(hop_config, dict):
                continue
            for key in hop_config.keys():
                if key == 'default' or not isinstance(key, str):
                    continue

                arrow_count = key.count('->')
                if arrow_count == 0:
                    invalid_keys.append(
                        f"  {repr(key)}: missing the '->' separator.\n"
                        f"  Fix by adding '->' between source and destination.")
                elif arrow_count > 1:
                    invalid_keys.append(
                        f"  {repr(key)}: too many '->'.\n"
                        f"  Fix by removing the extra arrow(s).")
                elif '->' in key:
                    src, dst = key.split('->')
                    if not src.strip():
                        invalid_keys.append(
                            f"  {repr(key)}: empty source (before '->').\n"
                            f"  Fix by adding a table.column before '->'.")
                    elif not dst.strip():
                        invalid_keys.append(
                            f"  {repr(key)}: empty destination (after '->').\n"
                            f"  Fix by adding a table.column after '->'.")
                    elif '.' not in src:
                        invalid_keys.append(
                            f"  {repr(key)}: source is missing the dot separator.\n"
                            f"  Fix by adding a dot between table and column.")
                    elif '.' not in dst:
                        invalid_keys.append(
                            f"  {repr(key)}: destination is missing the dot separator.\n"
                            f"  Fix by adding a dot between table and column.")

    if invalid_keys:
        # No `ctx` is passed, so PydanticCustomError leaves any literal
        # braces in the message untouched (the hop keys are `repr`'d).
        errors.append({
            'type':
            PydanticCustomError(
                'model_plan_syntax',
                # The template is built from user input rather than a literal,
                # which is safe here only because no `ctx` is passed.
                str(  # type: ignore[arg-type]
                    ModelPlanSyntaxError("Invalid hop key(s):\n\n" +
                                         "\n".join(invalid_keys))),
            ),
            'loc': ('neighbor_sampling', 'num_neighbors'),
            'input':
            options_dict.get('num_neighbors'),
        })

    return errors


class _PrettyDumper(yaml.SafeDumper):
    r"""A custom dumper to deal with enums and prettify nested list
    representations."""
    def represent_data(self, data):
        if data is None:
            # Represent empty data as `attr:` instead of `attr: null`:
            return self.represent_scalar('tag:yaml.org,2002:null', '')
        if isinstance(data, Enum):
            return self.represent_scalar('tag:yaml.org,2002:str', data.value)
        if type(data) is not str and isinstance(data, str):
            return self.represent_scalar('tag:yaml.org,2002:str', str(data))
        if is_dataclass(data):
            return self.represent_dataclass(data)
        return super().represent_data(data)

    def represent_dataclass(self, data):
        if isinstance(data, Encoder):
            # Represent encoder classes as strings:
            return self.represent_scalar('tag:yaml.org,2002:str', str(data))

        if isinstance(data, NumNeighborsConfig):
            out = {  # Represent number of neighbors with extra keys:
                f'hop{key + 1}':
                dict(default=value.default) | value.__pydantic_extra__
                for key, value in data.to_dict().items()
            }
            return self.represent_mapping('tag:yaml.org,2002:map', out)

        return self.represent_mapping('tag:yaml.org,2002:map', asdict(data))

    def represent_sequence(self, tag, sequence, flow_style=None):
        node = super().represent_sequence(tag, sequence, flow_style)

        # Decode the children in nested lists via "flow" format:
        for child in node.value:
            if isinstance(child, yaml.SequenceNode):
                child.flow_style = True

        return node
