# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import Any, Dict, Optional

import yaml
from omegaconf import OmegaConf


def to_yaml(dataclass: Any) -> str:
    """Converts a pydantic dataclass into a yaml string"""
    return OmegaConf.to_yaml(OmegaConf.structured(dataclass))


# converts dict to minimal possible yaml string by dropping keys with None
# value and returning None if all values are None
def dict_to_yaml(arg_dict: Dict[str, Any]) -> Optional[str]:
    # need to cast keys to str to make sorting work reliably
    arg_dict = {str(key): value for key, value in arg_dict.items() if value}
    # Return None rather than yaml of '{}\n'
    if not arg_dict:
        return None
    # make result deterministic regardless of input dict order (note that this
    # seems to be inconsistent if arg_dict keys were not all strings)
    return yaml.dump(arg_dict, sort_keys=True)


def parse_ast_node(node, locals):
    """Parses a node from a call to ast.parse"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                      (ast.UAdd, ast.USub)):
        operand = parse_ast_node(node.operand, locals)
        if isinstance(node.op, ast.UAdd):
            return operand
        else:
            return -operand  # type: ignore
    elif isinstance(node, ast.Tuple):
        return tuple(parse_ast_node(v, locals) for v in node.elts)
    elif isinstance(node, ast.List):
        return [parse_ast_node(v, locals) for v in node.elts]
    elif isinstance(node, ast.Call):
        class_name = node.func.id  # type: ignore
        args = [parse_ast_node(n, locals) for n in node.args]
        kwargs = {
            kw.arg: parse_ast_node(kw.value, locals)
            for kw in node.keywords
        }
        return locals[class_name](*args, **kwargs)
    else:
        raise ValueError(f'Object type not supported. Must be one of '
                         f'{ast.Constant, ast.Name, ast.List, ast.Call} '
                         f'(got {type(node)}')


def repr_to_object(object_str: str, locals: Dict[str, Any]) -> Any:
    """
    Turns a string representation of an object, e.g.,
    Index(num_categories=3, min_occ=5) into the actual object. Class object
    mappings must be passed into locals.
    """
    tree = ast.parse(object_str)
    call = tree.body[0].value  # type: ignore
    return parse_ast_node(call, locals)
