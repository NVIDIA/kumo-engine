# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed Arrow values and calculations shared by the arena query backends."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from kumo.pytorch.contract.dtypes import DType, arrow_float64
from kumo.pytorch.contract.sample import (
    CalendarPart,
    MapValues,
    _typed_map_groups,
    evaluate_derived,
)


def typed_array(values, dtype):
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if dtype.name == "timestamp" and pa.types.is_timestamp(values.type):
        # Stored fractional instants round up; request cutoffs round down.
        if values.type.unit != "s":
            factor = {"ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}[
                values.type.unit
            ]
            ticks = pc.cast(values, pa.int64())
            quotient = pc.divide(ticks, factor)
            remainder = pc.subtract(ticks, pc.multiply(quotient, factor))
            values = pc.add(quotient, pc.cast(pc.greater(remainder, 0), pa.int64()))
        return pc.cast(values, dtype.to_arrow())
    return pc.cast(values, dtype.to_arrow(), safe=True)


def derived_array(spec, values, *, path):
    operands = [values[name] for name in spec.inputs]
    operation = spec.operation
    try:
        if spec.dtype.name.startswith("decimal") and operation in (
            "add",
            "subtract",
            "multiply",
            "abs",
        ):
            types = {name: DType.from_arrow(values[name].type) for name in spec.inputs}
            return pa.array(
                [
                    evaluate_derived(
                        spec,
                        dict(zip(spec.inputs, row, strict=True)),
                        path=path,
                        input_types=types,
                    )
                    for row in zip(*(v.to_pylist() for v in operands), strict=True)
                ],
                type=spec.dtype.to_arrow(),
            )
        if isinstance(operation, CalendarPart):
            result = getattr(
                pc, {"weekday": "day_of_week"}.get(operation.part, operation.part)
            )(operands[0])
        elif isinstance(operation, MapValues):
            fallback = operation.fallback if operation.unknown == "fallback" else None
            result = pa.repeat(
                pa.scalar(fallback, type=spec.dtype.to_arrow()), len(operands[0])
            )
            known = pc.is_null(operands[0])
            for output, members in _typed_map_groups(
                operation, DType.from_arrow(operands[0].type)
            ):
                selected = pc.fill_null(
                    pc.is_in(
                        operands[0], value_set=pa.array(members, type=operands[0].type)
                    ),
                    False,
                )
                known = pc.or_(known, selected)
                result = pc.if_else(
                    selected, pa.scalar(output, type=spec.dtype.to_arrow()), result
                )
            if operation.unknown == "error" and not pc.all(known).as_py():
                raise ValueError("unmapped value in MapValues")
            result = pc.if_else(pc.is_valid(operands[0]), result, None)
        elif operation in ("add", "subtract", "multiply"):
            operands = [
                pc.cast(
                    arrow_float64(v)
                    if spec.dtype.name.startswith("float")
                    and pa.types.is_decimal(v.type)
                    else v,
                    spec.dtype.to_arrow(),
                    safe=spec.dtype.integer,
                )
                for v in operands
            ]
            result = getattr(pc, operation + "_checked")(*operands)
        elif operation == "abs":
            result = pc.abs_checked(typed_array(operands[0], spec.dtype))
        elif operation == "sign":
            result = pc.sign(operands[0])
        elif operation == "divide":
            left, right = map(arrow_float64, operands)
            result = pc.divide(left, pc.if_else(pc.equal(right, 0), None, right))
        elif operation == "log1p":
            value = arrow_float64(operands[0])
            result = pc.log1p(pc.if_else(pc.less_equal(value, -1), None, value))
        else:
            raise ValueError(f"unsupported derived operation {operation!r}")
        return typed_array(result, spec.dtype)
    except (pa.ArrowException, ValueError, OverflowError) as error:
        raise ValueError(f"{path}: {error}") from error
