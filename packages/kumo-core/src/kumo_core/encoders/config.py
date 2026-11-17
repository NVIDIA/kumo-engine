# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from dataclasses import field

from kumoapi.encoder import EncoderType
from pydantic.dataclasses import dataclass


@dataclass
class TargetEncoderConfig:
    col_name: str
    encoder: EncoderType


@dataclass(repr=False)
class EncodersConfig:
    # Mapping from table and column name to the encoder:
    encoders: dict[str, dict[str, EncoderType]] = field(default_factory=dict)

    # The target encoder of the training table (optional):
    target: TargetEncoderConfig | None = None

    def __getitem__(self, table_name: str) -> dict[str, EncoderType]:
        if table_name not in self.encoders:
            self.encoders[table_name] = {}
        return self.encoders[table_name]

    def __setitem__(
        self,
        table_name: str,
        encoders: dict[str, EncoderType],
    ) -> None:
        self.encoders[table_name] = encoders

    def __contains__(
        self,
        table_name: str,
    ) -> bool:
        return table_name in self.encoders

    def keys(self) -> Iterable[str]:
        return self.encoders.keys()

    def values(self) -> Iterable[dict[str, EncoderType]]:
        return self.encoders.values()

    def items(self) -> Iterable[tuple[str, dict[str, EncoderType]]]:
        return self.encoders.items()

    def __repr__(self) -> str:
        col_encoders_repr = {
            table_name: "\n".join(
                [
                    f"      {col_name}: {encoder},"
                    for col_name, encoder in encoders.items()
                ]
            )
            for table_name, encoders in self.items()
        }
        encoders_repr = "\n".join(
            [
                f"    {table_name}: {{\n{encoders}\n    }},\n"
                for table_name, encoders in col_encoders_repr.items()
            ]
        )
        target_repr = None
        if self.target is not None:
            target_repr = f"({self.target.col_name}, {self.target.encoder})"

        return (
            f"{self.__class__.__name__}(\n"
            f"  encoders={{\n"
            f"{encoders_repr}"
            f"  }},\n"
            f"  target={target_repr},\n"
            f")"
        )
