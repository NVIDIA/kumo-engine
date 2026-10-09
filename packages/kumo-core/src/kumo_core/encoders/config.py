# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from dataclasses import field

import pandas as pd
from kumoapi.encoder import Encoder, EncoderType, Index, MultiCategorical
from kumoapi.typing import ColStatType, Stype
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

    @classmethod
    def from_stypes(
        cls,
        df_dict: dict[str, pd.DataFrame],
        table_stype_dict: dict[str, dict[str, Stype]],
        train_table_name: str | None = None,
        target_col_name: str | None = None,
        target_stype: Stype | None = None,
    ) -> 'EncodersConfig':
        from kumo_core.encoders.selection import infer_encoder

        out = cls()

        for table_name, stype_dict in table_stype_dict.items():
            df = df_dict[table_name]
            out.encoders[table_name] = {}
            for col_name, stype in stype_dict.items():
                encoder = infer_encoder(df[col_name], stype)
                assert isinstance(encoder, EncoderType)
                out.encoders[table_name][col_name] = encoder

        target_encoder: Encoder | None = None
        if target_col_name is not None:
            assert train_table_name is not None
            assert target_stype is not None

            if target_stype == Stype.numerical:
                ser = df_dict[train_table_name][target_col_name]
                target_encoder = infer_encoder(ser, target_stype)
            elif target_stype in {Stype.categorical, Stype.ID}:
                target_encoder = Index()
            elif target_stype == Stype.multicategorical:
                target_encoder = MultiCategorical(na_strategy='zero')
                target_encoder._stats[
                    ColStatType.MULTI_CATEGORIES_SEPARATOR] = '|'
            else:
                raise NotImplementedError(f"Unsupported target semantic type "
                                          f"'{target_stype}'")

        if target_encoder is not None:
            assert target_col_name is not None
            out.target = TargetEncoderConfig(target_col_name, target_encoder)

        return out

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
            table_name:
            '\n'.join([
                f'      {col_name}: {encoder},'
                for col_name, encoder in encoders.items()
            ])
            for table_name, encoders in self.items()
        }
        encoders_repr = '\n'.join([
            f'    {table_name}: {{\n{encoders}\n    }},\n'
            for table_name, encoders in col_encoders_repr.items()
        ])
        target_repr = None
        if self.target is not None:
            target_repr = f'({self.target.col_name}, {self.target.encoder})'

        return (f'{self.__class__.__name__}(\n'
                f'  encoders={{\n'
                f'{encoders_repr}'
                f'  }},\n'
                f'  target={target_repr},\n'
                f')')
