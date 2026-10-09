# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from kumoapi.encoder import Datetime
from torch import Tensor
from torch_frame import TensorFrame

from kumo_core.encoders.config import EncodersConfig
from kumo_core.features.stores.feature import FeatureStore, FeatureStoreAttr, FeatureStoreTableSchema, FeatureStoreTrainingTableSchema
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.encoders.selection import compute_missing_stats_
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.types.features import FeatureStoreData
from kumo_core.encoders.index_mapping import Mapper
from kumo_core.features.time import to_datetime


class InMemoryFeatureStore(FeatureStore):
    r"""An in-memory implementation of a feature store.

    Args:
        df_dict: A dictionary of raw data frames.
        table_schema_dict: A dictionary describing the schema of each table in
            the graph.
        train_table_schema: The schema of the training table.
    """
    def __init__(
        self,
        df_dict: dict[str, pd.DataFrame],
        table_schema_dict: Mapping[
            str,
            FeatureStoreTableSchema | dict[str, Any],
        ],
        train_table_schema: FeatureStoreTrainingTableSchema | dict[str, Any]
        | None = None,
        pmm: bool = True,
    ) -> None:
        super().__init__()

        self._table_schema_dict = {
            table_name: FeatureStoreTableSchema.cast(table_schema)
            for table_name, table_schema in table_schema_dict.items()
        }
        self._train_table_schema = FeatureStoreTrainingTableSchema.cast(
            train_table_schema)
        self._pmm = pmm

        # Collect tensor data #################################################

        self._data: dict[str, dict[str, FeatureStoreData]] = {}

        for table_name, table_schema in self._table_schema_dict.items():
            self._data[table_name] = _get_table_data(
                schema=table_schema,
                df=df_dict[table_name],
                pmm=pmm,
            )

        self._entity_table_mappings: list[Mapper] = []
        if self._train_table_schema is not None:
            for _, dst_table_name in self._train_table_schema.fkeys:
                table_schema = self._table_schema_dict[dst_table_name]
                assert table_schema.pkey_name is not None
                mapping = Mapper(
                    df_dict[dst_table_name][table_schema.pkey_name])
                self._entity_table_mappings.append(mapping)

            self.set_train_table(df_dict[self._train_table_schema.table_name])

    @staticmethod
    def sanitize(
        df_dict: dict[str, pd.DataFrame],
        pkey_name_dict: dict[str, str],
        time_col_dict: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        r"""Sanitizes raw data according to table schema definition.

        In particular, it:
        * drops duplicate primary keys
        * removes rows with missing primary keys or time values
        """
        # Drop duplicated primary keys:
        for table_name, pkey_name in pkey_name_dict.items():
            df = df_dict[table_name]
            df = df.drop_duplicates(subset=pkey_name)
            df = df.dropna(subset=pkey_name)
            df_dict[table_name] = df

        # Collect subset of columns we need to sanitize against N/A values:
        na_cols_dict = defaultdict(list)
        for table_name, pkey_name in pkey_name_dict.items():
            na_cols_dict[table_name].append(pkey_name)
        for table_name, time_col in time_col_dict.items():
            na_cols_dict[table_name].append(time_col)

        for table_name, na_cols in na_cols_dict.items():
            df = df_dict[table_name]
            df = df.dropna(how='any', subset=na_cols)
            df_dict[table_name] = df

        return df_dict

    def _get_tensor(self, attr: FeatureStoreAttr) -> FeatureStoreData | None:
        if attr.attr_name == 'tf':
            return self._get_tf(attr.table_name, attr.index)

        if attr.attr_name == 'df':
            return self._get_df(attr.table_name, attr.index)

        value = self._data[attr.table_name].get(attr.attr_name, None)

        if value is None:
            return None

        if attr.index is None:
            return value

        if isinstance(value, np.ndarray) and isinstance(attr.index, Tensor):
            return value[attr.index.numpy()]

        return value[attr.index]

    def _get_tf(
        self,
        table_name: str,
        index: Tensor | slice | Any | None,
    ) -> TensorFrame:
        r"""Returns the encoded tensor frame of table ``table_name`` at
        specified indices ``index``.
        """
        if self._pmm:
            tf = self._data[table_name]['tf']
            return tf if index is None else tf[index]

        df = self._get_df(table_name, index)

        if table_name == self.train_table_name:
            mapper_dict = self._train_table_schema.mapper_dict
        else:
            mapper_dict = self._table_schema_dict[table_name].mapper_dict

        return get_tf(df, mapper_dict)

    def _get_df(
        self,
        table_name: str,
        index: Tensor | slice | Any | None,
    ) -> pd.DataFrame:
        r"""Returns the raw data frame of table ``table_name`` at specified
        indices ``index``.
        """
        if not self._pmm:
            df = self._data[table_name]['df']
            if isinstance(index, Tensor):
                df = df.iloc[index.numpy()]
            elif index is not None:
                df = df[index]
            df = df.reset_index(drop=True)

            if table_name == self.train_table_name:
                mapper_dict = self._train_table_schema.mapper_dict
            else:
                mapper_dict = self._table_schema_dict[table_name].mapper_dict
            length = len(df)
            df = df[list(mapper_dict.keys())]
            data = df.to_dict(orient='series')
        else:
            tf = self._get_tf(table_name, index)
            length = len(tf)

            data = {}
            for stype, feats in tf.feat_dict.items():
                # Iterate over stype-wise encoded tensor data and map to
                # original raw values via `TensorMapper.backward`:
                for i, col_name in enumerate(tf.col_names_dict[stype]):
                    mapper = self.get_tensor_mapper(table_name, col_name)
                    data[col_name] = mapper.backward(feats[:, i])

        time_col_name = self.get_time_col_name(table_name)
        if time_col_name is not None and time_col_name not in data:
            attr = FeatureStoreAttr(table_name, 'time', index)
            time = self._get_tensor(attr)
            assert isinstance(time, Tensor)
            # `time` cannot hold N/A values, so no further processing needed.
            data[time_col_name] = to_datetime(time.numpy())

        end_time_col_name = self.get_end_time_col_name(table_name)
        if end_time_col_name is not None and end_time_col_name not in data:
            attr = FeatureStoreAttr(table_name, 'end_time', index)
            end_time = self._get_tensor(attr)
            assert isinstance(end_time, Tensor)
            ser = to_datetime(end_time.numpy())
            ser[end_time.numpy() == -1] = pd.NaT  # Handle N/A gracefully.
            data[end_time_col_name] = ser

        return pd.DataFrame(data, index=pd.RangeIndex(0, length))

    def _get_tensor_size(
        self,
        attr: FeatureStoreAttr,
    ) -> tuple[int, ...] | None:
        return (self._data[attr.table_name]['num_nodes'], )

    @property
    def table_names(self) -> list[str]:
        return list(self._table_schema_dict.keys())

    @property
    def train_table_name(self) -> str | None:
        if self._train_table_schema is None:
            return None
        return self._train_table_schema.table_name

    @property
    def entity_table_names(self) -> list[str]:
        return [dst_table for _, dst_table in self._train_table_schema.fkeys]

    @property
    def train_fkey_names(self) -> list[str]:
        return [fkey for fkey, _ in self._train_table_schema.fkeys]

    def get_pkey_name(self, table_name: str) -> str | None:
        return self._table_schema_dict[table_name].pkey_name

    def get_time_col_name(self, table_name: str) -> str | None:
        if (self._train_table_schema is not None
                and self._train_table_schema.table_name == table_name):
            return self._train_table_schema.time_col_name
        if table_name not in self._table_schema_dict:
            return None
        return self._table_schema_dict[table_name].time_col_name

    def get_end_time_col_name(self, table_name: str) -> str | None:
        if table_name not in self._table_schema_dict:
            return None
        return self._table_schema_dict[table_name].end_time_col_name

    @property
    def has_train_table_test_mask(self) -> bool:
        if self._train_table_schema is None:
            return False
        return self._train_table_schema.test_mask_name is not None

    @property
    def has_train_table_target_col(self) -> bool:
        if self._train_table_schema is None:
            return False
        return self._train_table_schema.target_mapper is not None

    @property
    def has_train_table_weight_col(self) -> bool:
        if self._train_table_schema is None:
            return False
        return self._train_table_schema.weight_col_name is not None

    def get_node_id(self, table_name: str, pkey: np.ndarray) -> Tensor:
        if self._table_schema_dict[table_name].pkey_name is None:
            raise KeyError(f"Table '{table_name}' does not have primary keys")

        index_map = self._data[table_name]['pkey_map']

        try:
            pkey = pkey.astype(index_map.index.dtype)
        except ValueError as e:
            raise ValueError(f"Could not cast primary keys {pkey.tolist()} to "
                             f"the expected '{index_map.index.dtype}' data "
                             f"type.") from e

        try:
            return torch.from_numpy(index_map.loc[pkey]['arange'].to_numpy())
        except KeyError as e:
            missing = ~np.isin(pkey, index_map.index)
            raise KeyError(f"The primary keys {pkey[missing].tolist()} do not "
                           f"exist in the '{table_name}' table.") from e

    @classmethod
    def from_encoders(
        cls,
        df_dict: dict[str, pd.DataFrame],
        encoders: EncodersConfig,
        pkey_name_dict: dict[str, str],
        time_col_dict: dict[str, str],
        end_time_col_dict: dict[str, str],
        train_table_name: str | None = None,
        train_fkeys: list[tuple[str, str]] | None = None,
        train_mask_names: tuple[str, str, str | None] | None = None,
        train_weight_col_name: str | None = None,
        pmm: bool = True,
    ) -> 'InMemoryFeatureStore':
        for table_name, encoder_dict in encoders.items():
            for col_name, encoder in encoder_dict.items():
                df = df_dict[table_name]
                if isinstance(encoder, Datetime):
                    df[col_name] = to_datetime(df[col_name])
                compute_missing_stats_(encoder, df[col_name])

        for table_name, time_col in time_col_dict.items():
            df = df_dict[table_name]
            df[time_col] = to_datetime(df[time_col])

        for table_name, end_time_col in end_time_col_dict.items():
            df = df_dict[table_name]
            df[end_time_col] = to_datetime(df[end_time_col])

        if encoders.target is not None:
            assert train_table_name is not None
            compute_missing_stats_(
                encoders.target.encoder,
                df_dict[train_table_name][encoders.target.col_name],
            )

        table_schema_dict = {}
        for table_name, encoder_dict in encoders.items():
            if table_name == train_table_name:
                continue

            table_schema = FeatureStoreTableSchema(
                mapper_dict={
                    col_name: TensorMapper.from_encoder(encoder)
                    for col_name, encoder in encoder_dict.items()
                },
                pkey_name=pkey_name_dict.get(table_name),
                time_col_name=time_col_dict.get(table_name),
                end_time_col_name=end_time_col_dict.get(table_name),
            )
            table_schema_dict[table_name] = table_schema

        target_mapper = None
        if encoders.target is not None:
            target_mapper = (
                encoders.target.col_name,
                TensorMapper.from_encoder(encoders.target.encoder),
            )

        train_table_schema: FeatureStoreTrainingTableSchema | None = None
        if train_table_name is not None:
            assert train_fkeys is not None
            assert train_mask_names is not None
            train_table_schema = FeatureStoreTrainingTableSchema(
                table_name=train_table_name,
                fkeys=train_fkeys,
                train_mask_name=train_mask_names[0],
                val_mask_name=train_mask_names[1],
                test_mask_name=train_mask_names[2],
                mapper_dict={
                    col_name: TensorMapper.from_encoder(encoder)
                    for col_name, encoder in
                    encoders[train_table_name].items()
                },
                target_mapper=target_mapper,
                time_col_name=time_col_dict.get(train_table_name),
                weight_col_name=train_weight_col_name,
            )

        return cls(df_dict, table_schema_dict, train_table_schema, pmm)

    def set_train_table(self, df: pd.DataFrame) -> FeatureStore:
        assert self._train_table_schema is not None
        train_table_name = self._train_table_schema.table_name
        self._data[train_table_name] = _get_train_table_data(
            schema=self._train_table_schema,
            df=df,
            entity_table_mappings=self._entity_table_mappings,
            pmm=self._pmm,
        )
        return self

    def get_tensor_mapper(
        self,
        table_name: str,
        col_name: str,
    ) -> TensorMapper:
        if table_name == self.train_table_name:
            return self._train_table_schema.mapper_dict[col_name]
        return self._table_schema_dict[table_name].mapper_dict[col_name]

    def get_target_tensor_mapper(self) -> TensorMapper | None:
        if self._train_table_schema is None:
            return None
        if self._train_table_schema.target_mapper is None:
            return None
        return self._train_table_schema.target_mapper[1]


# Helpers #####################################################################


def _get_table_data(
    schema: FeatureStoreTableSchema,
    df: pd.DataFrame,
    pmm: bool = True,
) -> dict[str, FeatureStoreData]:

    data: dict[str, Any] = dict(num_nodes=len(df))

    if pmm:
        data['tf'] = get_tf(df, schema.mapper_dict)
    else:
        data['df'] = df.reset_index(drop=True)

    if schema.pkey_name is not None:
        pkey = df[schema.pkey_name].to_numpy()
        data['pkey'] = pkey
        data['pkey_map'] = pd.DataFrame(
            dict(arange=range(len(pkey))),
            index=pkey,
        )

    if schema.time_col_name is not None:  # Guaranteed to have no `NaN` values.
        ser = to_datetime(df[schema.time_col_name])
        data['time'] = torch.from_numpy(ser.astype(int).to_numpy())

    if schema.end_time_col_name is not None:
        # NOTE `NaN` values are given as `-1` to encode them as integers.
        ser = to_datetime(df[schema.end_time_col_name])
        mask = torch.from_numpy(ser.isna().to_numpy())
        data['end_time'] = torch.from_numpy(ser.astype(int).to_numpy())
        data['end_time'][mask] = -1

    return data


def _get_train_table_data(
    schema: FeatureStoreTrainingTableSchema,
    df: pd.DataFrame,
    entity_table_mappings: list[Mapper],
    pmm: bool = True,
) -> dict[str, FeatureStoreData]:

    data: dict[str, Any] = dict(num_nodes=len(df))

    if pmm:
        data['tf'] = get_tf(df, schema.mapper_dict)
    else:
        data['df'] = df.reset_index(drop=True)

    if schema.time_col_name is not None:  # Guaranteed to have no `NaN` values.
        ser = to_datetime(df[schema.time_col_name])
        data['time'] = torch.from_numpy(ser.astype(int).to_numpy())

    fkeys = []
    for fkey, mapping in zip(schema.fkeys, entity_table_mappings):
        fkey_col_name, dst_table_name = fkey
        fkeys.append(mapping.map(df[fkey_col_name]))
    data['fkey'] = torch.stack(fkeys, dim=-1)

    if schema.train_mask_name in df:
        mask = torch.from_numpy(df[schema.train_mask_name].to_numpy())
        data['train_mask'] = mask.bool()

    if schema.val_mask_name in df:
        mask = torch.from_numpy(df[schema.val_mask_name].to_numpy())
        data['val_mask'] = mask.bool()

    if schema.test_mask_name is not None and schema.test_mask_name in df:
        mask = torch.from_numpy(df[schema.test_mask_name].to_numpy())
        data['test_mask'] = mask.bool()

    if schema.target_mapper is not None:
        col_name, mapper = schema.target_mapper
        if col_name in df:
            data['y'] = mapper(df[col_name])

    if schema.weight_col_name is not None and schema.weight_col_name in df:
        weight = torch.from_numpy(df[schema.weight_col_name].to_numpy())
        data['weight'] = weight.float()

    return data
