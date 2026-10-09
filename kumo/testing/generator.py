# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""
This class defines a generic data generator that enables users to test varying
parts of the Kumo pipeline with a clear and easy-to-use graph specification.
Users can either use the `GraphSpec` dataclass to specify an entire
graph (with heterogeneous nodes and edges between them), or the
`NodeSpec` dataclass to generate a particular dataframe.

In order to set the random seed for generator.py, one can simply execute
torch_geometric.seed.seed_everything(seed: int) which sets the requisite
seeds for the random generators used in this class.
"""
import os.path as osp
import random
import string
import time
from abc import ABC, abstractclassmethod
from datetime import datetime
from functools import partial
from typing import Optional, Union

import numpy as np
import pandas as pd
from kumo_core.training.options import allow_array_targets
from kumo_core.types.conversions import dtype_to_numpy, dtype_to_pandas
from kumo_core.utils.mixins import CastMixin
from kumoapi.typing import Dtype
from pydantic import ValidationInfo, field_validator

from io_utils import fsutil, kumo_dataclass
from kumo.connector import ParquetConnector
from kumo.graph import Graph
from kumo.home import get_home_dir

# Random value generators #####################################################


def _bool_generator(n: int) -> pd.Series:
    return pd.Series(
        np.random.uniform(size=n) > 0.5, dtype=dtype_to_pandas(Dtype.bool))


def _int_generator(
    n: int,
    dtype: Dtype,
    low: int | None = None,
    high: int | None = None,
) -> pd.Series:
    np_dtype = dtype_to_numpy(dtype)
    dtype_info = np.iinfo(np_dtype)
    return pd.Series(
        np.random.randint(
            low=low or dtype_info.min,
            high=high or dtype_info.max,
            size=n,
            dtype=np_dtype,
        ), dtype=dtype_to_pandas(dtype))


def _float_generator(n: int, dtype: Dtype) -> pd.Series:
    return pd.Series(np.random.random(size=n).astype(dtype_to_numpy(dtype)))


def _datetime_generator(n: int, dtype: Dtype = Dtype.date) -> pd.Series:
    return pd.Series(np.random.randint(1, int(time.time()), size=n)).apply(
        lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d')).astype(
            dtype_to_pandas(dtype))


def _timedelta_generator(n: int) -> pd.Series:
    return pd.Series((_datetime_generator(n) - _datetime_generator(n)).astype(
        dtype_to_numpy(Dtype.timedelta)))


def _string_generator(n: int, size: int | None = 8):
    random_str = lambda: ''.join(random.choices(string.ascii_lowercase, k=size))
    return pd.Series(np.array([random_str() for _ in range(n)]))


def _category_generator(n: int, dtype: Dtype, distinct: int) -> pd.Series:
    if dtype == Dtype.bool:
        val_set = _bool_generator(n)
        return val_set
    else:
        val_set = DTYPE_TO_GENERATOR[dtype](distinct)
        if distinct == n:
            return val_set
        assert distinct < n
        return pd.concat([val_set,
                          val_set.sample(n - distinct,
                                         replace=True)]).reset_index(drop=True)


def _stringlist_generator(
    n: int,
    size: int | None = 8,
    distinct: int = 2,
) -> pd.Series:
    random_str = lambda: ''.join(random.choices(string.ascii_lowercase, k=size))
    distinct_strings = [random_str() for _ in range(distinct)]
    out = []
    for row in range(n):
        out.append(random.sample(distinct_strings, k=random.randint(1, 3)))

    return pd.Series(out)


def _multilabel_generator(
    n: int,
    distinct: int,
    max_labels: int,
    delimiter: str,
) -> pd.Series:
    assert max_labels > 1, ("max_labels must be greater than 1 as it is a "
                            "multilabel column")
    if allow_array_targets():
        data = _stringlist_generator(n, distinct=distinct)
        # Always have at least one empty list for testing:
        data.iloc[0] = []
        return data

    val_list = list(_string_generator(distinct))
    random_multilabel = lambda: delimiter.join(
        random.sample(val_list, k=random.randint(1, max_labels)))
    data = np.array([random_multilabel() for _ in range(n)])

    # Always have at least one empty string row for testing:
    data[0] = ""

    # Always have each label appear at least once:
    assert n >= 2
    data[1] = delimiter.join(val_list)

    return pd.Series(data)


DTYPE_TO_GENERATOR = {
    Dtype.bool: _bool_generator,
    Dtype.int: partial(_int_generator, dtype=Dtype.int),
    Dtype.byte: partial(_int_generator, dtype=Dtype.byte),
    Dtype.int16: partial(_int_generator, dtype=Dtype.int16),
    Dtype.int32: partial(_int_generator, dtype=Dtype.int32),
    Dtype.int64: partial(_int_generator, dtype=Dtype.int64),
    Dtype.float: partial(_float_generator, dtype=Dtype.float),
    Dtype.float32: partial(_float_generator, dtype=Dtype.float32),
    Dtype.float64: partial(_float_generator, dtype=Dtype.float64),
    Dtype.string: _string_generator,
    Dtype.date: partial(_datetime_generator, dtype=Dtype.date),
    Dtype.time: partial(_datetime_generator, dtype=Dtype.time),
    Dtype.timedelta: _timedelta_generator
}

# Implementations #############################################################

TableIdType = Union[int, str]


@kumo_dataclass
class ColumnSpec(CastMixin):
    r"""A specifiction of a generated column.

    Attributes:
        dtype (Dtype): the data type of the values in the column.
        na_frac (float, Optional): the fraction of NA values in the column.
        (default: 0).
        distinct_vals (int, Optional): the number of distinct values the column
            is allowed. (default: 0)
        max_labels (int, Optional): the maximum number of labels assigned to a
            mulitilabel column, assumed lower than distinct_vals. (default: 1)
        delimiter (str, Optional): delimiter used for multilabel columns.
            (default: '|')

    TODO: a distribution parameter that specifies the distribution of generated
    values.
    """
    dtype: Dtype
    na_frac: float | None = 0
    distinct_vals: int | None = 0
    max_labels: int | None = 1
    delimiter: str | None = "|"

    # pylint: disable=no-self-argument
    @field_validator('na_frac')
    @classmethod
    def na_frac_validator(cls, v):
        if v < 0 or v > 1:
            raise ValueError("'na_frac' must be between 0 and 1")
        return v

    # ``distinct_vals`` intentionally precedes ``max_labels`` because Pydantic
    # v2 exposes previously validated fields through ``info.data``.
    @field_validator('max_labels')
    @classmethod
    def multilabel_validator(cls, v, info: ValidationInfo):
        distinct_vals = info.data.get('distinct_vals')
        if distinct_vals and distinct_vals < v:
            raise ValueError(
                "'max_labels' must be lower or equal to 'distinct_vals'")
        return v


@kumo_dataclass
class EdgeSpec(CastMixin):
    r"""A specification of an edge between two or more tables.

    Attributes:
        density (float): the number of edges that are created between the
        tables. In particular,
            * density < 1: defines a PK/FK relationship between tables where
                some foreign keys do not point to a primary key. Edge must be
                between two tables.
            * density == 1: defines a PK/FK relationship with no dangling
                primary or foreign keys. Edge must be between two tables.
            * density > 1: defines a FK/FK relationship with multiple
                overlapping values between each pair of tables in the edge.
                The number of overlapping values is indicated by the density
                value; for example, density = 2 indicates both FKs have a value
                repeated twice, resulting in four edges (instead of 2).
    """
    density: float

    # pylint: disable=no-self-argument
    @field_validator('density')
    @classmethod
    def density_validator(cls, v):
        if v < 0:
            raise ValueError("'density' must be greater than 0")
        if v > 1 and not v.is_integer():
            raise ValueError("'density' values greater than 1 must be integral")
        return v


class TableSpec(CastMixin):
    r"""A specification of a table.

    Attributes:
        num_rows (int): the number of rows in the table.
        col_specs (List[ColumnSpec]): a list of the speficiations of each
            column.
    """
    num_rows: int
    col_specs: list[ColumnSpec]

    def __init__(self, num_rows, col_specs):
        self.num_rows = num_rows
        self.col_specs = [ColumnSpec.cast(x) for x in col_specs]

    def generate_feature_columns(self, data):
        for idx, col_spec in enumerate(self.col_specs):
            if col_spec.distinct_vals:
                if col_spec.max_labels > 1:
                    col_data = _multilabel_generator(self.num_rows,
                                                     col_spec.distinct_vals,
                                                     col_spec.max_labels,
                                                     col_spec.delimiter)
                else:
                    col_data = _category_generator(self.num_rows,
                                                   col_spec.dtype,
                                                   col_spec.distinct_vals)
            else:
                col_data = DTYPE_TO_GENERATOR[col_spec.dtype](self.num_rows)

            # Handle na_frac (not None or 0)
            if col_spec.na_frac:
                na_indices = np.random.choice(
                    col_data.size, replace=False,
                    size=int(col_data.size * col_spec.na_frac))
                col_data[na_indices] = pd.NA

            data[str(idx)] = col_data

    def generate_neighbor_lists(self, edge, table_id):
        out_neighbors = []
        in_neighbors = []
        for src_index, src_name in enumerate(edge[:-1]):
            for dst_name in edge[src_index + 1:]:
                is_src = (table_id == src_name)
                is_dst = (table_id == dst_name)
                if is_src:
                    out_neighbors.append(dst_name)
                if is_dst:
                    in_neighbors.append(src_name)
        return out_neighbors, in_neighbors

    def fill_fkey_data(self, fkey_data, edge_density, wrap):
        for val, i in enumerate(range(0, self.num_rows, int(edge_density))):
            fkey_data[i:i + int(edge_density)] = val % wrap

    def generate_foreign_keys(self, data, table_id, graph_spec):
        for idx, (edge, spec) in enumerate(graph_spec.edge_specs.items()):
            if table_id not in edge:
                continue

            edge_min_rows = min([
                graph_spec.table_specs[table_id].num_rows for table_id in edge
            ])
            edge_density = spec.density

            # wrap to be used in determining column data if density > 1
            wrap = edge_min_rows // edge_density

            # determine which pairs of tables defined by the edge contain
            # this table
            out_neighbors, in_neighbors = self.generate_neighbor_lists(
                edge, table_id)

            # from creates fk, to uses pk if possible
            for dst_name in out_neighbors:
                fkey_data = np.empty(shape=self.num_rows)
                has_fkey_data = False

                # Density < 1: source fkey data is edge_density% of the
                # minimum number of rows between the source and
                # destination tables
                if edge_density < 1:
                    dst_rows = min(graph_spec.table_specs[dst_name].num_rows,
                                   self.num_rows)
                    end_index = int(edge_density * dst_rows)
                    fkey_data[:end_index] = np.arange(end_index)
                    fkey_data[end_index:] = np.nan
                    has_fkey_data = True

                # Density == 1: source fkey data is all the rows it has
                if edge_density == 1:
                    fkey_data[:] = np.arange(self.num_rows)
                    has_fkey_data = True

                # if a Density <= 1 case results in adding fkey data:
                if has_fkey_data:
                    data[f'fkey_{idx}'] = fkey_data

                # Density > 1: repeat values based on input density.
                # Wrap by the minimum number of rows / the density
                # so we can deal with tables of different numbers
                # of rows. Creates bicliques K_(d,d) across vertices
                # from two tables.
                if edge_density > 1:
                    self.fill_fkey_data(fkey_data, edge_density, wrap)
                    # Add density > 1 fkey data
                    key_col_name = f'fkey_{idx}'

                    # For a hyper-edge specification, every table
                    # requires a source column aside from the final
                    # table in the specification
                    if len(edge) > 2:
                        key_col_name = (key_col_name + f'__src_to__{dst_name}')

                    data[key_col_name] = fkey_data

            if len(in_neighbors) > 0 and edge_density > 1:
                fkey_data = np.empty(shape=self.num_rows)
                self.fill_fkey_data(fkey_data, edge_density, wrap)
                key_col_name = f'fkey_{idx}'
                if len(edge) > 2:
                    key_col_name = key_col_name + '_dst'
                data[key_col_name] = fkey_data

    def generate(
        self,
        table_id: TableIdType | None = None,
        graph_spec: Optional['GraphSpec'] = None,
        pkey_as_index: bool | None = True,
    ):
        r"""Generates a dataframe representing a table that matches the table
        specification, and (if provided) the graph specification."""
        data = {}

        # Primary key
        index_col = np.arange(self.num_rows)
        data['index'] = index_col

        # Feature columns
        self.generate_feature_columns(data)

        # Foreign keys, as needed
        if graph_spec:
            assert table_id is not None
            self.generate_foreign_keys(data, table_id, graph_spec)

        df = pd.DataFrame(data)

        if pkey_as_index:
            df.set_index('index', inplace=True)
        # fkeys get generated as float initially b/c of NaN's;
        # convert to int to match pkeys
        for col in df.columns:
            if df.dtypes[col] == np.float64 or 'fkey' in col:
                df[col] = df[col].astype("Int64")
        return df


class GraphSpec:
    r"""A specification of a graph.

    Attributes:
        table_specs (Dict[TableIdType, TableSpec]): a mapping from a table
            ID (either an int or a string) to a table specification.
        edge_specs (Dict[Tuple[TableIdType, TableIdType]], EdgeSpec): a mapping
            from a tuple of Table IDs to an edge specification for the
            corresponding relationship between the tables. n-tuples for n > 2
            specify hyperedges which are interpretted as n-cliques.
    """
    table_specs: dict[TableIdType, TableSpec]
    edge_specs: dict[tuple[TableIdType, TableIdType], EdgeSpec]

    def __init__(self, table_specs, edge_specs):
        self.table_specs = {
            _safe_id(k): TableSpec.cast(v)
            for k, v in table_specs.items()
        }
        self.edge_specs = {
            tuple(map(_safe_id, k)): EdgeSpec.cast(v)
            for k, v in edge_specs.items()
        }

    @property
    def num_tables(self) -> int:
        return len(self.table_specs)

    def generate(self, name: str) -> Graph:
        r"""Generates a kumo.Graph from a specification."""
        root_dir = osp.join(get_home_dir(), 'graphs', name)
        fsutil.makedirs(root_dir, exist_ok=True)

        parquet_dir = osp.join(root_dir, 'generated_tables')
        fsutil.makedirs(parquet_dir, exist_ok=True)

        for table_id, table_spec in self.table_specs.items():
            df = table_spec.generate(table_id=table_id, graph_spec=self)
            df.reset_index(inplace=True)
            df.to_parquet(osp.join(parquet_dir, f'{table_id}.parquet'))
            del df

        graph = Graph(name=name)
        graph.connect(ParquetConnector('parquet', parquet_dir))

        for table_id in self.table_specs:
            # Adding foreign keys is done via graph.link
            graph.add(str(table_id), 'parquet', pkey='index')

        for idx, edge in enumerate(self.edge_specs):
            edge_density = self.edge_specs[edge].density
            if len(edge) > 2 and edge_density <= 1:
                raise ValueError(f"""'density' must be > 1 for hyperedges
                    (was {edge_density} for edge {edge}).""")

            for src_index, src_name in enumerate(edge[:-1]):
                for dst_name in edge[src_index + 1:]:

                    # Default src/dest keys for normal edge
                    src_key = f'fkey_{idx}'
                    dst_key = None
                    if edge_density > 1:
                        dst_key = f'fkey_{idx}'

                    # Hyperedge specifics appended to keys
                    if len(edge) > 2:
                        src_key = src_key + f'__src_to__{dst_name}'
                        dst_key = dst_key + '_dst'
                    graph.link(src_table=str(src_name), src_key=src_key,
                               dst_table=str(dst_name), dst_key=dst_key)

        # Use information from specification to set column types in the graph
        for table_id, table_spec in self.table_specs.items():
            for col_id, col_spec in enumerate(table_spec.col_specs):
                # TODO (matthias) Add support for semantic types.
                graph[str(table_id)].add_col(str(col_id), dtype=col_spec.dtype)

        return graph


def _safe_id(id: TableIdType) -> str:
    # Table IDs may not begin with numbers, so prefix numerical IDs with a
    # string prefix
    if isinstance(id, int):
        return "t" + str(id)
    return id


class PDSeriesGenerator(ABC):
    @abstractclassmethod
    def generate_series(self, num_rows: int) -> pd.Series:
        """generate a pd.Series which is used for parameterized synthetic
        data generation.

        Raises:
            NotImplementedError: _description_

        Returns:
            pd.Series: _description_
        """
        raise NotImplementedError


class CategoricalData(PDSeriesGenerator):
    def __init__(self, vocab_size: int, width: int,
                 probs: list[int] | None = None):
        self.width = width
        self.vocab_size = vocab_size
        assert self.width > 0 and self.vocab_size > 0
        self.vocab = self._generate_vocab()
        self.probs = probs

    def generate_series(self, num_rows) -> pd.Series:
        return pd.Series(
            random.choices(self.vocab, weights=self.probs, k=num_rows))

    def _generate_vocab(self) -> list[str]:
        characters = string.ascii_letters
        return [
            ''.join(random.choice(characters) for _ in range(self.width))
            for _ in range(self.vocab_size)
        ]
