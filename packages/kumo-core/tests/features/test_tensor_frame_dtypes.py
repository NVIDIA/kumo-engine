# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from encoder_stats import fit_encoder_stats
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_frame import get_tf
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumoapi.encoder import (
    Datetime,
    Image,
    Index,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
)
from kumoapi.typing import Stype


def test_all_dtypes() -> None:
    """Tensor-frame encoding accepts all supported raw column dtypes."""
    df_dict = {}
    df_dict["test"] = pd.DataFrame(
        {
            "bool": [True, False, True],
            "int": [1, 2, 3],
            "byte": np.array([1, 2, 3], dtype="int8"),
            "int16": np.array([1, 2, 3], dtype="int16"),
            "int32": np.array([1, 2, 3], dtype="int32"),
            "int64": np.array([1, 2, 3], dtype="int64"),
            "float": [1.1, 2.2, 3.3],
            "float32": np.array([1.1, 2.2, 3.3], dtype="float32"),
            "float64": np.array([1.1, 2.2, 3.3], dtype="float64"),
            "string": ["foo", "bar", "baz"],
            "binary": [b"abc", b"def", b"ghi"],
            "image": ["foo", "bar", "baz"],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                ]
            ).date,
            "time": [
                datetime.time(12, 0),
                datetime.time(13, 30),
                datetime.time(14, 45),
            ],
            "timedelta": [
                pd.Timedelta(days=1),
                pd.Timedelta(hours=2),
                None,
            ],
            "floatlist": [[1.1, 2.2], [3.3, 4.4], [5.5]],
            "intlist": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "stringlist": [["a", "b"], ["c"], ["d", "e", "f"]],
            "unsupported": [3 + 4j, 3 + 4j, 3 + 4j],
        }
    )

    encoders = EncodersConfig(
        encoders={
            "test": {
                "bool": Index(),
                "int": Numerical(),
                "byte": Index(),
                "int16": Numerical(),
                "int32": Index(),
                "int64": Numerical(),
                "float": Numerical(),
                "float32": Index(),
                "float64": Numerical(),
                "string": MultiCategorical(),
                "binary": Index(),
                "image": Image(),
                "date": Datetime(),
                "time": Datetime(),
                "timedelta": Numerical(),
                "floatlist": MultiCategorical(),
                "intlist": NumericalList(),
                "stringlist": MultiCategorical(),
                "unsupported": Null(),
            }
        }
    )
    mappers = {
        name: TensorMapper.from_encoder(
            fit_encoder_stats(encoder, df_dict["test"][name])
        )
        for name, encoder in encoders["test"].items()
    }
    table = pa.table(
        {
            name: pa.array(df_dict["test"][name], from_pandas=True)
            for name, mapper in mappers.items()
            if mapper.stype != Stype.unsupported
        }
    )
    frame = get_tf(table, mappers)
    assert frame.num_rows == 3
