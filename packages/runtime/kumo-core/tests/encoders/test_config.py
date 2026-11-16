# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from kumoapi.encoder import Index, Numerical


def test_encoders() -> None:
    encoders = EncodersConfig()

    encoders["TABLE_A"]["COL_1"] = Numerical()
    encoders["TABLE_A"]["COL_2"] = Index()
    encoders.target = TargetEncoderConfig(col_name="TARGET", encoder=Index())
    assert "TABLE_A" in encoders
    assert "TABLE_B" not in encoders

    assert list(encoders.keys()) == ["TABLE_A"]
    assert list(encoders.values()) == [{"COL_1": Numerical(), "COL_2": Index()}]
    assert list(encoders.items()) == [
        ("TABLE_A", {"COL_1": Numerical(), "COL_2": Index()})
    ]

    assert str(encoders) == (
        "EncodersConfig(\n"
        "  encoders={\n"
        "    TABLE_A: {\n"
        "      COL_1: Numerical(scaler=None, na_strategy=mean),\n"
        "      COL_2: Index(min_occ=1, na_strategy=separate),\n"
        "    },\n"
        "  },\n"
        "  target=(TARGET, Index(min_occ=1, na_strategy=separate)),\n"
        ")"
    )
