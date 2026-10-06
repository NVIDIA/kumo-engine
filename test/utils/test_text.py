# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.utils.text import Tokenizer


def test_tokenizer_remains_service_independent():
    assert Tokenizer()("Hello, world!") == ["hello", "world"]
