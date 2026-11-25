# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa

from kumo.pytorch.engines.graph_projection import GraphProjectionCache


def test_graph_projection_duplicate_overlapping_colliding_and_null_nodes():
    cache = GraphProjectionCache()
    vec = pa.list_(pa.float32(), 2)
    rows = {0: ([1, 2], 0), 1: (None, 10), 2: ([2, 4], None), 8192: ([8, 9], 20)}
    calls = []

    def read(ids):
        calls.append(ids)
        return {
            "vec": pa.array([rows[i][0] for i in ids], type=vec),
            "key": pa.array([rows[i][1] for i in ids], type=pa.int64()),
        }

    def check(ids):
        out = cache.read("pins", ids, read)
        assert out["vec"].to_pylist() == [rows[i][0] for i in ids]
        assert out["key"].to_pylist() == [rows[i][1] for i in ids]
        return out

    check([8192, 1, 0, 1])
    assert calls == [[0, 1, 8192]]
    check([0, 1])
    assert len(calls) == 1
    check([0, 2, 8192, 1, 2])
    assert calls[-1] == [2, 8192]
    returned = check([8192, 1, 2])
    # A torch tensor sharing this Arrow buffer may be modified by its consumer.
    buf = returned["vec"].values.buffers()[1]
    np.frombuffer(buf, dtype=np.float32)[0] = 99
    check([8192, 2, 1, 8192])
    assert len(calls) == 2
