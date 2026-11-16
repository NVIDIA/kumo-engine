# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def brute_bin(value, edges):
    return sum(edge <= value for edge in edges)


def brute_histogram(values, edges, weights=None):
    counts = [0] * (len(edges) + 1)
    for i, value in enumerate(values):
        if np.isfinite(value):
            counts[brute_bin(value, edges)] += 1 if weights is None else (
                weights[i])
    return tuple(counts)
