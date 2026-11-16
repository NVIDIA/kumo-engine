# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model-independent contract: what the data means, what one sample is, what the model receives.

Three declarations, one per question:

* `schema` — what the relational data *means*: tables, keys, clocks, joins.
* `sample` — what one entity is allowed to know at one prediction time.
* `batch` — what the model actually receives: arrays, encodings, layouts, targets.

`DataLoaderSpec` (in `kumo.pytorch.spec`) binds all three into one serializable
artifact. `Dataset` says where the bytes are and `Engine` says what reads them;
both are deliberately outside this package, because the same contract must be
executable by more than one engine over more than one physical layout.

Import from `kumo.pytorch.api` for the supported public surface. This package is
the internal organization behind it.
"""

from kumo.pytorch.contract import batch, sample, schema

__all__ = ["batch", "sample", "schema"]
