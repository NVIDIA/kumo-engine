# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What test.kumo_pytorch.reference_engine.gpu does not build yet, declared in one list, as upstream Kumo PyTorch declares its DiskGraph engine's.

Every entry is refused with its code and next step by upstream's `check` and by `test.kumo_pytorch.reference_engine.gpu.prepare`, before any
data is read (`test.kumo_pytorch.reference_engine.gpu.plan.check._Engine`: the sources bound, then the parts of the contract). The code is the
`ValidationIssue.code` of each refusal. docs/reference/refusals.md lists the entries.

The list is plain data here, built into upstream's own `Unsupported` objects (`declared`) when first asked
for, so importing this module needs neither Kumo PyTorch nor a GPU library.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

__all__ = ["ENTRIES", "declared"]

#: (code, what is refused, what to do instead), all refused by `check()`, before any data is read.
ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("GPU-SOURCE-KIND",
     "Tables and anchor tables must be bound to Parquet files; other sources, such as `InMemoryRows`, are "
     "refused.",
     "write the rows to local Parquet files and bind them with `Parquet(path)`"),
    ("GPU-REMOTE-PARQUET",
     "Remote Parquet paths, such as `s3://`, are not read yet, for tables or anchor tables.",
     "copy the files to a local disk and bind the local path"),
    ("GPU-REPEATED-FIELD",
     "Repeated field values, a `Field` with `selection=\"all\"`, are not built yet.",
     "read one value (`selection=\"one\"` or `\"latest\"`), or aggregate the rows"),
    ("GPU-CANDIDATE-SETS",
     "Candidate sets (`CandidateSetSpec`, `CandidateSource` inputs, `CandidateTarget` targets) are not built "
     "yet.",
     "leave them out of the contract"),
    ("GPU-DATED-PATH",
     "Dated values (`DatedEntityValue`) through more than one relation are not built yet.",
     "date the entity's own rows, or one relation's"),
    ("GPU-DATED-FIELD",
     "Dated values whose field is read through relations, on another clock or with another selection are not "
     "built yet.",
     "date a column of the rows the dated value selects"),
    ("GPU-DATED-TYPE",
     "Dated values of types other than integers, text, floats and booleans are not built yet.",
     "date an integer, text, float or boolean column"),
    ("GPU-DECIMAL-COMPOSITE",
     "Raw decimal components of a `Composite` row are not built yet.",
     "encode the decimal component (Buckets, ZScore), or store it as a float"),
    ("GPU-DECIMAL-TARGET",
     "Threshold labels and matching groups of decimal values are not built yet.",
     "store the values as floats"),
    ("GPU-TARGET-ENCODING",
     "Target encodings other than Raw, Vocabulary, ZScore and Buckets (Text) are not built yet.",
     "encode the target with Raw, Vocabulary, ZScore or Buckets"),
    ("GPU-OTHER",
     "A kind of value, input, representation, row, encoding or target this backend has no builder for, such "
     "as one a newer upstream Kumo PyTorch adds, is refused by name.",
     "leave it out of the contract"),
)


@cache
def declared() -> Mapping[str, Any]:
    """Each entry as upstream's `Unsupported`, by code (the form `DiskGraphEngine.unsupported` has)."""

    from types import MappingProxyType

    from kumo.pytorch.engines.unsupported import Unsupported

    return MappingProxyType({code: Unsupported(code, description, next_step)
                             for code, description, next_step in ENTRIES})
