# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelStore


@contextmanager
def no_encoder_inference_warnings() -> Iterator[None]:
    # Only fit reports what inference chose: training tables do not use it,
    # and a trained model encodes columns as it was trained to.
    def drop(record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING

    inference = logging.getLogger('kumo_core.encoder_selection')
    inference.addFilter(drop)
    try:
        yield
    finally:
        inference.removeFilter(drop)


def require_embedding_models(store: EmbeddingModelStore,
                             feature_schema: dict[str, Any],
                             encoder_overrides: Optional[Any] = None) -> None:
    from kumoapi.encoder import EncoderType
    from pydantic import TypeAdapter

    # What training will use: the inferred encoders, then the overrides.
    adapter: TypeAdapter = TypeAdapter(EncoderType)
    inferred = {
        f"{table}.{spec['column_name']}": adapter.validate_python(
            spec["encoder"])
        for table, schema in feature_schema["node_types"].items()
        for spec in schema["column_specs"]
    }
    embedding_models.require_embedding_models(store, inferred,
                                              encoder_overrides)


def require_trained_embedding_models(store: EmbeddingModelStore,
                                     encoders: Any) -> None:
    # A trained model is matched by the identity of its models, not names.
    if encoders is not None:
        embedding_models.require_encoder_models(encoders, store, trained=True)
