# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Optional

from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelStore


@contextmanager
def no_encoder_inference_warnings(
        overridden: Optional[Collection[str]] = None) -> Iterator[None]:
    # Only fit reports what inference chose: training tables do not use it,
    # and a trained model encodes columns as it was trained to. Given the
    # columns whose encoders the model plan overrides, fit stays silent on
    # those alone.
    def drop(record: logging.LogRecord) -> bool:
        if overridden is None:
            return record.levelno < logging.WARNING
        return (record.levelno < logging.WARNING
                or not isinstance(record.args, tuple) or not record.args
                or record.args[0] not in overridden)

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


class TextEncoderAsset:
    r"""The word vectors of a text encoder, as a model dependency."""
    asset_kind = 'text_encoder'

    def __init__(self, name: str, model: Any) -> None:
        self.name, self.model = name, model
        self.version = model.label
        self.path = getattr(model, 'path', None)

    def fingerprint(self) -> str:
        return str(self.model.fingerprint())

    def files(self) -> Dict[str, Path]:
        vocab, vectors = self.model._files()
        return {vocab.name: vocab, vectors.name: vectors}

    @classmethod
    def load(cls, directory: Path, dependency: Any) -> 'TextEncoderAsset':
        from kumo_core.embedding_models import GloVeVectors

        return cls(dependency.name, GloVeVectors.in_directory(directory)[0])


def _text_encoders(encoders: Any) -> List[Any]:
    from kumoapi.encoder import TextEmbedding

    found: Dict[str, Any] = {}
    for table in (encoders.encoders.values() if encoders else ()):
        for encoder in table.values():
            if isinstance(encoder, TextEmbedding) and encoder._identity:
                found.setdefault(encoder._identity['fingerprint'], encoder)
    return list(found.values())


def require_text_encoders(deps: Any, store: EmbeddingModelStore,
                          encoders: Any) -> bool:
    r"""Records the word vectors of a fitted model's text encoders in
    ``deps``; returns whether it has any."""
    from kumo_core.embedding_models import GloVeVectors, resolve_encoder

    from kumo.engine._manifest import Dependency

    text = _text_encoders(encoders)
    for encoder in text:
        model = resolve_encoder(encoder, store)
        if isinstance(model, GloVeVectors):
            deps.require(TextEncoderAsset(encoder.model, model))
        else:
            deps.add(
                Dependency(kind='asset', name=encoder.model,
                           version=model.label, asset_kind='text_encoder',
                           sha256=encoder._identity['fingerprint'],
                           packing='external'))
    return bool(text)


def with_text_encoders(store: EmbeddingModelStore, encoders: Any,
                       deps: Any) -> EmbeddingModelStore:
    r"""``store`` plus the stored word vectors of the model's text encoders,
    resolved where ``deps`` looks for assets. Unless a place was given to
    look, ``store`` alone serves, as before; a missing model it does not
    have either is an error."""
    from kumo.engine._deps import MissingDependencyError, loaded

    added: Dict[str, Any] = {}
    for encoder in _text_encoders(encoders):
        known = encoder._identity
        record = next((record for record in deps.assets('text_encoder')
                       if record.sha256 == known['fingerprint']), None)
        if record is None or not record.entry_point or not deps.candidates(
                record):
            continue
        try:
            directory = deps.resolve('text_encoder', record.name)
        except MissingDependencyError:
            if store._find(known, encoder.model) is not None:
                continue
            raise
        added[f"{record.name}@{record.sha256[:12]}"] = loaded(
            directory, record).model
    if not added:
        return store
    extended = EmbeddingModelStore()
    extended._models = {**store._models, **added}
    return extended
