# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Word-embedding models for text columns, and the store that names them."""
from __future__ import annotations

import difflib
import errno
import glob
import hashlib
import logging
import os
import pickle
import re
import sys
import uuid
import weakref
import zipfile
from dataclasses import dataclass
from importlib.metadata import entry_points
from itertools import islice
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from filelock import FileLock

if TYPE_CHECKING:
    from torch import Tensor

    from kumo_core.glove import VocabularyIndex

logger = logging.getLogger(__name__)

# Each entry point resolves to the path of a directory of Kumo-format files.
ENTRY_POINT_GROUP = "kumo.embedding_models"
DEFAULT_MODEL = "glove.6B-50d"
# How to install the package that provides DEFAULT_MODEL.
INSTALL_DEFAULT_MODEL = 'pip install "kumo-engine[text]"'
GLOVE_SOURCE = "https://nlp.stanford.edu/projects/glove/"
_MEMBERS = ("family", "dim", "label", "fingerprint()", "tokenize()",
            "vectors()", "vocabulary()")


class EmbeddingModelNotFoundError(RuntimeError):
    pass


@runtime_checkable
class EmbeddingModel(Protocol):
    """A model that embeds text into ``dim``-dimensional vectors; see
    :class:`TokenEmbeddingModel`."""
    @property
    def family(self) -> str:
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def label(self) -> Optional[str]:
        ...

    def fingerprint(self) -> str:
        ...


@runtime_checkable
class TokenEmbeddingModel(EmbeddingModel, Protocol):
    """A model that embeds a text as the mean of its words' vectors.
    Implement it with any class (no subclassing needed) and register an
    instance with ``EmbeddingModelStore.add``.

    ``dim`` is the vector size; ``family`` names the kind of model and its
    version, e.g. ``"my-words/1"``; ``label`` describes it in messages, or
    is ``None``. ``fingerprint()`` returns a string that changes whenever
    anything that changes the embeddings does, including the vectors, the
    vocabulary and how ``tokenize`` splits text, e.g. a digest of them all:
    models of the same family, dim and fingerprint are treated as the same
    model and used interchangeably, and a trained model loads only with
    one, whatever name it is registered under.

    ``vocabulary()`` is a ``pd.Index`` of words; ``vectors()`` is a float32
    array of shape ``[len(vocabulary()), dim]``, row ``r`` being the vector
    of ``vocabulary()[r]``. ``tokenize(texts)`` takes a ``pd.Series`` of raw
    column values (strings, or null) and returns ``(rowptr, rows)``, two
    integer tensors (or arrays): the words of ``texts.iloc[i]`` are the
    vocabulary rows ``rows[rowptr[i]:rowptr[i + 1]]``, so ``rowptr`` has
    ``len(texts) + 1`` entries starting at 0; unknown words are dropped.
    With vocabulary ``["red", "shoe"]``, ``["Red shoe", None, "sock"]``
    gives ``rowptr = [0, 2, 2, 2]`` and ``rows = [0, 1]``.

    Models are pickled to worker processes, so define the class in a .py
    module (not a notebook), and load large data lazily instead of
    pickling it.
    """
    def tokenize(self, texts: pd.Series) -> tuple[Tensor, Tensor]:
        ...

    def vectors(self) -> np.ndarray:
        ...

    def vocabulary(self) -> pd.Index:
        ...


@runtime_checkable
class TextVectorModel(EmbeddingModel, Protocol):
    """A model that embeds each text as a whole: ``encode(texts)`` returns
    a float32 array of shape ``[len(texts), dim]``. Not yet supported by
    the trainer.
    """
    def encode(self, texts: pd.Series) -> np.ndarray:
        ...


@dataclass(frozen=True)
class Release:
    model: str
    archive: str
    members: dict[int, str]


# Official Stanford releases, by the SHA-256 of their .zip.
RELEASES: dict[str, Release] = {
    "617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb":
    Release("glove.6B", "glove.6B.zip",
            {dim: f"glove.6B.{dim}d.txt"
             for dim in (50, 100, 200, 300)}),
    "03d5d7fa28e58762ace4b85fb71fe86a345ef0b5ff39f5390c14869da0fc1970":
    Release("glove.42B", "glove.42B.300d.zip", {300: "glove.42B.300d.txt"}),
    "c06db255e65095393609f19a4cfca20bf3a71e20cc53e892aafa490347e3849f":
    Release("glove.840B", "glove.840B.300d.zip", {300: "glove.840B.300d.txt"}),
    "792af52f795d1a32c9842a3240f5f3fe5e941a8ff6df5eb0f9d668092ebc019c":
    Release("glove_twitter.27B", "glove.twitter.27B.zip",
            {dim: f"glove.twitter.27B.{dim}d.txt"
             for dim in (25, 50, 100, 200)}),
}


def default_cache_dir() -> Optional[Path]:
    try:
        return Path.home() / ".cache" / "kumo" / "embeddings"
    except (RuntimeError, KeyError):  # No home directory.
        return None


class _VocabularyModel:
    family: str
    dim: int
    _vocabulary: Optional[pd.Index] = None
    _index: Optional[VocabularyIndex] = None
    _fingerprint: Optional[str] = None

    def _read_vocabulary(self) -> pd.Index:
        raise NotImplementedError

    def _compute_fingerprint(self) -> str:
        raise NotImplementedError

    def vocabulary(self) -> pd.Index:
        if self._vocabulary is None:
            self._vocabulary = self._read_vocabulary()
        return self._vocabulary

    def _vocabulary_index(self) -> VocabularyIndex:
        if self._index is None:
            from kumo_core.glove import VocabularyIndex
            self._index = VocabularyIndex(self.vocabulary())
        return self._index

    def tokenize(self, texts: pd.Series) -> tuple[Tensor, Tensor]:
        return self._vocabulary_index().tokenize(texts)

    def lookup(self, words: Any) -> tuple[Tensor, Tensor]:
        index = self._vocabulary_index()
        return index.map_index(words, return_mask=True)

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = self._compute_fingerprint()
        return self._fingerprint

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_vocabulary", None)
        state.pop("_index", None)
        return state

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"

    @property
    def label(self) -> Optional[str]:
        return None


class GloVeVectors(_VocabularyModel):
    """GloVe word vectors from an official release .zip, used unchanged
    (https://nlp.stanford.edu/projects/glove/).

    >>> kumo.engine.GloVeVectors("glove.6B.zip", dim=100)
    >>> kumo.engine.GloVeVectors("glove.42B.300d.zip")

    ``dim`` picks the size in glove.6B.zip (50, 100, 200 or 300) and
    glove.twitter.27B.zip (25, 50, 100 or 200). The .zip is converted on
    first use into ``cache_dir`` (default ``~/.cache/kumo/embeddings``). A
    directory of converted files also works; ``model`` picks one of several
    models in it.
    """
    family = "glove/1"

    def __init__(
        self,
        path: Union[str, os.PathLike],
        dim: Optional[int] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        given = str(path)
        self.path = Path(path).expanduser().resolve()
        self.cache_dir = (Path(cache_dir).expanduser().resolve()
                          if cache_dir is not None else None)
        self._release: Optional[Release] = None
        self._digest: Optional[str] = None
        if dim is not None and (isinstance(dim, bool)
                                or not isinstance(dim, int)):
            raise TypeError(f"dim must be an int, e.g. dim=50 (got {dim!r}).")
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            if self._cache() is None:
                raise ValueError(f"There is no home directory to convert "
                                 f"{given} into; pass cache_dir=.")
            self._digest = _archive_digest(self.path, self._cache())
            if self._digest not in RELEASES:
                known = ", ".join(
                    dict.fromkeys(r.archive for r in RELEASES.values()))
                raise ValueError(
                    f"{given} is not an official GloVe release. Download one "
                    f"of {known} from {GLOVE_SOURCE} and pass it unchanged; "
                    f"for other vectors, register your own "
                    f"TokenEmbeddingModel.")
            self._release = RELEASES[self._digest]
            if model not in (None, self._release.model):
                raise ValueError(f"{given} is {self._release.archive}, which "
                                 f"holds {self._release.model}, not {model}; "
                                 f"remove model=.")
            model, dims = self._release.model, sorted(self._release.members)
        elif self.path.is_dir():
            found = _scan(self.path)
            if not found:
                texts = sorted(self.path.glob("*.txt"))
                if texts:
                    raise ValueError(_unzipped_message(given, texts))
                raise ValueError(
                    f"{given} holds no GloVe files. Pass an official GloVe "
                    f"release .zip from {GLOVE_SOURCE}, e.g. "
                    f"GloVeVectors(\"glove.6B.zip\", dim=50).")
            if model is None:
                if len(found) > 1:
                    raise ValueError(
                        f"{given} holds several GloVe models "
                        f"({', '.join(found)}); pass model= to choose one, "
                        f"e.g. GloVeVectors({given!r}, "
                        f"model={next(iter(found))!r}).")
                model = next(iter(found))
            elif model not in found:
                raise ValueError(f"{given} holds no {model} files; it holds "
                                 f"{', '.join(found)}.")
            dims = found[model]
        elif self.path.suffix == ".txt":
            raise ValueError(_unzipped_message(given, [self.path]))
        elif self.path.exists():
            raise ValueError(
                f"{given} is not an official GloVe release .zip. Download "
                f"one from {GLOVE_SOURCE} and pass it unchanged.")
        else:
            raise FileNotFoundError(
                f"{given} does not exist. Download an official GloVe "
                f"release .zip from {GLOVE_SOURCE} and pass its path, e.g. "
                f"GloVeVectors(\"glove.6B.zip\", dim=50).")
        self.model: str = model
        self.dim: int = _choose_dim(given, model, dims, dim)

    @classmethod
    def in_directory(
        cls,
        path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> list["GloVeVectors"]:
        """Every model in a directory of converted files."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{path} is not a directory.")
        return [
            cls(root, dim=dim, model=model, cache_dir=cache_dir)
            for model, dims in _scan(root).items() for dim in dims
        ]

    @property
    def label(self) -> str:
        return f"{self.model} {self.dim}d"

    def __repr__(self) -> str:
        model = "" if self._release else f", model={self.model!r}"
        return f"GloVeVectors({str(self.path)!r}, dim={self.dim}{model})"

    def vectors(self) -> np.ndarray:
        # Mapped: the files are column-major, and the caller copies them.
        return np.load(self._files()[1], mmap_mode="r")

    def _cache(self) -> Optional[Path]:
        return self.cache_dir or default_cache_dir()

    def _read_vocabulary(self) -> pd.Index:
        words = pd.read_parquet(self._files()[0])["index"]
        return pd.Index(words, dtype=object, name="index")

    def _compute_fingerprint(self) -> str:
        vocab, vectors = self._files()
        return _remembered(
            "fingerprints", self._cache(), [vocab, vectors],
            lambda: _fingerprint(self.vocabulary(),
                                 np.load(vectors, mmap_mode="r")))

    def _files(self) -> tuple[Path, Path]:
        vocab_name, vectors_name = _file_names(self.model, self.dim)
        if self._release is None:
            return self.path / vocab_name, self.path / vectors_name
        cache = self._cache()
        assert self._digest is not None and cache is not None
        root = cache / self._digest
        vocab, vectors = root / vocab_name, root / vectors_name
        if not vectors.is_file():
            try:
                root.mkdir(parents=True, exist_ok=True)
                with FileLock(str(root / ".lock")):
                    if not vectors.is_file():
                        _convert(self.path, self._release.members[self.dim],
                                 self.dim, vocab, vectors)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                    raise
                raise PermissionError(
                    f"{self.path.name} must be converted on first use, but "
                    f"{cache} is not writable ({error.strerror}); pass a "
                    f"writable cache_dir=.") from error
        return vocab, vectors


class EmbeddingModelStore:
    """Word-embedding models for text columns, by name.

    >>> store = kumo.engine.EmbeddingModelStore()
    >>> store.add("desc", kumo.engine.GloVeVectors("glove.42B.300d.zip"))
    >>> plan.column_processing.encoder_overrides = {
    ...     "articles.detail_desc": TextEmbedding("desc")}
    >>> trainer = kumo.engine.Trainer(plan, embedding_models=store)

    Text columns without an override use ``"glove.6B-50d"``, which
    ``pip install "kumo-engine[text]"`` installs. Installed models are in
    every store under their default names.
    """
    # Set by deployments whose models are baked into an image: what to do
    # about a missing model, instead of the Kumo Local advice.
    _missing_advice: Optional[str] = None

    def __init__(self) -> None:
        self._models: dict[str, TokenEmbeddingModel] = {}

    def add(self, name: str, model: TokenEmbeddingModel) -> None:
        """Registers ``model`` (a :class:`GloVeVectors` or your own
        :class:`TokenEmbeddingModel`) under ``name``."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"A model name must be a non-empty string "
                             f"(got {name!r}).")
        if name in self._models:
            raise ValueError(f"{name!r} is already registered as "
                             f"{_describe(self._models[name])}; pick another "
                             f"name.")
        if isinstance(model, (str, os.PathLike)):
            raise TypeError(f"store.add takes a model, not a path: "
                            f"store.add({name!r}, GloVeVectors("
                            f"{str(model)!r})).")
        missing = [member for member in _MEMBERS if _lacks(model, member)]
        if missing:
            raise TypeError(
                f"{type(model).__name__} cannot be registered as {name!r}: "
                f"it lacks {', '.join(missing)}. Register a GloVeVectors or "
                f"your own model (see help(kumo.engine.TokenEmbeddingModel)).")
        dim = model.dim
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{_describe(model)} cannot be registered as "
                             f"{name!r}: its dim must be a positive integer "
                             f"(got {dim!r}).")
        if not isinstance(model.family, str) or not model.family:
            raise ValueError(f"{_describe(model)} cannot be registered as "
                             f"{name!r}: its family must be a non-empty "
                             f"string, e.g. \"my-words/1\" (got "
                             f"{model.family!r}).")
        if name == DEFAULT_MODEL and not (isinstance(model, GloVeVectors) and
                                          (model.model, model.dim)
                                          == ("glove.6B", 50)):
            raise ValueError(
                f"{DEFAULT_MODEL!r} is the default model, GloVe 6B 50d, so "
                f"{_describe(model)} cannot be registered under it. Register "
                f"it under another name, e.g. \"mine\", and set it per "
                f"column: encoder_overrides={{\"table.column\": "
                f"TextEmbedding(\"mine\")}}.")
        try:
            pickle.loads(pickle.dumps(model))
        except Exception as error:
            raise TypeError(
                f"{_describe(model)} cannot be registered as {name!r}: it "
                f"cannot be pickled ({error}), and models are sent to worker "
                f"processes. Define its class at module level and keep "
                f"only picklable attributes.") from error
        try:
            weakref.ref(model)
        except TypeError as error:
            raise TypeError(
                f"{_describe(model)} cannot be registered as {name!r}: it "
                f"cannot be weakly referenced; add \"__weakref__\" to the "
                f"__slots__ of its class.") from error
        self._models[name] = model

    @property
    def names(self) -> list[str]:
        """The names of the registered and installed models."""
        return sorted(set(self._models) | set(_installed()))

    def get(self, name: str) -> TokenEmbeddingModel:
        """The model registered or installed under ``name``."""
        model = self._get(name)
        if model is None:
            close = difflib.get_close_matches(name, self.names, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise KeyError(f"No embedding model is named {name!r}.{hint} "
                           f"The store has {self.names}.")
        return model

    def __repr__(self) -> str:
        models = {**_installed(), **self._models}
        entries = ", ".join(f"{name!r}: {models[name]!r}"
                            for name in self.names)
        return f"EmbeddingModelStore({{{entries}}})"

    def _find(self, known: dict[str, Any],
              name: str = "") -> Optional[TokenEmbeddingModel]:
        # Only models of the same family and dim are fingerprinted, as that
        # may convert a .zip or hash gigabytes; the likeliest go first.
        family, dim, _ = _key(known)
        for refresh in (False, True):
            entries = [*self._models.items(), *_installed(refresh).items()]
            entries.sort(key=lambda entry: (entry[0] != name, entry[1].label
                                            != known.get("label")))
            for _, model in entries:
                if (model.family == family and model.dim == dim
                        and model.fingerprint() == known["fingerprint"]):
                    return model
        return None

    def _get(self, name: str) -> Optional[TokenEmbeddingModel]:
        if name in self._models:
            return self._models[name]
        installed = _installed()
        if name not in installed:
            installed = _installed(refresh=True)
        return installed.get(name)


_default_store: Optional[EmbeddingModelStore] = None
_INSTALLED: Optional[dict[str, TokenEmbeddingModel]] = None
# Models resolved by a trainer's store, by identity. Models of one identity
# are interchangeable, so trainers with different stores can share this.
# Weak: the stores, mappers and encoder modules using a model own it.
_RESOLVED: weakref.WeakValueDictionary[tuple, TokenEmbeddingModel] = (
    weakref.WeakValueDictionary())


def set_default_store(
        store: Optional[EmbeddingModelStore]) -> Optional[EmbeddingModelStore]:
    """Sets the store used when none is given (tests); returns the last."""
    global _default_store
    previous, _default_store = _default_store, store
    return previous


def default_store() -> EmbeddingModelStore:
    return (_default_store
            if _default_store is not None else EmbeddingModelStore())


def default_name(model: GloVeVectors) -> str:
    return f"{model.model}-{model.dim}d"


def identity(model: EmbeddingModel) -> dict[str, Any]:
    return {
        "family": model.family,
        "dim": model.dim,
        "fingerprint": model.fingerprint(),
        "label": model.label,
    }


def _key(known: dict[str, Any]) -> tuple:
    return (known["family"], known["dim"], known["fingerprint"])


def resolve_encoder(
    encoder: Any,
    store: Optional[EmbeddingModelStore] = None,
    columns: Sequence[str] = (),
    why: str = "",
) -> TokenEmbeddingModel:
    """The model for a ``TextEmbedding`` encoder, from ``store``; without
    one, the model its trainer resolved, else the default store. ``why``
    says where the encoder of ``columns`` came from, for messages:
    ``"inferred"``, ``"override"`` or ``"bundle"`` (a trained model)."""
    name = encoder.model
    known = encoder._identity
    if known and store is None:
        model = _RESOLVED.get(_key(known))
        if model is not None:
            return model
    store = store if store is not None else default_store()
    if known:
        model = store._find(known, name)
        if model is None:
            raise EmbeddingModelNotFoundError(
                _trained_message(store, known, name, columns))
        _RESOLVED[_key(known)] = model
        return model
    found = store._get(name)
    if found is None:
        raise EmbeddingModelNotFoundError(
            _missing_message(store, name, columns, why))
    return found


def require_embedding_models(
    store: Optional[EmbeddingModelStore],
    inferred: dict[str, Any],
    encoder_overrides: Optional[dict] = None,
) -> None:
    """Fails with the fix if a text column's model is missing. ``inferred``
    maps ``table.column`` to the encoder training infers for it."""
    from kumo_core.model_plan import ColumnProcessingPlan

    overrides = ColumnProcessingPlan.get_encoder_overrides(encoder_overrides)
    why = {column: "inferred" for column in inferred}
    why.update({column: "override" for column in overrides})
    _require(store, {**inferred, **overrides}, why)


def require_encoder_models(
    encoders: Any,
    store: Optional[EmbeddingModelStore] = None,
    trained: bool = False,
) -> bool:
    """Resolves every text encoder, and records the identity of the model
    on encoders that lack it. Returns whether it did. ``trained``: the
    encoders are those of a trained model."""
    columns = {
        f"{table}.{column}": encoder
        for table, table_encoders in encoders.encoders.items()
        for column, encoder in table_encoders.items()
    }
    why = dict.fromkeys(columns, "bundle" if trained else "")
    stamped = False
    for model, group in _require(store, columns, why):
        for encoder in group:
            if not encoder._identity:
                encoder._identity = identity(model)
                _RESOLVED[_key(encoder._identity)] = model
                stamped = True
    return stamped


def require_importable(store: Optional[EmbeddingModelStore]) -> None:
    """Fails if a spawned process could not import a model's class."""
    main = sys.modules.get("__main__")
    for name, model in (store._models.items() if store else ()):
        if (type(model).__module__ == "__main__"
                and not getattr(main, "__file__", None)):
            raise TypeError(
                f"{type(model).__name__}, the model {name!r}, is defined in "
                f"this session, so worker processes cannot load it. Move the "
                f"class into a .py file and import it from there, or use "
                f"num_workers=1.")


def _require(
    store: Optional[EmbeddingModelStore],
    encoders: dict[str, Any],
    why: dict[str, str],
) -> list[tuple[TokenEmbeddingModel, list[Any]]]:
    from kumoapi.encoder import TextEmbedding

    store = store if store is not None else default_store()
    groups: dict[tuple, tuple[list[str], list[Any]]] = {}
    for column, encoder in encoders.items():
        if not isinstance(encoder, TextEmbedding):
            continue
        known = encoder._identity or {}
        key = (encoder.model, known.get("family"), known.get("fingerprint"),
               why.get(column, ""))
        names, group = groups.setdefault(key, ([], []))
        names.append(column)
        group.append(encoder)
    resolved, missing = [], []
    for key, (names, group) in groups.items():
        try:
            resolved.append(
                (resolve_encoder(group[0], store, names, key[-1]), group))
        except EmbeddingModelNotFoundError as error:
            missing.append(str(error))
    if missing:
        raise EmbeddingModelNotFoundError("\n".join(missing))
    return resolved


def _lacks(model: Any, member: str) -> bool:
    if member.endswith("()"):
        return not callable(getattr(model, member[:-2], None))
    return not hasattr(model, member)


def _describe(model: Any) -> str:
    label = getattr(model, "label", None)
    return f"{type(model).__name__} ({label})" if label else repr(model)


def _columns(columns: Sequence[str]) -> str:
    quoted = [repr(column) for column in columns[:5]]
    if len(columns) > 5:
        quoted.append(f"{len(columns) - 5} others")
    return (quoted[0] if len(quoted) == 1 else
            f"{', '.join(quoted[:-1])} and {quoted[-1]}")


def _release_for(name: str) -> Optional[tuple[Release, int]]:
    for release in RELEASES.values():
        for dim in release.members:
            if name == f"{release.model}-{dim}d":
                return release, dim
    return None


def _glove_call(release: Release, dim: int) -> str:
    extra = f", dim={dim}" if len(release.members) > 1 else ""
    return f'GloVeVectors("{release.archive}"{extra})'


def _register(name: str, release: Optional[tuple[Release, int]],
              then: str) -> str:
    if release is None:
        return (f'register a model under that name, e.g. store.add("{name}", '
                f'GloVeVectors("glove.6B.zip", dim=100)), and {then}')
    return (f"download {release[0].archive} from {GLOVE_SOURCE} and "
            f'register it: store.add("{name}", {_glove_call(*release)}), '
            f"and {then}")


def _missing_message(store: EmbeddingModelStore, name: str,
                     columns: Sequence[str], why: str) -> str:
    one = len(columns) == 1
    default = name == DEFAULT_MODEL
    listed = f"{'' if one else 's'} {_columns(columns)}" if columns else ""
    if not columns:
        lead = f"The embedding model {name!r}"
    elif why == "bundle":
        lead = (f"This model uses the embedding model {name!r} for "
                f"{_columns(columns)}")
    elif default and why == "inferred":
        lead = (f"Text column{listed} use{'s' if one else ''} the default "
                f"embedding model {name}")
    else:
        lead = (f"Column{listed} use{'s' if one else ''} the embedding "
                f"model {name!r}")
    if store._missing_advice:
        return _deployment_message(store, lead)
    then = ("pass the store: Trainer.load(bundle, embedding_models=store)"
            if why == "bundle" else
            "pass the store: Trainer(plan, embedding_models=store)")
    if default:
        message = (f"{lead}, which is not installed. Install it with "
                   f"{INSTALL_DEFAULT_MODEL}, or "
                   f"{_register(name, _release_for(name), then)}.")
        if why == "inferred":
            nulls = ", ".join(f'"{column}": Null()' for column in columns)
            message += (f" To train without {'its' if one else 'their'} "
                        f"text, set encoder_overrides={{{nulls}}} (Null is "
                        f"in kumoapi.encoder).")
        return message
    names, release = store.names, _release_for(name)
    close = [] if release else difflib.get_close_matches(name, names, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    has = (f"The store has {', '.join(map(repr, names))}."
           if names else "The store is empty.")
    return (f"{lead}, which is not in the store.{hint} To use it, "
            f"{_register(name, release, then)}. {has}")


def _trained_message(store: EmbeddingModelStore, known: dict, name: str,
                     columns: Sequence[str]) -> str:
    release = None
    match = re.fullmatch(r"(\S+) (\d+)d", str(known.get("label")))
    if known["family"] == GloVeVectors.family and match:
        release = _release_for(f"{match.group(1)}-{match.group(2)}d")
    label = known.get("label") or f"a {known['family']} model"
    if release is None:
        label += f" (fingerprint {str(known['fingerprint'])[:12]})"
    target = f" for {_columns(columns)}" if columns else ""
    if store._missing_advice:
        return _deployment_message(
            store, f"This model was trained with {label}, as {name!r}{target}")
    message = (f"This model was trained with {label}, as {name!r}{target}, "
               f"and the store does not have it")
    other = store._get(name)
    if other is not None:
        message += f"; its {name!r} is {_describe(other)}"
        if release is None or other.label == known.get("label"):
            message += f" (fingerprint {other.fingerprint()[:12]})"
    then = "pass the store: Trainer.load(bundle, embedding_models=store)"
    if release is None:
        return (f"{message}. To predict, register the same model under any "
                f'name, e.g. store.add("{name}", model), and {then}.')
    fix = (f"register the same vectors under any name, e.g. store.add("
           f'"{name}", {_glove_call(*release)}), and {then}')
    if release[0].model == "glove.6B" and release[1] == 50:
        fix = f"run {INSTALL_DEFAULT_MODEL}, or {fix}"
    return f"{message}. To predict, {fix}."


def _deployment_message(store: EmbeddingModelStore, lead: str) -> str:
    names = ", ".join(map(repr, store.names)) or "none"
    return (f"{lead}, which this deployment's image does not include (it has "
            f"{names}). {store._missing_advice}")


def _installed(refresh: bool = False) -> dict[str, TokenEmbeddingModel]:
    global _INSTALLED
    if _INSTALLED is None or refresh:
        found: dict[str, TokenEmbeddingModel] = {}
        for entry_point in entry_points(group=ENTRY_POINT_GROUP):
            try:
                models = GloVeVectors.in_directory(entry_point.load())
            except Exception as error:
                logger.warning("Skipping embedding-model package %s: %s",
                               entry_point.value, error)
                continue
            for model in models:
                found.setdefault(default_name(model), model)
        _INSTALLED = found
    return _INSTALLED


def _file_names(model: str, dim: int) -> tuple[str, str]:
    prefix = model.replace(".", "_")
    return f"{prefix}_index.parquet", f"{prefix}_{dim}d_emb.npy"


def _scan(root: Path) -> dict[str, list[int]]:
    names = {
        _file_names(r.model, 0)[0]: r.model
        for r in RELEASES.values()
    }
    found: dict[str, list[int]] = {}
    for index in sorted(root.glob("glove_*_index.parquet")):
        prefix = index.name[:-len("_index.parquet")]
        pattern = re.escape(prefix) + r"_(\d+)d_emb\.npy"
        dims = sorted(
            int(match.group(1))
            for path in root.glob(f"{glob.escape(prefix)}_*d_emb.npy")
            if (match := re.fullmatch(pattern, path.name)))
        if dims:
            found[names.get(index.name, prefix.replace("_", ".", 1))] = dims
    return found


def _unzipped_message(given: str, texts: list[Path]) -> str:
    call = 'GloVeVectors("glove.6B.zip", dim=50)'
    for release in RELEASES.values():
        for dim, member in release.members.items():
            if any(text.name == member for text in texts):
                call = _glove_call(release, dim)
    return (f"{given} is an unzipped GloVe release; pass the original .zip "
            f"from {GLOVE_SOURCE} instead, e.g. {call}.")


def _choose_dim(given: str, model: str, dims: list[int],
                dim: Optional[int]) -> int:
    held = ", ".join(f"{d}d" for d in dims)
    if dim is None:
        if len(dims) > 1:
            raise ValueError(
                f"{given} holds {model} in several sizes ({held}); pass dim= "
                f"to choose one, e.g. GloVeVectors({given!r}, dim={dims[0]}).")
        return dims[0]
    if dim not in dims:
        if len(dims) == 1:
            raise ValueError(
                f"{given} holds only {held}, so dim={dim} cannot be used. "
                f"Remove dim= (it is inferred) or pass dim={dims[0]}.")
        raise ValueError(f"{given} holds {held}, so dim={dim} cannot be "
                         f"used; pass one of those.")
    return dim


def _fingerprint(words: pd.Index, vectors: np.ndarray) -> str:
    digest = hashlib.sha256("\n".join(map(str, words)).encode("utf-8"))
    for start in range(0, len(vectors), 1 << 16):
        chunk = np.ascontiguousarray(vectors[start:start + (1 << 16)],
                                     dtype="<f4")
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _remembered(kind: str, cache_dir: Optional[Path], paths: list[Path],
                compute: Callable[[], str]) -> str:
    # Remembered by path, size and mtime, so each process does not rehash.
    if cache_dir is None:
        return compute()
    key = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        key.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    marker = cache_dir / kind / key.hexdigest()
    if marker.is_file():
        return marker.read_text()
    value = compute()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        partial = marker.with_name(f"{marker.name}.{uuid.uuid4().hex}")
        partial.write_text(value)
        os.replace(partial, marker)
    except OSError as error:
        logger.debug("Not remembering %s of %s: %s", kind, paths, error)
    return value


def _archive_digest(archive: Path, cache_dir: Path) -> str:
    def compute() -> str:
        digest = hashlib.sha256()
        with open(archive, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 24), b""):
                digest.update(block)
        return digest.hexdigest()

    return _remembered("archives", cache_dir, [archive], compute)


def _lines(archive: Path, member: str) -> Iterator[str]:
    with zipfile.ZipFile(archive) as opened:
        with opened.open(member) as raw:
            for line in raw:
                yield line.decode("utf-8").rstrip("\n")


def _convert(archive: Path, member: str, dim: int, vocab: Path,
             vectors: Path) -> None:
    logger.info("Converting %s from %s", member, archive)
    num_rows = sum(1 for _ in _lines(archive, member))
    partial = vectors.with_name(vectors.name + ".partial")
    partial_vocab = vocab.with_name(vocab.name + ".partial")
    try:
        # Column order, as in the files Kumo has always shipped.
        matrix = np.lib.format.open_memmap(partial, mode="w+",
                                           dtype=np.float32,
                                           shape=(num_rows, dim),
                                           fortran_order=True)
        words: list[str] = []
        lines = _lines(archive, member)
        while chunk := list(islice(lines, 100_000)):
            rows = [line.rsplit(" ", dim) for line in chunk]
            for offset, row in enumerate(rows):
                if len(row) != dim + 1:
                    raise ValueError(f"{member} line {len(words) + offset + 1} "
                                     f"does not have {dim} values.")
            start = len(words)
            words.extend(row[0] for row in rows)
            matrix[start:len(words)] = np.array([row[1:] for row in rows],
                                                dtype=np.float32)
            logger.info("Converted %d of %d rows of %s", len(words), num_rows,
                        member)
        matrix.flush()
        del matrix
        if vocab.is_file():
            if pd.read_parquet(vocab)["index"].tolist() != words:
                raise ValueError(f"{member} has a different vocabulary than "
                                 f"{vocab}.")
        else:
            pd.DataFrame({
                "index": pd.Series(words, dtype=object)
            }).to_parquet(partial_vocab)
            os.replace(partial_vocab, vocab)
        os.replace(partial, vectors)
    finally:
        partial.unlink(missing_ok=True)
        partial_vocab.unlink(missing_ok=True)
