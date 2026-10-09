# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import pickle
import sys
import types
import weakref
import zipfile
from importlib.metadata import EntryPoint

import numpy as np
import pandas as pd
import pytest
from kumoapi.encoder import Null, TextEmbedding

from kumo_core import embedding_models
from kumo_core.embedding_models import (
    INSTALL_DEFAULT_MODEL,
    EmbeddingModelNotFoundError,
    EmbeddingModelStore,
    GloVeVectors,
    Release,
    TokenEmbeddingModel,
    require_embedding_models,
    require_encoder_models,
    resolve_encoder,
)
from kumo_core.encoders.config import EncodersConfig
from kumo_core.glove import VocabularyIndex
from kumo_core.io.json import load_json

WORDS = ["the", "nan", "null", "n/a", "new york"]


class ArrayModel:
    family = "array/1"
    label = None

    def __init__(self, words, vectors):
        self.words, self.array = pd.Index(words), vectors
        self.dim = vectors.shape[1]

    def fingerprint(self):
        return embedding_models._fingerprint(self.words, self.array)

    def tokenize(self, texts):
        return VocabularyIndex(self.words).tokenize(texts)

    def vectors(self):
        return self.array

    def vocabulary(self):
        return self.words


class Labelled(ArrayModel):
    def __init__(self, path, vectors):
        super().__init__(["a", "b"], vectors)
        self.label = f"words from {path}"


class Unhashable(ArrayModel):
    family = "glove/1"

    def fingerprint(self):
        raise AssertionError("fingerprinted")


@pytest.fixture(autouse=True)
def default_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(embedding_models, "default_cache_dir",
                        lambda: tmp_path / "default-cache")
    monkeypatch.setattr(embedding_models, "_RESOLVED",
                        weakref.WeakValueDictionary())


@pytest.fixture
def installed(monkeypatch):
    points = []
    monkeypatch.setattr(embedding_models, "entry_points",
                        lambda group: [p for p in points if p.group == group])
    monkeypatch.setattr(embedding_models, "_INSTALLED", None)
    return points


@pytest.fixture
def store(monkeypatch, installed):
    store = EmbeddingModelStore()
    monkeypatch.setattr(embedding_models, "_default_store", store)
    return store


def _write_dir(root, words=("hello", "world"), model="6B", dims=(50, ),
               value=1.0, vectors=None):
    root.mkdir(exist_ok=True)
    pd.DataFrame({"index": list(words)}).to_parquet(
        root / f"glove_{model}_index.parquet")
    for dim in dims:
        array = np.full((len(words), dim), value, dtype=np.float32)
        np.save(root / f"glove_{model}_{dim}d_emb.npy",
                array if vectors is None else vectors)
    return root


def _install(installed, monkeypatch, root, module="kumo_fake_glove"):
    monkeypatch.setitem(sys.modules, module,
                        types.SimpleNamespace(ROOT=str(root)))
    installed.append(
        EntryPoint("glove", f"{module}:ROOT",
                   embedding_models.ENTRY_POINT_GROUP))


def _write_release(path, monkeypatch, model="glove.6B", dims=(2, 3)):
    vectors = {
        dim: np.arange(len(WORDS) * dim, dtype=np.float32).reshape(-1, dim) / 7
        for dim in dims
    }
    stem = model.replace("_", ".")
    with zipfile.ZipFile(path, "w") as archive:
        for dim, matrix in vectors.items():
            archive.writestr(
                f"{stem}.{dim}d.txt", "".join(
                    f"{word} {' '.join(repr(float(v)) for v in row)}\n"
                    for word, row in zip(WORDS, matrix)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        embedding_models.RELEASES, digest,
        Release(model, path.name, {dim: f"{stem}.{dim}d.txt"
                                   for dim in dims}))
    return vectors


def test_construction_refuses_what_it_cannot_use(tmp_path, monkeypatch):
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as opened:
        opened.writestr("glove.6B.50d.txt", "the 1.0\n")
    unzipped = tmp_path / "unzipped"
    unzipped.mkdir()
    (unzipped / "glove.6B.50d.txt").write_text("the 1.0\n")
    both = _write_dir(tmp_path / "both")
    _write_dir(both, model="42B", dims=(300, ))
    _write_release(tmp_path / "glove.6B.zip", monkeypatch)
    cache = tmp_path / "cache"

    with pytest.raises(FileNotFoundError, match="absent does not exist"):
        GloVeVectors(tmp_path / "absent")
    with pytest.raises(ValueError, match="other.zip is not an official GloVe "
                       "release. Download one of glove.6B.zip"):
        GloVeVectors(other, cache_dir=cache)
    for path in (unzipped, unzipped / "glove.6B.50d.txt"):
        with pytest.raises(ValueError, match=r"unzipped GloVe release; pass "
                           r"the original \.zip .* GloVeVectors\("
                           r"\"glove.6B.zip\", dim=50\)"):
            GloVeVectors(path)
    with pytest.raises(TypeError, match="dim must be an int"):
        GloVeVectors(both, dim="50")
    with pytest.raises(ValueError, match="several GloVe models .*model="):
        GloVeVectors(both)
    with pytest.raises(ValueError, match=r"2d, 3d\); pass dim=.*dim=2\)"):
        GloVeVectors(tmp_path / "glove.6B.zip", cache_dir=cache)
    with pytest.raises(
            ValueError, match=r"holds only 300d, so dim=100 cannot be used\. "
            r"Remove dim= \(it is inferred\) or pass dim=300\."):
        GloVeVectors(both, model="glove.42B", dim=100)


def test_a_directory_is_read_in_place(tmp_path):
    root = _write_dir(tmp_path / "glove", dims=(50, 100))
    model = GloVeVectors(root, dim=100, cache_dir=tmp_path / "cache")

    assert model.label == "glove.6B 100d"
    assert model.vectors().shape == (2, 100)
    assert model.tokenize(pd.Series(["world hello"]))[1].tolist() == [1, 0]
    assert not (tmp_path / "cache").exists()
    assert [(m.model, m.dim) for m in GloVeVectors.in_directory(root)
            ] == [("glove.6B", 50), ("glove.6B", 100)]


def test_a_release_zip_is_converted_once_per_dimension(tmp_path, monkeypatch):
    archive = tmp_path / "glove.6B.zip"
    vectors = _write_release(archive, monkeypatch)
    model = GloVeVectors(archive, dim=3, cache_dir=tmp_path / "cache")

    loaded = model.vectors()
    assert loaded.flags.f_contiguous
    np.testing.assert_array_equal(loaded, vectors[3])
    assert model.vocabulary().tolist() == WORDS
    vocab, matrix = model._files()
    assert [path.name for path in vocab.parent.glob("*.npy")] == [matrix.name]

    converted_at = matrix.stat().st_mtime_ns
    again = GloVeVectors(archive, dim=3, cache_dir=tmp_path / "cache")
    assert again._files()[1].stat().st_mtime_ns == converted_at
    np.testing.assert_array_equal(
        GloVeVectors(archive, dim=2, cache_dir=tmp_path / "cache").vectors(),
        vectors[2])


def test_the_fingerprint_is_the_vocabulary_and_vectors(tmp_path, monkeypatch):
    archive = tmp_path / "glove.6B.zip"
    vectors = _write_release(archive, monkeypatch)
    converted = GloVeVectors(archive, dim=3, cache_dir=tmp_path / "cache")
    same = _write_dir(tmp_path / "same", words=WORDS, dims=(3, ),
                      vectors=np.ascontiguousarray(vectors[3]))
    renamed = _write_dir(tmp_path / "renamed", words=["a", *WORDS[1:]],
                         dims=(3, ), vectors=vectors[3])

    fingerprint = converted.fingerprint()
    assert GloVeVectors(same).fingerprint() == fingerprint
    assert GloVeVectors(archive, dim=3, cache_dir=tmp_path /
                        "cache").fingerprint() == fingerprint
    assert GloVeVectors(renamed).fingerprint() != fingerprint
    assert GloVeVectors(archive, dim=2, cache_dir=tmp_path /
                        "cache").fingerprint() != fingerprint


def test_a_failed_conversion_leaves_nothing_behind(tmp_path, monkeypatch):
    archive = tmp_path / "glove.6B.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr("glove.6B.2d.txt", "the 1.0 2.0\nbroken 1.0\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(embedding_models.RELEASES, digest,
                        Release("glove.6B", "glove.6B.zip",
                                {2: "glove.6B.2d.txt"}))
    model = GloVeVectors(archive, cache_dir=tmp_path / "cache")

    with pytest.raises(ValueError, match="line 2 does not have 2 values"):
        model.vectors()
    assert not list((tmp_path / "cache" / digest).glob("*.npy*"))
    assert not list((tmp_path / "cache" / digest).glob("*.parquet*"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_read_only_cache_still_works(tmp_path, monkeypatch):
    archive = tmp_path / "glove.6B.zip"
    vectors = _write_release(archive, monkeypatch)
    cache = tmp_path / "cache"
    GloVeVectors(archive, dim=3, cache_dir=cache).vectors()
    for path in [cache, *cache.rglob("*")]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        model = GloVeVectors(archive, dim=3, cache_dir=cache)
        np.testing.assert_array_equal(model.vectors(), vectors[3])
        assert len(model.fingerprint()) == 64
        with pytest.raises(PermissionError, match=r"glove.6B.zip must be "
                           r"converted on first use, but .*cache is not "
                           r"writable .*; pass a writable cache_dir="):
            GloVeVectors(archive, dim=2, cache_dir=cache).vectors()
    finally:
        for path in [cache, *cache.rglob("*")]:
            path.chmod(0o755)


def test_only_a_zip_needs_a_home_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_models, "default_cache_dir", lambda: None)
    archive = tmp_path / "glove.6B.zip"
    _write_release(archive, monkeypatch)

    with pytest.raises(ValueError, match="no home directory .*cache_dir="):
        GloVeVectors(archive, dim=3)
    model = GloVeVectors(_write_dir(tmp_path / "dir"))
    assert len(model.fingerprint()) == 64


def test_the_store_validates_what_it_is_given(tmp_path):
    store = EmbeddingModelStore()
    six = GloVeVectors(_write_dir(tmp_path / "six"))
    big = GloVeVectors(_write_dir(tmp_path / "big", model="42B", dims=(3, )))
    unpicklable = ArrayModel(["a"], np.ones((1, 2), dtype=np.float32))
    unpicklable.hook = lambda: None
    flat = ArrayModel(["a"], np.ones((1, 2), dtype=np.float32))
    flat.dim = 0

    with pytest.raises(TypeError, match="takes a model, not a path"):
        store.add("x", str(tmp_path / "six"))
    with pytest.raises(TypeError, match=r"lacks family, dim, label, "
                       r"fingerprint\(\), tokenize\(\)"):
        store.add("x", object())
    with pytest.raises(ValueError, match="positive integer"):
        store.add("x", flat)
    with pytest.raises(TypeError, match="cannot be pickled"):
        store.add("x", unpicklable)
    with pytest.raises(ValueError, match="default model.*glove.42B 3d"):
        store.add("glove.6B-50d", big)
    store.add("glove.6B-50d", six)
    store.add("big", big)
    with pytest.raises(ValueError, match="already registered as "
                       r"GloVeVectors \(glove.42B 3d\)"):
        store.add("big", six)
    assert store.get("big") is big
    assert store.names == ["big", "glove.6B-50d"]


def test_installed_packages_contribute_default_names(tmp_path, monkeypatch,
                                                     store, installed):
    installed.append(
        EntryPoint("glove", "kumo_missing_glove:ROOT",
                   embedding_models.ENTRY_POINT_GROUP))
    _install(installed, monkeypatch, _write_dir(tmp_path / "pkg",
                                                dims=(50, 100)))
    mine = GloVeVectors(_write_dir(tmp_path / "mine", value=2.0))
    store.add("glove.6B-50d", mine)

    assert store.names == ["glove.6B-100d", "glove.6B-50d"]
    assert store.get("glove.6B-50d") is mine
    assert resolve_encoder(TextEmbedding("glove.6B-100d")).dim == 100
    assert "pkg" in str(store.get("glove.6B-100d").path)


def test_resolution_by_name_and_by_identity(tmp_path, store):
    one = GloVeVectors(_write_dir(tmp_path / "one", value=1.0))
    two = GloVeVectors(_write_dir(tmp_path / "two", value=2.0))
    store.add("one", one)

    assert resolve_encoder(TextEmbedding("one")) is one
    encoders = EncodersConfig({"users": {"bio": TextEmbedding("one")}})
    assert require_encoder_models(encoders)
    stamped = encoders.encoders["users"]["bio"]
    assert stamped._identity == {
        "family": "glove/1",
        "dim": 50,
        "fingerprint": one.fingerprint(),
        "label": "glove.6B 50d",
    }
    assert not require_encoder_models(encoders)

    other = EmbeddingModelStore()
    other.add("renamed", GloVeVectors(tmp_path / "one"))
    assert resolve_encoder(stamped, other).fingerprint() == one.fingerprint()
    other = EmbeddingModelStore()
    other.add("one", two)
    with pytest.raises(EmbeddingModelNotFoundError) as error:
        resolve_encoder(stamped, other, columns=["users.bio"], why="bundle")
    assert str(error.value) == (
        f"This model was trained with glove.6B 50d, as 'one' for 'users.bio', "
        f"and the store does not have it; its 'one' is GloVeVectors (glove.6B "
        f"50d) (fingerprint {two.fingerprint()[:12]}). To predict, run "
        f"{INSTALL_DEFAULT_MODEL}, or register the same vectors under any "
        f"name, e.g. store.add(\"one\", GloVeVectors(\"glove.6B.zip\", "
        f"dim=50)), and pass the store: Trainer.load(bundle, "
        f"embedding_models=store).")


def test_only_candidates_are_fingerprinted(tmp_path, monkeypatch, store,
                                           installed):
    _install(installed, monkeypatch, _write_dir(tmp_path / "pkg"))
    store.add("glove.6B-50d", GloVeVectors(_write_dir(tmp_path / "mine",
                                                      value=2.0)))
    store.add("other", Unhashable(["a"], np.ones((1, 50), dtype=np.float32)))
    installed_model = embedding_models._installed()["glove.6B-50d"]
    encoder = TextEmbedding(_identity=embedding_models.identity(
        installed_model))

    # The installed model is found although the store's has its name.
    assert resolve_encoder(encoder, store) is installed_model


def test_trainers_with_different_stores_keep_their_models(tmp_path):
    from kumo_core.encoders.tensor_mapper import TensorMapper
    from kumo_core.train.trainer import Trainer

    models, trainers = [], []
    for value in (1.0, 2.0):
        model = GloVeVectors(_write_dir(tmp_path / str(value), value=value))
        store = EmbeddingModelStore()
        store.add("desc", model)
        trainer = Trainer.__new__(Trainer)
        trainer._embedding_models = store
        trainer.encoders = EncodersConfig(
            {"users": {"bio": TextEmbedding("desc")}})
        models.append(model)
        trainers.append(trainer)

    for trainer, model in zip(trainers, models):
        encoder = trainer.encoders.encoders["users"]["bio"]
        assert TensorMapper.from_encoder(encoder).model is model


def test_models_and_stores_survive_pickling(tmp_path, store):
    model = GloVeVectors(_write_dir(tmp_path / "glove"))
    model.tokenize(pd.Series(["hello"]))
    fingerprint = model.fingerprint()
    store.add("glove.6B-50d", model)
    custom = ArrayModel(["a", "b"], np.eye(2, dtype=np.float32))
    store.add("custom", custom)

    copy = pickle.loads(pickle.dumps(model))
    assert copy._vocabulary is None and copy._index is None
    assert copy.fingerprint() == fingerprint
    restored = pickle.loads(pickle.dumps(store))
    assert resolve_encoder(TextEmbedding(), restored).fingerprint() == (
        fingerprint)
    custom_copy = resolve_encoder(TextEmbedding("custom"), restored)
    assert custom_copy.tokenize(pd.Series(["b a"]))[1].tolist() == [1, 0]
    assert isinstance(custom, TokenEmbeddingModel)


def test_identity_ignores_the_label(tmp_path):
    vectors = np.eye(2, dtype=np.float32)
    store = EmbeddingModelStore()
    store.add("desc", Labelled(tmp_path / "old", vectors))
    encoders = EncodersConfig({"users": {"bio": TextEmbedding("desc")}})
    require_encoder_models(encoders, store)
    stamped = encoders.encoders["users"]["bio"]

    # The files moved: the label differs, the vectors do not.
    moved = Labelled(tmp_path / "new", vectors)
    other = EmbeddingModelStore()
    other.add("desc", Labelled(tmp_path / "new", 2 * vectors))
    other.add("moved", moved)
    assert resolve_encoder(stamped, other) is moved


def test_missing_models_name_the_columns_and_the_fix(store):
    inferred = {
        column: TextEmbedding()
        for column in ("users.bio", "items.title")
    }
    store.add("desc", ArrayModel(["a"], np.ones((1, 2), dtype=np.float32)))
    with pytest.raises(EmbeddingModelNotFoundError) as error:
        require_embedding_models(store, inferred)
    assert str(error.value) == (
        "Text columns 'users.bio' and 'items.title' use the default embedding "
        "model glove.6B-50d, which is not installed. Install it with "
        f"{INSTALL_DEFAULT_MODEL}, or download glove.6B.zip from "
        "https://nlp.stanford.edu/projects/glove/ and register it: "
        "store.add(\"glove.6B-50d\", GloVeVectors(\"glove.6B.zip\", dim=50)), "
        "and pass the store: Trainer(plan, embedding_models=store). To train "
        "without their text, set encoder_overrides={\"users.bio\": Null(), "
        "\"items.title\": Null()} (Null is in kumoapi.encoder).")

    with pytest.raises(EmbeddingModelNotFoundError) as error:
        require_embedding_models(
            store, inferred, {
                "users.bio": Null(),
                "items.title": "TextEmbedding(model='glove.42B-300d')",
                "items.desc": TextEmbedding("dsec"),
            })
    first, second = str(error.value).split("\n")
    assert first == (
        "Column 'items.title' uses the embedding model 'glove.42B-300d', "
        "which is not in the store. To use it, download glove.42B.300d.zip from "
        "https://nlp.stanford.edu/projects/glove/ and register it: "
        "store.add(\"glove.42B-300d\", GloVeVectors(\"glove.42B.300d.zip\")), "
        "and pass the store: Trainer(plan, embedding_models=store). The store "
        "has 'desc'.")
    assert second.startswith("Column 'items.desc' uses the embedding model "
                             "'dsec', which is not in the store. Did you mean "
                             "'desc'? To use it, register a model under that "
                             "name, e.g. store.add(\"dsec\", GloVeVectors(")

    require_embedding_models(store, {"users.bio": TextEmbedding()},
                             {"users.bio": Null()})


def test_auto_trainer_checks_and_stamps_before_any_trial(
        tmp_path, monkeypatch, store):
    from kumo_core.automl.auto_trainer import AutoTrainer

    trainer = AutoTrainer.__new__(AutoTrainer)
    runtime_store = EmbeddingModelStore()
    trainer.runtime = types.SimpleNamespace(
        trainer_runtime_kwargs=lambda: {"embedding_models": runtime_store})
    monkeypatch.setattr(AutoTrainer, "log_dir", str(tmp_path))
    monkeypatch.setattr(
        AutoTrainer, "_resolve_encoders", lambda self: EncodersConfig(
            {"users": {
                "bio": TextEmbedding(),
                "name": TextEmbedding()
            }}))

    for _ in range(2):
        with pytest.raises(EmbeddingModelNotFoundError,
                           match="'users.bio' and 'users.name'"):
            _ = trainer.encoders

    model = GloVeVectors(_write_dir(tmp_path / "six"))
    runtime_store.add("glove.6B-50d", model)
    assert trainer.encoders.encoders["users"]["bio"]._identity
    saved = json.loads((tmp_path / "encoders.json").read_text())
    assert saved["encoders"]["users"]["bio"]["_identity"]["fingerprint"] == (
        model.fingerprint())
    # Encoders built without a store, as in trials, get the same model.
    loaded = load_json(str(tmp_path / "encoders.json"), EncodersConfig)
    assert resolve_encoder(loaded.encoders["users"]["name"]) is model
