# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
import weakref
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core import embedding_models
from kumo_core.embedding_models import EmbeddingModelNotFoundError, Release
from kumo_core.glove import TokenEmbeddingTable
from kumoapi.encoder import TextEmbedding
from kumoapi.typing import Stype

from kumo.engine import (
    EmbeddingModelStore,
    FileConnector,
    GloVeVectors,
    Graph,
    PredictiveQuery,
    Table,
    Trainer,
)

WORDS = ['hello', 'world', 'graph', 'network', 'text', 'cool', 'test']


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    monkeypatch.setattr(embedding_models, 'entry_points', lambda group: [])
    monkeypatch.setattr(embedding_models, '_INSTALLED', None)
    monkeypatch.setattr(embedding_models, '_default_store', None)
    monkeypatch.setattr(TokenEmbeddingTable, 'instances',
                        weakref.WeakValueDictionary())


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('fitted')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr('kumo.engine._scratch._root', root / 'scratch')
        monkeypatch.setattr(embedding_models, 'entry_points', lambda group: [])
        monkeypatch.setattr(embedding_models, '_INSTALLED', None)
        archive = _release_zip(root / 'glove.6B.zip', monkeypatch)
        graph = Graph([_users_table(root / 'source')])
        snapshot = graph.snapshot()  # Snapshots do not need the models.
        query = PredictiveQuery(graph, 'PREDICT users.label FOR EACH users.id')
        plan = query.suggest_model_plan(snapshot)
        plan.training_job.num_experiments = 1
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        warned = _Warnings()
        logging.getLogger('kumo_core.encoder_selection').addHandler(warned)
        table = query.generate_training_table(snapshot)
        prediction_table = query.generate_prediction_table(snapshot)
        warnings = {'tables': warned.take()}

        with pytest.raises(EmbeddingModelNotFoundError,
                           match=r"^Text column 'users.bio' uses the default "
                           r"embedding model glove.6B-50d, which is not "
                           r"installed. Install it with pip install "
                           r'"kumo-engine\[text\]"'):
            Trainer(plan, accelerator='cpu').fit(table, snapshot=snapshot)
        warnings['failed fit'] = warned.take()

        plan.column_processing.encoder_overrides = {
            'users.bio': TextEmbedding('words')
        }
        store = EmbeddingModelStore()
        store.add('words', GloVeVectors(archive, cache_dir=root / 'emb'))
        trainer = Trainer(plan, embedding_models=store, accelerator='cpu')
        result = trainer.fit(table, snapshot=snapshot)
        warnings['fit'] = warned.take()
        logging.getLogger('kumo_core.encoder_selection').removeHandler(warned)
        yield (archive, root, snapshot, table, prediction_table, trainer,
               result, warnings)


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def take(self):
        taken, self.messages = self.messages, []
        return taken


def _release_zip(path, monkeypatch):
    with zipfile.ZipFile(path, 'w') as opened:
        opened.writestr(
            'glove.6B.50d.txt', ''.join(
                f"{word} {' '.join([str(i / 10)] * 50)}\n"
                for i, word in enumerate(WORDS)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        embedding_models.RELEASES, digest,
        Release('glove.6B', path.name, {50: 'glove.6B.50d.txt'}))
    return path


def _users_table(root):
    rng = np.random.default_rng(0)
    n = 200
    root.mkdir()
    pq.write_table(
        pa.table({
            'id':
            pa.array(range(n), type=pa.int64()),
            'bio': [
                ' '.join(rng.choice(WORDS, 25 if i % 10 == 0 else 5))
                for i in range(n)
            ],
            # Text that training leaves unencoded, so needs no model.
            'code': [f'x{i}q{i * 7}z' for i in range(n)],
            'label':
            pa.array([bool(i % 2) if i < 180 else None for i in range(n)]),
        }), root / 'users.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(root))['users'], primary_key='id')
    return table.with_column('bio', stype=Stype.text).with_column(
        'code', stype=Stype.text)


def test_a_missing_model_fails_fit_before_any_work(fitted, monkeypatch):
    _, _, snapshot, table, _, trainer, _, _ = fitted
    monkeypatch.setattr(
        'kumo_core.training.workflow.materialize_generated_training_task',
        lambda *args, **kwargs: pytest.fail('fit did work'))
    unfitted = Trainer(trainer.model_plan, accelerator='cpu')
    with pytest.raises(EmbeddingModelNotFoundError,
                       match=r"^Column 'users.bio' uses the embedding model "
                       r"'words', which is not in the store\. .*The store is "
                       r"empty\.$"):
        unfitted.fit(table, snapshot=snapshot)
    assert not (unfitted.scratch_dir / 'runs').exists()


def test_each_fit_reports_what_inference_leaves_unencoded(fitted):
    unencoded = ("Text column 'users.code' will not be encoded by default: its "
                 "values are too short or have too few known words. To "
                 "encode it anyway, set its encoder to TextEmbedding().")
    assert fitted[-1] == {
        'tables': [],
        'failed fit': [unencoded],
        'fit': [unencoded],
    }


def test_fit_and_predict_use_the_store(fitted):
    archive, root, snapshot, _, prediction_table, trainer, result, _ = fitted
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (root / 'emb' / digest / 'glove_6B_50d_emb.npy').is_file()
    assert 'test_auroc' in result.metrics
    # Spawned scorers get the store only through the trainer.
    predictions = trainer.predict(prediction_table, snapshot=snapshot,
                                  num_workers=2,
                                  binary_classification_threshold=.5)
    assert predictions.num_rows == 20


def test_load_matches_models_by_identity_not_name(fitted):
    archive, root, snapshot, _, prediction_table, _, result, _ = fitted
    renamed = EmbeddingModelStore()
    renamed.add('renamed', GloVeVectors(archive, cache_dir=root / 'emb'))
    loaded = Trainer.load(result.bundle_dir, embedding_models=renamed,
                          accelerator='cpu')
    predictions = loaded.predict(prediction_table, snapshot=snapshot,
                                 binary_classification_threshold=.5)
    assert predictions.num_rows == 20

    with pytest.raises(EmbeddingModelNotFoundError,
                       match=r"^This model was trained with glove.6B 50d, as "
                       r"'words' for 'users.bio'.*store.add\(\"words\", "
                       r'GloVeVectors\("glove.6B.zip", dim=50\)\)'):
        Trainer.load(result.bundle_dir, accelerator='cpu').predict(
            prediction_table, snapshot=snapshot,
            binary_classification_threshold=.5)
