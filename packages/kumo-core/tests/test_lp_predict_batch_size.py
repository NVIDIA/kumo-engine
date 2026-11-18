# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Link prediction honors an explicit ``predict(batch_size=...)``.

The node path applies ``batch_size`` to the evaluation loader; link prediction
used to dispatch to ``_predict_lp`` without it, so batches followed the
evaluation batch size and DataLoader worker sharding. The unit tests drive the
real ``_predict_lp`` against a scripted Lightning trainer; the end-to-end test
trains a small hybrid model through ``kumo.engine`` when it is installed.
"""
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from kumo_core.train.diskgraph_nested_trainer import DiskgraphNestedTrainer
from kumo_core.training.config import FitMode
from kumo_core.types import TaskType

SEEDS = 10
CATALOG = 6
TOP_K = 3
DEFAULT_EVAL_BATCH = 4


# Scripted trainer state ######################################################


class _Loader:
    def __init__(self, seeds: list[int], batch_size: int) -> None:
        self.batch_size = batch_size
        self.batches = [
            seeds[start:start + batch_size]
            for start in range(0, len(seeds), batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class _RecordingDict(dict):
    """An ``eval_loader_kwargs`` that counts writes."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.writes = 0

    def __setitem__(self, key, value) -> None:
        self.writes += 1
        super().__setitem__(key, value)


class _DataModule:
    def __init__(self) -> None:
        self.eval_loader_kwargs = _RecordingDict(batch_size=DEFAULT_EVAL_BATCH,
                                                 num_workers=2)
        self.task_manifest = SimpleNamespace(target_node_type='items')
        self._seed_node_type = 'users'
        self.time_col_dict: dict[str, str] = {}
        self.loaders: list[_Loader] = []

    def predict_dataloader(self) -> _Loader:
        loader = _Loader(list(range(SEEDS)),
                         int(self.eval_loader_kwargs['batch_size']))
        self.loaders.append(loader)
        return loader

    def defer_dataloader_worker_shutdown(self, stage: str):
        return contextlib.nullcontext()

    def shutdown_dataloader_workers(self, stage: str) -> None:
        pass


class _Module:
    num_rhs_nodes = CATALOG
    pred_top_k = None
    eval_top_k = None
    entity_table_names = ('users', 'items')

    def _get_register_embedding_hook_fn(self):
        return None

    def has_embedding_hook(self) -> bool:
        return False

    def set_rhs_index_map(self, frame: pd.DataFrame) -> None:
        self.rhs_index_map = frame


class _Lightning:
    """Runs the writer callbacks over the loader, as Lightning predict does.

    Scores depend only on the anchor, so every batching gives one frame.
    """
    def __init__(self, fail_at: int | None = None) -> None:
        self.callbacks: list = []
        self.fail_at = fail_at
        self.anchors_per_step: list[int] = []
        self.batch_size_seen: list[int] = []

    def predict(self, *, model, datamodule, ckpt_path, return_predictions):
        writers = [cb for cb in self.callbacks if hasattr(cb, 'write_prediction')]
        for index, seeds in enumerate(datamodule.predict_dataloader()):
            self.batch_size_seen.append(
                datamodule.eval_loader_kwargs['batch_size'])
            if index == self.fail_at:
                raise RuntimeError('scripted failure')
            self.anchors_per_step.append(len(seeds))
            pkey = np.asarray(seeds, dtype=np.int64)
            target = (pkey[:, None] + np.arange(TOP_K)) % CATALOG
            score = (-np.arange(TOP_K, dtype=np.float32)[None, :] +
                     pkey[:, None].astype(np.float32))
            for writer in writers:
                writer.write_on_batch_end(self, model, {
                    'pkey': pkey, 'target': target, 'score': score
                }, None, None, index, 0)
        for writer in writers:
            writer.on_predict_end(self, model)
        return None


def _trainer(monkeypatch, fail_at: int | None = None) -> DiskgraphNestedTrainer:
    monkeypatch.setattr(DiskgraphNestedTrainer, 'best_ckpt',
                        property(lambda self: 'best.ckpt'))
    trainer = DiskgraphNestedTrainer.__new__(DiskgraphNestedTrainer)
    trainer.task = SimpleNamespace(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION)
    trainer.fit_mode = FitMode.fit
    trainer._fully_initialized = True
    trainer._prediction_mode = False
    trainer.datamodule = _DataModule()
    trainer.pl_model = _Module()
    trainer.pl_trainer = _Lightning(fail_at)
    trainer._rhs_primary_key_node_ids = np.arange(CATALOG, dtype=np.int64)
    trainer._rhs_primary_key_catalog = 100 + np.arange(CATALOG,
                                                       dtype=np.int64)
    return trainer


def _predict(trainer, output_dir: Path, **options) -> pd.DataFrame:
    """Run ``predict`` and read back every written prediction row."""
    task = SimpleNamespace(train_table=SimpleNamespace(
        foreign_key_columns=['ENTITY']))
    trainer.predict(output_dir=str(output_dir), task=task, top_k=TOP_K,
                    ignore_time_col=True, **options)
    parts = sorted(part for part in output_dir.rglob('*.parquet')
                   if part.is_file())
    assert parts
    return pd.concat([pd.read_parquet(part) for part in parts],
                     ignore_index=True)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(['ENTITY', 'SCORE'], ascending=[True, False],
                             ignore_index=True)


@pytest.mark.parametrize('batch_size', [1, 3, SEEDS, 2 * SEEDS])
def test_explicit_batch_size_sets_the_anchors_of_every_predict_step(
        monkeypatch, tmp_path, batch_size):
    trainer = _trainer(monkeypatch)
    frame = _predict(trainer, tmp_path / 'explicit', batch_size=batch_size)
    lightning = trainer.pl_trainer
    expected = [min(batch_size, SEEDS - start)
                for start in range(0, SEEDS, batch_size)]
    assert lightning.anchors_per_step == expected
    assert set(lightning.batch_size_seen) == {batch_size}
    # Every loader the call built (progress total, writer flush, predict)
    # used the requested size.
    assert {loader.batch_size for loader in trainer.datamodule.loaders} == {
        batch_size}
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == (
        DEFAULT_EVAL_BATCH)
    assert len(frame) == SEEDS * TOP_K
    default = _predict(_trainer(monkeypatch), tmp_path / 'default')
    pd.testing.assert_frame_equal(_sorted(frame), _sorted(default))


def test_default_call_leaves_the_loader_batch_size_untouched(
        monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    kwargs = trainer.datamodule.eval_loader_kwargs
    _predict(trainer, tmp_path, batch_size=None)
    assert trainer.datamodule.eval_loader_kwargs is kwargs
    assert kwargs.writes == 0 and kwargs['batch_size'] == DEFAULT_EVAL_BATCH
    assert trainer.pl_trainer.anchors_per_step == [4, 4, 2]


def test_progress_total_counts_the_requested_batches(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    _predict(trainer, tmp_path / 'out', batch_size=1,
             progress_log_dir=str(tmp_path / 'progress'))
    assert trainer.datamodule.loaders[0].batch_size == 1
    assert len(trainer.datamodule.loaders[0]) == SEEDS


def test_batch_size_is_restored_when_prediction_fails(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch, fail_at=1)
    with pytest.raises(RuntimeError, match='scripted failure'):
        _predict(trainer, tmp_path, batch_size=1)
    assert trainer.pl_trainer.batch_size_seen == [1, 1]
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == (
        DEFAULT_EVAL_BATCH)


@pytest.mark.parametrize('batch_size', [0, -1, 1.0, True, '1'])
def test_invalid_batch_size_is_rejected_before_any_state_change(
        monkeypatch, tmp_path, batch_size):
    trainer = _trainer(monkeypatch)

    def refuse_init(**_):
        raise AssertionError('init must not run')

    trainer.init = refuse_init  # type: ignore[method-assign]
    with pytest.raises(ValueError, match='batch_size must be a positive'):
        _predict(trainer, tmp_path, batch_size=batch_size)
    assert trainer._prediction_mode is False
    assert trainer.datamodule.eval_loader_kwargs.writes == 0


# End to end: a small hybrid model through kumo.engine ########################

QUERY = ('PREDICT LIST_DISTINCT(transactions.article_id, 0, 30, days) '
         'RANK TOP 10 FOR EACH customer.customer_id '
         'WHERE COUNT(transactions.*, -90, 0, days) > 1')
ANCHOR = pd.Timestamp('2020-06-01')
#: Tolerance for float32 scores of one model under another batch composition.
SCORE_TOLERANCE = 1e-5


def _write_hybrid_tables(root, *, customers=48, articles=30, events=2400,
                         seed=3):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    pd.DataFrame({
        'customer_id': np.arange(customers, dtype=np.int64),
        'age': rng.integers(18, 70, customers).astype(np.float64),
    }).to_parquet(root / 'customer.parquet')
    pd.DataFrame({
        'article_id': np.arange(500, 500 + articles, dtype=np.int64),
        'colour': rng.choice([f'c{i}' for i in range(5)], articles),
        'kind': rng.choice([f'k{i}' for i in range(7)], articles),
    }).to_parquet(root / 'article.parquet')
    weights = 1.0 / np.arange(1, articles + 1)
    pd.DataFrame({
        'transaction_id': np.arange(events, dtype=np.int64),
        'customer_id': rng.integers(0, customers, events).astype(np.int64),
        'article_id': (500 + rng.choice(articles, events,
                                        p=weights / weights.sum())).astype(
                                            np.int64),
        't_dat': (pd.Timestamp('2019-06-01') +
                  pd.to_timedelta(rng.integers(0, 380, events), unit='D')),
        'price': rng.random(events),
    }).to_parquet(root / 'transactions.parquet')


def _hybrid_graph(kumo, connector):
    from kumoapi.typing import Stype

    customer = kumo.Table.from_source_table(
        connector['customer.parquet'], ['customer_id', 'age'],
        primary_key='customer_id').with_column('age', stype=Stype.numerical)
    article = kumo.Table.from_source_table(
        connector['article.parquet'], ['article_id', 'colour', 'kind'],
        primary_key='article_id')
    for column in ('colour', 'kind'):
        article = article.with_column(column, stype=Stype.categorical)
    transactions = kumo.Table.from_source_table(
        connector['transactions.parquet'],
        ['transaction_id', 'customer_id', 'article_id', 't_dat', 'price'],
        primary_key='transaction_id',
        time_column='t_dat').with_column('price', stype=Stype.numerical)
    return kumo.Graph(
        tables={'customer': customer, 'article': article,
                'transactions': transactions},
        edges=[kumo.Edge(src_table='transactions', fkey='customer_id',
                         dst_table='customer'),
               kumo.Edge(src_table='transactions', fkey='article_id',
                         dst_table='article')])


def _hybrid_plan(query, snapshot):
    from kumoapi.model_plan import (
        AggregationType,
        DistanceMeasureType,
        GNNModelPlan,
        LinkPredOutputType,
        RHSEmbeddingMode,
        RunMode,
    )

    plan = query.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    arch = plan.model_architecture
    arch.module = LinkPredOutputType.CONTEXT
    arch.handle_new_target_entities = True
    arch.distance_measure = [DistanceMeasureType.COSINE]
    arch.target_embedding_mode = [RHSEmbeddingMode.FEATURE]
    arch.model = [GNNModelPlan(channels=[16], aggregation=[[
        AggregationType.MEAN]], dropout=[0.0])]
    arch.normalization = [None]
    job = plan.training_job
    job.num_experiments = 1
    job.manual_seed = 7
    job.metrics = ['ndcg@10']
    job.tune_metric = 'ndcg@10'
    job.disable_explain = True
    job.refit_trainval = False
    job.refit_full = False
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    plan.optimization.batch_size = [16]
    plan.neighbor_sampling.num_neighbors = [[4, 4]]
    return plan


@contextlib.contextmanager
def _scratch(kumo, path):
    from kumo.engine._scratch import get_scratch_dir

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=path)
    try:
        yield
    finally:
        kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def hybrid(tmp_path_factory):
    kumo = pytest.importorskip(
        'kumo.engine', reason='the end-to-end check trains with kumo.engine')
    root = tmp_path_factory.mktemp('lp-batch-size')
    _write_hybrid_tables(root / 'data')
    with _scratch(kumo, root / 'scratch'):
        connector = kumo.FileConnector(root_dir=str(root / 'data'))
        graph = _hybrid_graph(kumo, connector)
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(graph=graph, query=QUERY)
        trainer = kumo.Trainer(_hybrid_plan(query, snapshot),
                               accelerator='cpu')
        trainer.fit(query.generate_training_table(snapshot),
                    snapshot=snapshot)
        transactions = pd.read_parquet(root / 'data' / 'transactions.parquet')
        window = transactions[
            (transactions.t_dat > ANCHOR - pd.Timedelta(90, unit='D'))
            & (transactions.t_dat <= ANCHOR)]
        counts = window.groupby('customer_id').size()
        customers = np.sort(counts[counts > 1].index.to_numpy())
        pd.DataFrame({
            'ENTITY1': customers.astype(np.int64),
            'TIMESTAMP': pd.Series([ANCHOR] * len(customers),
                                   dtype='datetime64[ns]'),
        }).to_parquet(root / 'data' / 'rows.parquet')
        table = kumo.PredictionTable.from_source_table(
            connector['rows.parquet'], pq=query, snapshot=snapshot,
            plan=trainer.training_table_generation_plan)
        yield SimpleNamespace(kumo=kumo, trainer=trainer, snapshot=snapshot,
                              table=table, rows=len(customers), root=root)


@contextlib.contextmanager
def _float32_matmuls(patch: pytest.MonkeyPatch):
    """Hold float32 matmuls at full precision for the whole call.

    Engine sets ``medium`` when it builds its Lightning trainer; on CPUs with
    bfloat16 matmul support that rounds linears of more than a few rows, which
    would mix precision into a comparison of batch compositions. The process
    settings are restored afterwards.
    """
    from kumo_core.explain.walk_graph import mkldnn_matmul_fp32_precision

    def keep(precision: str) -> None:
        del precision

    setter = torch.set_float32_matmul_precision
    previous = torch.get_float32_matmul_precision()
    previous_mkldnn = mkldnn_matmul_fp32_precision()
    setter('highest')
    patch.setattr(torch, 'set_float32_matmul_precision', keep)
    try:
        yield
    finally:
        setter(previous)
        if previous_mkldnn is not None:
            torch.backends.mkldnn.matmul.fp32_precision = previous_mkldnn


def _scored(hybrid, monkeypatch, *, workers, **options):
    """Predict with ``workers`` DataLoader workers; record seeds per step."""
    from kumo_core.train.hybrid_link_gnn_module import (
        HybridLinkGNNLightningModule,
    )
    from kumo_core.training.runtime import LocalDiskgraphTrainingRuntime

    auto_config = LocalDiskgraphTrainingRuntime.auto_config

    def with_workers(self, **kwargs):
        return {**auto_config(self, **kwargs), 'num_workers': workers}

    predict_step = HybridLinkGNNLightningModule.predict_step
    seeds: list[int] = []

    def recording(module, batch, batch_idx, *args, **kwargs):
        seeds.append(int(batch[module.lhs_name].batch_size))
        return predict_step(module, batch, batch_idx, *args, **kwargs)

    with monkeypatch.context() as patch, \
            _float32_matmuls(patch), \
            _scratch(hybrid.kumo, hybrid.root / 'scratch'):
        patch.setattr(LocalDiskgraphTrainingRuntime, 'auto_config',
                      with_workers)
        patch.setattr(HybridLinkGNNLightningModule, 'predict_step', recording)
        frame = hybrid.trainer.predict(hybrid.table, snapshot=hybrid.snapshot,
                                       num_classes_to_return=10,
                                       **options).frame
    return frame, seeds


def _assert_same_lists(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Per anchor: equal score lists within tolerance and equal item sets
    except among items tied (within tolerance) with the last listed score."""
    assert set(left.ENTITY) == set(right.ENTITY)
    for entity, mine in left.groupby('ENTITY'):
        theirs = right[right.ENTITY == entity]
        a = mine.sort_values('SCORE', ascending=False)
        b = theirs.sort_values('SCORE', ascending=False)
        assert len(a) == len(b)
        np.testing.assert_allclose(a.SCORE.to_numpy(np.float64),
                                   b.SCORE.to_numpy(np.float64), rtol=0,
                                   atol=SCORE_TOLERANCE)
        last = min(a.SCORE.min(), b.SCORE.min())
        certain_a = set(a.CLASS[a.SCORE > last + SCORE_TOLERANCE])
        certain_b = set(b.CLASS[b.SCORE > last + SCORE_TOLERANCE])
        assert certain_a == certain_b
        assert certain_a <= set(b.CLASS) and certain_b <= set(a.CLASS)


@pytest.mark.parametrize('workers', [0, 1, 2])
def test_hybrid_link_prediction_scores_one_anchor_per_step(
        hybrid, monkeypatch, workers):
    default, default_seeds = _scored(hybrid, monkeypatch, workers=workers)
    single, single_seeds = _scored(hybrid, monkeypatch, workers=workers,
                                   batch_size=1)
    assert single_seeds == [1] * hybrid.rows
    assert sum(default_seeds) == hybrid.rows
    # The default batches hold several anchors, so the option took effect.
    assert max(default_seeds) > 1
    assert len(single) == len(default) == 10 * hybrid.rows
    _assert_same_lists(single, default)
