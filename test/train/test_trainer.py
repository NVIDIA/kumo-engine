# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os.path as osp
import pathlib
import re
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.target.multicategorical import (
    MultiCategoricalTargetEncoder,
)
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.tensor_mapper import MultiCategoricalTensorMapper
from kumo_core.io.json import load_json as load_training_config
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.public_error import PublicTrainingError
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumo_core.train.output import (
    EvalOutput,
    PredOutput,
    StreamingPredOutputWriter,
)
from kumo_core.train.stats import (
    AllTargetRegressionStats,
    TargetHistogramAndStats,
    TargetHistogramAndStatsPair,
)
from kumo_core.train.trainer import (
    Trainer,
    _table_encoder_dict_for_model,
    create_multilabel_classification_prediction_output,
    create_multilabel_ranking_prediction_output,
)
from kumo_core.training.config import FitMode, TrainerConfig
from kumo_core.training.task_config import TaskConfig
from kumo_core.types import TaskType
from kumoapi.encoder import MultiCategorical, Numerical, NumericalList
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch_geometric.sampler import NumNeighbors

from io_utils import fsutil, parquet_util
from io_utils.json import load_json, save_json
from kumo.config.explain import StrategyMappings
from kumo.home import HomeDirManager, get_home_dir
from kumo.pquery.AST import ast_node
from kumo.testing.fixtures.task import TestTableSchema, create_task_data
from rest.backends.ml_backend.activities import generate_regression_histogram

# Setup #######################################################################

ROOT_DIR = osp.join(osp.dirname(osp.realpath(__file__)))
TEST_RESULT_DIR = osp.join(ROOT_DIR, 'test_results')
DEFAULT_TRAINER_KWARGS = dict(
    root_dir=TEST_RESULT_DIR,
    max_epochs=2,
    max_steps_per_epoch=2,
    max_val_steps=2,
    max_test_steps=2,
    loader_kwargs=dict(batch_size=3),
    optimizer_kwargs=dict(lr=0.001),
)


def test_explain_starts_and_clears_ttng_selection_capture() -> None:
    class _Datamodule:
        def __init__(self) -> None:
            self.eval_loader_kwargs = {'batch_size': 8}
            self.active_stage = None
            self.events = []

        def begin_explain_selection_capture(self, stage: str) -> None:
            self.active_stage = stage
            self.events.append(('begin', stage))

        def clear_explain_selection_capture(self) -> None:
            self.events.append(('clear', self.active_stage))
            self.active_stage = None

    class _PLTrainer:
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        limit_predict_batches = 1.0

        def test(self, *, model, datamodule, ckpt_path):
            datamodule.events.append((
                'eval',
                datamodule.active_stage,
                datamodule.eval_loader_kwargs['batch_size'],
            ))
            return []

    class _Trainer:
        _explain_runtime = Trainer._explain_runtime
        reset_explain = Trainer.reset_explain

        def __init__(self) -> None:
            self.refit_cfg = SimpleNamespace(refitted_trainval=False,
                                             refitted_full=False)
            self.pl_model = SimpleNamespace(explainers=[], outputs=None)
            self._fully_initialized = True
            self.task = SimpleNamespace(
                task_type=TaskType.BINARY_CLASSIFICATION)
            self.datamodule = _Datamodule()
            self.pl_trainer = _PLTrainer()
            self.best_ckpt = '/tmp/fake.ckpt'
            self.explain_runtime = SimpleNamespace(max_batches=1)

        def _check_readiness_for_prediction(self, fit_mode_to_use):
            pass

        def init(self):
            pass

        def set_explainers(self, stage, explain_methods, root_dir=None):
            self.pl_model.explainers = []

    trainer = _Trainer()

    Trainer.explain(trainer, explain_methods=[], stage=TrainingStage.TEST)

    assert trainer.datamodule.events == [
        ('begin', 'test'),
        ('eval', 'test', 2),
        ('clear', 'test'),
    ]
    assert trainer.datamodule.active_stage is None
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 8
    assert trainer.pl_trainer.limit_test_batches == 1.0


def test_explain_pred_ttng_uses_capped_predict_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Datamodule:
        def __init__(self) -> None:
            self.eval_loader_kwargs = {'batch_size': 8}
            self.events = []

        def begin_explain_selection_capture(self, stage: str) -> None:
            self.events.append(('begin', stage))

        def clear_explain_selection_capture(self) -> None:
            self.events.append(('clear', None))

        def predict_dataloader(self):
            self.events.append(('predict_dataloader', None))
            return 'predict-loader'

    class _PLTrainer:
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        limit_predict_batches = 1.0

        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append((self.limit_predict_batches, kwargs))
            if 'datamodule' in kwargs:
                return [{
                    'pred': torch.tensor([1, 0]),
                    'prob': torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                }]
            return []

    class _Trainer:
        _explain_runtime = Trainer._explain_runtime
        reset_explain = Trainer.reset_explain
        subgraph_collector_dir = Trainer.subgraph_collector_dir

        def __init__(self) -> None:
            self.refit_cfg = SimpleNamespace(refitted_trainval=False,
                                             refitted_full=False)
            self.pl_model = SimpleNamespace(explainers=[], outputs=None,
                                            disable_eval_metrics=False)
            self._fully_initialized = True
            self.task = SimpleNamespace(
                task_type=TaskType.BINARY_CLASSIFICATION)
            self.datamodule = _Datamodule()
            self.pl_trainer = _PLTrainer()
            self.best_ckpt = '/tmp/fake.ckpt'
            self.feature_store = object()
            self.log_dir = str(tmp_path)
            self.explain_runtime = SimpleNamespace(
                max_batches=500,
                prediction_selection_nodes=1000,
                get_explain_node_inputs=fake_get_explain_node_inputs,
                select_input_nodes=fake_select_input_nodes,
                get_eval_dataloader=fake_get_eval_dataloader,
                mapping_file_name='strategy_mappings.json',
            )

        def _check_readiness_for_prediction(self, fit_mode_to_use):
            pass

        def init(self):
            pass

        def set_explainers(self, stage, explain_methods, root_dir=None):
            from kumo_core.types import ExplainMethod
            self.pl_model.explainers = ([
                object()
            ] if ExplainMethod.SUBGRAPH_COLLECTOR in explain_methods else [])

    selected_stage_ids = torch.tensor([100, 101])
    replay_calls = []

    def fake_get_explain_node_inputs(datamodule, stage):
        assert stage == TrainingStage.PRED
        return torch.tensor([7, 7]), selected_stage_ids

    def fake_select_input_nodes(task_type, feature_store, input_nodes, input_id,
                                outputs, stage):
        assert task_type == TaskType.BINARY_CLASSIFICATION
        assert torch.equal(input_id, selected_stage_ids)
        assert stage == TrainingStage.PRED
        return selected_stage_ids, StrategyMappings(
            mappings={'first': ['0', '1']},
            orig_mappings={'first': [1007, 1007]},
        )

    def fake_get_eval_dataloader(datamodule, stage, seed_nodes):
        replay_calls.append((stage, seed_nodes))
        return 'selected-predict-loader'

    trainer = _Trainer()

    Trainer.explain(
        trainer,
        explain_methods=['subgraph_collector'],
        stage=TrainingStage.PRED,
        root_dir=str(tmp_path),
    )

    assert trainer.datamodule.events == [
        ('begin', 'pred'),
        ('predict_dataloader', None),
    ]
    assert len(trainer.pl_trainer.calls) == 2
    assert trainer.pl_trainer.calls[0][0] == 500
    assert 'datamodule' in trainer.pl_trainer.calls[0][1]
    assert trainer.pl_trainer.calls[1][0] == 500
    assert trainer.pl_trainer.calls[1][1]['dataloaders'] == 'predict-loader'
    assert replay_calls == []
    assert trainer.datamodule.eval_loader_kwargs['batch_size'] == 8
    assert trainer.pl_trainer.limit_predict_batches == 1.0


def test_forecasting_model_encoder_dict_omits_train_table_features():
    encoders = EncodersConfig(
        encoders={
            'customers': {
                'age': Numerical(),
            },
            'train_table': {
                'autoregression_labels': NumericalList(),
                'year_over_year_labels': NumericalList(),
            },
        })

    table_encoder_dict = _table_encoder_dict_for_model(
        TaskType.FORECASTING,
        encoders,
        train_table_name='train_table',
    )

    assert table_encoder_dict == {
        'customers': {
            'age': Numerical(),
        },
        'train_table': {},
    }
    assert encoders.encoders['train_table'] == {
        'autoregression_labels': NumericalList(),
        'year_over_year_labels': NumericalList(),
    }


def test_non_forecasting_model_encoder_dict_uses_encoders_directly():
    encoders = EncodersConfig(
        encoders={'train_table': {
            'target': MultiCategorical()
        }})

    table_encoder_dict = _table_encoder_dict_for_model(
        TaskType.BINARY_CLASSIFICATION,
        encoders,
        train_table_name='train_table',
    )

    assert table_encoder_dict is encoders.encoders


@pytest.fixture(autouse=True)
def cleanup():
    yield

    if osp.exists(TEST_RESULT_DIR):
        shutil.rmtree(TEST_RESULT_DIR, ignore_errors=True)

    folder = osp.join(get_home_dir(), 'graphs')
    shutil.rmtree(folder, ignore_errors=True)

    folder = osp.join(get_home_dir(), 'tasks')
    shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize('task_type', [TaskType.BINARY_CLASSIFICATION])
def test_trainer_compare_config(temporal_task, task_type):
    feature_store, graph_store, encoders = temporal_task

    trainer = Trainer(
        task=TaskConfig(task_type, name='TEST_COMPARE_CONFIG'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
        sampler_kwargs=dict(num_neighbors=[10, 10]),
        **DEFAULT_TRAINER_KWARGS,
    )

    # Change the sampler arguments:
    trainer2 = Trainer(
        task=TaskConfig(task_type, name='TEST_COMPARE_CONFIG'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        sampler_kwargs=dict(num_neighbors=[20, 20]),
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
        **DEFAULT_TRAINER_KWARGS,
    )

    buffer = io.StringIO()
    trainer.compare_config(trainer, file=buffer)
    out = buffer.getvalue()
    assert 'No differences found.' in out

    buffer = io.StringIO()
    trainer.compare_config(trainer2, file=buffer)
    out = buffer.getvalue()
    assert re.search(r'\+.*num_neighbors.*', out)
    assert re.search(r'-.*num_neighbors.*', out)


def test_trainer_load_preserves_edge_type_with_trailing_underscore_rel(
        tmp_path: pathlib.Path):
    """Regression test for EdgeTypeStr JSON round-trip bug (#27593): a
    trailing underscore in a FK column name (e.g. 'USER_ID_') must not
    get corrupted when ``Trainer.load()`` unqualifies an edge-type key
    that was serialized before the ``KUMO_EDGE_KEY``-delimited encoding
    existed, i.e. plain ``'__'``-joined (``NumNeighbors(values={EdgeTypeStr
    single-string ...})`` re-splits on ``'__'``, which is ambiguous for a
    table/column name containing an underscore right before the
    separator). ``Trainer.load()``'s backward-compat unqualification path
    (``_convert_key`` in ``Trainer.load``) recovers the correct edge type
    by matching the corrupted string form against ``graph_store.edge_types``.
    """
    df_dict = {
        'USERS':
        pd.DataFrame({
            'USER_ID': [0, 1, 2, 3],
            'AGE': [20, 30, 40, 50],
        }),
        'ORDERS':
        pd.DataFrame({
            'ORDER_ID': [0, 1, 2, 3, 4, 5],
            'USER_ID_': [0, 0, 1, 1, 2, 3],  # trailing underscore in FK column
            'TIME': ['1990', '1991', '1992', '1990', '1991', '1992'],
        }),
    }
    schema_dict = {
        'USERS':
        TestTableSchema(
            stypes={'AGE': Stype.numerical},
            pkey_name='USER_ID',
        ),
        'ORDERS':
        TestTableSchema(
            stypes={'TIME': Stype.timestamp},
            fkeys=[('USER_ID_', 'USERS')],
            time_col='TIME',
        ),
    }
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3] * 3,
        'TIME': ['1993'] * 4 + ['1994'] * 4 + ['1995'] * 4,
        'TARGET': [True, True, False, True] * 3,
        'TRAIN_MASK': [True] * 4 + [False] * 8,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 8 + [True] * 4,
    })
    feature_store, graph_store, encoders = create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_df=train_table_df,
    )

    edge_types = graph_store.edge_types
    num_neighbors_dict = {et: [-1, -1] for et in edge_types}

    trainer = Trainer(
        task=TaskConfig(TaskType.BINARY_CLASSIFICATION,
                        name='TRAILING_UNDERSCORE'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
        sampler_kwargs=dict(num_neighbors=NumNeighbors(
            values=num_neighbors_dict, default=[-1, -1])),
        root_dir=str(tmp_path),
        max_epochs=1,
    )

    config_path = tmp_path / 'config.json'
    save_json(trainer.config(), str(config_path))

    # Rewrite the on-disk config to the pre-KUMO_EDGE_KEY, plain
    # '__'-joined format that #27593 was filed against (a real model
    # directory saved by an older Trainer version could look like this).
    raw = json.loads(config_path.read_text())
    raw['sampler_kwargs']['num_neighbors']['values'] = {
        '__'.join(edge_type): value
        for edge_type, value in
        zip(edge_types, raw['sampler_kwargs']['num_neighbors']['values'].
            values())
    }
    config_path.write_text(json.dumps(raw))

    loaded_trainer = Trainer.load(
        TaskConfig(TaskType.BINARY_CLASSIFICATION,
                   name='TRAILING_UNDERSCORE'),
        str(tmp_path),
        version=None,
        infer_config_path=False,
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
    )

    loaded_tuples = {
        k if isinstance(k, tuple) else k.to_tuple()
        for k in loaded_trainer.sampler_kwargs.num_neighbors.values
    }
    assert loaded_tuples == set(num_neighbors_dict.keys())


def test_cuda_oom_exception_handling(tmp_path: pathlib.Path):
    fake_error_message = "RuntimeError: CUDA out of memory"

    trainer = Trainer(
        task=TaskConfig(TaskType.BINARY_CLASSIFICATION, name='TEST_CUDA_OOM'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        raise_fake_trial_error_msg=fake_error_message,
        root_dir=str(tmp_path),
        max_epochs=1,
    )

    log_dir = trainer.log_dir
    assert fsutil.exists(log_dir)

    # Trainer should return a CUDA out of memory error message string:
    with pytest.raises(RuntimeError, match=fake_error_message):
        trainer.fit()

    # The log directory should be deleted:
    assert not fsutil.exists(log_dir)

    trainer.refit_cfg.best_val_epoch = 999
    with pytest.raises(
            PublicTrainingError,
            match="GPU out-of-memory error during refit.",
    ):
        trainer.refit(fit_mode=FitMode.refit_trainval)


@pytest.mark.parametrize('task_type', [TaskType.BINARY_CLASSIFICATION])
def test_load_model_from_checkpoint(temporal_task, task_type):
    feature_store, graph_store, encoders = temporal_task
    entity_table_names = feature_store.entity_table_names
    train_table_name = feature_store.train_table_name

    def init_pl_model(trainer):
        # Exercises exactly the checkpoint-dispatch steps `Trainer.init()`
        # runs, without going through the DiskGraph-only graph-sampler /
        # datamodule setup that precedes them (irrelevant to this test).
        trainer.initialize_model(
            node_types=sorted(feature_store.table_names),
            edge_types=graph_store.edge_types,
            train_table_name=train_table_name,
            has_train_table_time_col=feature_store.has_train_table_time_col,
            time_col_dict=feature_store.time_col_dict,
            entity_table_names=entity_table_names,
        )
        with feature_store.safe_batch_mode(batch_size=50_000):
            rhs_index = feature_store[entity_table_names[-1], 'pkey', :]
        rhs_index_map = pd.DataFrame(index=rhs_index)
        rhs_index_map[NODE_INDEX_COLUMN] = range(len(rhs_index_map))
        trainer.create_pl_model(
            entity_table_names=entity_table_names,
            train_table_name=train_table_name,
            rhs_index_map=rhs_index_map,
            ckpt_path=trainer.checkpoint_path,
        )

    method_name = (
        'kumo_core.train.base_module.BaseLightningModule.load_checkpoint')
    trainer = Trainer(
        task=TaskConfig(task_type, name='TEST_CUDA_OOM'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
        checkpoint_path=None,
        **DEFAULT_TRAINER_KWARGS,
    )
    with patch(method_name) as mock:
        init_pl_model(trainer)

    assert mock.call_count == 0

    trainer = Trainer(
        task=TaskConfig(task_type, name='TEST_CUDA_OOM'),
        encoder=HeteroTensorFrameEncoder(),
        model=HeteroGNN(channels=8, num_mp_layers=2),
        head=NodePredHead(),
        feature_store=feature_store,
        graph_store=graph_store,
        encoders=encoders,
        checkpoint_path='path/to/checkpoint',
        **DEFAULT_TRAINER_KWARGS,
    )
    with patch(method_name) as mock:
        init_pl_model(trainer)
    assert mock.call_count == 1
    assert 'path/to/checkpoint' in mock.call_args[0][0]


def test_pred_output_save(tmp_path: pathlib.Path):
    temp_path: str = tmp_path.resolve().as_posix()
    # Create a random DF with 100 rows and 5 columns
    # Memory usage: 4128 bytes
    pred_df = pd.DataFrame(np.random.rand(100, 5), columns=list('ABCDE'))

    # Create another random DF with 100 rows and 3 columns
    # Memory usage: 2528 bytes
    embedding_df = pd.DataFrame(np.random.rand(100, 3), columns=list('XYZ'))

    pred_output = PredOutput(pred_df, embedding_df, max_size_per_file=1000)

    pred_output.save(osp.join(temp_path, 'pred_output'))
    pred_path = osp.join(temp_path, 'pred_output/predictions.parquet')
    embedding_path = osp.join(temp_path, 'pred_output/embeddings.parquet')
    # This should produce 5 files for predictions and 3 files for embeddings:
    assert len(fsutil.listdir(pred_path)) == 5
    assert len(fsutil.listdir(embedding_path)) == 3

    # Load the saved prediction and embedding output and compare:
    predictions_loaded = parquet_util.read_parquet(pred_path)
    embeddings_loaded = parquet_util.read_parquet(embedding_path)
    pd.testing.assert_frame_equal(predictions_loaded, pred_df)
    pd.testing.assert_frame_equal(embeddings_loaded, embedding_df)


def test_pred_output_save_mixed_type_class_column(tmp_path: pathlib.Path):
    pred_df = pd.DataFrame({
        'ENTITY': ['a', 'a', 'b'],
        'CLASS': [np.int64(1), '', np.int64(2)],
        'SCORE': [0.7, 0.2, 0.1],
        'PREDICTED': [True, False, False],
    })

    pred_output = PredOutput(pred_df)
    pred_output.save(str(tmp_path / 'pred_output'))

    pred_path = tmp_path / 'pred_output' / 'predictions.parquet'
    predictions_loaded = parquet_util.read_parquet(str(pred_path))
    assert predictions_loaded['CLASS'].tolist() == ['1', '', '2']


def test_streaming_pred_output_stabilizes_class_schema_across_batches(
        tmp_path: pathlib.Path):
    output_dir = str(tmp_path / 'pred_output')
    with StreamingPredOutputWriter(output_dir, partition_id=0) as writer:
        writer.write(
            PredOutput(
                pd.DataFrame({
                    'ENTITY': ['a', 'b'],
                    'CLASS': pd.Series([1, 2], dtype=object),
                })))
        writer.write(
            PredOutput(
                pd.DataFrame({
                    'ENTITY': ['c'],
                    'CLASS': pd.Series(['other'], dtype=object),
                })))

    predictions = parquet_util.read_parquet(
        osp.join(output_dir, 'predictions.parquet'))
    assert predictions['CLASS'].tolist() == ['1', '2', 'other']


def test_streaming_pred_output_writes_artifact_specific_final_schema(
        tmp_path: pathlib.Path):
    output_dir = str(tmp_path / 'pred_output')
    predictions_dir = tmp_path / 'pred_output' / 'predictions.parquet'
    embeddings_dir = tmp_path / 'pred_output' / 'embeddings.parquet'
    predictions_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)
    stale_files = [
        predictions_dir / 'part-00000-99999.parquet',
        embeddings_dir / 'part-00000-99999.parquet',
    ]
    preserved_files = [
        predictions_dir / 'part-00001-99999.parquet',
        embeddings_dir / 'part-00001-99999.parquet',
    ]
    for path in stale_files + preserved_files:
        path.write_bytes(b'stale')
    job_start_time = pd.Timestamp('2026-01-02T03:04:05Z')
    with StreamingPredOutputWriter(
            output_dir,
            partition_id=0,
            output_extra_fields={
                'PREDICTIONS': ['ANCHOR_TIMESTAMP'],
                'EMBEDDINGS': ['JOB_TIMESTAMP'],
            },
            job_start_time=job_start_time,
    ) as writer:
        writer.write(
            PredOutput(
                pd.DataFrame({
                    'ID': ['a'],
                    'SCORE': [0.5],
                    'ANCHOR_TIMESTAMP': [pd.Timestamp('2026-01-01T00:00:00')],
                    'JOB_TIMESTAMP': [pd.Timestamp('2026-01-02T03:04:05')],
                }),
                pd.DataFrame({
                    'ID': ['a'],
                    'EMBEDDING': [[1.0, 2.0]],
                    'ANCHOR_TIMESTAMP': [pd.Timestamp('2026-01-01T00:00:00')],
                }),
            ))

    assert all(not path.exists() for path in stale_files)
    assert all(path.exists() for path in preserved_files)
    for path in preserved_files:
        path.unlink()
    predictions = parquet_util.read_parquet(
        osp.join(output_dir, 'predictions.parquet'))
    embeddings = parquet_util.read_parquet(
        osp.join(output_dir, 'embeddings.parquet'))
    assert list(predictions.columns) == ['ID', 'SCORE', 'ANCHOR_TIMESTAMP']
    assert list(embeddings.columns) == ['ID', 'EMBEDDING', 'JOB_TIMESTAMP']
    assert '__index_level_0__' not in embeddings.columns
    assert embeddings['JOB_TIMESTAMP'].iloc[0] == job_start_time.tz_localize(
        None)


def test_streaming_pred_output_rejects_missing_requested_metadata(
        tmp_path: pathlib.Path):
    with StreamingPredOutputWriter(
            str(tmp_path / 'pred_output'),
            partition_id=0,
            output_extra_fields={
                'PREDICTIONS': ['ANCHOR_TIMESTAMP'],
            },
    ) as writer:
        with pytest.raises(
                ValueError,
                match='PREDICTIONS.*ANCHOR_TIMESTAMP',
        ):
            writer.write(PredOutput(pd.DataFrame({'ID': ['a']})))


def test_generate_regression_histogram(tmp_path: pathlib.Path):
    log_dir = str(tmp_path / "log_dir")
    job_dir = str(tmp_path / "job_dir")
    mgr = HomeDirManager(job_dir)

    # Case 1: all files missing
    generate_regression_histogram(log_dir, job_dir)
    result = load_json(mgr.all_target_regression_stats,
                       AllTargetRegressionStats)
    assert result == AllTargetRegressionStats.empty()

    # Case 2: only val stats available
    stats = TargetHistogramAndStatsPair(
        label=TargetHistogramAndStats(target_mean=1),
        prediction=TargetHistogramAndStats(target_mean=2))
    save_json(
        stats,
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.VAL))
    generate_regression_histogram(log_dir, job_dir)
    result = load_json(mgr.all_target_regression_stats,
                       AllTargetRegressionStats)
    assert result.label_distribution.val_table_stats is not None
    assert result.label_distribution.val_table_stats.target_mean == 1
    assert result.prediction_distribution.val_table_stats is not None
    assert result.prediction_distribution.val_table_stats.target_mean == 2

    # Case 2: all stats available
    stats = TargetHistogramAndStatsPair(
        label=TargetHistogramAndStats(target_mean=3),
        prediction=TargetHistogramAndStats(target_mean=4))
    save_json(
        stats,
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.TEST))
    save_json(
        stats,
        EvalOutput.get_prediction_histogram_path(log_dir, TrainingStage.TRAIN))
    generate_regression_histogram(log_dir, job_dir)
    result = load_json(mgr.all_target_regression_stats,
                       AllTargetRegressionStats)
    assert result.label_distribution.test_table_stats is not None
    assert result.label_distribution.train_table_stats is not None
    assert result.prediction_distribution.test_table_stats is not None
    assert result.prediction_distribution.train_table_stats is not None


@pytest.mark.parametrize('with_target', [True, False])
@pytest.mark.parametrize('is_array_targets', [True, False])
def test_create_multilabel_ranking_prediction_output(
    with_target,
    is_array_targets,
    monkeypatch,
):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        is_array_targets,
    )

    num_rows = 3
    num_classes = 4
    df = pd.DataFrame({
        'ENTITY': [0, 1, 2],
        'TIMESTAMP': ['2025-01-01', '2025-01-02', '2025-01-03'],
    })
    target = None
    if with_target:  # when not BP
        target = F.one_hot(torch.tensor([0, 1, 2]), num_classes=num_classes)
        if is_array_targets:
            df['TARGET'] = [['A', 'B'], ['B'], ['C']]
        else:
            df['TARGET'] = ['A|B', 'B', 'C']

    prob = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.8, 0.7, 0.6, 0.5],
        [0.9, 0.8, 0.7, 0.6],
    ])
    category_counts = [['A', 4], ['B', 3], ['C', 2], ['D', 1]]
    target_encoder = MultiCategoricalTargetEncoder(
        tensor_mapper=MultiCategoricalTensorMapper(
            category_counts=category_counts,
            sep=None if is_array_targets else '|',
        ))

    for top_k, expected_top_k in [
        (1, 1),
        (4, 4),
        (999, num_classes),
    ]:
        out = create_multilabel_ranking_prediction_output(
            df=df,
            target_encoder=target_encoder,
            prob=prob,
            target=target,
            top_k=top_k,
        )
        assert len(out) == num_rows * expected_top_k
        expected_cols = {'ENTITY', 'TIMESTAMP', 'CLASS', 'SCORE'}
        if with_target:
            expected_cols.add('LABEL')
        assert set(out.columns) == expected_cols

    with pytest.raises(ValueError, match='must be greater than 0'):
        create_multilabel_ranking_prediction_output(
            df=df,
            target_encoder=target_encoder,
            prob=prob,
            target=None,
            top_k=0,
        )


@pytest.mark.parametrize('with_target', [True, False])
@pytest.mark.parametrize('is_array_targets', [True, False])
def test_create_multilabel_classification_prediction_output(
    with_target,
    is_array_targets,
    monkeypatch,
):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        is_array_targets,
    )

    num_rows = 3
    num_classes = 4
    df = pd.DataFrame({
        'ENTITY': [0, 1, 2],
        'TIMESTAMP': ['2025-01-01', '2025-01-02', '2025-01-03'],
    })
    target = None
    if with_target:  # when not BP
        target = F.one_hot(torch.tensor([0, 1, 2]), num_classes=num_classes)

    prob = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.8, 0.7, 0.6, 0.5],
        [0.9, 0.8, 0.7, 0.6],
    ])
    pred = prob > 0.5
    category_counts = [['A', 4], ['B', 3], ['C', 2], ['D', 1]]
    target_encoder = MultiCategoricalTargetEncoder(
        tensor_mapper=MultiCategoricalTensorMapper(
            category_counts=category_counts,
            sep=None if is_array_targets else '|',
        ))

    for top_k, expected_top_k in [
        (1, 1),
        (4, 4),
        (999, num_classes),
    ]:
        out = create_multilabel_classification_prediction_output(
            df=df,
            target_encoder=target_encoder,
            prob=prob,
            pred=pred,
            target=target,
            top_k=top_k,
        )
        assert len(out) == num_rows * expected_top_k
        expected_cols = {'ENTITY', 'TIMESTAMP', 'CLASS', 'SCORE', 'PREDICTED'}
        if with_target:
            expected_cols.add('LABEL')
        assert set(out.columns) == expected_cols

    with pytest.raises(ValueError, match='must be greater than 0'):
        create_multilabel_classification_prediction_output(
            df=df,
            target_encoder=target_encoder,
            prob=prob,
            pred=pred,
            target=None,
            top_k=0,
        )


@pytest.mark.parametrize(
    ('edge_type', 'expected_edge_type'),
    [
        (
            ('transactions__KUMO_EDGE_KEY__customer_id'
             '__KUMO_EDGE_KEY__customers'),
            'transactions__customer_id__customers',
        ),
        (
            'source.transactions__customer_id__source.customers',
            'source.transactions__customer_id__source.customers',
        ),
    ],
)
def test_trainer_config_loads_serialized_edge_type_sampler_key(
    edge_type, expected_edge_type, tmp_path
):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({
        'encoder': {'table_encoder_dict': {}},
        'model': None,
        'head': None,
        'sampler_kwargs': {
            'num_neighbors': {
                'values': {edge_type: [64, 12]},
                'default': [0, 0],
            },
        },
    }))
    config = load_training_config(str(path), TrainerConfig)

    assert config is not None
    assert config.sampler_kwargs is not None
    assert config.sampler_kwargs.num_neighbors.values == {
        expected_edge_type: [64, 12],
    }
