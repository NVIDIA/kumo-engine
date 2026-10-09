# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import shutil
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from kumo_core.train.trainer import load_encoders

import kumo.engine as kumo


def _predict(trainer, table, snapshot):
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=.5)
    frame = pd.concat(pd.read_parquet(path) for path in result.files)
    return frame.sort_values('ENTITY').reset_index(drop=True)


def _as_written_before_hparams_dropped_encoder_stats(bundle):
    checkpoint_path = bundle / 'model' / 'checkpoint'
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    encoder = checkpoint['hyper_parameters']['encoder']
    assert all('_stats' not in column
               for table in encoder['table_encoder_dict'].values()
               for column in table.values())
    encoders = load_encoders(str(bundle / 'model'))
    encoder['table_encoder_dict'] = {
        table: {name: asdict(value)
                for name, value in columns.items()}
        for table, columns in encoders.encoders.items()
    }
    assert any(column['_stats']
               for table in encoder['table_encoder_dict'].values()
               for column in table.values())
    torch.save(checkpoint, checkpoint_path)
    manifest_path = bundle / 'publication-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest['artifacts']:
        if artifact['relative_path'] == 'model/checkpoint':
            artifact['sha256'] = hashlib.sha256(
                checkpoint_path.read_bytes()).hexdigest()
            artifact['size_bytes'] = checkpoint_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


def test_bundle_with_encoder_stats_in_checkpoint_hparams_loads(
        fitted, tmp_path):
    trainer, table, snapshot = fitted
    expected = _predict(trainer, table, snapshot)
    bundle = tmp_path / 'old-format'
    shutil.copytree(trainer._training_result.bundle_dir, bundle)
    _as_written_before_hparams_dropped_encoder_stats(bundle)

    restored = _predict(kumo.Trainer.load(bundle), table, snapshot)

    assert restored['ENTITY'].tolist() == expected['ENTITY'].tolist()
    assert restored['TARGET_PRED'].tolist() == expected['TARGET_PRED'].tolist()
    np.testing.assert_allclose(restored['True_PROB'], expected['True_PROB'],
                               atol=1e-4)
