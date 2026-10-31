# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The predictor a model bundle's manifest names."""

from pathlib import Path
from typing import Any, Mapping, Protocol, Tuple, Union

import pandas as pd

from kumo.engine._deps import imported

TRAINER_PREDICTOR = 'kumo.engine._predictor:TrainerPredictor'


class Predictor(Protocol):
    r"""Scores ``rows`` (a frame or `PredictionTable`) on ``snapshot``."""
    def predict(self, rows: Any, snapshot: Any,
                params: Mapping[str, Any]) -> Union[pd.DataFrame,
                                                    Tuple[str, ...]]:
        ...


def load_predictor(bundle_dir: Union[str, Path]) -> Predictor:
    r"""The predictor the bundle at ``bundle_dir`` names, loaded."""
    from kumo.engine.trainer._provenance import read_provenance

    manifest = read_provenance(Path(bundle_dir))
    entry = next((record.name
                  for record in getattr(manifest, 'dependencies', None) or ()
                  if record.kind == 'predictor'), TRAINER_PREDICTOR)
    try:
        predictor = imported(entry)
    except (ImportError, AttributeError, ValueError) as error:
        raise ValueError(
            f"The model bundle at {bundle_dir} names predictor {entry!r}, "
            f"which cannot be imported here: {error}") from error
    return predictor.load(bundle_dir)


class TrainerPredictor:
    r"""Scores with the `Trainer` the bundle restores."""
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    @classmethod
    def load(cls, bundle_dir: Union[str, Path]) -> 'TrainerPredictor':
        from kumo.engine.trainer.trainer import Trainer

        return cls(Trainer.load(bundle_dir))

    def predict(self, rows: Any, snapshot: Any,
                params: Mapping[str, Any]) -> Union[pd.DataFrame,
                                                    Tuple[str, ...]]:
        from kumo.engine.integrations._mlflow_model import predict
        from kumo.engine.pquery.prediction_table import PredictionTable

        if isinstance(rows, PredictionTable):
            return self.trainer.predict(snapshot, prediction_table=rows,
                                        **params).files
        return predict(self.trainer, snapshot, rows, params)
