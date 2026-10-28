# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A Kumo model bundle as a Snowflake Model Registry version that scores in
Snowpark Container Services, e.g. with ``ModelVersion.run_batch``.

snowflake-ml pickles this module by value into the version, so the container
must run the Python the version was registered from. kumo itself is embedded
in the version as code, on the path before the model loads; its third-party
requirements install from PyPI or a pip artifact repository.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from kumo.engine.connector._extras import import_provider

custom_model = import_provider('snowflake.ml.model.custom_model', 'snowflake')

DEFAULTS = {
    'cuda_version': '12.6',
    'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
    'function_type': 'TABLE_FUNCTION',
}
#: Packages embedded as code: kumo's own, which are not on PyPI, and
#: torchmetrics, whose pinned version the CUDA index the image installs from
#: first does not carry.
EMBEDDED = ('kumo.engine', 'kumo_core', 'kumoapi', 'kumopql', 'kumo_snowflake',
            'diskgraph', 'torchmetrics')
_EMBEDDED_DISTS = ('diskgraph', 'torchmetrics')
#: Modules whose distributions' requirements the embedded code needs, with
#: the distribution of a workspace install; a bundled kumo-engine wheel
#: provides both modules itself.
REQUIRING = (('kumo_core', 'kumo-core'), ('kumopql', 'kumo-pql'))
#: kumo-engine's ``text`` extra, for models that embed text columns.
TEXT_REQUIREMENT = 'kumo-glove-6b-50d>=1.0.0,<2'
#: Installed by the Snowflake inference image; their own pins there win.
PROVIDED = ('snowflake-snowpark-python', 'snowflake-connector-python')


class KumoModel(custom_model.CustomModel):
    r"""Scores entity ids (an ``ENTITY`` column) against a graph snapshot."""
    @custom_model.inference_api
    def predict(self, df: pd.DataFrame, *, snapshot: str = '',
                prediction_time: str = '',
                binary_classification_threshold: float = -1.0,
                num_classes_to_return: int = 0) -> pd.DataFrame:
        from kumo.engine.integrations._serving import score_registered

        params = {
            'snapshot': snapshot or None,
            'prediction_time': prediction_time or None,
            'binary_classification_threshold':
            (binary_classification_threshold
             if binary_classification_threshold >= 0 else None),
            'num_classes_to_return': num_classes_to_return or None,
        }
        return score_registered(self.context.path('bundle'),
                                self.context.path('serving'), df, params)


def log_model_args(bundle_dir: str, metadata: Mapping[str, str],
                   target: Mapping[str, Optional[str]],
                   workdir: str) -> Dict[str, Any]:
    r"""``Registry.log_model`` arguments that register the bundle as a
    :class:`KumoModel` for Snowpark Container Services on GPU. ``target`` is
    the registering connector's warehouse, database and schema."""
    from kumo.engine.integrations._serving import output_columns
    from kumo.engine.trainer._spec import read_bundle_artifact_paths
    from kumo.engine.trainer.output_config import (
        SERVING_OPTION_PREFIX,
        SERVING_SNAPSHOT_KEY,
    )

    bundle = Path(bundle_dir)
    serving = Path(workdir) / 'serving.json'
    serving.write_text(
        json.dumps({
            'snapshot': metadata.get(SERVING_SNAPSHOT_KEY),
            'columns': output_columns(bundle),
            **target
        }))
    options = {
        **DEFAULTS,
        **{
            key[len(SERVING_OPTION_PREFIX):]: value
            for key, value in metadata.items()
            if key.startswith(SERVING_OPTION_PREFIX)
        }
    }
    repository = options.pop('pip_repository', None)
    python = options.pop('python_version')
    if python != DEFAULTS['python_version']:
        raise ValueError(
            f"serving_options python_version={python!r} differs from the "
            f"registering Python {DEFAULTS['python_version']}: snowflake-ml "
            f"pickles the model by value, so the container must run the same "
            f"Python. Register from Python {python} instead.")
    paths, _ = read_bundle_artifact_paths(bundle)
    environment: Dict[str, Any] = dict(
        pip_requirements=pip_requirements() +
        text_requirements(paths.get('encoders')),
        code_paths=embedded_code(Path(workdir) / 'code'))
    if repository:
        environment['artifact_repository_map'] = {'pip': repository}
    return dict(
        model=KumoModel(
            custom_model.ModelContext(artifacts={
                'bundle': str(bundle),
                'serving': str(serving),
            })),
        signatures={'predict': signature(bundle)},
        target_platforms=['SNOWPARK_CONTAINER_SERVICES'],
        python_version=python,
        options=options,
        **environment,
    )


def embedded_code(directory: Path) -> List[str]:
    r"""Copies the :data:`EMBEDDED` packages into ``directory``, without
    caches and tests, and returns the package directories."""
    import importlib

    paths = []
    for name in EMBEDDED:
        source = Path(importlib.import_module(name).__file__).parent
        if name == 'kumo.engine':
            source, name = source.parent, 'kumo'
        target = directory / name
        shutil.copytree(
            source, target,
            ignore=shutil.ignore_patterns('__pycache__', 'test', 'tests'))
        paths.append(str(target))
    return paths


def pip_requirements() -> List[str]:
    r"""The third-party requirements of the distributions providing
    :data:`REQUIRING`, in a form the Snowflake image build accepts: torch at
    this installation's minor version, faiss-cpu for kumo's own faiss build,
    one line per package, and no ``==X.*`` pins."""
    from importlib.metadata import packages_distributions, requires, version

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    provided = packages_distributions()
    dists = sorted({
        dist
        for module, fallback in REQUIRING
        for dist in provided.get(module) or [fallback]
    })
    found: Dict[str, Any] = {}
    faiss = None
    for dist in dists:
        for line in requires(dist) or []:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate(
                    {'extra': 'faiss-cpu'}):
                continue
            name = canonicalize_name(req.name)
            if name.startswith('faiss'):
                faiss = req.specifier
            elif name in found:
                found[name].specifier &= req.specifier
            elif not name.startswith('kumo') and name not in _EMBEDDED_DISTS:
                found[name] = req
    major, minor = (int(part) for part in version('torch').split('.')[:2])
    lines = [_pinned(req) for name, req in sorted(found.items())
             if name != 'torch']
    if faiss is not None:
        lines.append(f'faiss-cpu{faiss}')
    return [*lines, f'torch>={major}.{minor},<{major}.{minor + 1}', *PROVIDED]


def text_requirements(encoders: Optional[str]) -> List[str]:
    r"""The word-embedding package, from PyPI, when ``encoders`` embeds
    text columns."""
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import load_json
    from kumoapi.encoder import TextEmbedding

    config = (load_json(encoders, EncodersConfig)
              if encoders and Path(encoders).is_file() else None)
    if config is not None and any(
            isinstance(encoder, TextEmbedding)
            for table in config.encoders.values()
            for encoder in table.values()):
        return [TEXT_REQUIREMENT]
    return []


def _pinned(req: Any) -> str:
    specs = []
    for spec in req.specifier:
        if spec.operator == '==' and spec.version.endswith('.*'):
            parts = spec.version[:-2].split('.')
            upper = [*parts[:-1], str(int(parts[-1]) + 1)]
            specs += [f'>={".".join(parts)}', f'<{".".join(upper)}']
        else:
            specs.append(str(spec))
    return req.name + ','.join(sorted(specs))


def signature(bundle: Path) -> Any:
    from kumo_core.types import TaskType
    ms = import_provider('snowflake.ml.model.model_signature', 'snowflake')

    from kumo.engine.integrations._serving import output_columns, task_type

    binary = task_type(bundle) is TaskType.BINARY_CLASSIFICATION
    dtypes = {
        'PREDICTED': ms.DataType.BOOL,
        'CLASS': ms.DataType.STRING,
        'TIMESTAMP': ms.DataType.TIMESTAMP_NTZ,
        'TARGET_PRED':
        ms.DataType.BOOL if binary else ms.DataType.DOUBLE,
    }
    return ms.ModelSignature(
        inputs=[ms.FeatureSpec('ENTITY', ms.DataType.STRING)],
        outputs=[
            ms.FeatureSpec(
                name, ms.DataType.STRING if name == 'ENTITY' else dtypes.get(
                    name, ms.DataType.DOUBLE))
            for name in output_columns(bundle)
        ],
        params=[
            ms.ParamSpec('snapshot', ms.DataType.STRING, ''),
            ms.ParamSpec('prediction_time', ms.DataType.STRING, ''),
            ms.ParamSpec('binary_classification_threshold',
                         ms.DataType.DOUBLE, -1.0),
            ms.ParamSpec('num_classes_to_return', ms.DataType.INT64, 0),
        ],
    )
