# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A Kumo bundle as a Snowflake Model Registry version."""

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from kumo.engine.connector._extras import import_provider

custom_model = import_provider('snowflake.ml.model.custom_model', 'snowflake')
_registry = import_provider('kumo_snowflake.registry', 'snowflake')

CPU_TORCH_CUDA = '12.6'
DEFAULTS = {
    'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
    'function_type': 'TABLE_FUNCTION',
}
EMBEDDED = ('kumo.engine', 'kumo_core', 'kumoapi', 'kumopql', 'kumo_snowflake',
            'diskgraph', 'torchmetrics')
_EMBEDDED_DISTS = ('diskgraph', 'torchmetrics')
REQUIRING = (('kumo_core', 'kumo-core'), ('kumopql', 'kumo-pql'))
TEXT_REQUIREMENT = 'kumo-glove-6b-50d>=1.0.0,<2'
PROVIDED = ('snowflake-snowpark-python', 'snowflake-connector-python')


class KumoModel(custom_model.CustomModel):
    r"""Scores entity ids (an ``ENTITY`` column) against a graph snapshot."""
    @custom_model.inference_api
    def predict(self, df: pd.DataFrame, *, snapshot: str = '',
                prediction_time: str = '',
                binary_classification_threshold: float = -1.0,
                num_classes_to_return: int = 0) -> pd.DataFrame:
        from kumo.engine.integrations._snowflake_serving import (
            score_registered,
        )

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
    r"""``Registry.log_model`` arguments for the bundle as a `KumoModel`."""
    from kumo_core.artifacts import bundle_task_type

    from kumo.engine.trainer._spec import read_bundle_artifact_paths
    from kumo.engine.trainer.output_config import (
        SERVING_OPTION_PREFIX,
        SERVING_SNAPSHOT_KEY,
    )

    check_host()
    bundle = Path(bundle_dir)
    paths, _ = read_bundle_artifact_paths(bundle)
    task = bundle_task_type(paths)
    serving = Path(workdir) / 'serving.json'
    serving.write_text(
        json.dumps({
            'snapshot': metadata.get(SERVING_SNAPSHOT_KEY),
            'columns': output_columns(task),
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
    options['cuda_version'] = cuda_version(options.get('cuda_version'))
    if python != DEFAULTS['python_version']:
        raise _registry.ServingEnvironmentError(
            f"serving_options python_version={python!r} differs from the "
            f"registering Python {DEFAULTS['python_version']}; the container "
            f"must run the same Python, so register from Python {python}.")
    args = dict(
        model=KumoModel(
            custom_model.ModelContext(artifacts={
                'bundle': str(bundle),
                'serving': str(serving),
            })),
        signatures={'predict': signature(task)},
        target_platforms=['SNOWPARK_CONTAINER_SERVICES'],
        python_version=python,
        options=options,
        pip_requirements=pip_requirements() +
        text_requirements(paths.get('encoders')),
        code_paths=embedded_code(Path(workdir) / 'code'),
    )
    if repository:
        args['artifact_repository_map'] = {'pip': repository}
    return args


def check_host() -> None:
    r"""Rejects hosts whose native code the Linux x86_64 image cannot run."""
    host = f'{sys.platform} {platform.machine()}'
    if host != 'linux x86_64':
        raise _registry.ServingEnvironmentError(
            f"Snowpark Container Services runs linux x86_64, but this host "
            f"is {host}: the version embeds this installation's DiskGraph "
            f"native module, so register from a Linux x86_64 host.")


def cuda_version(requested: Optional[str]) -> str:
    r"""The installed torch's CUDA, which ``requested`` must equal."""
    import torch

    def minor(version: str) -> str:
        return '.'.join(version.split('.')[:2])

    built = minor(torch.version.cuda) if torch.version.cuda else None
    if requested is None:
        return built or CPU_TORCH_CUDA
    if built is not None and minor(requested) != built:
        raise _registry.ServingEnvironmentError(
            f"serving_options cuda_version={requested!r}, but torch "
            f"{torch.__version__} is built for CUDA {built}: register from a "
            f"torch built for CUDA {minor(requested)} or drop cuda_version.")
    return requested


def embedded_code(directory: Path) -> List[str]:
    r"""Copies the :data:`EMBEDDED` packages into ``directory``."""
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
    r"""Third-party requirements of :data:`REQUIRING`, as the image takes."""
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
    r"""The word-embedding package, if ``encoders`` embeds text."""
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


def output_columns(task: Any) -> List[str]:
    from kumo_core.train.formatting import prediction_output_columns

    return ['ENTITY', *prediction_output_columns(task)]


def signature(task: Any) -> Any:
    from kumo_core.types import TaskType
    ms = import_provider('snowflake.ml.model.model_signature', 'snowflake')

    binary = task is TaskType.BINARY_CLASSIFICATION
    dtypes = {
        'ENTITY': ms.DataType.STRING,
        'PREDICTED': ms.DataType.BOOL,
        'CLASS': ms.DataType.STRING,
        'TIMESTAMP': ms.DataType.TIMESTAMP_NTZ,
        'TARGET_PRED':
        ms.DataType.BOOL if binary else ms.DataType.DOUBLE,
    }
    return ms.ModelSignature(
        inputs=[ms.FeatureSpec('ENTITY', ms.DataType.STRING)],
        outputs=[
            ms.FeatureSpec(name, dtypes.get(name, ms.DataType.DOUBLE))
            for name in output_columns(task)
        ],
        params=[
            ms.ParamSpec('snapshot', ms.DataType.STRING, ''),
            ms.ParamSpec('prediction_time', ms.DataType.STRING, ''),
            ms.ParamSpec('binary_classification_threshold',
                         ms.DataType.DOUBLE, -1.0),
            ms.ParamSpec('num_classes_to_return', ms.DataType.INT64, 0),
        ],
    )
