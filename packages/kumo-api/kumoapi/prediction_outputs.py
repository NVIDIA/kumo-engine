# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Mapping


def resolve_prediction_output_paths(
    output_paths: Mapping[str, str],
    *,
    output_types: object = None,
    output_table_name: object = None,
) -> dict[str, str]:
    if output_types is not None or output_table_name is not None:
        raise ValueError(
            'output_paths cannot be combined with output_types or output_table_name'
        )
    if not output_paths:
        raise ValueError('output_paths must contain predictions or embeddings')
    result = dict(output_paths)
    for artifact, path in result.items():
        if artifact not in {'predictions', 'embeddings'}:
            raise ValueError(f'Unsupported output artifact: {artifact!r}')
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                f'Output path for {artifact} must be a nonempty string')
        if not path.endswith('/') and not path.lower().endswith(
            ('.parquet', '.csv')):
            result[artifact] = path + '/'
    if len(set(result.values())) != len(result):
        raise ValueError(
            'Predictions and embeddings require distinct output paths')
    return result
