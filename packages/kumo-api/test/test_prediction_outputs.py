# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from kumoapi.prediction_outputs import resolve_prediction_output_paths


def test_exact_paths_select_artifacts_and_preserve_file_names():
    assert resolve_prediction_output_paths({
        'predictions':
        's3://bucket/predictions/project',
        'embeddings':
        's3://bucket/embeddings/project.parquet',
    }) == {
        'predictions': 's3://bucket/predictions/project/',
        'embeddings': 's3://bucket/embeddings/project.parquet',
    }


@pytest.mark.parametrize('paths,kwargs', [
    ({}, {}),
    ({
        'unknown': 's3://bucket/out/'
    }, {}),
    ({
        'predictions': ''
    }, {}),
    ({
        'predictions': 'a',
        'embeddings': 'a/'
    }, {}),
    ({
        'predictions': 'a'
    }, {
        'output_types': {'predictions'}
    }),
    ({
        'predictions': 'a'
    }, {
        'output_table_name': 'legacy'
    }),
])
def test_invalid_output_paths_are_rejected(paths, kwargs):
    with pytest.raises(ValueError):
        resolve_prediction_output_paths(paths, **kwargs)
