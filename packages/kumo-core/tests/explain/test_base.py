# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any
from unittest.mock import patch

import pytest
from kumoapi.task import TaskType

from kumo_core.explain.base import Explainer, continue_on_error


class ExplainerWithException(Explainer):
    supported_task_types = {TaskType.BINARY_CLASSIFICATION}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.has_raised_error = False

    @continue_on_error
    def explain_post_step(self, *args: Any, **kwargs: Any) -> None:
        self.has_raised_error = True
        raise AttributeError("Something is wrong")


def test_explainer_with_exception() -> None:
    explainer = ExplainerWithException(
        task_type=TaskType.BINARY_CLASSIFICATION,
        root_dir='.',
    )
    assert str(explainer) == 'ExplainerWithException()'

    # In test, we raise an error by default:
    with patch.dict(os.environ, {}):
        with pytest.raises(AttributeError):
            explainer.explain_post_step()

    # If we continue, the error is suppressed:
    with patch.dict(os.environ, {'KUMO_EXPLAINER_CONTINUE_ON_ERROR': 'true'}):
        explainer.explain_post_step()
    assert explainer.has_raised_error

    # If we raise, the error is raised:
    with patch.dict(os.environ, {'KUMO_EXPLAINER_RAISE_ON_ERROR': 'true'}):
        with pytest.raises(AttributeError):
            explainer.explain_post_step()
