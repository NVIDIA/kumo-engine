# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Dict, List, Optional

import yaml
from kumo_core.callbacks.train_progress_logger import AutoTrainerProgress
from kumoapi.jobs import TrainerProgress

from async_utils.sync import blocking_method
from io_utils import json, logging
from kumo.home import HomeDirManager
from rest.config.kumo_tensorboard import (
    JobScalarTags,
    ScalarEvent,
    ScalarsResponse,
)

log = logging.getLogger(__name__)


class TensorboardApiDriver:
    def __init__(self):
        pass

    async def list_trials(self, training_job_path: str) -> List[str]:
        return await self._list_trials.coroutine(training_job_path)

    @blocking_method()
    def _list_trials(self, training_job_path: str) -> List[str]:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            return []

        atp = AutoTrainerProgress.load(home.get_automl_dir(task_name))
        return sorted(atp.trial_progress.keys())

    async def list_scalars_tags(self, training_job_path: str,
                                trial: Optional[str] = None) -> JobScalarTags:
        return await self._list_scalars_tags.coroutine(training_job_path, trial)

    @blocking_method()
    def _list_scalars_tags(self, training_job_path: str,
                           trial: Optional[str] = None) -> JobScalarTags:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            return JobScalarTags(trial_tags={})

        atp = AutoTrainerProgress.load(home.get_automl_dir(task_name))
        if trial is not None and trial not in atp.trial_progress:
            return JobScalarTags(trial_tags={trial: []})
        elif trial is not None:
            return JobScalarTags(
                trial_tags={trial: _list_trial_tags(atp.trial_progress[trial])})
        else:
            return JobScalarTags(trial_tags=_list_all_trial_tags(atp))

    async def get_scalars(self, training_job_path: str, trial: str,
                          tag: str) -> ScalarsResponse:
        return await self._get_scalars.coroutine(training_job_path, trial, tag)

    @blocking_method()
    def _get_scalars(self, training_job_path: str, trial: str,
                     tag: str) -> ScalarsResponse:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            return ScalarsResponse()

        atp: AutoTrainerProgress = AutoTrainerProgress.load(
            home.get_automl_dir(task_name))
        if trial not in atp.trial_progress:
            return ScalarsResponse()

        tp = atp.trial_progress[trial]
        metrics = _get_metrics(tp)
        if tag not in metrics:
            return ScalarsResponse()

        return ScalarsResponse(
            [ScalarEvent(v[0], v[0], v[1]) for v in metrics[tag]])

    async def get_many_scalars(
        self,
        training_job_path: str,
        trial: str,
        tags: List[str],
    ) -> Dict[str, ScalarsResponse]:
        r"""Similar to `get_scalars` but get many scalars for a single trial"""
        return await self._get_many_scalars.coroutine(
            training_job_path,
            trial,
            tags,
        )

    @blocking_method()
    def _get_many_scalars(
        self,
        training_job_path: str,
        trial: str,
        tags: List[str],
    ) -> Dict[str, ScalarsResponse]:
        home = HomeDirManager(training_job_path)
        task_name = home.first_task_name()
        if task_name is None:
            return {}

        atp: AutoTrainerProgress = AutoTrainerProgress.load(
            home.get_automl_dir(task_name))
        if trial not in atp.trial_progress:
            return {}

        tp = atp.trial_progress[trial]
        metrics = _get_metrics(tp)

        return {
            tag:
            ScalarsResponse(
                [ScalarEvent(v[0], v[0], v[1]) for v in metrics.get(tag, [])])
            for tag in tags
        }

    async def get_hparams(self, training_job_path: str) -> str:
        return await self._get_hparams.coroutine(training_job_path)

    @blocking_method()
    def _get_hparams(self, training_job_path: str) -> str:
        home = HomeDirManager(training_job_path)
        try:
            json_hparams = json.read_json(home.advanced_options_file)
            yaml_string = yaml.dump(json_hparams, default_flow_style=False)
            return yaml_string
        except Exception:
            return ""


def _list_trial_tags(tp: TrainerProgress) -> List:
    '''Return a list of tags in a `TrainerProgress`.
    '''
    tags = set()
    for _, val in tp.metrics.items():
        tags.update(val.train_metrics.keys())
        tags.update(val.validation_metrics.keys())
    return sorted(list(tags))


def _list_all_trial_tags(atp: AutoTrainerProgress) -> Dict:
    '''Return all tags in a `AutoTrainerProgress`.
    '''
    trial_tags = dict()
    for name, tp in atp.trial_progress.items():
        tags = set()
        for _, val in tp.metrics.items():
            tags.update(val.train_metrics.keys())
            tags.update(val.validation_metrics.keys())
        trial_tags[name] = sorted(list(tags))
    return trial_tags


def _get_metrics(tp: TrainerProgress) -> Dict:
    '''Convert the metrics from `dict[int, EpochMetrics]` to
    `dict[str, List[ScalarEvent]]`.
    '''
    raw = defaultdict(list)
    for epoch, metrics in tp.metrics.items():
        for name, value in metrics.train_metrics.items():
            raw[name].append((int(epoch), value))
        for name, value in metrics.validation_metrics.items():
            raw[name].append((int(epoch), value))

    ret = dict()
    for name, metrics in raw.items():
        ret[name] = sorted(metrics)
    return ret
