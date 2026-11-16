# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

import pytest

from rest.driver.tensorboard_api_driver import TensorboardApiDriver


def test_list_trials():
    root_dir = osp.dirname(osp.realpath(__file__))
    home_dir = osp.join(root_dir, '..', '..', 'tensorboard_data')

    driver = TensorboardApiDriver()
    ret = driver._list_trials(home_dir)

    assert len(ret) == 2
    assert ret[0] == '0'
    assert ret[1] == '1'


def test_list_scalars():
    root_dir = osp.dirname(osp.realpath(__file__))
    home_dir = osp.join(root_dir, '..', '..', 'tensorboard_data')

    driver = TensorboardApiDriver()
    job_scalars_tags = driver._list_scalars_tags(home_dir, '1')
    assert job_scalars_tags.trial_tags["1"] == [
        'train_loss', 'train_map@1', 'val_map@1'
    ]

    job_scalars_tags2 = driver._list_scalars_tags(home_dir)
    assert len(job_scalars_tags2.trial_tags) == 2
    assert job_scalars_tags2.trial_tags["0"] == [
        'train_loss', 'train_map@1', 'val_map@1'
    ]

    job_scalars_tags3 = driver._list_scalars_tags(home_dir, 'not_exist')
    assert len(job_scalars_tags3.trial_tags) == 1
    assert job_scalars_tags3.trial_tags["not_exist"] == []


def test_get_scalars():
    root_dir = osp.dirname(osp.realpath(__file__))
    home_dir = osp.join(root_dir, '..', '..', 'tensorboard_data')

    driver = TensorboardApiDriver()
    train_loss = driver._get_scalars(home_dir, '0', 'train_loss')
    assert train_loss.scalars[0].step == 0
    assert train_loss.scalars[0].value == pytest.approx(42.41317367553711)
    assert train_loss.scalars[-1].step == 3
    assert train_loss.scalars[-1].value == pytest.approx(35.628990173339844)
    assert len(train_loss.scalars) == 4

    train_map1 = driver._get_scalars(home_dir, '0', 'train_map@1')
    assert train_map1.scalars[0].step == 1
    assert train_map1.scalars[0].value == pytest.approx(0.022065918892621994)
    assert train_map1.scalars[-1].step == 3
    assert train_map1.scalars[-1].value == pytest.approx(0.04124853387475014)
    assert len(train_map1.scalars) == 3

    # Non-existent tag should return empty scalars
    random_tag = driver._get_scalars(home_dir, '0', 'random_tag')
    assert random_tag.scalars == []


def test_get_hparams(tmp_path):
    inputs_dir = tmp_path / 'inputs'
    inputs_dir.mkdir()
    (inputs_dir / 'advance_options.json').write_text(
        '{"learning_rate": 0.01, "num_layers": 3}',
        encoding='utf-8',
    )
    driver = TensorboardApiDriver()
    assert driver._get_hparams(
        str(tmp_path)) == ('\'{"learning_rate": 0.01, "num_layers": 3}\'\n')


def test_get_many_scalars():
    # Similar to test_get_scalars but testing the batch version.
    root_dir = osp.dirname(osp.realpath(__file__))
    home_dir = osp.join(root_dir, '..', '..', 'tensorboard_data')
    driver = TensorboardApiDriver()

    result = driver._get_many_scalars(
        home_dir,
        trial="0",
        tags=["train_loss", "train_map@1", "random_tag"],
    )
    # Check train_loss
    train_loss = driver._get_scalars(home_dir, '0', 'train_loss')
    assert train_loss.scalars[0].step == 0
    assert train_loss.scalars[0].value == pytest.approx(42.41317367553711)
    assert train_loss.scalars[-1].step == 3
    assert train_loss.scalars[-1].value == pytest.approx(35.628990173339844)
    assert len(train_loss.scalars) == 4

    # Check train_map1
    train_map1 = driver._get_scalars(home_dir, '0', 'train_map@1')
    assert train_map1.scalars[0].step == 1
    assert train_map1.scalars[0].value == pytest.approx(0.022065918892621994)
    assert train_map1.scalars[-1].step == 3
    assert train_map1.scalars[-1].value == pytest.approx(0.04124853387475014)
    assert len(train_map1.scalars) == 3

    # Check non-existent tag
    assert result["random_tag"].scalars == []
