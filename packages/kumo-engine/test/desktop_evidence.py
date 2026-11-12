# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run existing recipes with explicit CPU configuration and observe real devices."""
import json
import os
from pathlib import Path

DEVICES = []
ORIGINALS = None
CURRENT_TEST = 'session setup'


def pytest_runtest_logstart(nodeid, location):
    global CURRENT_TEST
    CURRENT_TEST = nodeid


def pytest_sessionstart(session):
    global ORIGINALS
    from kumo_core.train.trainer import Trainer as CoreTrainer

    import kumo.engine as kumo
    original_init = kumo.Trainer.__init__
    original_create = CoreTrainer.create_pl_trainer
    ORIGINALS = (kumo.Trainer, original_init, CoreTrainer, original_create)

    def initialize(trainer, *args, **kwargs):
        if kwargs.get('accelerator', 'auto') == 'auto':
            kwargs['accelerator'] = 'cpu'
        return original_init(trainer, *args, **kwargs)

    def observe(trainer):
        original_create(trainer)
        device = trainer.pl_trainer.strategy.root_device.type
        DEVICES.append(dict(test=CURRENT_TEST, device=device))
        assert device == 'cpu', 'Recipe executed on a different device'

    kumo.Trainer.__init__ = initialize
    CoreTrainer.create_pl_trainer = observe


def pytest_sessionfinish(session, exitstatus):
    if ORIGINALS is not None:
        trainer, initialize, core, create = ORIGINALS
        trainer.__init__ = initialize
        core.create_pl_trainer = create
    Path(os.environ['DESKTOP_DEVICE_RECEIPT']).write_text(
        json.dumps(dict(devices=DEVICES, exitstatus=exitstatus), indent=2) +
        '\n')
