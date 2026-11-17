# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated cases: each (property × engine × seed batch) is a required obligation.

So are the regression corpus, per engine, and every recorded known gap. A
failure writes a repro file and names it in the assertion message.
"""

import re

import pytest

from conformance.assertions import require_engine
from conformance.cases import ENGINES, REPOSITORY
from conformance.properties import PROPERTIES, obligation_id, obligations, replay, run_obligation
from conformance.registry import REPLAYED
from conformance.repro import load_corpus, load_repro


def _engines(config):
    selected = config.getoption("--conformance-engine")
    return ENGINES if selected == "all" else (selected,)


def pytest_generate_tests(metafunc):
    engines = _engines(metafunc.config)
    if "obligation" in metafunc.fixturenames:
        metafunc.parametrize(
            "obligation",
            [
                pytest.param(item, id=obligation_id(*item))
                for item in obligations(
                    engines,
                    metafunc.config.getoption("--conformance-seeds"),
                    metafunc.config.getoption("--conformance-seed-start"),
                )
            ],
        )
    if "corpus_engine" in metafunc.fixturenames:
        metafunc.parametrize("corpus_engine", engines)
    if "gap" in metafunc.fixturenames:
        gaps = [gap for gap in REPLAYED if gap.engine in engines]
        metafunc.parametrize("gap", gaps, ids=[gap.id for gap in gaps])


def test_property(obligation):
    name, engine, seed = obligation
    require_engine(engine)
    run_obligation(name, engine, seed)


def test_corpus(corpus_engine):
    """Every corpus entry holds on each selected engine its property applies to."""
    require_engine(corpus_engine)
    failures = []
    for entry_id, entry in load_corpus().items():
        if corpus_engine not in PROPERTIES[entry.property].engines:
            continue
        try:
            replay(entry, corpus_engine)
        except AssertionError as error:
            failures.append(f"{entry_id} ({entry.issue}): {error}")
    assert not failures, "\n".join(failures)


def test_known_gap(gap):
    """Each known gap's or in-memory divergence's repro still fails as recorded; when one
    passes, lift the entry."""
    require_engine(gap.engine)
    repro = load_repro(REPOSITORY / gap.repro)
    assert (repro.property, repro.engine) == (gap.property, gap.engine)
    assert gap.covers(repro.case.dimensions), f"{gap.id}: repro lies outside the gap"
    with pytest.raises(AssertionError, match=re.escape(gap.symptom)):
        replay(repro)
