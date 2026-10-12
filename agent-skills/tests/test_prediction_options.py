# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the skills say about `num_classes_to_return` must match `kumo_core.prediction_options`.

The SDK requires the count for link prediction, caps it at `MAX_LINKS_TO_PREDICT`, and never
compares it with the query's `RANK TOP K`. The skills once told the agent the two must match,
which made a request for 50 results from a `RANK TOP 10` model look impossible.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / 'agent-skills' / 'skills'
SDK_REF = SKILLS / 'kumoai-reference' / 'references' / 'kumo_finetuning_sdk.md'
OPTIONS = REPO / 'packages' / 'kumo-core' / 'src' / 'kumo_core' / 'prediction_options.py'

# "must match/equal RANK TOP N", in either order, within one sentence or table row.
COUPLED = re.compile(
    r'num_classes_to_return[^.\n]*\bmust\s+(?:match|equal|be\s+equal\s+to)\b[^.\n]*RANK\s+TOP'
    r'|RANK\s+TOP[^.\n]*\bmust\s+(?:match|equal)\b[^.\n]*num_classes_to_return', re.IGNORECASE)


def sdk_cap() -> int:
    for node in ast.parse(OPTIONS.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'MAX_LINKS_TO_PREDICT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('MAX_LINKS_TO_PREDICT is no longer defined in prediction_options.py')


def test_count_is_not_tied_to_rank_top_k() -> None:
    hits = [f'{p.relative_to(SKILLS)}: {m.group(0)}' for p in sorted(SKILLS.rglob('*.md'))
            for m in COUPLED.finditer(p.read_text())]
    assert hits == []


def test_coupling_pattern_matches_the_old_wording() -> None:
    assert COUPLED.search('- `num_classes_to_return` must match `RANK TOP N` for a ranking query.')
    assert COUPLED.search('| `num_classes_to_return=N` | Must match `RANK TOP N` |')


def test_sdk_reference_states_the_link_prediction_rule() -> None:
    text = ' '.join(SDK_REF.read_text().split())
    cap = sdk_cap()
    assert f'1 to {cap}' in text, f'the §9 bullet must state the 1–{cap} range'
    assert 'Link prediction requires it' in text
    assert 'independent of the query\'s `RANK TOP K`' in text
