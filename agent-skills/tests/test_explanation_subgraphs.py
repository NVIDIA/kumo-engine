# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the explanations skill says about sampled subgraphs must match the SDK that writes them.

The skill tells the agent where `subgraph_json/<ordinal>.json` lives and which fields to read. A
renamed field or directory would make the agent's loader fail on a real run, so the names and
limits it states are checked against `kumoapi.explain.subgraph` and the `kumo_core` producer.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = (REPO / 'agent-skills' / 'skills' / 'kumo-model-explanations' / 'references' /
             'artifacts.md')
KUMOAPI = REPO / 'packages' / 'kumo-api' / 'kumoapi' / 'explain' / 'subgraph.py'
CORE = REPO / 'packages' / 'kumo-core' / 'src' / 'kumo_core'
SUBGRAPH_TYPES = ('Subgraph', 'SubgraphStore', 'SubgraphNode', 'AggregatedNodes',
                  'SubgraphFeatureStore', 'Feature')


def section() -> str:
    text = ARTIFACTS.read_text()
    assert '## Sampled subgraphs' in text
    return text.split('## Sampled subgraphs', 1)[1].split('\n## ', 1)[0]


def sdk_fields() -> set[str]:
    fields = set()
    for node in ast.parse(KUMOAPI.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in SUBGRAPH_TYPES:
            fields |= {s.target.id for s in node.body
                       if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    return fields


def constant(path: Path, name: str):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} is no longer assigned in {path.relative_to(REPO)}')


def test_artifacts_names_the_subgraph_files() -> None:
    text = section()
    for name in ('subgraph_collector/test/', 'subgraph_json/', 'strategy_mappings.json',
                 'best_trainer.json', '_refit_trainval', 'disable_explain'):
        assert name in text, name


def test_field_names_exist_in_the_sdk_types() -> None:
    fields = sdk_fields()
    assert {'subgraph_store', 'feature_store', 'root_id', 'nodes', 'edges', 'tables',
            'features', 'score'} <= fields
    prose, _, code = section().partition('```python')
    named = set(re.findall(r'`([a-z][a-z_]*)`', prose))
    named |= set(re.findall(r'\["([a-z][a-z_]*)"\]', code))
    # Field lists such as `{id, pkey_name, pkey}`.
    named |= {f for group in re.findall(r'\{([a-z_]+(?:, [a-z_]+)+)\}', prose)
              for f in group.split(', ')}
    # Strategy names and the other files' keys (EntityMappings, BestTrainerConfig).
    not_fields = {'random', 'unsure', 'old_entity', 'recent_entity', 'mappings', 'orig_mappings',
                  'best_trainer'}
    assert named - not_fields <= fields, sorted(named - not_fields - fields)


def test_stated_limits_match_the_producer() -> None:
    text = ' '.join(section().split())
    max_facts = constant(CORE / 'training' / 'runtime.py', 'max_facts')
    assert f'At most {max_facts} neighbours' in text
    per_strategy = constant(CORE / 'training' / 'explain_selection.py', 'EVAL_NUM_SELECTION_NODES')
    assert f'up to {per_strategy} entities' in text
    assert constant(CORE / 'training' / 'runtime.py', 'mapping_file_name') in text
    collector = (CORE / 'explain' / 'subgraph_collector.py').read_text()
    assert "'subgraph_json'" in collector
