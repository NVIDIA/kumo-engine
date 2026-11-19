# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Example subgraph card, from the retained typed graph samples."""

import math
import posixpath
from typing import Any, Optional

from kumoapi.explain.subgraph import AggregatedNodes, Subgraph

from kumo.engine._explainability._run import (
    FORECASTING,
    LINK_PREDICTION,
    Run,
    mapping,
    read_json,
)


def _json_value(value: Any) -> Any:
    """Strict JSON with integers JavaScript keeps exact and missing numbers
    as null."""
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _output(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(label): score
            for label, score in value.items()
            if isinstance(score, (int, float)) and math.isfinite(score)
        }
    if isinstance(value, list):
        return [
            node['id'] for node in value
            if isinstance(node, dict) and 'id' in node
        ]
    return value if isinstance(value, (int, float)) else None


def _sample(graph: Subgraph, task_type: str) -> Optional[dict[str, Any]]:
    store = graph.subgraph_store
    if store is None or store.task_type.value != task_type:
        return None
    tables = {
        node: table
        for table, ids in store.tables.items()
        for node in ids
    }
    features = graph.feature_store.features if graph.feature_store else {}
    nodes = []
    for node_id, node in store.nodes.items():
        aggregate = isinstance(node, AggregatedNodes)
        start = node.start_time if aggregate else None
        end = node.end_time if aggregate else None
        nodes.append(
            dict(
                id=str(node_id), table=tables.get(node_id, 'Records'),
                key=node.pkey_name, value=None if aggregate else node.pkey,
                count=node.count if aggregate else None, start=start
                and start.isoformat(), end=end and end.isoformat(), features=[
                    dict(name=name, value=feature.value, score=feature.score)
                    for name, feature in features.get(node_id, {}).items()
                ]))
    edges = [
        dict(source=str(source), target=str(target), relation=relation)
        for source, relations in store.edges.items()
        for relation, targets in relations.items() for target in targets
    ]
    return dict(root=str(store.root_id),
                time=store.seed_time.isoformat() if store.seed_time else None,
                nodes=nodes, edges=edges)


def card(run: Run) -> Optional[dict[str, Any]]:
    """Saved TEST examples; unavailable records never start model work."""
    if run.model_type == FORECASTING:
        return None
    try:
        references = run.explanations.samples()
    except (OSError, ValueError):
        return None
    collector = posixpath.join(str(run.explanations.trial_root),
                               'subgraph_collector', 'test')
    # Each category keeps the collector's selection order.
    categories = mapping(
        mapping(read_json(posixpath.join(
            collector, 'strategy_mappings.json'))).get('mappings'))
    samples = []
    for reference in references:
        try:
            sample = reference.available and _sample(
                run.explanations.sample(reference.artifact_id), run.task_type)
        except (OSError, ValueError):
            sample = None
        if sample:
            output = mapping(
                read_json(
                    posixpath.join(collector, 'output',
                                   reference.artifact_id + '.json')))
            samples.append(
                dict(id=reference.artifact_id,
                     original_ids=reference.original_ids, **sample,
                     prediction=_output(output.get('prediction')),
                     target=_output(output.get('target')),
                     categories=list(reference.categories) or ['saved']))
    if not samples:
        return None
    binary = run.task_type == 'binary_classification'
    ranking = run.model_type == LINK_PREDICTION
    return _json_value(
        dict(
            title='Example subgraph', samples=samples, task_type=run.task_type,
            categories=list(categories) or ['saved'],
            category_samples=categories,
            entity_label=run.entities if run.entities != 'rows' else 'Entity',
            prediction_label=('Recommended records'
                              if ranking else 'Predicted probability of True'
                              if binary else 'Predicted value'),
            target_label=('Observed records' if ranking else
                          'Observed outcome' if binary else 'Observed value'),
            class_labels=['False', 'True'] if binary else None,
            unavailable_count=len(references) - len(samples)))
