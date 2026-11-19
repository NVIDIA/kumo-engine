# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt retained collector graphs through the existing typed artifact reader."""

import logging
import math
import posixpath
from typing import Any, Optional

from kumoapi.explain.subgraph import AggregatedNodes

from kumo.engine._explainability import _files
from kumo.engine._explainability._run import LINK_PREDICTION, Run
from kumo.engine.explain._artifacts import list_samples, read_sample

logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    """Preserve nested feature values as strict JSON, with missing numbers null."""
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
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def card(run: Run) -> Optional[dict[str, Any]]:
    """Read saved TEST examples; unavailable records never start model work."""
    if run.files.trial is None:
        return None
    root = posixpath.join(run.files.trial, 'subgraph_collector', 'test')
    return collector_card(root, run.model_type, run.task_type, run.entities)


def collector_card(root: str, model_type: str, task_type: str,
                   entities: str) -> Optional[dict[str, Any]]:
    """Normalize saved typed samples for the explanation card."""
    try:
        references = list_samples(root)
    except (OSError, ValueError) as error:
        logger.debug('Subgraph selection unavailable: %s', error)
        return None
    metadata = _files.read_json(posixpath.join(root, 'strategy_mappings.json'))
    mappings = metadata.get('mappings', {}) if isinstance(metadata,
                                                          dict) else {}
    allowed = {ref.artifact_id for ref in references}
    mappings = {
        name: [item for item in ids if item in allowed]
        for name, ids in mappings.items()
    }
    mappings = {name: ids for name, ids in mappings.items() if ids}
    samples = []
    unavailable = 0
    for reference in references:
        if not reference.available:
            unavailable += 1
            continue
        try:
            graph = read_sample(root, reference.artifact_id)
        except (OSError, ValueError) as error:
            logger.debug('Subgraph unavailable: %s', error)
            unavailable += 1
            continue
        store = graph.subgraph_store
        if store is None:
            unavailable += 1
            continue
        if store.task_type.value != task_type:
            unavailable += 1
            continue
        tables = {
            node: table
            for table, ids in store.tables.items()
            for node in ids
        }
        features = (graph.feature_store.features
                    if graph.feature_store is not None else {})
        nodes = []
        for node_id, node in store.nodes.items():
            aggregate = isinstance(node, AggregatedNodes)
            nodes.append({
                'id':
                str(node_id),
                'table':
                tables.get(node_id, 'Records'),
                'key':
                node.pkey_name,
                'value':
                None if aggregate else node.pkey,
                'count':
                node.count if aggregate else None,
                'start': (node.start_time.isoformat()
                          if aggregate and node.start_time else None),
                'end': (node.end_time.isoformat()
                        if aggregate and node.end_time else None),
                'features': [{
                    'name':
                    name,
                    'value':
                    feature.value,
                    'score': (feature.score if feature.score is not None
                              and math.isfinite(feature.score) else None)
                } for name, feature in features.get(node_id, {}).items()],
            })
        output = _files.read_json(
            posixpath.join(root, 'output', reference.artifact_id + '.json'))
        if not isinstance(output, dict):
            output = {}
        samples.append({
            'id':
            reference.artifact_id,
            'original_ids':
            reference.original_ids,
            'root':
            str(store.root_id),
            'time':
            store.seed_time.isoformat() if store.seed_time else None,
            'nodes':
            nodes,
            'edges': [{
                'source': str(source),
                'target': str(target),
                'relation': relation
            } for source, relations in store.edges.items()
                      for relation, targets in relations.items()
                      for target in targets],
            'prediction':
            _output(output.get('prediction')),
            'target':
            _output(output.get('target')),
            'categories':
            list(reference.categories) or ['saved'],
        })
    if not samples:
        return None
    binary = task_type == 'binary_classification'
    ranking = model_type == LINK_PREDICTION
    return _json_value({
        'model_type':
        model_type,
        'title':
        'Example subgraph',
        'samples':
        samples,
        'task_type':
        task_type,
        'categories':
        list(mappings) or ['saved'],
        'category_samples':
        mappings,
        'entity_label':
        entities if entities != 'rows' else 'Entity',
        'prediction_label':
        ('Recommended records' if ranking else
         'Predicted probability of True' if binary else 'Predicted value'),
        'target_label': ('Observed records' if ranking else
                         'Observed outcome' if binary else 'Observed value'),
        'class_labels': ['False', 'True'] if binary else None,
        'unavailable_count':
        unavailable,
    })
