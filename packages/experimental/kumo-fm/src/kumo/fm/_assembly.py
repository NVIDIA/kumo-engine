# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and assemble selected columnar graph occurrences into FM tables."""
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ._phase import phase
from ._types import TYPES


def empty_features(columns):
    return pa.table({name: pa.array([], type=TYPES[spec["dtype"]]) for name, spec in columns.items()})


def normalize_features(table, columns):
    return pa.table({name: pc.cast(table[name], TYPES[columns[name]["dtype"]]) for name in table.column_names})


def indices(column):
    return column.to_numpy(zero_copy_only=False)


def _parent_positions(pairs, child_count, relation):
    """Canonicalize validated local endpoints without sorting edge pairs.

    A child has at most one selected parent per relation. Scatter all batches
    first, then compare *every* original pair with the final map: this catches
    conflicts both within and across batches regardless of which repeated
    assignment NumPy retains. Local endpoint bounds/ownership are checked by
    the caller; -1 denotes an unselected child, not a source primary key.
    """
    with phase('native_multihop_parent_map_allocate', 'graph_assembly', relation=relation):
        positions = np.full(child_count, -1, dtype=np.int64)
    with phase('native_multihop_parent_map_scatter', 'graph_assembly', relation=relation):
        for child, parent in pairs:
            positions[child] = parent
    with phase('native_multihop_parent_map_validate', 'graph_assembly', relation=relation):
        for child, parent in pairs:
            if not np.array_equal(positions[child], parent):
                raise ValueError('Selected child has conflicting parents')
    with phase('native_multihop_parent_map_count', 'graph_assembly', relation=relation):
        count = int(np.count_nonzero(positions >= 0))
    return positions, count


class GraphAssembler:
    def __init__(self, bundle):
        self.bundle = bundle
        self.node_types = list(bundle.tables)

    def _adapt(self, workload, seeds, ids, nodes, features, edge_batches, directions):
        """Build final related tables using columnar gathers, not Python edges."""
        from ._contract import schema_for
        schema = schema_for(workload)['related_tables']
        frames, owners, node_tables, orders = {}, {}, {}, {}
        for name in workload.table_names:
            frames[name] = empty_features(self.bundle.tables[name]['columns'])
            owners[name] = np.array([], dtype=np.int64)
        for type_id in set(nodes) | set(features):
            if not nodes[type_id] or not features[type_id]:
                raise ValueError('Native fused node/feature alignment mismatch')
            name = self.node_types[type_id]
            with phase('native_multihop_feature_normalize', 'graph_assembly', table=name):
                ns = pa.Table.from_batches(nodes[type_id])
                fs = normalize_features(pa.Table.from_batches(features[type_id]), self.bundle.tables[name]['columns'])
            with phase('native_multihop_node_validate', 'graph_assembly', table=name):
                if (len(ns) != len(fs) or ns['id_a'].type != pa.uint64() or ns['id_b'].type != pa.uint64()
                        or ns['id_a'].null_count or ns['id_b'].null_count):
                    raise ValueError('Native fused node/feature alignment mismatch')
                owner = indices(ns['id_a'])
                if np.any(owner >= len(seeds)) or np.any(owner < 0):
                    raise ValueError('Native seed position out of bounds')
                if len(fs) > workload.budgets['max_rows_per_table']:
                    raise ValueError('Expanded related table exceeds row budget')
            with phase('native_multihop_identity_sort', 'graph_assembly', table=name):
                pk = fs[self.bundle.tables[name]['primary_key']]
                identity = pa.table({'i': owner, 'k': pk})
                order = pc.sort_indices(identity, sort_keys=[('i', 'ascending'), ('k', 'ascending')])
            with phase('native_multihop_identity_validate', 'graph_assembly', table=name):
                ordered = identity.take(order)
                duplicate = pc.and_(pc.equal(ordered['i'].slice(1), ordered['i'].slice(0, max(0, len(fs)-1))),
                                    pc.equal(ordered['k'].slice(1), ordered['k'].slice(0, max(0, len(fs)-1))))
                if pk.null_count or pc.any(duplicate).as_py():
                    raise ValueError('Native returned duplicate or null node occurrences')
            frames[name], owners[name], node_tables[name], orders[name] = fs, owner, ns, order
        with phase('native_multihop_root_validate', 'graph_assembly'):
            root = frames[workload.entity]
            root_spec = self.bundle.tables[workload.entity]
            root_keys = root[root_spec['primary_key']].slice(0, len(seeds))
            if (len(root_keys) != len(seeds)
                    or not np.array_equal(owners[workload.entity][:len(seeds)], np.arange(len(seeds)))
                    or not pc.all(pc.fill_null(pc.equal(root_keys, pa.array(
                        [s['entity_key'] for s in seeds], type=root_keys.type)), False)).as_py()):
                raise ValueError('Native root prefix differs from input seeds')
            if not np.array_equal(indices(node_tables[workload.entity]['id_b'].slice(0, len(seeds))), np.asarray(ids)):
                raise ValueError('Native root IDs differ from resolved seeds')
            if root_spec.get('time_column'):
                from ._types import TYPES
                anchors = pa.array([s['anchor_time'] for s in seeds], TYPES['timestamp[us]'])
                if not pc.all(pc.fill_null(pc.less_equal(root[root_spec['time_column']].slice(0, len(seeds)), anchors), False)).as_py():
                    raise ValueError('Task entity is not eligible at its anchor')

        # Positions stay in the engine's original node order until final take().
        pairs = defaultdict(list)
        for rel_id, batches in edge_batches.items():
            relation, direction, source, dest = directions[rel_id]
            with phase('native_multihop_edge_validate', 'graph_assembly', relation=relation, direction=direction):
                batch = pa.Table.from_batches(batches)
                if (batch['id_a'].type != pa.uint64() or batch['id_b'].type != pa.uint64()
                        or batch['id_a'].null_count or batch['id_b'].null_count):
                    raise ValueError('Native edge has null endpoints')
                aa, bb = indices(batch['id_a']), indices(batch['id_b'])
                if np.any(aa >= len(frames[source])) or np.any(bb >= len(frames[dest])):
                    raise ValueError('Native edge position out of bounds')
                if not np.array_equal(owners[source][aa], owners[dest][bb]):
                    raise ValueError('Native edge crosses seed instances')
                child, parent = (aa, bb) if direction == 'forward' else (bb, aa)
                pairs[relation].append((child, parent))
        edge_columns, edge_count = {}, 0
        for relation in workload.relation_names:
            rel = self.bundle.relations[relation]
            parent = frames[rel['parent']][self.bundle.tables[rel['parent']]['primary_key']]
            positions, count = _parent_positions(pairs[relation], len(frames[rel['child']]), relation)
            edge_count += count
            with phase('native_multihop_parent_key_gather', 'graph_assembly', relation=relation):
                edge_columns[relation] = pc.take(parent, pa.array(positions, mask=positions < 0))
        with phase('native_multihop_instance_ids', 'graph_assembly'):
            instance_ids = pa.array([s.get('instance_id', i) for i, s in enumerate(seeds)], pa.int64())
        ready = {}
        for name, spec in schema.items():
            columns = {}
            for col in spec['columns']:
                if col == 'instance_id':
                    with phase('native_multihop_instance_gather', 'graph_assembly', table=name):
                        columns[col] = pc.take(instance_ids, pa.array(owners[name]))
                elif col.startswith('__fm_edge_'):
                    columns[col] = edge_columns[col.removeprefix('__fm_edge_')]
                else:
                    columns[col] = frames[name][col]
            with phase('native_multihop_table_build', 'graph_assembly', table=name):
                table = pa.table(columns)
            with phase('native_multihop_output_gather', 'graph_assembly', table=name):
                ready[name] = table.take(orders[name]) if name in orders else table
        return ready
