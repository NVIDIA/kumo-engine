# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded prediction readers and comparisons for nightly cases."""

import math
from pathlib import Path

_BATCH_ROWS = 65536
_SORT_ROWS = 262144
_SORT_BYTES = 64 * 1024**2
_SAMPLE_KEYS = 1024


def ordered_batches(files, directory, *, ranked=True, sort_keys=None):
    """Sort bounded Arrow partitions without materializing Python row records.

    Range partitions keep every entity together. Only leaves below both sort
    bounds enter memory. The sample chooses a split, never a numerical key
    conversion. At most one input and two spill writers are open at a time.
    """
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    custom = sort_keys is not None
    order = list(sort_keys) if custom else [('ENTITY', 'ascending')]
    if ranked and not custom:
        order += [('SCORE', 'descending'), ('CLASS', 'ascending')]
    if (not order or any(not isinstance(key, (tuple, list)) or len(key) != 2
                         or not isinstance(key[0], str) or key[1] not in
                         ('ascending', 'descending')
                         for key in order) or order[0][1] != 'ascending'
            or len({key[0]
                    for key in order}) != len(order)):
        raise ValueError('Sort keys need unique columns and ascending entities')
    entity = order[0][0]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='parquet-sort-', dir=directory) as name:
        work, sequence = Path(name), 0

        def next_path():
            nonlocal sequence
            path = work / f'{sequence}.parquet'
            sequence += 1
            return path

        def extend_bounds(bounds, table):
            if not table.num_rows:
                return
            limits = pc.min_max(table[entity]).as_py()
            bounds[0] += table.num_rows
            bounds[1] = limits['min'] if bounds[1] is None else min(
                bounds[1], limits['min'])
            bounds[2] = limits['max'] if bounds[2] is None else max(
                bounds[2], limits['max'])

        def pivot(reader):
            # Evenly spaced positions bound the sample and avoid float casts.
            count = min(_SAMPLE_KEYS, reader.metadata.num_rows)
            positions = [
                i * reader.metadata.num_rows // count for i in range(count)
            ]
            samples, offset, position = [], 0, 0
            for group in range(reader.metadata.num_row_groups):
                end = offset + reader.metadata.row_group(group).num_rows
                local = []
                while position < count and positions[position] < end:
                    local.append(positions[position] - offset)
                    position += 1
                if local:
                    column = reader.read_row_group(group, columns=[entity],
                                                   use_threads=False)[entity]
                    samples.extend(column.take(pa.array(local)).chunks)
                offset = end
            values = pa.chunked_array(samples)
            values = pc.take(values, pc.sort_indices(values))
            return values[len(values) // 2].as_py()

        def ordered(path, bounds):
            with pq.ParquetFile(path) as reader:
                size = sum(
                    reader.metadata.row_group(i).total_byte_size
                    for i in range(reader.metadata.num_row_groups))
                if bounds[0] <= _SORT_ROWS and size <= _SORT_BYTES:
                    table = reader.read(use_threads=False).sort_by(order)
                    leaf = True
                else:
                    if bounds[1] == bounds[2]:
                        raise ValueError(
                            'One entity exceeds the bounded Arrow sort capacity'
                        )
                    split = pivot(reader)
                    span = bounds[2] - bounds[1]
                    if not bounds[1] + span // 4 <= split < (bounds[2] -
                                                             span // 4):
                        # Bound recursion even when the sample is badly skewed.
                        split = bounds[1] + span // 2
                    scalar = pa.scalar(split, type=schema.field(entity).type)
                    children = [next_path(), next_path()]
                    child_bounds = [[0, None, None], [0, None, None]]
                    with ExitStack() as stack:
                        writers = [
                            stack.enter_context(pq.ParquetWriter(child, schema))
                            for child in children
                        ]
                        for batch in reader.iter_batches(
                                batch_size=_BATCH_ROWS, use_threads=False):
                            table = pa.Table.from_batches([batch],
                                                          schema=schema)
                            mask = pc.less_equal(table[entity], scalar)
                            for side, selection in enumerate(
                                (mask, pc.invert(mask))):
                                part = table.filter(selection)
                                if part.num_rows:
                                    writers[side].write_table(
                                        part, row_group_size=_BATCH_ROWS)
                                    extend_bounds(child_bounds[side], part)
                    del batch, table, mask, part, selection
                    leaf = False
            path.unlink()
            if leaf:
                for batch in table.to_batches(max_chunksize=_BATCH_ROWS):
                    yield batch.to_pandas(ignore_metadata=True)
            else:
                for child, limits in zip(children, child_bounds, strict=True):
                    yield from ordered(child, limits)

        source, schema, bounds = next_path(), None, [0, None, None]
        with ExitStack() as stack:
            writer = None
            for path in files:
                with pq.ParquetFile(path) as reader:
                    observed = reader.schema_arrow.remove_metadata()
                    if schema is None:
                        schema = observed
                    if (not schema.equals(observed, check_metadata=False)
                            or len(set(schema.names)) != len(schema.names) or
                            not {column
                                 for column, _ in order} <= set(schema.names)):
                        raise ValueError('Prediction columns or schemas differ')
                    keys = (entity,
                            'CLASS') if ranked and not custom else (entity, )
                    for column in keys:
                        if not pa.types.is_integer(schema.field(column).type):
                            raise ValueError(
                                f'Prediction changed the integer {column.lower()} key type'
                            )
                    if ranked and not custom and not (pa.types.is_integer(
                            schema.field('SCORE').type) or pa.types.is_floating(
                                schema.field('SCORE').type)):
                        raise ValueError(
                            'Prediction scores must be finite numbers')
                    if writer is None:
                        writer = stack.enter_context(
                            pq.ParquetWriter(source, schema))
                    for batch in reader.iter_batches(batch_size=_BATCH_ROWS,
                                                     use_threads=False):
                        table = pa.Table.from_batches([batch], schema=schema)
                        for column in keys:
                            if table[column].null_count:
                                raise ValueError(
                                    f'Prediction changed the integer {column.lower()} key type'
                                )
                        if ranked and not custom and (
                                table['SCORE'].null_count or not pc.all(
                                    pc.is_finite(table['SCORE'])).as_py()):
                            raise ValueError(
                                'Prediction scores must be finite numbers')
                        writer.write_table(table, row_group_size=_BATCH_ROWS)
                        extend_bounds(bounds, table)
        if bounds[0]:
            yield from ordered(source, bounds)


def ranked_difference(before, after, *, atol, measure_only=False):
    """Bound scores, inverted pairs, and proven changes at the top-K boundary."""
    if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (atol, )):
        raise ValueError('Prediction tolerances must be finite and nonnegative')
    groups = [
        value[key] for value in (before, after)
        for key in ('ranked_ids', 'ranked_scores')
    ]
    if not groups[0] or len({len(group) for group in groups}) != 1:
        raise ValueError('Ranked populations are incomplete')

    def coordinates(names, scores):
        if (not names or len(names) != len(scores)
                or len(set(names)) != len(names) or any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in scores)
                or any(x < y for x, y in zip(scores, scores[1:]))):
            raise ValueError('Ranked coordinates are incomplete or invalid')
        return dict(zip(names, scores, strict=True))

    maximum, maximum_gap, reordered, inversions, added, removed = 0., 0., 0, 0, 0, 0
    unproven = 0
    for index, (left, a, right, b) in enumerate(zip(*groups, strict=True)):
        old, observed = coordinates(left, a), coordinates(right, b)
        if len(left) != len(right):
            raise ValueError('Ranked output cardinality changed')
        changed = left != right
        added += len(observed.keys() - old.keys())
        removed += len(old.keys() - observed.keys())
        if set(left) != set(right):
            if any(
                    len(value.get(key, [])) != len(groups[0])
                    for value in (before, after)
                    for key in ('boundary_ids', 'boundary_scores')):
                raise ValueError(
                    'Membership changes require retained cutoff coordinates')
            union = set(left) | set(right)
            orders, values = [], []
            for value, names, scores in ((before, left, a), (after, right, b)):
                boundary, numbers = value['boundary_ids'][index], value[
                    'boundary_scores'][index]
                mapped = coordinates(boundary, numbers)
                if (boundary[:len(names)] != names
                        or numbers[:len(scores)] != scores
                        or len(boundary) < len(names) + 1):
                    raise ValueError(
                        'Cutoff coordinates do not prove the membership change')
                orders.append([name for name in boundary if name in union])
                values.append(mapped)
            if not all(union <= mapped.keys() for mapped in values):
                if not measure_only:
                    raise ValueError(
                        'Cutoff coordinates do not prove the membership change')
                unproven += 1
                common = union & values[0].keys() & values[1].keys()
                orders = [[name for name in order if name in common]
                          for order in orders]
            left, right = orders
            old, observed = values
        for name in left:
            delta = abs(old[name] - observed[name])
            if delta > atol and not measure_only:
                raise ValueError('Ranked scores exceed the reviewed tolerance')
            maximum = max(maximum, delta)
        if changed:
            reordered += 1
            positions = {name: position for position, name in enumerate(right)}
            for i, first in enumerate(left):
                for second in left[i + 1:]:
                    if positions[first] > positions[second]:
                        maximum_gap = max(maximum_gap, old[first] - old[second],
                                          observed[second] - observed[first])
                        if maximum_gap > atol and not measure_only:
                            raise ValueError(
                                'Rank change exceeds the reviewed score-gap budget'
                            )
                        inversions += 1
    return dict(
        status='PASS' if not unproven and max(maximum, maximum_gap) <= atol else
        'FAIL', policy='measurement' if measure_only else 'enforced',
        reordered_entities=reordered, rank_inversions=inversions,
        added_items=added, removed_items=removed,
        max_abs_aligned_score_difference=maximum,
        max_inverted_score_gap=maximum_gap,
        unproven_membership_changes=unproven, atol=atol, rtol=0.)


def ranked_rows(batches, ids, catalog, exclusions, k, raw_limit, *, anchor=None,
                retry=None):
    """Certify top-K from a bounded, entity/score/key-ordered raw stream."""
    import pandas as pd
    if (not ids or len(set(ids)) != len(ids)
            or len(set(catalog)) != len(catalog) or len(exclusions) != len(ids)
            or type(raw_limit) is not int
            or not 0 < raw_limit <= min(500, len(catalog))):
        raise ValueError('Invalid ranked population or bounded output request')
    positions, members = {v: i for i, v in enumerate(ids)}, set(catalog)
    items, scores = [None] * len(ids), [None] * len(ids)
    boundary_ids, boundary_scores = [None] * len(ids), [None] * len(ids)
    current, pairs, count = None, [], 0
    seen, pending, requests = set(), [], []

    def finish(entity):
        index = positions.get(entity)
        if index is None or entity in seen:
            raise ValueError('Repeated or foreign ranked entity')
        seen.add(entity)
        blocked = set(exclusions[index])
        if (len(blocked) != len(exclusions[index]) or not blocked <= members
                or len(pairs) != raw_limit or len({p[0]
                                                   for p in pairs}) != raw_limit
                or any(p[0] not in members or not math.isfinite(p[1])
                       for p in pairs)):
            raise ValueError(
                'Incomplete, duplicate or invalid ranked coordinates')
        wanted = min(k, len(catalog) - len(blocked))
        eligible = sorted((p for p in pairs if p[0] not in blocked),
                          key=lambda p: (-p[1], p[0]))[:2 * wanted]
        if (not wanted or len(eligible) < wanted or raw_limit < len(catalog)
                and min(p[1] for p in pairs) >= eligible[wanted - 1][1]):
            if not wanted or retry is None or raw_limit >= min(
                    500, len(catalog)):
                raise ValueError(
                    'Capped ranked output cannot certify the eligible cutoff')
            pending.append(entity)
            return
        boundary_ids[index], boundary_scores[index] = [
            p[0] for p in eligible
        ], [p[1] for p in eligible]
        items[index], scores[index] = boundary_ids[
            index][:wanted], boundary_scores[index][:wanted]

    for frame in batches:
        if not {'ENTITY', 'CLASS', 'SCORE'} <= set(frame):
            raise ValueError('Ranked columns are missing')
        for col in ('ENTITY', 'CLASS'):
            if (not pd.api.types.is_integer_dtype(frame[col])
                    or pd.api.types.is_bool_dtype(frame[col])
                    or frame[col].isna().any()):
                raise ValueError(
                    'Ranked IDs must retain their integer business type')
        if anchor is not None and (
                'ANCHOR_TIMESTAMP' not in frame or not pd.to_datetime(
                    frame.ANCHOR_TIMESTAMP).eq(pd.Timestamp(anchor)).all()):
            raise ValueError('Ranked prediction anchors differ')
        for entity, item, score in frame[['ENTITY', 'CLASS', 'SCORE'
                                          ]].itertuples(index=False, name=None):
            if current is not None and entity != current:
                finish(current)
                pairs = []
            current = entity
            pairs.append((int(item), float(score)))
            count += 1
            if len(pairs) > raw_limit:
                raise ValueError('Too many ranked rows for one entity')
    if current is not None:
        finish(current)
    if len(seen) != len(ids):
        raise ValueError('Ranked output is missing entities')
    if pending:
        limit = min(500, len(catalog), 2 * raw_limit)
        extra = ranked_rows(
            retry(pending, limit), pending, catalog,
            [exclusions[positions[entity]]
             for entity in pending], k, limit, anchor=anchor, retry=retry)
        for offset, entity in enumerate(pending):
            index = positions[entity]
            items[index], scores[index] = (extra['ranked_ids'][offset],
                                           extra['ranked_scores'][offset])
            boundary_ids[index], boundary_scores[index] = (
                extra['boundary_ids'][offset], extra['boundary_scores'][offset])
        count += extra['raw_rows']
        requests = [
            dict(ids=pending, raw_limit=limit), *extra['cutoff_retries']
        ]
    return dict(ranked_ids=items, ranked_scores=scores, raw_rows=count,
                boundary_ids=boundary_ids, boundary_scores=boundary_scores,
                cutoff_retries=requests)


def quality(task, truth, values, classes):
    import numpy as np
    from sklearn import metrics
    targets = truth.TARGET1.tolist()
    if task.endswith('link'):
        return float(
            np.mean([
                len(set(y) & set(p)) / len(set(y))
                for y, p in zip(targets, values['ranked_ids'], strict=True)
            ]))
    if task == 'binary':
        return float(metrics.roc_auc_score(targets, values[:, 1])) if len(
            set(targets)) == 2 else None
    if task == 'multiclass':
        return float(
            np.mean(np.asarray(classes)[values.argmax(axis=1)] == targets))
    if task in ('multilabel', 'multilabel_ranking'):
        labels = [
            set(v.split('|') if isinstance(v, str) else v) for v in targets
        ]
        y = np.asarray([[c in row for c in classes] for row in labels],
                       dtype=int)
        if task == 'multilabel':
            return float(
                metrics.average_precision_score(y, values, average='macro'))
        return float(
            np.mean([
                sum(y[i, np.lexsort((np.arange(len(classes)), -row))[:5]]) /
                sum(y[i]) if sum(y[i]) else 0 for i, row in enumerate(values)
            ]))
    target = np.asarray(truth.future_labels.tolist() if task ==
                        'forecasting' else targets)
    return float(np.mean(np.abs(target.reshape(values.shape) - values)))
