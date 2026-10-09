# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""How much of the multiclass score is just "which capture is this?".

Row 2 S3 measured this for the binary target and found the answer alarming: a
model that knows nothing but the capture scores 0.9256 AUROC, and the graph
hands it the capture for free, because 99.87% of hosts appear in exactly one
capture and both endpoint IPs are edges into ``hosts``.

**That finding is worse here, not better.** An IoT-23 capture is one malware
sample run against one device, and the classes of *this* task are malware
families. So where row 2's structural channel leaked "malicious or not", row
4's leaks the answer itself.

This measures the multiclass form of it, without any model:

1. **Capture purity.** For each capture, the share of its rows that are its own
   most common class. Row 3 measured 1.000000 for BCCC's 30 sessions; the
   question is whether IoT-23's 23 are the same.
2. **Class concentration -- the inverted view, and the one that matters.** For
   each class, the share of its rows living in a single capture. A class that
   is 100% inside one capture is perfectly predicted by recovering the capture.
3. **The no-model baseline.** Predict, for every held-out flow, the majority
   class of its capture as computed on the *training* rows only. Report the
   accuracy and macro-F1 of that. A model can beat this by learning flow
   features; it cannot be credited for the part this already explains.
4. **Host capture-privacy**, re-measured on this tier, since it is the
   mechanism that makes 1--3 reachable from the graph.

``flow_id`` is literally ``'<scenario_id>:<uid>'``, so the dropped column is
recoverable from the key. Keys are ID-typed and never featurized, so this is
not itself a leak -- it is what makes the leak measurable.
"""
import argparse
from typing import Any

from snowflake.snowpark import Session

SCHEMA = 'HYU.IOT23'

#: The whole fixture, for ``--raw``: row 2's typed fact table and the two-column
#: target table this row added beside it.
FLOWS_RAW = f'{SCHEMA}.NETWORK_FLOWS_RAW'
LABELS_RAW = f'{SCHEMA}.DETAILED_LABELS_RAW'

#: `flow_id` is '<scenario_id>:<uid>'.
SCENARIO = "SPLIT_PART(flow_id, ':', 1)"


def probe_raw(session: Session, show: Any) -> None:
    r"""The fixture-wide facts, measured over all 325,309,946 rows.

    Separate from the per-tier probe because these are properties of the
    fixture, not of a slice: the class distribution, and the exactness of the
    ``is_malicious`` collapse. A tier can only bound the second, and this row's
    whole leakage story turns on whether it is exact or merely close.
    """
    show(
        'full class distribution', f"""
        SELECT detailed_label, COUNT(*) AS n,
               COUNT(*) / SUM(COUNT(*)) OVER () AS share
        FROM {LABELS_RAW} GROUP BY 1 ORDER BY 2 DESC
        """)

    # `is_malicious` was row 2's *target*. If it is exactly
    # `detailed_label <> '-'` then keeping it would collapse 16 classes to 2 for
    # free, which is why the build excludes it. This counts the exceptions
    # rather than asserting there are none.
    show(
        "rows where is_malicious <> (detailed_label <> '-')", f"""
        SELECT COUNT(*) AS rows_compared,
               COUNT_IF(f.is_malicious <> (d.detailed_label <> '-'))
                   AS disagreements
        FROM {FLOWS_RAW} f JOIN {LABELS_RAW} d ON f.flow_id = d.flow_id
        """)

    # Two directions, and they answer different questions.
    #
    # *Concentration* -- how much of a class sits in one capture -- says whether
    # recovering the capture hands you that class.
    # *Purity* -- how much of a capture is one class -- says whether recovering
    # the capture hands you a class at all.
    #
    # Row 3 measured BCCC's captures at purity 1.000000, which makes both
    # answers "yes". IoT-23 need not behave the same way: a capture there is a
    # real device left running, so it can contain benign background traffic and
    # more than one phase of the malware.
    show(
        'per-class concentration over the whole fixture', f"""
        WITH per_class AS (
            SELECT detailed_label AS cls, {SCENARIO} AS scenario,
                   COUNT(*) AS n
            FROM {LABELS_RAW} GROUP BY 1, 2
        )
        SELECT cls, SUM(n) AS rows_in_class, COUNT(*) AS n_captures,
               MAX(n) / SUM(n) AS share_in_largest_capture
        FROM per_class GROUP BY 1 ORDER BY 2 DESC
        """)

    show(
        'per-capture purity over the whole fixture', f"""
        WITH per_class AS (
            SELECT {SCENARIO} AS scenario, detailed_label AS cls,
                   COUNT(*) AS n
            FROM {LABELS_RAW} GROUP BY 1, 2
        )
        SELECT scenario, SUM(n) AS rows_in_capture, COUNT(*) AS n_classes,
               MAX(n) / SUM(n) AS purity,
               MAX_BY(cls, n) AS majority_class
        FROM per_class GROUP BY 1 ORDER BY purity
        """)

    # The ceiling that capture recovery alone buys, at fixture scale: predict
    # every row as its capture's majority class. This is the number the model
    # has to beat before any of its accuracy is its own.
    show(
        'accuracy of "predict the capture majority class", whole fixture', f"""
        WITH per_class AS (
            SELECT {SCENARIO} AS scenario, detailed_label AS cls,
                   COUNT(*) AS n
            FROM {LABELS_RAW} GROUP BY 1, 2
        ), per_capture AS (
            SELECT scenario, SUM(n) AS total, MAX(n) AS top
            FROM per_class GROUP BY 1
        )
        SELECT SUM(top) / SUM(total) AS capture_majority_accuracy,
               COUNT(*) AS captures
        FROM per_capture
        """)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='small')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--raw', action='store_true',
        help='measure the whole fixture instead of one tier: '
        'class distribution, the is_malicious collapse, '
        'and per-class capture concentration')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows = f'{SCHEMA}.NETWORK_FLOWS_DL_{suffix}'
    truth = f'{flows}_TRUTH'

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    def show(title: str, sql: str) -> None:
        print(f'\n--- {title} ---', flush=True)
        rows = session.sql(sql).collect()
        if rows:
            print('   ', ' | '.join(rows[0].as_dict().keys()), flush=True)
        for row in rows:
            print('   ', ' | '.join(str(v) for v in row), flush=True)

    if args.raw:
        probe_raw(session, show)
        return

    show(
        'held-out set: size and class count', f"""
        SELECT COUNT(*) AS n, COUNT(DISTINCT detailed_label) AS classes
        FROM {truth}
        """)

    # 1. Capture purity. Row 3 found BCCC's captures perfectly single-label;
    #    IoT-23's need not be, since a capture can contain benign background
    #    traffic alongside the malware.
    show(
        'per-capture purity (share of rows in the capture majority class)',
        f"""
        WITH labeled AS (
            SELECT {SCENARIO} AS scenario, detailed_label AS cls
            FROM {flows} WHERE detailed_label IS NOT NULL
        ), per_class AS (
            SELECT scenario, cls, COUNT(*) AS n FROM labeled GROUP BY 1, 2
        ), per_scenario AS (
            SELECT scenario, SUM(n) AS total, MAX(n) AS top,
                   COUNT(*) AS n_classes
            FROM per_class GROUP BY 1
        )
        SELECT scenario, total, n_classes, top / total AS purity
        FROM per_scenario ORDER BY purity
        """)

    # 2. The inverted view, and the reason this row is worse than row 2. A class
    #    whose rows all live in one capture is *given away* by recovering the
    #    capture -- and the capture is recoverable from graph structure alone.
    show(
        'per-class concentration (share of the class in its largest capture)',
        f"""
        WITH labeled AS (
            SELECT {SCENARIO} AS scenario, detailed_label AS cls
            FROM {flows} WHERE detailed_label IS NOT NULL
        ), per_class AS (
            SELECT cls, scenario, COUNT(*) AS n FROM labeled GROUP BY 1, 2
        )
        SELECT cls, SUM(n) AS rows_in_class,
               COUNT(*) AS n_captures,
               MAX(n) / SUM(n) AS share_in_largest_capture
        FROM per_class GROUP BY 1 ORDER BY 2 DESC
        """)

    # 3. The no-model baseline: majority class of the capture, fitted on train
    #    rows only, so it never sees a held-out label. Per-class hits and misses
    #    come back raw so the macro-F1 is computed here rather than in SQL.
    print(
        '\n--- baseline: predict the capture\'s training-set majority '
        'class ---', flush=True)
    rows = session.sql(f"""
        WITH train_top AS (
            SELECT scenario, cls FROM (
                SELECT {SCENARIO} AS scenario, detailed_label AS cls,
                       COUNT(*) AS n,
                       ROW_NUMBER() OVER (PARTITION BY {SCENARIO}
                                          ORDER BY COUNT(*) DESC) AS rank
                FROM {flows} WHERE detailed_label IS NOT NULL
                GROUP BY 1, 2
            ) WHERE rank = 1
        ), scored AS (
            SELECT t.detailed_label AS truth,
                   COALESCE(b.cls, '<unknown>') AS predicted
            FROM {truth} t
            LEFT JOIN train_top b ON b.scenario = {SCENARIO}
        )
        SELECT truth, predicted, COUNT(*) AS n FROM scored GROUP BY 1, 2
    """).collect()

    support: dict = {}
    n_pred: dict = {}
    hits: dict = {}
    total = correct = 0
    for truth_cls, pred_cls, n in rows:
        support[truth_cls] = support.get(truth_cls, 0) + n
        n_pred[pred_cls] = n_pred.get(pred_cls, 0) + n
        total += n
        if truth_cls == pred_cls:
            hits[truth_cls] = hits.get(truth_cls, 0) + n
            correct += n

    print(
        f'\n{"class":<34} {"support":>10} {"pred":>10} '
        f'{"prec":>7} {"recall":>7} {"f1":>7}', flush=True)
    f1s = []
    for cls in sorted(set(support) | set(n_pred)):
        hit = hits.get(cls, 0)
        sup, pred = support.get(cls, 0), n_pred.get(cls, 0)
        precision = hit / pred if pred else 0.0
        recall = hit / sup if sup else float('nan')
        f1 = (2 * precision * recall / (precision + recall) if precision +
              recall else 0.0)
        if sup:
            f1s.append(f1)
        print(
            f'{str(cls):<34} {sup:>10,} {pred:>10,} '
            f'{precision:>7.4f} {recall:>7.4f} {f1:>7.4f}', flush=True)
    print(f'\nbaseline accuracy  {correct / total:.4f}', flush=True)
    print(
        f'baseline macro F1  {sum(f1s) / len(f1s):.4f} '
        f'over {len(f1s)} classes with support', flush=True)
    print(f'majority rate      {max(support.values()) / total:.4f}',
          flush=True)

    # 4. The mechanism. Both endpoint IPs are edges into `hosts`, so a host that
    #    appears in exactly one capture is a direct channel to the capture, and
    #    through 1-3 to the class.
    show(
        'how many hosts appear in more than one capture', f"""
        WITH endpoint AS (
            SELECT source_ip AS ip, {SCENARIO} AS scenario FROM {flows}
            UNION
            SELECT destination_ip AS ip, {SCENARIO} AS scenario FROM {flows}
        )
        SELECT COUNT(*) AS hosts,
               SUM(IFF(scenarios = 1, 1, 0)) AS in_one_scenario_only,
               AVG(scenarios) AS mean_scenarios
        FROM (SELECT ip, COUNT(DISTINCT scenario) AS scenarios
              FROM endpoint GROUP BY 1)
        """)


if __name__ == '__main__':
    main()
