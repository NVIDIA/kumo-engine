# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the BCCC-CSE-CIC-IDS2018 fixture as Snowflake tables.

The local backend's source area reads Snowflake *tables*, not stages, so the
staged parquet has to land in a schema before the SDK can see it. This builds
one scale at a time into ``HYU.BCCC``. Same shape as
``../iot23/build_iot23_tables.py``; what differs is noted below.

Shape of the build
------------------

Everything derives from one typed copy of the fact table,
``NETWORK_FLOWS_RAW``. Building it costs one pass over the staged parquet; every
scale after that is a subset of a native Snowflake table, which is far cheaper
than re-parsing 20 GB of variant-typed parquet per scale. ``--rebuild-raw``
forces that first pass again, and ``--one-part`` restricts it to a single part
file so the 325-column cast list can be validated in ninety seconds rather than
after an hour.

Scales are nested deterministic samples of ``flow_id``::

    MOD(ABS(HASH(flow_id)), 10000) < permille10

so ``smoke`` is a subset of ``small`` is a subset of ``part``, and a given
``flow_id`` lands in the same tier on every rebuild. The sample is uniform over
the whole fixture rather than a prefix, for the same reason as iot23: the parts
are capture-aligned. ``part-00003`` is 544,081 flows of
``friday_16_02_2018_benign_part2`` and nothing else -- one capture, one label,
zero attacks. A prefix slice would be a fixture with no positive class at all.

Multiclass raises the stakes on the sample width, and on this fixture it
decides what the tiers even mean. Measured over all 48,120,094 rows:

===================  ============  ===========
class                        rows      share
===================  ============  ===========
Benign                 43,029,445     89.421%
DDoS_HOIC               2,164,586      4.498%
DoS_HULK                1,824,450      3.791%
DDoS_LOIC_HTTP            289,527      0.602%
Bot                       286,099      0.595%
Brute_Force_FTP           193,360      0.402%
Brute_Force_SSH           188,444      0.392%
DoS_SlowHTTP              105,550      0.219%
DoS_Golden_Eye             29,485      0.061%
DoS_Slowloris               8,269      0.017%
Brute_Force_Web               536      0.0011%
Brute_Force_XSS               232      0.0005%
SQL_Injection                 106      0.0002%
Infiltration                    5    0.00001%
===================  ============  ===========

A uniform hash sample keeps each class in proportion, so ``Infiltration``'s
**five rows in forty-eight million** survive a 1% sample with probability 0.05.
The four rarest classes together are 879 rows, 0.0018% of the fixture, and no
tier below ``full`` is expected to contain all of them. **Only the full fixture
is a fourteen-class problem**; every sample is a smaller one, and the tier's
class count has to be read alongside its metrics rather than assumed.

The sample is left uniform anyway, deliberately. Over-sampling the tail would
make each tier a fourteen-class problem, but it would also make the tiers stop
nesting and stop being samples of the fixture -- the prior would be an artefact
of the sampler, and the majority-class baseline every accuracy figure is
measured against would move with it. What the build does instead is *report*
which classes survived, so the loss is a stated fact rather than a silent one.

Two floors remain, because below them the tier is not merely smaller but a
different kind of task: a class present but never held out cannot be scored, and
a tier that drops to two classes has the trainer silently rewrite the task type
to binary (``kumo_core/train/trainer.py``: "Setting the task type to binary
classification since there only exists two classes").

Columns
-------

325 columns, against iot23's 21. The vast majority are per-flow traffic
statistics and are kept. What is dropped, and why:

``is_attack``
    The binary collapse of the target: ``False`` iff ``label = 'Benign'``. Pure
    leak, and it is the row-2 target in disguise.
``capture_id``
    Names the capture session of the CSE-CIC-IDS2018 testbed. Each session ran
    one scripted attack against a fixed victim set, so capture is close to
    deterministic for the label -- this is iot23's ``scenario_id`` under another
    name. As a categorical feature it is the answer, not a hint.

``flow_id`` stays as the primary key. It is
``'<capture_id>:<src>_<sport>_<dst>_<dport>_<proto>_<timestamp>'``, so the
dropped ``capture_id`` survives inside the key exactly as it did in iot23. Keys
are ID-typed and never featurized, so this should be inert, and the run script
checks the trained model's encoder list to confirm it.

Types and non-finite values
---------------------------

``INFER_SCHEMA`` cannot settle on a single physical type for 18 of the columns:
they are float statistics that some part files write as JSON strings
(``"delta_start": "0.0001"``) and others as numbers. ``TRY_TO_DOUBLE`` accepts
both and returns NULL rather than raising on anything else.

Separately, the ``cov_*`` and ``skewness_*`` statistics are computed as
``std/mean`` and are literally ``Infinity`` in the source whenever the mean is
zero -- visible in a single sampled row, so not rare. An infinity in a feature
column is worse than a null: it survives the cast, then destroys the mean and
standard deviation used to normalise the whole column. Every float column is
therefore passed through ``IFF(v BETWEEN -1e308 AND 1e308, v, NULL)``, which
nulls +inf, -inf and NaN in one predicate -- Snowflake orders NaN above every
finite value, so it fails the range test too. Note that Snowflake's ``NaN =
NaN`` is *true*, so the usual ``v != v`` idiom does not work here.

Deduplication
-------------

``flow_id`` is **not unique**, contrary to the matrix: ``part-00003`` alone has
544,081 rows for 543,740 distinct keys. The key is a 5-tuple plus a microsecond
timestamp, so two flows opened in the same microsecond between the same ports
collide. Ethereum's declared key had the same problem. ``RAW`` keeps one row per
``flow_id``, chosen by ``ROW_NUMBER`` so the choice is deterministic.

Holdout
-------

``--holdout-pct`` is what makes the fixture scoreable at all. For a *static*
column target Kumo compiles the prediction table to ``WHERE target IS NULL`` --
train on the labeled rows, score the unlabeled ones -- and the fixture labels
every flow, so batch prediction over it as published selects nothing. This is
the third confirmed instance; ethereum predicted it for exactly this row.
Blanking a deterministic slice of ``label`` restores the intended shape, and the
blanked rows keep their true label in ``*_TRUTH``.
"""
import argparse
import time
from typing import List

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.BCCC_IDS_20GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.BCCC'
RAW = f'{SCHEMA}.NETWORK_FLOWS_RAW'
ONE_PART = 'part-00003.parquet'

#: Sample width in parts per 10,000 of `flow_id`, against 48,182,782 flows
#: (48,140,000-odd after deduplication). `full` is the whole fact table; the
#: rest nest inside each other.
SCALES = {
    'smoke': 5,  # ~24K flows
    'small': 100,  # ~482K flows
    'part': 500,  # ~2.4M flows
    'half': 5000,  # ~24.1M flows
    'full': 10000,  # 48.1M flows
}

#: Deterministic in the key itself, so the same rows are sampled and the same
#: rows are held out on every rebuild and at every scale.
#:
#: The two predicates must hash *different* strings. Nesting two thresholds on
#: one hash makes them perfectly dependent, not independent: taking the sample
#: as ``MOD(h, 10000) < 5`` means every sampled row also has ``MOD(h, 100) < 5``,
#: so a ``< 20`` holdout on the same ``h`` selects all of them. The iot23 build
#: did exactly that and produced a smoke slice with zero labeled rows. Salting
#: the holdout hash decorrelates them.
#: The fourteen classes, measured over all 48,120,094 rows. Kept here so a
#: tier can report which of them it lost without re-scanning the fact table.
FIXTURE_CLASSES = [
    'Benign',
    'DDoS_HOIC',
    'DoS_HULK',
    'DDoS_LOIC_HTTP',
    'Bot',
    'Brute_Force_FTP',
    'Brute_Force_SSH',
    'DoS_SlowHTTP',
    'DoS_Golden_Eye',
    'DoS_Slowloris',
    'Brute_Force_Web',
    'Brute_Force_XSS',
    'SQL_Injection',
    'Infiltration',
]

SAMPLED = 'MOD(ABS(HASH(flow_id)), 10000) < {permille10}'
HELD_OUT = "MOD(ABS(HASH(flow_id || '/holdout')), 100) < {pct}"

#: 54 integer feature columns. Integers are the only ones whose stype can flip
#: between tiers -- `kumo_core.stype_inference` calls an integer column
#: categorical when it is under MAX_CATEGORIES (4000) distinct *and* under
#: UNIQUE_INT_THRESHOLD (5%) unique, and a 10,000-row inference sample can put a
#: column on either side of that. Floats are numerical unconditionally.
INT_FEATURES = [
    'packets_count', 'fwd_packets_count', 'bwd_packets_count',
    'total_payload_bytes', 'fwd_total_payload_bytes',
    'bwd_total_payload_bytes', 'payload_bytes_max', 'payload_bytes_min',
    'fwd_payload_bytes_max', 'fwd_payload_bytes_min', 'bwd_payload_bytes_max',
    'bwd_payload_bytes_min', 'total_header_bytes', 'max_header_bytes',
    'min_header_bytes', 'fwd_total_header_bytes', 'fwd_max_header_bytes',
    'fwd_min_header_bytes', 'bwd_total_header_bytes', 'bwd_max_header_bytes',
    'bwd_min_header_bytes', 'fwd_init_win_bytes', 'bwd_init_win_bytes',
    'fwd_bulk_state_count', 'fwd_bulk_total_size', 'fwd_bulk_per_packet',
    'bwd_bulk_state_count', 'bwd_bulk_total_size', 'bwd_bulk_per_packet',
    'fin_flag_counts', 'psh_flag_counts', 'urg_flag_counts', 'ece_flag_counts',
    'syn_flag_counts', 'ack_flag_counts', 'cwr_flag_counts', 'rst_flag_counts',
    'fwd_fin_flag_counts', 'fwd_psh_flag_counts', 'fwd_urg_flag_counts',
    'fwd_ece_flag_counts', 'fwd_syn_flag_counts', 'fwd_ack_flag_counts',
    'fwd_cwr_flag_counts', 'fwd_rst_flag_counts', 'bwd_fin_flag_counts',
    'bwd_psh_flag_counts', 'bwd_urg_flag_counts', 'bwd_ece_flag_counts',
    'bwd_syn_flag_counts', 'bwd_ack_flag_counts', 'bwd_cwr_flag_counts',
    'bwd_rst_flag_counts', 'handshake_state'
]
#: 261 float feature columns, including the 18 that ``INFER_SCHEMA`` reported as
#: VARIANT. All go through the same ``TRY_TO_DOUBLE`` + finite-range guard.
FLOAT_FEATURES = [
    'duration', 'payload_bytes_mean', 'payload_bytes_std',
    'payload_bytes_variance', 'payload_bytes_median', 'payload_bytes_skewness',
    'payload_bytes_cov', 'payload_bytes_mode', 'fwd_payload_bytes_mean',
    'fwd_payload_bytes_std', 'fwd_payload_bytes_variance',
    'fwd_payload_bytes_median', 'fwd_payload_bytes_skewness',
    'fwd_payload_bytes_cov', 'fwd_payload_bytes_mode',
    'bwd_payload_bytes_mean', 'bwd_payload_bytes_std',
    'bwd_payload_bytes_variance', 'bwd_payload_bytes_median',
    'bwd_payload_bytes_skewness', 'bwd_payload_bytes_cov',
    'bwd_payload_bytes_mode', 'mean_header_bytes', 'std_header_bytes',
    'median_header_bytes', 'skewness_header_bytes', 'cov_header_bytes',
    'mode_header_bytes', 'variance_header_bytes', 'fwd_mean_header_bytes',
    'fwd_std_header_bytes', 'fwd_median_header_bytes',
    'fwd_skewness_header_bytes', 'fwd_cov_header_bytes',
    'fwd_mode_header_bytes', 'fwd_variance_header_bytes',
    'bwd_mean_header_bytes', 'bwd_std_header_bytes', 'bwd_median_header_bytes',
    'bwd_skewness_header_bytes', 'bwd_cov_header_bytes',
    'bwd_mode_header_bytes', 'bwd_variance_header_bytes',
    'fwd_avg_segment_size', 'bwd_avg_segment_size', 'avg_segment_size',
    'active_min', 'active_max', 'active_mean', 'active_std', 'active_median',
    'active_skewness', 'active_cov', 'active_mode', 'active_variance',
    'idle_min', 'idle_max', 'idle_mean', 'idle_std', 'idle_median',
    'idle_skewness', 'idle_cov', 'idle_mode', 'idle_variance', 'bytes_rate',
    'fwd_bytes_rate', 'bwd_bytes_rate', 'packets_rate', 'bwd_packets_rate',
    'fwd_packets_rate', 'down_up_rate', 'avg_fwd_bytes_per_bulk',
    'avg_fwd_packets_per_bulk', 'avg_fwd_bulk_rate', 'avg_bwd_bytes_per_bulk',
    'avg_bwd_packets_bulk_rate', 'avg_bwd_bulk_rate', 'fwd_bulk_duration',
    'bwd_bulk_duration', 'fin_flag_percentage_in_total',
    'psh_flag_percentage_in_total', 'urg_flag_percentage_in_total',
    'ece_flag_percentage_in_total', 'syn_flag_percentage_in_total',
    'ack_flag_percentage_in_total', 'cwr_flag_percentage_in_total',
    'rst_flag_percentage_in_total', 'fwd_fin_flag_percentage_in_total',
    'fwd_psh_flag_percentage_in_total', 'fwd_urg_flag_percentage_in_total',
    'fwd_ece_flag_percentage_in_total', 'fwd_syn_flag_percentage_in_total',
    'fwd_ack_flag_percentage_in_total', 'fwd_cwr_flag_percentage_in_total',
    'fwd_rst_flag_percentage_in_total', 'bwd_fin_flag_percentage_in_total',
    'bwd_psh_flag_percentage_in_total', 'bwd_urg_flag_percentage_in_total',
    'bwd_ece_flag_percentage_in_total', 'bwd_syn_flag_percentage_in_total',
    'bwd_ack_flag_percentage_in_total', 'bwd_cwr_flag_percentage_in_total',
    'bwd_rst_flag_percentage_in_total',
    'fwd_fin_flag_percentage_in_fwd_packets',
    'fwd_psh_flag_percentage_in_fwd_packets',
    'fwd_urg_flag_percentage_in_fwd_packets',
    'fwd_ece_flag_percentage_in_fwd_packets',
    'fwd_syn_flag_percentage_in_fwd_packets',
    'fwd_ack_flag_percentage_in_fwd_packets',
    'fwd_cwr_flag_percentage_in_fwd_packets',
    'fwd_rst_flag_percentage_in_fwd_packets',
    'bwd_fin_flag_percentage_in_bwd_packets',
    'bwd_psh_flag_percentage_in_bwd_packets',
    'bwd_urg_flag_percentage_in_bwd_packets',
    'bwd_ece_flag_percentage_in_bwd_packets',
    'bwd_syn_flag_percentage_in_bwd_packets',
    'bwd_ack_flag_percentage_in_bwd_packets',
    'bwd_cwr_flag_percentage_in_bwd_packets',
    'bwd_rst_flag_percentage_in_bwd_packets', 'packets_IAT_mean',
    'packet_IAT_std', 'packet_IAT_max', 'packet_IAT_min', 'packet_IAT_total',
    'packets_IAT_median', 'packets_IAT_skewness', 'packets_IAT_cov',
    'packets_IAT_mode', 'packets_IAT_variance', 'fwd_packets_IAT_mean',
    'fwd_packets_IAT_std', 'fwd_packets_IAT_max', 'fwd_packets_IAT_min',
    'fwd_packets_IAT_total', 'fwd_packets_IAT_median',
    'fwd_packets_IAT_skewness', 'fwd_packets_IAT_cov', 'fwd_packets_IAT_mode',
    'fwd_packets_IAT_variance', 'bwd_packets_IAT_mean', 'bwd_packets_IAT_std',
    'bwd_packets_IAT_max', 'bwd_packets_IAT_min', 'bwd_packets_IAT_total',
    'bwd_packets_IAT_median', 'bwd_packets_IAT_skewness',
    'bwd_packets_IAT_cov', 'bwd_packets_IAT_mode', 'bwd_packets_IAT_variance',
    'subflow_fwd_packets', 'subflow_bwd_packets', 'subflow_fwd_bytes',
    'subflow_bwd_bytes', 'delta_start', 'handshake_duration',
    'min_bwd_packets_delta_time', 'max_bwd_packets_delta_time',
    'mean_packets_delta_time', 'mode_packets_delta_time',
    'variance_packets_delta_time', 'std_packets_delta_time',
    'median_packets_delta_time', 'skewness_packets_delta_time',
    'cov_packets_delta_time', 'mean_bwd_packets_delta_time',
    'mode_bwd_packets_delta_time', 'variance_bwd_packets_delta_time',
    'std_bwd_packets_delta_time', 'median_bwd_packets_delta_time',
    'skewness_bwd_packets_delta_time', 'cov_bwd_packets_delta_time',
    'min_fwd_packets_delta_time', 'max_fwd_packets_delta_time',
    'mean_fwd_packets_delta_time', 'mode_fwd_packets_delta_time',
    'variance_fwd_packets_delta_time', 'std_fwd_packets_delta_time',
    'median_fwd_packets_delta_time', 'skewness_fwd_packets_delta_time',
    'cov_fwd_packets_delta_time', 'min_packets_delta_len',
    'max_packets_delta_len', 'mean_packets_delta_len',
    'mode_packets_delta_len', 'variance_packets_delta_len',
    'std_packets_delta_len', 'median_packets_delta_len',
    'skewness_packets_delta_len', 'cov_packets_delta_len',
    'min_bwd_packets_delta_len', 'max_bwd_packets_delta_len',
    'mean_bwd_packets_delta_len', 'mode_bwd_packets_delta_len',
    'variance_bwd_packets_delta_len', 'std_bwd_packets_delta_len',
    'median_bwd_packets_delta_len', 'skewness_bwd_packets_delta_len',
    'cov_bwd_packets_delta_len', 'min_fwd_packets_delta_len',
    'max_fwd_packets_delta_len', 'mean_fwd_packets_delta_len',
    'mode_fwd_packets_delta_len', 'variance_fwd_packets_delta_len',
    'std_fwd_packets_delta_len', 'median_fwd_packets_delta_len',
    'skewness_fwd_packets_delta_len', 'cov_fwd_packets_delta_len',
    'min_header_bytes_delta_len', 'max_header_bytes_delta_len',
    'mean_header_bytes_delta_len', 'mode_header_bytes_delta_len',
    'variance_header_bytes_delta_len', 'std_header_bytes_delta_len',
    'median_header_bytes_delta_len', 'skewness_header_bytes_delta_len',
    'cov_header_bytes_delta_len', 'min_bwd_header_bytes_delta_len',
    'max_bwd_header_bytes_delta_len', 'mean_bwd_header_bytes_delta_len',
    'mode_bwd_header_bytes_delta_len', 'variance_bwd_header_bytes_delta_len',
    'std_bwd_header_bytes_delta_len', 'median_bwd_header_bytes_delta_len',
    'skewness_bwd_header_bytes_delta_len', 'cov_bwd_header_bytes_delta_len',
    'min_fwd_header_bytes_delta_len', 'max_fwd_header_bytes_delta_len',
    'mean_fwd_header_bytes_delta_len', 'mode_fwd_header_bytes_delta_len',
    'variance_fwd_header_bytes_delta_len', 'std_fwd_header_bytes_delta_len',
    'median_fwd_header_bytes_delta_len', 'skewness_fwd_header_bytes_delta_len',
    'cov_fwd_header_bytes_delta_len', 'min_payload_bytes_delta_len',
    'max_payload_bytes_delta_len', 'mean_payload_bytes_delta_len',
    'mode_payload_bytes_delta_len', 'variance_payload_bytes_delta_len',
    'std_payload_bytes_delta_len', 'median_payload_bytes_delta_len',
    'skewness_payload_bytes_delta_len', 'cov_payload_bytes_delta_len',
    'min_bwd_payload_bytes_delta_len', 'max_bwd_payload_bytes_delta_len',
    'mean_bwd_payload_bytes_delta_len', 'mode_bwd_payload_bytes_delta_len',
    'variance_bwd_payload_bytes_delta_len', 'std_bwd_payload_bytes_delta_len',
    'median_bwd_payload_bytes_delta_len',
    'skewness_bwd_payload_bytes_delta_len', 'cov_bwd_payload_bytes_delta_len',
    'min_fwd_payload_bytes_delta_len', 'max_fwd_payload_bytes_delta_len',
    'mean_fwd_payload_bytes_delta_len', 'mode_fwd_payload_bytes_delta_len',
    'variance_fwd_payload_bytes_delta_len', 'std_fwd_payload_bytes_delta_len',
    'median_fwd_payload_bytes_delta_len',
    'skewness_fwd_payload_bytes_delta_len', 'cov_fwd_payload_bytes_delta_len'
]

#: The columns handled by name rather than in bulk: the key, the two foreign
#: keys, the time column, the target, and the two that are dropped.
#:
#: ``timestamp`` is **microseconds** since epoch. There is no float-seconds
#: sibling to settle that against as iot23's ``ts`` was, but ``flow_id`` embeds
#: the same instant as a rendered string -- 1518766768399438 against
#: ``...TCP_2018-02-16 07:39:28.399438`` -- which pins it exactly. Reading it as
#: seconds gives the year 48,129,749, and Snowflake will build that timestamp
#: happily and fail later in the arrow decoder rather than at the cast.
KEY_COLUMNS = """
    $1:flow_id::STRING                AS flow_id,
    $1:label::STRING                  AS label,
    $1:src_ip::STRING                 AS src_ip,
    $1:dst_ip::STRING                 AS dst_ip,
    $1:src_port::NUMBER(38,0)         AS src_port,
    $1:dst_port::NUMBER(38,0)         AS dst_port,
    $1:protocol::STRING               AS protocol,
    TO_TIMESTAMP_NTZ($1:timestamp::NUMBER(38,0), 6) AS event_time
"""


def cast_list() -> str:
    r"""The full ``SELECT`` list: keys, then 54 integers, then 261 floats.

    Generated rather than written out because 315 near-identical cast
    expressions are not worth reading, and because the finite-range guard has to
    be applied to every float without exception -- see the module docstring for
    why an ``Infinity`` that reaches the model is worse than a null.
    """
    parts: List[str] = [KEY_COLUMNS.strip()]
    parts += [f'$1:{c}::NUMBER(38,0) AS {c}' for c in INT_FEATURES]
    # `TRY_TO_DOUBLE` refuses a VARIANT argument outright -- "Function TRY_CAST
    # cannot be used with arguments of types VARIANT and FLOAT" -- so the value
    # goes through its string form. `$1:c::FLOAT` would avoid the round trip and
    # keep the last bit of precision, but it *raises* on anything unparseable,
    # and finding that out is an hour into the full pass. A last-ULP difference
    # in a traffic statistic is not worth that trade.
    parts += [
        f"IFF(TRY_TO_DOUBLE($1:{c}::STRING) BETWEEN -1e308 AND 1e308, "
        f'TRY_TO_DOUBLE($1:{c}::STRING), NULL) AS {c}' for c in FLOAT_FEATURES
    ]
    return ',\n    '.join(parts)


#: Substrings that must not appear in a feature column's name, and what they
#: are rewritten to. See ``stat_safe_name`` for why.
#:
#: Applied longest-first so no rewrite can create another token, and every
#: replacement is itself token-free (``cnt``, ``mid``, ``avg``, ``lwr``,
#: ``upr``, ``sdv``).
STAT_TOKEN_REWRITES = [
    ('median', 'mid'),
    ('count', 'cnt'),
    ('mean', 'avg'),
    ('min', 'lwr'),
    ('max', 'upr'),
    ('std', 'sdv'),
]


def stat_safe_name(name: str) -> str:
    r"""Renames a column so ``kumo_core`` can find its statistics.

    ``kumo_core.pquery._column_stats`` resolves a column's statistics out of a
    flat dictionary whose keys look like::

        PACKETS_COUNT__kumo.numerical__MEAN__False

    It does that by testing every ``ColStatType`` value for membership **in the
    whole key**, longest first, and taking the first hit::

        for stat_type in sorted(ColStatType, key=lambda v: len(v.value),
                                reverse=True):
            if stat_type.value in key:
                result[stat_type.value] = ...
                break

    The column name is part of that key. So a column called ``PACKETS_COUNT``
    matches ``COUNT`` (5 characters) before ``MEAN`` (4), and its mean, min, max
    and standard deviation are *all* filed under ``COUNT``, each overwriting the
    last. On this fixture that surfaces as::

        PredictiveQueryPlanError: encoder inference for 'PACKETS_COUNT'
        requires MEAN

    which is the benign case, because it stops. The damaging case is silent.
    ``PAYLOAD_BYTES_MEAN`` contains ``MEAN``, which is longer than ``MIN``,
    ``MAX`` and ``STD``, so those three are filed as ``MEAN`` -- and measured on
    the smoke index the column's parsed ``MEAN`` is 136.17, which is its
    *standard deviation*; the true mean is 99.35. Nothing raises. The encoder is
    then chosen, and the column normalised, against another statistic's value.
    Worse still for a categorical column: ``MAX_HEADER_BYTES`` has no numerical
    statistics at all, yet parses to ``MEAN=1.163, MIN=-20.0, MAX=4.275``,
    invented out of its ``CATEGORY_COUNTS`` payload.

    **142 of this fixture's 323 columns contain such a token** -- 44% -- via
    ``COUNT`` (29), ``MEAN`` (23), ``STD`` (23), ``MEDIAN`` (23), ``MAX`` (22)
    and ``MIN`` (22). That is a property of BCCC's naming, which labels every
    per-flow statistic by the statistic it is; rows 1 and 2 never hit it because
    no ethereum or iot23 column name contains a ``ColStatType`` token.

    Renaming the columns in the tier tables is the only fix available from this
    repository. The alternative is a one-line change upstream -- match the stat
    token against the key's *suffix* rather than the whole key -- which is where
    the fix belongs, and this workaround should be deleted when it lands.
    """
    out = name.lower()
    for token, replacement in STAT_TOKEN_REWRITES:
        out = out.replace(token, replacement)
    return out


#: The RAW column order: keys first, then the integer features, then the float
#: ones, matching ``cast_list``.
RAW_COLUMNS = [
    'flow_id', 'label', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'event_time'
] + INT_FEATURES + FLOAT_FEATURES


def tier_select(held_out: str) -> str:
    r"""The tier ``SELECT`` list: every RAW column, renamed and holdout-blanked.

    ``SELECT * REPLACE (...)`` would be shorter but cannot rename, and the
    rename is not optional -- see ``stat_safe_name``.
    """
    parts = []
    for name in RAW_COLUMNS:
        safe = stat_safe_name(name)
        if name == 'label':
            parts.append(f'IFF({held_out}, NULL, label) AS label')
        elif safe != name:
            parts.append(f'{name} AS {safe}')
        else:
            parts.append(name)
    return ',\n    '.join(parts)


def run(session: Session, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f'[{label}] running...', flush=True)
    session.sql(sql).collect()
    print(f'[{label}] done in {time.monotonic() - start:.1f}s', flush=True)


def table_exists(session: Session, name: str) -> bool:
    database, schema, table = name.split('.')
    rows = session.sql(
        f"SHOW TABLES LIKE '{table}' IN SCHEMA {database}.{schema}").collect()
    return len(rows) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=list(SCALES), default='smoke')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--holdout-pct', type=int, default=20,
        help='percent of flows whose label is blanked so batch '
        'prediction has rows to score')
    parser.add_argument(
        '--rebuild-raw', action='store_true',
        help='re-read the staged parquet even if '
        'NETWORK_FLOWS_RAW already exists')
    parser.add_argument(
        '--one-part', action='store_true',
        help='build RAW from a single part file, to validate '
        'the 325-column cast list cheaply')
    parser.add_argument('--raw-only', action='store_true',
                        help='build RAW and stop')
    parser.add_argument(
        '--min-class-rows', type=int, default=25,
        help='below this many labeled rows a class is '
        'reported as too thin to score; it is not a '
        'failure, since a uniform sample of this fixture '
        'necessarily thins its tail')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows_table = f'{SCHEMA}.NETWORK_FLOWS_{suffix}'
    hosts_table = f'{SCHEMA}.HOSTS_{suffix}'
    truth_table = f'{flows_table}_TRUTH'

    session = Session.builder.config('connection_name', 'tlb').create()
    # The size has to be set on the session: `USE WAREHOUSE` is the only thing
    # that moves work, and the connection profile names none, so the default is
    # KUMO_XS.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    if args.rebuild_raw or not table_exists(session, RAW):
        source = (f"'@{STAGE}/network_flows/{ONE_PART}'"
                  if args.one_part else f"'@{STAGE}/network_flows/'")
        # QUALIFY is the deduplication: `flow_id` collides whenever two flows
        # share a 5-tuple within one microsecond. ORDER BY the key itself makes
        # the surviving row a deterministic function of the data rather than of
        # the scan order.
        run(
            session, f'staged parquet -> {RAW}', f"""
            CREATE OR REPLACE TABLE {RAW} AS
            SELECT {cast_list()}
            FROM {source} {FF}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY flow_id ORDER BY flow_id) = 1
        """)
        n_raw = session.sql(f'SELECT COUNT(*) FROM {RAW}').collect()[0][0]
        print(f'{RAW}: {n_raw:,} rows after deduplication', flush=True)

    if args.raw_only:
        return

    sampled = SAMPLED.format(permille10=SCALES[args.scale])
    held_out = HELD_OUT.format(pct=args.holdout_pct)

    run(
        session, f'flows ({args.holdout_pct}% held out) -> {flows_table}', f"""
        CREATE OR REPLACE TABLE {flows_table} AS
        SELECT {tier_select(held_out)}
        FROM {RAW}
        WHERE {sampled}
    """)

    run(
        session, f'held-out labels -> {truth_table}', f"""
        CREATE OR REPLACE TABLE {truth_table} AS
        SELECT flow_id, label FROM {RAW}
        WHERE {sampled} AND {held_out}
    """)

    # The published `hosts` table is 407,244 rows of nothing but `host_id`: it
    # is exactly the set of endpoint IPs, with no attributes to contribute. So
    # the host node set is derived from the flows in scope instead of read back
    # from the stage. That guarantees no foreign key points at a missing row and
    # no isolated node is ingested -- an unreferenced, featureless node cannot
    # affect a GNN, it can only cost an ingest.
    run(
        session, f'endpoint IPs -> {hosts_table}', f"""
        CREATE OR REPLACE TABLE {hosts_table} AS
        SELECT DISTINCT ip AS host_id FROM (
            SELECT src_ip AS ip FROM {flows_table}
            UNION ALL
            SELECT dst_ip FROM {flows_table}
        )
        WHERE ip IS NOT NULL
    """)

    for name in (flows_table, hosts_table, truth_table):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)

    # Per-class counts on both sides of the holdout. A class that is labeled but
    # never held out cannot be scored; a class that is held out but never
    # labeled cannot be learned. Both are silent failures if not printed here.
    rows = session.sql(f"""
        SELECT t.label,
               COUNT_IF(f.label IS NOT NULL) AS labeled,
               COUNT(*) - COUNT_IF(f.label IS NOT NULL) AS held_out
        FROM {flows_table} f
        JOIN (SELECT flow_id, label FROM {RAW} WHERE {sampled}) t
          ON f.flow_id = t.flow_id
        GROUP BY 1 ORDER BY 2 + 3 DESC
    """).collect()
    print(f'\n{"class":<28} {"labeled":>12} {"held out":>10}', flush=True)
    for label, labeled, held in rows:
        print(f'{label:<28} {labeled:>12,} {held:>10,}', flush=True)

    total_labeled = sum(r[1] for r in rows)
    total_held = sum(r[2] for r in rows)
    print(f'\n{"TOTAL":<28} {total_labeled:>12,} {total_held:>10,}',
          flush=True)

    # Which of the fixture's classes this tier lost outright. Printed rather
    # than raised on: see the module docstring -- a uniform sample of a fixture
    # whose rarest class has five rows in forty-eight million *will* lose
    # classes, and that is the honest tier rather than a broken one.
    present = {label for label, _, _ in rows}
    missing = [c for c in FIXTURE_CLASSES if c not in present]
    print(
        f'\n{len(present)} of {len(FIXTURE_CLASSES)} fixture classes '
        'survive this sample', flush=True)
    if missing:
        print('lost entirely: ' + ', '.join(missing), flush=True)
    unscoreable = [
        f'{label} ({labeled:,} labeled, none held out)'
        for label, labeled, held in rows if held < 1
    ]
    if unscoreable:
        print(
            'present but not held out, so not scoreable: ' +
            ', '.join(unscoreable), flush=True)
    thin = [
        f'{label} ({labeled:,})' for label, labeled, _ in rows
        if 0 < labeled < args.min_class_rows
    ]
    if thin:
        print(
            f'fewer than {args.min_class_rows} labeled rows, so any '
            'per-class metric for them is noise: ' + ', '.join(thin),
            flush=True)

    # Only these two are fatal. Everything above is a property of the tier.
    if not total_labeled or not total_held:
        raise SystemExit(f'degenerate slice: labeled={total_labeled:,} '
                         f'held_out={total_held:,}; the sample and holdout '
                         'predicates are not independent')
    if len(present) < 3:
        raise SystemExit(
            f'only {len(present)} classes survive this sample; the trainer '
            'rewrites a two-class multiclass task to binary without saying so, '
            'so this tier would not be the fixture at all')


if __name__ == '__main__':
    main()
