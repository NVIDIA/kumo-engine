"""The custom-TT path's split plan, checked against the contract
`ttgen_executor` actually reads.

A `SplitSpec` reaches Rust by two routes: the generated path through
`DiskGraphTTBuilder.split_ratio`, and the custom path through
`build_train_table_artifacts_for_existing_table`, which takes the split as its
own `split_spec_json` argument. Both now take the `SplitSpec` dataclass
itself and marshal through one Rust implementation (`make_split_spec` in
`diskgraph-pybind/src/tt_builder.rs`), which extracts it field by field.

They must agree, because `SplitSpec` in `diskgraph-core/src/tt_builder.rs`
declares its unit fields `#[serde(default)]`: an omitted field deserializes to
`None` rather than failing, so a divergence is silent and shows up only as a
differently-split training table.

That is not hypothetical. When this path hand-built its own JSON,
`offsets=[-30, 0]` with `offset_unit="days"` was serialized raw, read as -30
*seconds*, and collapsed a `DateOffsetSplit`'s validation window to nothing --
a run trained with no validation set rather than erroring. These tests pin the
behavior that regression cost us, and the parity that now prevents a second
copy of it.
"""
from __future__ import annotations

import json

import pytest
from diskgraph._c import (DiskGraphTTBuilder, DiskgraphConfigError,
                          build_split_spec_json)
from diskgraph.split_contract import SplitSpec
from diskgraph.tt_artifacts import _split_spec_to_plan_json

_DAY = 86_400


def _plan(spec: SplitSpec, *, hint: str | None = None) -> dict:
    return json.loads(_split_spec_to_plan_json(spec, split_family_hint=hint))


def _date_offset(**overrides) -> SplitSpec:
    r"""The split a temporal predictive query resolves to by default."""
    fields = dict(kind='DateOffsetSplit', is_temporal=True, offsets=[-30, 0],
                  offset_unit='days', unit='days', timeframe=30,
                  anchor_start_offset=0, flexible_boundaries=False)
    fields.update(overrides)
    return SplitSpec(**fields)


def test_offsets_are_converted_into_the_base_unit():
    r"""`ttgen_executor` reads `offsets` as `Dur<U>` and never rescales them.

    This is the regression: raw `[-30, 0]` means thirty *seconds* on the Rust
    side, which puts the validation window inside the final timeframe and
    leaves it empty.
    """
    plan = _plan(_date_offset())

    assert plan['offsets'] == [-30 * _DAY, 0]


@pytest.mark.parametrize('unit, factor', [
    ('seconds', 1),
    ('minutes', 60),
    ('hours', 3600),
    ('days', 86_400),
    ('weeks', 604_800),
])
def test_every_fixed_unit_scales_as_dur_from_unit_does(unit, factor):
    plan = _plan(_date_offset(offsets=[-3, 0], offset_unit=unit))

    assert plan['offsets'] == [-3 * factor, 0]


def test_unit_timeframe_and_anchor_cross_unconverted():
    r"""These three are rescaled Rust-side, by `dur_from_unit(v, unit)`.

    Omitting them left `timeframe` to be derived from the offsets, which only
    coincides with the caller's when the timeframe equals the offset spacing.
    """
    plan = _plan(_date_offset(timeframe=7, anchor_start_offset=2))

    assert plan['unit'] == 'days'
    assert plan['timeframe'] == 7
    assert plan['anchor_start_offset'] == 2


def test_absent_units_default_the_way_the_pyo3_path_defaults_them():
    plan = _plan(
        _date_offset(offset_unit=None, unit=None, anchor_start_offset=None))

    assert plan['offset_unit'] == 'days'
    assert plan['unit'] == 'days'
    assert plan['anchor_start_offset'] == 0
    assert plan['offsets'] == [-30 * _DAY, 0]


@pytest.mark.parametrize('unit, expected', [
    ('months', [-3, -1]),
    ('years', [-36, -12]),
])
def test_calendar_offsets_route_to_month_offsets(unit, expected):
    r"""Calendar arithmetic is not a fixed duration, so it travels separately.

    `month_offsets` was hardcoded to ``None`` here, so a calendar-offset split
    reached Rust with no offsets of either kind.
    """
    plan = _plan(_date_offset(offsets=[-3, -1], offset_unit=unit))

    assert plan['month_offsets'] == expected
    assert plan['offsets'] is None


def test_sub_second_offsets_truncate_toward_zero():
    r"""`Dur<Seconds>::from_nanos` is an i64 divide, which truncates.

    Python's ``//`` floors, so a negative offset would otherwise land one
    second earlier here than on the pyo3 path.
    """
    plan = _plan(
        _date_offset(offsets=[-1_500_000_000, 1_500_000_000],
                     offset_unit='nanoseconds'))

    assert plan['offsets'] == [-1, 1]


def test_an_unknown_offset_unit_is_refused():
    with pytest.raises(DiskgraphConfigError, match='fortnights'):
        _plan(_date_offset(offset_unit='fortnights'))


def test_every_field_the_rust_spec_declares_is_populated():
    r"""A structural guard, not a value check.

    `SplitSpec` in `diskgraph-core/src/tt_builder.rs` marks its newer fields
    `#[serde(default)]`, so a field added there and forgotten here defaults to
    ``None`` silently -- which is exactly how the unit fields went missing.
    Pinning the key set makes the next such omission a failing test rather
    than a differently-split training table.
    """
    plan = _plan(_date_offset())

    assert set(plan) == {
        'kind',
        'ratio',
        'rnd_seed',
        'is_temporal',
        'offsets',
        'offset_unit',
        'unit',
        'timeframe',
        'anchor_start_offset',
        'ranges',
        'backtesting_timestamp',
        'split_column',
        'split_output_column',
        'manual_train_values',
        'manual_val_values',
        'manual_test_values',
        'month_offsets',
        'flexible_boundaries',
        'timeframe_step',
    }


def test_a_split_that_names_no_units_is_unchanged():
    r"""Families without temporal boundary semantics must not acquire any.

    `make_split_spec` gates every unit field on
    ``kind in {DateOffsetSplit, TimeRangeSplit}``. Asserting only the offset
    fields let a `ManualSplit` pick up ``offset_unit='days'``, ``unit='days'``
    and ``anchor_start_offset=0`` while this test stayed green -- harmless
    today, since no consumer reads them outside the two gated kinds, but the
    kind of silent divergence this file exists to catch.
    """
    plan = _plan(
        SplitSpec(kind='ManualSplit', is_temporal=False,
                  split_column='split_col', manual_train_values=['0'],
                  manual_val_values=['1'], manual_test_values=['2']))

    assert plan['offsets'] is None
    assert plan['month_offsets'] is None
    assert plan['offset_unit'] is None
    assert plan['unit'] is None
    assert plan['timeframe'] is None
    assert plan['anchor_start_offset'] is None
    assert plan['flexible_boundaries'] is None
    assert plan['timeframe_step'] is None



# ── Unit names: what `parse_time_unit` accepts, and what it emits ────────────
#
# The Rust boundary parses unit strings with `parse_time_unit`, which lowercases
# and accepts aliases. Emitting them back through `time_unit_name` is what makes
# the plan safe for `TimeUnit::from_name`, which is strict: canonical lowercase
# long names only, no aliases, no case folding. A path that passed unit strings
# through verbatim would serialize a plan that fails at TT-gen time -- after the
# parquet scan has started -- rather than at config time.


@pytest.mark.parametrize('alias, canonical, factor', [
    ('s', 'seconds', 1),
    ('min', 'minutes', 60),
    ('h', 'hours', 3600),
    ('d', 'days', 86_400),
    ('w', 'weeks', 604_800),
    ('Days', 'days', 86_400),
    ('HOURS', 'hours', 3600),
])
def test_unit_aliases_and_casing_are_accepted_and_canonicalized(
        alias, canonical, factor):
    plan = _plan(_date_offset(offsets=[-3, 0], offset_unit=alias, unit=alias))

    assert plan['offsets'] == [-3 * factor, 0]
    assert plan['offset_unit'] == canonical
    assert plan['unit'] == canonical


@pytest.mark.parametrize('alias, expected', [
    ('mo', [-3, -1]),
    ('month', [-3, -1]),
    ('Months', [-3, -1]),
    ('y', [-36, -12]),
    ('year', [-36, -12]),
])
def test_calendar_offset_aliases_also_route_to_month_offsets(alias, expected):
    plan = _plan(_date_offset(offsets=[-3, -1], offset_unit=alias))

    assert plan['month_offsets'] == expected
    assert plan['offsets'] is None


def test_a_calendar_unit_on_date_offset_split_is_refused_at_config_time():
    r"""`ttgen_executor` rejects it too, but only once the scan is under way."""
    with pytest.raises(DiskgraphConfigError, match='months'):
        _plan(_date_offset(unit='months'))


# ── Absolute timestamps carry their own unit ────────────────────────────────


def test_timestamp_unit_scales_ranges_and_backtesting_timestamp():
    r"""`ranges` and `backtesting_timestamp` deserialize as `Abs<Seconds>`.

    kumo's canonical TIMESTAMP representation is int64 nanoseconds -- the same
    reason `input_timestamp_unit="nanoseconds"` is threaded through this
    module. Left unscaled, a `BacktestingSplit` cutoff of epoch-nanoseconds is
    read as epoch-seconds, putting every row on one side of the split.
    """
    cutoff_ns = 1_704_067_200_000_000_000
    plan = _plan(SplitSpec(kind='BacktestingSplit',
                           backtesting_timestamp=cutoff_ns,
                           ranges=[(0, cutoff_ns)],
                           timestamp_unit='nanoseconds'))

    assert plan['backtesting_timestamp'] == 1_704_067_200
    assert plan['ranges'] == [[0, 1_704_067_200]]


def test_timestamp_unit_defaults_to_seconds():
    plan = _plan(SplitSpec(kind='BacktestingSplit',
                           backtesting_timestamp=1_704_067_200))

    assert plan['backtesting_timestamp'] == 1_704_067_200


# ── The two routes agree ────────────────────────────────────────────────────


@pytest.mark.parametrize('spec', [
    _date_offset(),
    _date_offset(offsets=[-3, -1], offset_unit='months'),
    _date_offset(offset_unit='h', unit='h', timeframe=12),
    SplitSpec(kind='TimeRangeSplit', ranges=[(0, 100), (100, 200)],
              timeframe=30, unit='days'),
    SplitSpec(kind='BacktestingSplit', backtesting_timestamp=1_704_067_200),
    SplitSpec(kind='FixedSplit', ratio=[0.8, 0.1, 0.1]),
    SplitSpec(kind='ManualSplit', split_column='c', manual_train_values=['0'],
              manual_val_values=['1'], manual_test_values=['2']),
])
def test_both_marshalling_routes_produce_the_same_spec(spec):
    r"""The generated path and the custom-TT path must not diverge.

    They share `make_split_spec`, and take the same single argument, so there
    is one call shape rather than two hand-maintained argument lists --
    single-sourcing the rules but not the forwarding is the weaker version of
    the same failure mode.
    """
    builder = DiskGraphTTBuilder()
    builder.for_each('users', 'user_id')
    builder.split_ratio(spec)
    generated = json.loads(builder.build())['split']

    custom = json.loads(build_split_spec_json(spec))

    assert generated == custom



# ── Absolute-timestamp units are a narrower set than duration units ─────────
#
# `ranges` and `backtesting_timestamp` are `Abs<Seconds>`, scaled by
# `abs_from_unit`, which handles five units and `panic!`s on the other five --
# "months" has no fixed anchor to the epoch. `timestamp_unit` only became
# load-bearing when this marshalling was unified, so the unreachable panic
# became reachable; it has to be a config error at the boundary instead.


@pytest.mark.parametrize('unit, scale', [
    ('seconds', 1),
    ('milliseconds', 1_000),
    ('microseconds', 1_000_000),
    ('nanoseconds', 1_000_000_000),
])
def test_supported_timestamp_units_scale_absolute_values(unit, scale):
    plan = _plan(SplitSpec(kind='BacktestingSplit',
                           backtesting_timestamp=1_704_067_200 * scale,
                           ranges=[(0, 1_704_067_200 * scale)],
                           timestamp_unit=unit))

    assert plan['backtesting_timestamp'] == 1_704_067_200
    assert plan['ranges'] == [[0, 1_704_067_200]]


def test_days_is_a_supported_timestamp_unit():
    plan = _plan(SplitSpec(kind='BacktestingSplit', backtesting_timestamp=2,
                           timestamp_unit='days'))

    assert plan['backtesting_timestamp'] == 2 * _DAY


@pytest.mark.parametrize('unit', ['hours', 'minutes', 'weeks', 'months', 'years'])
def test_a_timestamp_unit_that_cannot_anchor_is_a_config_error(unit):
    r"""Not a `PanicException` -- a Rust panic must not cross into Python."""
    with pytest.raises(DiskgraphConfigError, match='timestamp_unit'):
        _plan(SplitSpec(kind='TimeRangeSplit', ranges=[(0, 100)],
                        timestamp_unit=unit))


# ── Shapes the contract accepts ────────────────────────────────────────────


@pytest.mark.parametrize('ranges', [
    [(0, 100), (100, 200)],
    [[0, 100], [100, 200]],
    [(0, 100), [100, 200]],
])
def test_range_pairs_may_be_lists_or_tuples(ranges):
    r"""`validate_for_compiler` accepts `(list, tuple)` pairs, so both reach here.

    A tuple-only extraction would reject the list form that a range parsed
    straight out of a JSON config arrives in.
    """
    plan = _plan(SplitSpec(kind='TimeRangeSplit', ranges=ranges))

    assert plan['ranges'] == [[0, 100], [100, 200]]


@pytest.mark.parametrize('field', ['unit', 'offset_unit'])
def test_an_empty_unit_string_falls_back_to_the_default(field):
    r"""`normalize_split_spec` propagates `unit=""` from a config verbatim.

    The Python callers spelled the default `spec.unit or "days"`, and `or`
    catches the empty string; a `None`-only default would turn it into
    ``unknown time unit: ``.
    """
    plan = _plan(_date_offset(**{field: ''}))

    assert plan[field] == 'days'


def test_a_non_splitspec_argument_names_what_it_should_have_been():
    r"""Bare `getattr` failure names neither the argument nor the expected type."""
    with pytest.raises(DiskgraphConfigError, match='must be a SplitSpec'):
        build_split_spec_json({'kind': 'FixedSplit', 'ratio': [0.8, 0.2]})
