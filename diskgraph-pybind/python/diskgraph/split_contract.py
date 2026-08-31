from dataclasses import dataclass
from typing import Any, Optional

try:
    from ._c import DiskgraphConfigError
except Exception:  # pragma: no cover - stale/missing wheel should fail fast at callsites
    DiskgraphConfigError = RuntimeError


_RATIO_REQUIRED_FAMILIES = {
    'FixedSplit',
    'RandomSplit',
    'RandomLinkSplit',
    'TemporalSplit',
    'ColumnSplit',
    'ApproxDateOffsetSplit',
}
_OFFSET_REQUIRED_FAMILIES = {'DateOffsetSplit'}
_RANGES_REQUIRED_FAMILIES = {'TimeRangeSplit'}
_BACKTESTING_REQUIRED_FAMILIES = {'BacktestingSplit'}
_COLUMN_REQUIRED_FAMILIES = {'ColumnSplit', 'ManualSplit', 'CustomSplit', 'CustomLinkSplit'}
_MANUAL_VALUES_REQUIRED_FAMILIES = {'ManualSplit', 'CustomSplit', 'CustomLinkSplit'}
_SUPPORTED_FAMILIES = {
    *_RATIO_REQUIRED_FAMILIES,
    *_OFFSET_REQUIRED_FAMILIES,
    *_RANGES_REQUIRED_FAMILIES,
    *_BACKTESTING_REQUIRED_FAMILIES,
    *_COLUMN_REQUIRED_FAMILIES,
}


@dataclass(frozen=True)
class SplitSpec:
    kind: Optional[str] = None
    ratio: Optional[list[float]] = None
    rnd_seed: Optional[int] = None
    is_temporal: Optional[bool] = None
    offsets: Optional[list[int]] = None
    ranges: Optional[list[tuple[Any, Any]]] = None
    backtesting_timestamp: Optional[Any] = None
    split_column: Optional[str] = None
    split_output_column: Optional[str] = None
    manual_train_values: Optional[list[str]] = None
    manual_val_values: Optional[list[str]] = None
    manual_test_values: Optional[list[str]] = None
    timeframe: Optional[int] = None
    unit: Optional[str] = None
    offset_unit: Optional[str] = None
    anchor_start_offset: Optional[int] = None
    # kumo DateOffsetSplit boundary-snapping (prod default True). None ⇒ the
    # Rust side applies the production default.
    flexible_boundaries: Optional[bool] = None
    timeframe_step: Optional[int] = None
    # Unit of range/backtesting_timestamp values: "seconds", "milliseconds",
    # "microseconds", "nanoseconds", or "days".  The Rust pybind scales them
    # into the plan's base unit; anything else is a config error, because these
    # are absolute timestamps and `abs_from_unit` cannot anchor e.g. "months"
    # to the epoch.
    timestamp_unit: str = "seconds"

    def to_assignment_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.kind:
            cfg['_target_'] = f'kumo.task.split.{self.kind}'
        if self.ratio is not None:
            cfg['ratio'] = [float(x) for x in self.ratio]
        if self.rnd_seed is not None:
            cfg['rnd_seed'] = int(self.rnd_seed)
        if self.split_column is not None:
            cfg['col'] = {'name': self.split_column}
            cfg['split_column'] = self.split_column
        if self.split_output_column is not None:
            cfg['split_output_column'] = self.split_output_column
            cfg['split_projection'] = self.split_output_column
        if self.manual_train_values is not None:
            cfg['manual_train_values'] = list(self.manual_train_values)
        if self.manual_val_values is not None:
            cfg['manual_val_values'] = list(self.manual_val_values)
        if self.manual_test_values is not None:
            cfg['manual_test_values'] = list(self.manual_test_values)
        if self.offsets is not None:
            cfg['offsets'] = list(self.offsets)
        if self.ranges is not None:
            cfg['ranges'] = list(self.ranges)
        if self.backtesting_timestamp is not None:
            cfg['backtesting_timestamp'] = self.backtesting_timestamp
        if self.timeframe is not None:
            cfg['timeframe'] = int(self.timeframe)
        if self.unit is not None:
            cfg['unit'] = str(self.unit)
        if self.offset_unit is not None:
            cfg['offset_unit'] = str(self.offset_unit)
        if self.anchor_start_offset is not None:
            cfg['anchor_start_offset'] = int(self.anchor_start_offset)
        if self.flexible_boundaries is not None:
            cfg['flexible_boundaries'] = bool(self.flexible_boundaries)
        if self.timeframe_step is not None:
            cfg['timeframe_step'] = int(self.timeframe_step)
        if self.is_temporal is not None:
            cfg['is_temporal'] = bool(self.is_temporal)
        cfg['timestamp_unit'] = str(self.timestamp_unit)
        return cfg

    def validate_for_compiler(self) -> None:
        family = self.kind
        if not family:
            raise DiskgraphConfigError('diskgraph split contract requires split kind')
        if family not in _SUPPORTED_FAMILIES:
            raise DiskgraphConfigError(
                f'unsupported split family for diskgraph compiler split planning: {family}'
            )

        if family in _RATIO_REQUIRED_FAMILIES:
            _validate_ratio(family, self.ratio)

        if family in _OFFSET_REQUIRED_FAMILIES:
            if not self.offsets:
                raise DiskgraphConfigError(f'{family} requires offsets')
            # Kumo parity: DateOffsetSplit supports 1 or 2 negative offsets.
            # 1 offset -> 2-way split, 2 offsets -> 3-way split.
            if len(self.offsets) < 1 or len(self.offsets) > 2:
                raise DiskgraphConfigError(f'{family} offsets must contain 1 or 2 entries')
            if any(offset > 0 for offset in self.offsets):
                raise DiskgraphConfigError(f'{family} offsets must be <= 0')
            if any(left > right for left, right in zip(self.offsets[:-1], self.offsets[1:])):
                raise DiskgraphConfigError(f'{family} offsets must be monotonically increasing')

        if family in _RANGES_REQUIRED_FAMILIES:
            if not self.ranges or len(self.ranges) < 2:
                raise DiskgraphConfigError(f'{family} requires at least two ranges')
            if len(self.ranges) > 3:
                raise DiskgraphConfigError(f'{family} supports at most three ranges')
            for pair in self.ranges:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise DiskgraphConfigError(f'{family} range entries must be [start, end]')

        if family in _BACKTESTING_REQUIRED_FAMILIES and self.backtesting_timestamp is None:
            raise DiskgraphConfigError(f'{family} requires backtesting_timestamp')

        if family in _COLUMN_REQUIRED_FAMILIES and not self.split_column:
            raise DiskgraphConfigError(f'{family} requires split_column')

        if family in _MANUAL_VALUES_REQUIRED_FAMILIES:
            if not self.manual_train_values:
                raise DiskgraphConfigError(f'{family} requires manual_train_values')
            if not self.manual_val_values:
                raise DiskgraphConfigError(f'{family} requires manual_val_values')


def _validate_ratio(family: str, ratio: Optional[list[float]]) -> None:
    if ratio is None:
        raise DiskgraphConfigError(f'{family} requires ratio')
    if len(ratio) < 2 or len(ratio) > 3:
        raise DiskgraphConfigError(f'{family} ratio must contain 2 or 3 entries')
    if any(val < 0 for val in ratio):
        raise DiskgraphConfigError(f'{family} ratio cannot contain negative values')


def _coerce_split_payload(raw: Optional[Any]) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, SplitSpec):
        return raw.to_assignment_config()
    if isinstance(raw, dict):
        return dict(raw)
    raise TypeError(
        f'diskgraph split contract requires SplitSpec or dict, got {type(raw).__name__}'
    )


def _as_list(v: Any) -> list[Any]:
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _split_family_from_target(target: Any) -> Optional[str]:
    if isinstance(target, str) and target:
        return target.rsplit('.', 1)[-1]
    return None


def normalize_split_spec(
    raw: Optional[Any],
    *,
    split_family_hint: Optional[str] = None,
    require: bool = False,
) -> Optional[SplitSpec]:
    payload = _coerce_split_payload(raw)
    if not payload:
        if require:
            raise RuntimeError('missing split config for diskgraph split contract')
        return None

    inferred_kind = (
        payload.get('kind')
        or payload.get('split_kind')
        or _split_family_from_target(payload.get('_target_'))
    )
    kind = split_family_hint or inferred_kind
    if split_family_hint and inferred_kind and split_family_hint != inferred_kind:
        raise RuntimeError(
            'diskgraph split contract mismatch: '
            f'split_family_hint={split_family_hint} conflicts with payload kind={inferred_kind}'
        )

    if require and not kind:
        raise RuntimeError('diskgraph split contract requires split kind')

    col_spec = payload.get('col')
    split_column = payload.get('split_column')
    if split_column is None:
        if isinstance(col_spec, dict):
            split_column = col_spec.get('name')
        elif isinstance(col_spec, str):
            split_column = col_spec
    split_output_column = payload.get('split_output_column') or payload.get('split_projection')

    ratio = payload.get('ratio')
    if ratio is not None:
        ratio = [float(x) for x in _as_list(ratio)]

    seed_raw = payload.get('rnd_seed')
    rnd_seed = int(seed_raw) if isinstance(seed_raw, int) else None

    def _string_list(*keys: str) -> Optional[list[str]]:
        for k in keys:
            if k in payload and payload[k] is not None:
                return [str(v) for v in _as_list(payload[k])]
        return None

    offsets = payload.get('offsets')
    if offsets is not None:
        offsets = [int(x) for x in _as_list(offsets)]

    ranges_raw = payload.get('ranges')
    ranges: Optional[list[tuple[Any, Any]]] = None
    if isinstance(ranges_raw, list):
        ranges = []
        for pair in ranges_raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                ranges.append((pair[0], pair[1]))

    timeframe_raw = payload.get('timeframe')
    timeframe = int(timeframe_raw) if isinstance(timeframe_raw, int) else None

    anchor_raw = payload.get('anchor_start_offset')
    anchor_start_offset = int(anchor_raw) if isinstance(anchor_raw, int) else None

    flexible_raw = payload.get('flexible_boundaries')
    flexible_boundaries = bool(flexible_raw) if isinstance(flexible_raw, bool) else None

    step_raw = payload.get('timeframe_step')
    timeframe_step = int(step_raw) if isinstance(step_raw, int) else None

    is_temporal_raw = payload.get('is_temporal')
    is_temporal = bool(is_temporal_raw) if isinstance(is_temporal_raw, bool) else None

    return SplitSpec(
        kind=str(kind) if isinstance(kind, str) and kind else None,
        ratio=ratio,
        rnd_seed=rnd_seed,
        is_temporal=is_temporal,
        offsets=offsets,
        ranges=ranges,
        backtesting_timestamp=payload.get('backtesting_timestamp'),
        split_column=str(split_column) if isinstance(split_column, str) and split_column else None,
        split_output_column=str(split_output_column) if isinstance(split_output_column, str) and split_output_column else None,
        manual_train_values=_string_list('manual_train_values'),
        manual_val_values=_string_list('manual_val_values'),
        manual_test_values=_string_list('manual_test_values'),
        timeframe=timeframe,
        unit=str(payload['unit']) if isinstance(payload.get('unit'), str) else None,
        offset_unit=str(payload['offset_unit']) if isinstance(payload.get('offset_unit'), str) else None,
        anchor_start_offset=anchor_start_offset,
        flexible_boundaries=flexible_boundaries,
        timeframe_step=timeframe_step,
    )
