from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SplitSpec:
    kind: Optional[str] = None
    ratio: Optional[list[float]] = None
    seed: Optional[int] = None
    is_temporal: Optional[bool] = None
    offsets: Optional[list[int]] = None
    ranges: Optional[list[tuple[Any, Any]]] = None
    backtesting_timestamp: Optional[Any] = None
    split_column: Optional[str] = None
    manual_train_values: Optional[list[str]] = None
    manual_val_values: Optional[list[str]] = None
    manual_test_values: Optional[list[str]] = None
    timeframe: Optional[int] = None
    unit: Optional[str] = None
    offset_unit: Optional[str] = None
    anchor_start_offset: Optional[int] = None

    def to_assignment_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.kind:
            cfg['_target_'] = f'kumo.task.split.{self.kind}'
        if self.ratio is not None:
            cfg['ratio'] = [float(x) for x in self.ratio]
        if self.seed is not None:
            cfg['seed'] = int(self.seed)
        if self.split_column is not None:
            cfg['col'] = {'name': self.split_column}
            cfg['split_column'] = self.split_column
        if self.manual_train_values is not None:
            cfg['manual_train_values'] = list(self.manual_train_values)
            cfg['train_value'] = list(self.manual_train_values)
        if self.manual_val_values is not None:
            cfg['manual_val_values'] = list(self.manual_val_values)
            cfg['val_value'] = list(self.manual_val_values)
        if self.manual_test_values is not None:
            cfg['manual_test_values'] = list(self.manual_test_values)
            cfg['test_value'] = list(self.manual_test_values)
        if self.offsets is not None:
            cfg['offsets'] = list(self.offsets)
        if self.ranges is not None:
            cfg['ranges'] = list(self.ranges)
        if self.backtesting_timestamp is not None:
            cfg['backtesting_timestamp'] = self.backtesting_timestamp
            cfg['timestamp'] = self.backtesting_timestamp
        if self.timeframe is not None:
            cfg['timeframe'] = int(self.timeframe)
        if self.unit is not None:
            cfg['unit'] = str(self.unit)
        if self.offset_unit is not None:
            cfg['offset_unit'] = str(self.offset_unit)
        if self.anchor_start_offset is not None:
            cfg['anchor_start_offset'] = int(self.anchor_start_offset)
        if self.is_temporal is not None:
            cfg['is_temporal'] = bool(self.is_temporal)
        return cfg


def _coerce_split_payload(raw: Optional[Any]) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, SplitSpec):
        return raw.to_assignment_config()
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, 'config') and callable(raw.config):
        payload = raw.config()
        if isinstance(payload, dict):
            return payload
    if hasattr(raw, 'model_dump') and callable(raw.model_dump):
        payload = raw.model_dump()
        if isinstance(payload, dict):
            return payload
    if hasattr(raw, '__dict__'):
        return {k: v for k, v in raw.__dict__.items() if not k.startswith('_')}
    return {}


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

    ratio = payload.get('ratio')
    if ratio is not None:
        ratio = [float(x) for x in _as_list(ratio)]

    seed_raw = payload.get('seed')
    seed = int(seed_raw) if isinstance(seed_raw, int) else None

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

    is_temporal_raw = payload.get('is_temporal')
    is_temporal = bool(is_temporal_raw) if isinstance(is_temporal_raw, bool) else None

    return SplitSpec(
        kind=str(kind) if isinstance(kind, str) and kind else None,
        ratio=ratio,
        seed=seed,
        is_temporal=is_temporal,
        offsets=offsets,
        ranges=ranges,
        backtesting_timestamp=payload.get('backtesting_timestamp', payload.get('timestamp')),
        split_column=str(split_column) if isinstance(split_column, str) and split_column else None,
        manual_train_values=_string_list('manual_train_values', 'train_value'),
        manual_val_values=_string_list('manual_val_values', 'val_value'),
        manual_test_values=_string_list('manual_test_values', 'test_value'),
        timeframe=timeframe,
        unit=str(payload['unit']) if isinstance(payload.get('unit'), str) else None,
        offset_unit=str(payload['offset_unit']) if isinstance(payload.get('offset_unit'), str) else None,
        anchor_start_offset=anchor_start_offset,
    )
