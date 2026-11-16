"""Laying values out as tensors: padded, packed and nested sequences, named and sliced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
if TYPE_CHECKING:
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import Flat, TokenRows
from test.kumo_pytorch.reference_engine.gpu.batching.inputs import _InputValues


def _offsets(lengths: Any) -> Any:
    import numpy as np

    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

def _sequence(arrays: dict[str, tuple[Any, Any]], offsets: Any, total: int, count: int, width: int, layout: Any,
              *, exists: bool = True) -> dict[str, Any]:
    """`count` sequences of CuPy `arrays` (each with its padding fill, keeping its type), stored as `layout`
    says (upstream's ``sequence_layout``): packed, flat plus ``valid`` and ``offsets``; padded, [count, width]
    arrays plus ``valid``. `offsets` (CuPy int64, count + 1, from 0) bound the sequences; `total` is their
    last. Without `exists`, no ``valid`` is added (upstream's ``layout_array`` per array)."""

    import cupy
    import torch
    from kumo.pytorch.contract.batch.layouts import Packed

    device = torch.device("cuda", torch.cuda.current_device())
    if isinstance(layout, Packed):
        result = {name: torch.from_dlpack(cupy.ascontiguousarray(values)) for name, (values, _) in arrays.items()}
        if exists:
            result["valid"] = torch.ones(total, dtype=torch.bool, device=device)
        result["offsets"] = torch.from_dlpack(cupy.ascontiguousarray(offsets))
        return result
    slot = cupy.arange(total, dtype=cupy.int64)
    owner = cupy.searchsorted(offsets[1:], slot, side="right")
    position = torch.from_dlpack(slot - offsets[owner])
    owner = torch.from_dlpack(owner)
    result = {}
    for name, (values, fill) in arrays.items():
        source = torch.from_dlpack(cupy.ascontiguousarray(values))
        laid = torch.full((count, width), fill, dtype=source.dtype, device=device)
        laid[owner, position] = source
        result[name] = laid
    if exists:
        valid = torch.zeros((count, width), dtype=torch.bool, device=device)
        valid[owner, position] = True
        result["valid"] = valid
    return result

def _kernel(result: dict[str, Any], prefix: str, total: int, longest: int) -> None:
    """Upstream's ``kernel_metadata`` for packed `prefix` + ``offsets``: ``cu_seqlens`` (int32) and
    ``max_seqlen`` (`longest`, a host int), added last."""

    import numpy as np
    import torch

    if total > np.iinfo(np.int32).max:
        raise ValueError(f"packed boundary {prefix}offsets exceeds int32 attention-kernel range")
    result[f"{prefix}cu_seqlens"] = result[f"{prefix}offsets"].to(torch.int32)
    result[f"{prefix}max_seqlen"] = longest

def _flat_batch(flat: Flat, row_bounds: Any, start: int, stop: int, layout: Any, pad: int,
                fields: Any = None) -> dict[str, Any]:
    """One batch of a flattened or entity token input: one sequence per sample, as upstream's
    ``TokenRows.output`` lays it out (padding: `pad`; ``source_event_indices`` count from the batch's first
    row, -1 for a marker). With `fields` (each token row's field-name ID), ``field_ids`` and ``field_valid``
    follow: a marker's 0 and False."""

    import cupy
    import numpy as np
    from kumo.pytorch.contract.batch.layouts import Packed

    begin, end = int(flat.bounds[start]), int(flat.bounds[stop])
    local = flat.bounds[start:stop + 1] - begin
    first = int(row_bounds[start])
    rows = flat.rows[begin:end]
    arrays = {"value_ids": (flat.values[begin:end], pad),
              "position_ids": (flat.positions[begin:end], 0),
              "source_event_indices": (cupy.where(rows >= 0, rows - first, -1), 0)}
    longest = int(np.diff(local).max(initial=0))
    result = _sequence(arrays, cupy.asarray(local), end - begin, stop - start, longest, layout)
    if fields is not None:
        source = flat.sources[begin:end]
        ids = fields[cupy.maximum(source, 0)] if fields.size else cupy.zeros(source.size, dtype=cupy.int64)
        owned = source >= 0
        result.update(_named_sequence(cupy.where(owned, ids, 0), owned, cupy.asarray(local), end - begin,
                                      stop - start, longest, layout))
    if isinstance(layout, Packed):
        _kernel(result, "", end - begin, longest if stop > start else 0)
    return result

def _named_sequence(ids: Any, owned: Any, offsets: Any, total: int, count: int, width: int,
                    layout: Any) -> dict[str, Any]:
    """Tokens' ``field_ids`` and ``field_valid``, laid out as their sequences are (padding 0 and False)."""

    laid = _sequence({"field_ids": (ids, 0), "field_valid": (owned, False)}, offsets, total, count, width, layout,
                     exists=False)
    laid.pop("offsets", None)
    return laid

@dataclass
class _Nested:
    """A grouped token input's tokens for a call, ready to be cut into batches (``Nested`` layout)."""

    tokens: TokenRows
    owner: Any  # each event's anchor position in the call (CuPy)
    place: Any  # each event's place among its anchor's (CuPy)
    starts: Any  # the tokens before each batch, and in all (host)
    longest: Any  # the most tokens of one event in each batch (host)
    cuts: Any  # where each batch starts, then the end, in samples (host)

    @classmethod
    def of(cls, tokens: TokenRows, row_bounds: Any, owner: Any, cuts: Any) -> _Nested:
        import cupy

        events = int(row_bounds[-1])
        starts = tokens.bounds[cupy.asarray(row_bounds[cuts])].get()
        longest = cupy.zeros(max(len(cuts) - 1, 0), dtype=cupy.int64)
        if events:
            number = cupy.searchsorted(cupy.asarray(cuts), owner, side="right") - 1
            cupy.maximum.at(longest, number, cupy.diff(tokens.bounds))
        place = cupy.arange(events, dtype=cupy.int64) - cupy.asarray(row_bounds)[owner]
        return cls(tokens, owner, place, starts, longest.get(), cuts)

    def batch(self, row_bounds: Any, start: int, stop: int, layout: Any, pad: int) -> dict[str, Any]:
        """One batch, as upstream's ``_token_output`` lays a grouped input out: tokens per event under
        ``layout.fields`` (packed boundaries prefixed ``field_``), each event's sample (``owner``) and place
        there (``history_position``), and events per sample under ``layout.events`` (``sequence``)."""

        import cupy
        import numpy as np
        import torch
        from kumo.pytorch.contract.batch.layouts import Packed

        number = int(np.searchsorted(self.cuts, start, side="left"))
        first, last = int(row_bounds[start]), int(row_bounds[stop])
        begin, end = int(self.starts[number]), int(self.starts[number + 1])
        tokens = self.tokens
        rows = tokens.rows[begin:end]
        arrays = {"value_ids": (tokens.values[begin:end], pad),
                  "position_ids": (tokens.positions[begin:end], 0),
                  "source_event_indices": (rows - first, 0)}
        widest = int(self.longest[number])
        offsets = tokens.bounds[first:last + 1] - begin
        result = _sequence(arrays, offsets, end - begin, last - first, widest, layout.fields)
        if tokens.fields is not None:  # each token's field, laid out as the tokens are
            result.update(_named_sequence(tokens.fields[begin:end], cupy.ones(end - begin, dtype=bool), offsets,
                                          end - begin, last - first, widest, layout.fields))
        if isinstance(layout.fields, Packed):
            result["field_offsets"] = result.pop("offsets")
        result["owner"] = torch.from_dlpack(self.owner[first:last] - start)
        result["history_position"] = torch.from_dlpack(cupy.ascontiguousarray(self.place[first:last]))
        local = row_bounds[start:stop + 1] - first
        events_longest = int(np.diff(local).max(initial=0))
        sequence = _sequence({"event_indices": (cupy.arange(last - first, dtype=cupy.int64), -1)},
                             cupy.asarray(local), last - first, stop - start, events_longest, layout.events)
        if isinstance(layout.events, Packed):
            _kernel(sequence, "", last - first, events_longest if stop > start else 0)
        result["sequence"] = sequence
        if isinstance(layout.fields, Packed):
            _kernel(result, "field_", end - begin, widest if last > first else 0)
        return result

def _laid_out(fields: dict[str, dict[str, Any]], bounds: Any, layout: Any, *, pads=None, field_pad=0) -> dict[str, Any]:
    """One batch's slice of an event input, stored as its layout says (upstream's ``layout_array`` and
    ``kernel_metadata``): packed, flat with ``offsets``, ``cu_seqlens`` and ``max_seqlen``; padded, as
    [samples, longest] arrays (values 0, validity False, Arrow entries NULL past each sequence's end)."""

    import cupy
    import numpy as np
    import torch
    from kumo.pytorch.contract.batch.layouts import Packed

    begin, end = int(bounds[0]), int(bounds[-1])
    local = bounds - begin
    lengths = np.diff(local)
    device = torch.device("cuda", torch.cuda.current_device())
    part = slice(begin, end)
    events = local  # a Text field's own bounds of each sample's events
    if isinstance(layout, Packed):
        result = {field: encoded.rows(begin, end, events) if isinstance(encoded, _Pieces) else
                  {key: _slice(value, part) for key, value in encoded.items()} for field, encoded in fields.items()}
        result["valid"] = torch.ones(end - begin, dtype=torch.bool, device=device)
        if end - begin > np.iinfo(np.int32).max:
            raise ValueError("packed boundary offsets exceeds int32 attention-kernel range")
        offsets = torch.as_tensor(local, dtype=torch.int64, device=device)
        result["offsets"] = offsets
        result["cu_seqlens"] = offsets.to(torch.int32)
        result["max_seqlen"] = int(lengths.max()) if lengths.size else 0
        return result
    count, width = lengths.size, int(lengths.max(initial=0))
    slot = cupy.arange(end - begin, dtype=cupy.int64)
    starts = cupy.asarray(local[:-1])
    owner = cupy.searchsorted(cupy.asarray(local[1:]), slot, side="right")
    position = slot - starts[owner]
    owner, position = torch.from_dlpack(owner), torch.from_dlpack(position)
    result = {}
    for field, encoded in fields.items():
        if isinstance(encoded, _Pieces):  # not padded: pieces, with each event's and each sample's bounds
            result[field] = encoded.rows(begin, end, events)
            continue
        result[field] = {key: _padded(value, part, owner, position, lengths, local, count, width,
                              False if key in {"valid", "field_valid"} else field_pad if key == "field_ids"
                              else (pads or {}).get(field, 0) if key == "values" else 0)
                         for key, value in encoded.items()}
    valid = torch.zeros((count, width), dtype=torch.bool, device=device)
    valid[owner, position] = True
    result["valid"] = valid
    return result

def _padded(value: Any, part: slice, owner: Any, position: Any, lengths: Any, local: Any, count: int, width: int,
            fill: Any = 0) -> Any:
    """One flat array's slice as a [samples, width] array: tensors filled with 0 (False for validity), Arrow
    values as fixed-size lists with NULL past each sequence's end."""

    import numpy as np
    import pyarrow as pa
    import torch

    if isinstance(value, torch.Tensor):
        values = value[part]
        result = torch.full((count, width), fill, dtype=values.dtype, device=values.device)
        result[owner, position] = values
        return result
    values = value.slice(part.start, part.stop - part.start)
    if not width:
        return pa.array([[] for _ in range(count)], type=pa.list_(values.type))
    indices = np.full((count, width), -1, dtype=np.int64)
    host_owner = np.repeat(np.arange(count), lengths)
    indices[host_owner, np.arange(int(local[-1])) - np.repeat(local[:-1], lengths)] = np.arange(len(values))
    flat = values.take(pa.array(indices.reshape(-1), mask=indices.reshape(-1) < 0))
    return pa.FixedSizeListArray.from_arrays(flat, width)

@dataclass
class _Lists:
    """A dated field encoded for a call's anchors (a list each): every occurrence's value (`values`: a tensor or
    an Arrow array; a Text field's pieces), each occurrence's validity, where each anchor's occurrences start
    (`offsets`, host int64) and, for Text, where each occurrence's pieces start (`pieces`, host int64)."""

    values: Any
    valid: Any
    offsets: Any
    pieces: Any = None
    field_ids: Any = None  # each of `values`' field-name ID and whether it has one, with field-name IDs
    field_valid: Any = None

    def rows(self, begin: int, end: int) -> dict[str, Any]:
        """Anchors `begin` to `end`, as upstream gives a list column: ``values``, ``valid``, ``piece_offsets`` (Text)
        and ``occurrence_offsets``, from 0, ``field_ids`` and ``field_valid`` with field-name IDs, then upstream's
        ``kernel_metadata`` for each."""

        import numpy as np
        import torch

        first, last = int(self.offsets[begin]), int(self.offsets[end])
        device = self.valid.device
        low, high = (first, last) if self.pieces is None else (int(self.pieces[first]), int(self.pieces[last]))
        result = {"values": _slice(self.values, slice(low, high)), "valid": self.valid[first:last]}
        bounds = {}
        if self.pieces is not None:
            bounds["piece_"] = self.pieces[first:last + 1] - low
        bounds["occurrence_"] = self.offsets[begin:end + 1] - first
        for prefix, host in bounds.items():
            result[f"{prefix}offsets"] = torch.as_tensor(host, dtype=torch.int64, device=device)
        if self.field_ids is not None:
            result["field_ids"], result["field_valid"] = self.field_ids[low:high], self.field_valid[low:high]
        for prefix, host in bounds.items():
            if int(host[-1]) > np.iinfo(np.int32).max:
                raise ValueError(f"packed boundary {prefix}offsets exceeds int32 attention-kernel range")
            result[f"{prefix}cu_seqlens"] = result[f"{prefix}offsets"].to(torch.int32)
            result[f"{prefix}max_seqlen"] = int(np.diff(host).max()) if host.size > 1 else 0
        return result

def _projected(contract: DataLoaderSpec, name: str, selector: str, tokens: TokenRows, parent: _InputValues) -> Values:
    """An ``OccurrencesOf`` input's field over its parent's tokens (upstream's ``_occurrence_rows``): a dated
    field's token takes its occurrence's value of the field, any other token NULL."""

    import cupy
    from kumo.pytorch.contract.batch.validation import resolved_fields

    parent_name = contract.batch.inputs[name].source.input
    fields = [use.field for use in contract.batch.inputs[parent_name].representation.fields]
    matches = resolved_fields(contract, name, selector)
    dtype = matches[0][1]
    size = int(tokens.values.size)
    values, present, texts = None, cupy.zeros(size, dtype=bool), None
    for (_, dated_name, field), _, _ in matches:
        for rank, parent_field in enumerate(fields):
            found = parent.dated.get(parent_field)
            if found is None or resolved_fields(contract, parent_name, parent_field)[0][0][1] != dated_name:
                continue
            source = found.fields[field]
            if source.values is None:
                raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not project {dtype.name} occurrence fields yet ({selector})")
            if texts is not None and source.texts is not texts:
                raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not project text occurrence fields of several dated values yet "
                                 f"({selector})")
            texts = source.texts
            if values is None:
                values = cupy.zeros(size, dtype=source.values.dtype)
            pick = cupy.flatnonzero(tokens.slots == rank)
            index = cupy.asarray(found.bounds)[tokens.rows[pick]] + tokens.occurrences[pick]
            values[pick] = source.values[index]
            present[pick] = source.valid()[index]
    if values is None:
        values = cupy.zeros(size, dtype=cupy.int64)
    return Values(dtype, values, present, texts)

@dataclass
class _Pieces:
    """A Text field encoded for a call's rows: every row's piece IDs (`values`), each row's validity, and where
    each row's pieces start (`offsets`, host int64, rows + 1)."""

    values: Any
    valid: Any
    offsets: Any
    field_ids: Any = None  # each piece's field-name ID and whether it has one, with field-name IDs
    field_valid: Any = None

    @classmethod
    def of(cls, ids: Any, valid: Any, offsets: Any) -> _Pieces:
        import torch

        return cls(torch.from_dlpack(ids), torch.from_dlpack(valid), offsets.get())

    def rows(self, begin: int, end: int, events: Any = None) -> dict[str, Any]:
        """Rows `begin` to `end`, as upstream gives a Text field: ``values``, ``valid``, ``piece_offsets`` (from 0),
        for an event input each sample's events (``offsets``, `events`: host int64), then upstream's
        ``kernel_metadata`` for each (``piece_cu_seqlens``, ``piece_max_seqlen``, ``cu_seqlens``, ``max_seqlen``)."""

        import numpy as np
        import torch

        low, high = int(self.offsets[begin]), int(self.offsets[end])
        device = self.values.device
        pieces = self.offsets[begin:end + 1] - low
        result = {"values": self.values[low:high], "valid": self.valid[begin:end],
                  "piece_offsets": torch.as_tensor(pieces, device=device)}
        if self.field_ids is not None:
            result["field_ids"], result["field_valid"] = self.field_ids[low:high], self.field_valid[low:high]
        bounds = {"piece_": pieces}
        if events is not None:
            result["offsets"] = torch.as_tensor(events, dtype=torch.int64, device=device)
            bounds[""] = np.asarray(events)
        for prefix, host in bounds.items():
            if int(host[-1]) > np.iinfo(np.int32).max:
                raise ValueError(f"packed boundary {prefix}offsets exceeds int32 attention-kernel range")
            result[f"{prefix}cu_seqlens"] = result[f"{prefix}offsets"].to(torch.int32)
            result[f"{prefix}max_seqlen"] = int(np.diff(host).max()) if host.size > 1 else 0
        return result

def _named(columns: dict[str, Any], field: str, lookup: Any, spec: Any, sampled: _InputValues) -> None:
    """A column field's field-name IDs (`test.kumo_pytorch.reference_engine.gpu.encoding.field_names`), beside its values as upstream gives them:
    ``field_ids``, each row's (an event's by its component), and ``field_valid``, whether the row's component
    declares the field; repeated over a dated field's occurrences and a text's pieces."""

    if lookup is None:
        return
    import cupy
    import numpy as np
    import torch

    encoded = columns[field]
    rank = [use.field for use in spec.representation.fields].index(field)
    if isinstance(encoded, _Lists):  # an entity's dated field: every occurrence (and piece) its one field
        size = int(encoded.offsets[-1]) if encoded.pieces is None else int(encoded.pieces[-1])
        encoded.field_ids = torch.from_dlpack(cupy.repeat(lookup[rank, :1], size))
        encoded.field_valid = torch.ones(size, dtype=torch.bool, device=encoded.valid.device)
        return
    rows = int(len(encoded.offsets) - 1) if isinstance(encoded, _Pieces) else int(len(encoded["valid"]))
    found, declared = sampled.components, sampled.declared.get(field)
    ids = cupy.repeat(lookup[rank, :1], rows) if found is None else lookup[rank][found]
    owned = cupy.ones(rows, dtype=bool) if declared is None else declared
    if isinstance(encoded, _Pieces):  # each row's, over its pieces
        total = int(encoded.offsets[-1])
        piece_row = cupy.searchsorted(cupy.asarray(np.asarray(encoded.offsets[1:])), cupy.arange(total),
                                      side="right")
        encoded.field_ids = torch.from_dlpack(ids[piece_row])
        encoded.field_valid = torch.from_dlpack(owned[piece_row])
        return
    encoded["field_ids"] = torch.from_dlpack(cupy.ascontiguousarray(ids))
    encoded["field_valid"] = torch.from_dlpack(cupy.ascontiguousarray(owned))

def _slice(value: Any, part: slice) -> Any:
    return value[part] if not hasattr(value, "slice") else value.slice(part.start, part.stop - part.start)

def _raw(key: str, values: Values, model: bool) -> dict[str, Any]:
    """A Raw field: its own torch type with NULL filled (0 or False); other types as Arrow arrays."""

    import cupy
    import torch

    valid = values.valid()
    kind = getattr(torch, values.dtype.name, None) if values.dtype.name not in {
        "string", "bytes", "date32", "timestamp", "decimal128", "decimal256"} else None
    if kind is None:
        if model:
            raise ValueError(f"encoder {key}: Raw {values.dtype.name} requires inspectable output or an "
                             "explicit encoding")
        return {"values": values.arrow() if values.values is None else _arrow(values), "valid": torch.from_dlpack(valid)}
    filled = cupy.where(valid, values.values, values.values.dtype.type(0))
    return {"values": torch.from_dlpack(filled), "valid": torch.from_dlpack(valid)}

def _arrow(values: Values) -> Any:
    """Raw values with no torch type, as an Arrow array of their declared type (a host copy)."""

    import pyarrow as pa

    present = values.valid().get()
    if values.texts is not None:
        codes = values.values.get()
        texts = values.texts.to_arrow()
        picked = texts.take(pa.array(codes, mask=~present)) if len(codes) else pa.array([], pa.string())
        return picked.cast(values.dtype.to_arrow())
    numbers = values.values.get()
    stored = pa.array(numbers, mask=~present)
    if values.dtype.name == "timestamp":
        return stored.cast(pa.int64()).cast(values.dtype.to_arrow())
    if values.dtype.name == "date32":
        return stored.cast(pa.int32()).cast(pa.date32())
    return stored.cast(values.dtype.to_arrow())

def _sample_ids(sample_ids: list[Any], splits: list[str | None], metadata: list[Any],
                padding: list[bool], first: int = 0) -> tuple[list[Any], list[dict[str, Any]]]:
    """Each row's sample ID (a padding row's made unique, as upstream does, numbered from its place in the call:
    `first` plus its index here) and request metadata, from each anchor's sample ID, split and metadata
    (`GpuAnchors.request_fields`)."""

    ids = list(sample_ids)
    if any(padding):
        used = set(sample_ids)
        for index, is_padding in enumerate(padding):
            if is_padding:
                suffix = first + index
                sample_id = f"{sample_ids[index]}#padding{suffix}"
                while sample_id in used:
                    suffix += 1
                    sample_id = f"{sample_ids[index]}#padding{suffix}"
                used.add(sample_id)
                ids[index] = sample_id
    return ids, [{"split": split, **extra} for split, extra in zip(splits, metadata, strict=True)]
