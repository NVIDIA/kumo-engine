"""Composite rows: one outer ID for each row's ordered tuple of component results, on the GPU.

Upstream Kumo PyTorch's rules (``assembly.py`` ``_eligible``, ``_component_key``,
``_fit_inputs``, ``_token_rows``; G-COMPOSITE-001):

- A row (an event, or the entity's row) where a ``required_fields`` component
  is NULL is dropped, before an ``EventsOf`` follower reads the events. The
  components' encoders are fitted on every row, dropped ones included.
- Each component's result is typed: a NULL is the typed missing component (not
  the encoder's missing ID); a Raw value is its typed value; another encoder's
  value is its result (an ID, or ZScore's number).
- Fitted, the kept rows' tuples are counted, and the ``top_k`` most frequent
  take IDs from 2, ties in canonical typed order. A supplied ``vocabulary`` is
  used as given. Every other tuple is unknown (1); ID 0 (missing) is never
  given.
- With ``id_space``, the outer IDs join that space as one member, with missing
  0 and unknown 1.

The same tuples, of Raw typed values only, give a ``MatchingGroupTarget``'s
group IDs (`fit_groups`): every distinct tuple of the fitting anchors, in
canonical typed order, numbered from 0, a NULL spelled ``("null", "")``; any
other tuple is -1.

Canonical typed order compares tuples component by component. A typed missing
component comes before or after the component's values as its spelling sorts
against theirs ("missing" against the encoder's name, or a Raw field's type
name). Values order by value (integers, booleans, times, dates, IDs, ZScore
numbers), text by its bytes, and Raw floats by the spelling of their hex form
(``float.hex``), which `hex_order` works out from the bits.

Everything here works on all of a call's rows at once; fitting reads back only
the number of distinct tuples kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from test.kumo_pytorch.reference_engine.gpu.data.text import is_text, python_texts, text_column

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import GpuEncoder, Values

__all__ = ["Component", "CompositeVocabulary", "ComponentSpec", "component", "component_specs", "composite_identity",
           "fit_composite", "fit_groups", "hex_order"]


def composite_identity(input_name: str) -> str:
    """A Composite input's member identity in its ID space, spelled as upstream spells it."""

    return json.dumps(("composite", input_name), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ComponentSpec:
    """One component of a Composite input: its field, encoder and typed spelling.

    :param raw: whether the component is the field's typed value (Raw), rather than an encoder's result.
    :param tag: the first part of the component's typed key: the type name for Raw, else the encoder's name.
    :param null_key: a NULL's typed key: None for a Composite's typed missing component,
        ``("missing", str(dtype))``; a matching group's is ``("null", "")``.
    """

    field: str
    encoder: GpuEncoder
    raw: bool
    tag: str
    null_key: tuple[str, str] | None = None

    @property
    def dtype(self) -> DType:
        return self.encoder.dtype

    @property
    def missing(self) -> tuple[str, str]:
        """A NULL's typed key."""

        return ("missing", str(self.dtype)) if self.null_key is None else self.null_key

    @property
    def missing_first(self) -> bool:
        """Whether a NULL's key sorts before the values' (its spelling sorts first)."""

        return self.missing[0] < self.tag

    def host_key(self, missing: bool, identity: int, text: str | None) -> tuple[str, Any]:
        """A component result in upstream's typed spelling (``_component_key``)."""

        import struct

        from kumo.pytorch.contract.batch.encodings import ZScore

        if missing:
            return self.missing
        if not self.raw:
            if isinstance(self.encoder.config, ZScore):
                return (self.tag, struct.unpack("<d", struct.pack("<q", identity))[0])
            return (self.tag, identity)
        name = self.dtype.name
        if name.startswith("float"):
            return (name, struct.unpack("<d", struct.pack("<q", identity))[0].hex())
        if name == "string":
            return (name, text)
        if name == "bytes":  # typed_key spells bytes by their hex form
            return (name, text.hex())
        if name == "bool":
            return (name, bool(identity))
        if name == "uint64":
            return (name, identity % 2**64)
        return (name, identity)  # integers; timestamps in seconds and dates in days, as typed_key spells them

    def device_identity(self, part: Any) -> tuple[bool, bool, int, str | None]:
        """A supplied vocabulary's component as (whether a row can give it, missing, identity, text): a part
        no row's typed key can equal (another tag, a non-canonical float spelling) is never matched."""

        import math
        import struct

        from kumo.pytorch.contract.batch.encodings import ZScore

        never = (False, False, 0, None)
        if not isinstance(part, tuple) or len(part) != 2 or not isinstance(part[0], str):
            return never
        tag, value = part
        if self.null_key is not None:  # a matching group's NULL, spelled ("null", "")
            if part == self.null_key:
                return (True, True, 0, None)
        elif tag == "missing":
            return (value == str(self.dtype), True, 0, None) if isinstance(value, str) else never
        if tag != self.tag:
            return never
        name = self.dtype.name
        if not self.raw and isinstance(self.encoder.config, ZScore) or self.raw and name.startswith("float"):
            if self.raw:  # typed_key spells a float by its hex form, -0.0 as 0.0
                if not isinstance(value, str):
                    return never
                try:
                    number = float.fromhex(value)
                except (ValueError, OverflowError):
                    return never
                if not math.isfinite(number) or number.hex() != value or value.startswith("-0x0.0p"):
                    return never
            else:
                try:
                    number = float(value) if _number(value) else math.nan
                except OverflowError:  # an integer no float equals
                    return never
                if not math.isfinite(number) or number != value:
                    return never
            number = 0.0 if number == 0 else number
            return (True, False, struct.unpack("<q", struct.pack("<d", number))[0], None)
        if self.raw and name == "string":
            return (True, False, 0, value) if isinstance(value, str) else never
        if self.raw and name == "bytes":  # a canonical hex spelling, as typed_key gives it
            try:
                data = bytes.fromhex(value) if isinstance(value, str) else None
            except ValueError:
                data = None
            return (True, False, 0, data) if data is not None and data.hex() == value else never
        if not _number(value) or isinstance(value, float) and not value.is_integer():
            return never
        number = int(value)
        if name == "uint64" and 2**63 <= number < 2**64:
            number -= 2**64
        if not -(2**63) <= number < 2**63:
            return never
        return (True, False, number, None)


def _number(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def component_specs(representation: Any, input_name: str, encoders: dict[str, GpuEncoder]) -> list[ComponentSpec]:
    """A Composite input's components, in declared order."""

    from kumo.pytorch.contract.batch.encodings import Raw

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import encoder_identity

    specs = []
    for use in representation.fields:
        encoder = encoders[encoder_identity(input_name, use)]
        raw = isinstance(encoder.config, Raw)
        specs.append(ComponentSpec(use.field, encoder, raw, encoder.dtype.name if raw else type(encoder.config).__name__))
    return specs


@dataclass
class Component:
    """One component's typed results for a call's rows (CuPy).

    :param missing: whether each row's component is the typed missing one.
    :param identity: int64, equal exactly where the typed results are equal (0 where missing): an ID or
        integer, a float's bits (-0.0 as 0.0), a text's code into `texts`.
    :param texts: a Raw text component's sorted dictionary (a ``pylibcudf.Column``).
    """

    spec: ComponentSpec
    missing: Any
    identity: Any
    texts: Any = None

    def order(self) -> list[Any]:
        """int64 keys, most significant first, sorting the results in canonical typed order."""

        import cupy

        placement = cupy.where(self.missing, 0 if self.spec.missing_first else 2, 1).astype(cupy.int64)
        name = self.spec.dtype.name
        if self.spec.raw and name.startswith("float"):
            return [placement, *hex_order(self.identity)]
        if self.spec.raw and name == "uint64":
            return [placement, self.identity ^ cupy.int64(-(2**63))]
        from kumo.pytorch.contract.batch.encodings import ZScore

        if isinstance(self.spec.encoder.config, ZScore) and not self.spec.raw:
            return [placement, _float_order(self.identity)]
        return [placement, self.identity]


def component(spec: ComponentSpec, values: Values, encoded: tuple[Any, Any] | None,
              flags: list[tuple[Any, Any]]) -> Component:
    """A component's typed results for `values` (`encoded`: its encoder's (result, valid), None for Raw)."""

    import cupy

    valid = values.valid()
    missing = ~valid
    texts = None
    name = spec.dtype.name
    if not spec.raw:
        result = encoded[0]
        if result.dtype.kind == "f":
            identity = _float_bits(result)
        else:
            identity = result.astype(cupy.int64)
    elif name.startswith("float"):
        flags.append((f"{name} vocabulary values must be finite", (~cupy.isfinite(values.values) & valid).any()))
        identity = _float_bits(values.values)
    elif name == "uint64":
        identity = values.values.view(cupy.int64)
    elif name in {"string", "bytes", "bool", "date32", "timestamp"} or spec.dtype.integer:
        identity = values.values.astype(cupy.int64)
        texts = values.texts
    else:
        raise ValueError(f"encoder {spec.encoder.identity}: test.kumo_pytorch.reference_engine.gpu does not build Composite components of "
                         f"{name} values yet")
    return Component(spec, missing, cupy.where(missing, 0, identity), texts)


def _float_bits(values: Any) -> Any:
    """Floats as float64 bits (int64), -0.0 as 0.0: equal exactly where the values are."""

    import cupy

    numbers = values.astype(cupy.float64)
    return cupy.where(numbers == 0, 0.0, numbers).view(cupy.int64)


def _float_order(bits: Any) -> Any:
    """float64 bits (int64) as int64 keys sorting as the numbers do."""

    import cupy

    return cupy.where(bits < 0, bits ^ cupy.int64(2**63 - 1), bits)


def hex_order(bits: Any) -> tuple[Any, Any]:
    """Finite float64 bits (int64, no -0.0) as two int64 keys (most significant first) sorting as the
    numbers' ``float.hex`` spellings sort as text.

    The spelling is ``[-]0x1.<13 hex digits>p<+|-><decimal exponent>`` (zero: ``0x0.0p+0``; subnormal:
    ``0x0.<13 digits>p-1022``). So: negatives first ('-' sorts before '0'); then zero and subnormals before
    normal numbers; then the mantissa's digits, zero's single "0" followed by 'p', which sorts after every
    hex digit; then a non-negative exponent before a negative one ('+' before '-'); then the exponent's
    decimal digits as text ("10" before "2").
    """

    import cupy

    unsigned = bits.view(cupy.uint64)
    negative = (unsigned >> cupy.uint64(63)).astype(cupy.int64)
    biased = ((unsigned >> cupy.uint64(52)) & cupy.uint64(0x7FF)).astype(cupy.int64)
    mantissa = (unsigned & cupy.uint64((1 << 52) - 1)).astype(cupy.int64)
    zero = (biased == 0) & (mantissa == 0)
    normal = biased != 0
    # zero's "0p" sits after the subnormal mantissas starting "0" (below 16**12) and before the others
    digits = cupy.where(zero, 2 * 16**12 - 1, 2 * mantissa)
    high = ((1 - negative) << 56) | (normal.astype(cupy.int64) << 55) | digits
    exponent = cupy.where(normal, biased - 1023, cupy.where(zero, 0, -1022))
    size = cupy.abs(exponent)
    length = 1 + (size >= 10).astype(cupy.int64) + (size >= 100) + (size >= 1000)
    text = cupy.zeros_like(size)
    for place in range(4):  # each decimal digit (plus 1; 0 past the end), compared as text
        shift = length - 1 - place
        digit = (size // cupy.where(shift >= 0, 10 ** cupy.maximum(shift, 0), 1)) % 10
        text = text * 11 + cupy.where(shift >= 0, digit + 1, 0)
    low = ((exponent < 0).astype(cupy.int64) << 20) | text
    return high, low


@dataclass
class CompositeVocabulary:
    """A Composite input's outer IDs: each tuple it knows, by component, and the tuple's ID.

    :param specs: the components.
    :param ids: each known tuple's ID (CuPy int64).
    :param missing: per component, whether each known tuple's component is the typed missing one (CuPy bool).
    :param identity: per component, each known tuple's identity (CuPy int64; see `Component`).
    :param texts: per component, each known tuple's text (a ``pylibcudf.Column``) for Raw text, else None.
    :param live: whether a row can give each known tuple (a supplied one may be unreachable), or None for all.
    :param all_ids: every outer ID, unreachable supplied ones included (host ints, for the ID space).
    :param supplied: a supplied vocabulary (or reused fitted IDs), normalized as upstream normalizes it, or None.
    :param unknown: the ID of a tuple it does not know (1; a matching group's -1).
    """

    specs: list[ComponentSpec]
    ids: Any
    missing: list[Any]
    identity: list[Any]
    texts: list[Any]
    live: Any
    all_ids: tuple[int, ...]
    supplied: tuple[tuple[tuple[Any, ...], int], ...] | None = None
    unknown: int = 1
    _host: Any = field(default=None, repr=False)

    def entries(self) -> tuple[tuple[tuple[Any, ...], int], ...]:
        """Every known tuple in upstream's typed spelling, with its ID, by ID (as a fitted contract's
        ``Composite.vocabulary``). Inspection only: it reads the vocabulary back to the host."""

        if self.supplied is not None:
            return self.supplied
        if self._host is None:
            ids = self.ids.get().tolist()
            columns = []
            for spec, missing, identity, texts in zip(self.specs, self.missing, self.identity, self.texts,
                                                      strict=True):
                words = python_texts(texts, spec.dtype) if texts is not None else [None] * len(ids)
                columns.append([spec.host_key(bool(m), int(i), w) for m, i, w in
                                zip(missing.get().tolist(), identity.get().tolist(), words, strict=True)])
            self._host = tuple(sorted(((tuple(column[row] for column in columns), ids[row])
                                       for row in range(len(ids))), key=lambda entry: entry[1]))
        return self._host

    @classmethod
    def from_supplied(cls, specs: list[ComponentSpec], vocabulary: Any) -> CompositeVocabulary:
        """A supplied ``Composite.vocabulary``, used exactly as given (upstream's dictionary of it), or the IDs
        a prior task fitted, reused unchanged."""

        import cupy

        normalized = {tuple(tuple(part) if isinstance(part, list) else part for part in key): index
                      for key, index in vocabulary}
        live, missing, identity, words = [], [[] for _ in specs], [[] for _ in specs], [[] for _ in specs]
        for key in normalized:
            reachable = len(key) == len(specs)
            for position, spec in enumerate(specs):
                ok, gap, number, text = spec.device_identity(key[position]) if reachable else (False, False, 0, None)
                reachable &= ok
                missing[position].append(gap)
                identity[position].append(number)
                words[position].append(text)
            live.append(reachable)
        texts = [text_column(words[position]) if spec.raw and is_text(spec.dtype) else None
                 for position, spec in enumerate(specs)]
        return cls(specs, cupy.asarray(list(normalized.values()), dtype=cupy.int64),
                   [cupy.asarray(column, dtype=bool) for column in missing],
                   [cupy.asarray(column, dtype=cupy.int64) for column in identity], texts,
                   cupy.asarray(live, dtype=bool), tuple(normalized.values()),
                   supplied=tuple(sorted(normalized.items(), key=lambda entry: entry[1])))

    def lookup(self, components: list[Component]) -> Any:
        """Each row's ID (CuPy int64): its tuple's, or `unknown`."""

        import cupy

        rows = int(components[0].missing.size)
        known = int(self.ids.size)
        if rows == 0 or known == 0:
            return cupy.full(rows, self.unknown, dtype=cupy.int64)
        dead = cupy.zeros(known, dtype=bool) if self.live is None else ~self.live
        keys = []
        for position, part in enumerate(components):
            missing, identity = self.missing[position], self.identity[position]
            if self.texts[position] is not None:  # a known text as a code of this call's dictionary
                code, found = _codes(self.texts[position], part.texts)
                identity = cupy.where(missing, 0, code)
                dead = dead | (~missing & ~found)
            keys.append(cupy.concatenate([missing, part.missing]).astype(cupy.int64))
            keys.append(cupy.concatenate([identity, part.identity]))
        keys.insert(0, cupy.concatenate([dead, cupy.zeros(rows, dtype=bool)]).astype(cupy.int64))
        is_row = cupy.concatenate([cupy.zeros(known, dtype=cupy.int64), cupy.ones(rows, dtype=cupy.int64)])
        order = cupy.lexsort(cupy.stack([is_row, *keys[::-1]]))  # a group's known tuple first
        change = cupy.zeros(order.size, dtype=bool)
        change[0] = True
        for key in keys:
            ordered = key[order]
            change[1:] |= ordered[1:] != ordered[:-1]
        starts = cupy.flatnonzero(change)
        first = order[starts[cupy.cumsum(change) - 1]]
        found = (first < known) & (keys[0][first] == 0)
        value = cupy.where(found, self.ids[cupy.minimum(first, known - 1)], self.unknown)
        result = cupy.empty(rows, dtype=cupy.int64)
        at_row = order >= known
        result[order[at_row] - known] = value[at_row]
        return result


def _codes(words: Any, texts: Any) -> tuple[Any, Any]:
    """Each of `words` as a code of the sorted dictionary `texts`, and whether it is there (CuPy)."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

    size = words.size()
    if texts is None or texts.size() == 0 or size == 0:
        return cupy.zeros(size, dtype=cupy.int64), cupy.zeros(size, dtype=bool)
    where = _view(plc.search.lower_bound(plc.Table([texts]), plc.Table([words]), [plc.types.Order.ASCENDING],
                                         [plc.types.NullOrder.AFTER])).astype(cupy.int64)
    clipped = cupy.minimum(where, texts.size() - 1)
    at = plc.copying.gather(plc.Table([texts]), _column(clipped.astype(cupy.int32)),
                            plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    same = plc.binaryop.binary_operation(at, words, plc.binaryop.BinaryOperator.EQUAL,
                                         plc.DataType(plc.TypeId.BOOL8))
    found = (where < texts.size()) & _view(same).view(cupy.bool_)
    if words.null_count():
        found &= _view(plc.unary.is_valid(words)).view(cupy.bool_)
    return clipped, found


def _distinct(components: list[Component]) -> tuple[Any, Any, Any]:
    """The rows' distinct tuples in canonical typed order: (row order, where each tuple starts in it, how many
    rows give it)."""

    import cupy

    size = int(components[0].missing.size)
    if not size:
        empty = cupy.zeros(0, dtype=cupy.int64)
        return empty, empty, empty
    keys = [key for part in components for key in part.order()]
    order = cupy.lexsort(cupy.stack(keys[::-1]))
    change = cupy.zeros(size, dtype=bool)
    change[0] = True
    for key in keys:
        ordered = key[order]
        change[1:] |= ordered[1:] != ordered[:-1]
    starts = cupy.flatnonzero(change)
    return order, starts, cupy.diff(cupy.append(starts, size))


def _known(specs: list[ComponentSpec], components: list[Component], chosen: Any, first: int,
           unknown: int) -> CompositeVocabulary:
    """The tuples of the rows at `chosen` (CuPy), with IDs from `first` in that order."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    known = int(chosen.size)
    texts = []
    for part in components:
        if not (part.spec.raw and is_text(part.spec.dtype)):
            texts.append(None)
        elif part.texts is None or part.texts.size() == 0:
            texts.append(text_column([None] * known))
        else:
            codes = cupy.where(part.missing[chosen], 0, part.identity[chosen]).astype(cupy.int32)
            texts.append(plc.copying.gather(plc.Table([part.texts]), _column(codes),
                                            plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0])
    return CompositeVocabulary(specs, cupy.arange(first, first + known, dtype=cupy.int64),
                               [part.missing[chosen] for part in components],
                               [part.identity[chosen] for part in components], texts, None,
                               tuple(range(first, first + known)), unknown=unknown)


def fit_composite(specs: list[ComponentSpec], row: Any, components: list[Component]) -> CompositeVocabulary:
    """The outer IDs fitted on the kept rows' `components` (upstream's ``_fit_inputs``): the ``top_k`` most
    frequent tuples, ties in canonical typed order, take IDs from 2."""

    import cupy

    if row.vocabulary is not None:
        return CompositeVocabulary.from_supplied(specs, row.vocabulary)
    order, starts, counts = _distinct(components)
    by = cupy.lexsort(cupy.stack([cupy.arange(starts.size), -counts]))[: row.top_k]
    return _known(specs, components, order[starts[by]], 2, 1)


def fit_groups(specs: list[ComponentSpec], components: list[Component]) -> CompositeVocabulary:
    """A matching group target's groups (upstream's ``_fit_targets``): every distinct tuple of typed values, in
    canonical typed order, numbered from 0; any other tuple is -1."""

    order, starts, _ = _distinct(components)
    return _known(specs, components, order[starts], 0, -1)
