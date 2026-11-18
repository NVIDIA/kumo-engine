# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar encoders fitted and applied on the GPU: Raw, Vocabulary, ZScore and Buckets.

The rules are upstream Kumo PyTorch's (``kumo.pytorch.encoding``, G-ENCODING-001):

- Vocabulary IDs follow descending count, then Kumo PyTorch's canonical typed key. That
  key is the value itself for integers, booleans, times and text (whose codes
  already sort as their bytes do), so those are counted and ordered on the GPU.
  Floats order by the hex spelling of the value and are ordered on the host,
  distinct values only. Missing and unknown keep their reserved IDs; learned
  IDs take the lowest others. ``-0.0`` and ``0.0`` are one category; a
  non-finite float is refused.
- ZScore uses the population mean and standard deviation of the non-NULL
  values (scale 0 becomes 1; no values: mean 0, scale 1), and computes
  ``(x - mean) / scale`` in float64 before casting to its dtype.
- Learned Buckets cuts are numpy's linear-interpolation quantiles with
  duplicates collapsed: the values are sorted on the GPU, and numpy's own
  formula is applied on the host to the few order statistics it needs, so the
  cuts equal upstream's exactly. A value equal to a cut enters the higher range.

A shared encoder (``UseEncoding``, `encoder_identity` ``["shared", name]``) is
fitted once over every field that uses it, their texts put into one dictionary.
An encoder's `roles` and `local_ids` are upstream's, for numbering ID spaces
(`test.kumo_pytorch.reference_engine.gpu.encoding.tokens`).

Every encoder gives NULL its missing ID (0) or a zero value, with ``valid``
false. Nothing here reads a value back per row; checks are queued as GPU flags
and read with the batch's other flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from test.kumo_pytorch.reference_engine.gpu.data.text import (
    is_text,
    python_texts,
    text_column,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from kumo.pytorch.contract.dtypes import DType

__all__ = ["GpuEncoder", "Values", "concatenated", "encoder_identity", "fit_encoder", "recoded", "reused_encoder",
           "shared_texts"]


@dataclass(frozen=True)
class Values:
    """One field's sampled values on the GPU.

    :param values: a CuPy array; a text field's int32 codes into `texts`.
        Undefined where `present` is false.
    :param present: CuPy bool array of valid rows, or None when every row is valid.
    :param texts: a text field's sorted distinct texts (a ``pylibcudf.Column``).
    :param host: values with no CuPy type (decimals), as an Arrow array; `values` is then None.
    :param numbers: a decimal field's values as float64 (CuPy), rounded as upstream rounds them
        (`test.kumo_pytorch.reference_engine.gpu.data.tables.GpuTable.numbers`), for numeric transforms; None otherwise.
    :param column: a decimal field's values as a ``pylibcudf`` column, made an Arrow array (`host`) only when
        something asks for it (`arrow`); None otherwise.
    :param build: makes a decimal field's Arrow array (NULLs not yet applied) when something asks for it
        (`arrow`), where there is no single column (an event field read from several components); None otherwise.
    :param words: a decimal's unscaled integers as 256-bit words (`test.kumo_pytorch.reference_engine.gpu.data.decimals`), where they are held (a
        calculation's result, a constant); None otherwise.
    """

    dtype: DType
    values: Any
    present: Any = None
    texts: Any = None
    host: Any = None
    numbers: Any = None
    column: Any = None
    build: Any = None
    words: Any = None

    @property
    def size(self) -> int:
        if self.values is not None:
            return int(self.values.shape[0])
        if self.numbers is not None:
            return int(self.numbers.size)
        return int(self.column.size()) if self.column is not None else len(self.host)

    def valid(self) -> Any:
        import cupy

        if self.present is not None:
            return self.present
        return cupy.ones(self.size, dtype=bool)

    def arrow(self) -> Any:
        """A decimal field's values as an Arrow array of its type (NULL where not present), made once."""

        if self.host is None:
            from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
                decimal_arrow,
            )

            import pyarrow as pa
            import pyarrow.compute as pc

            host = self.build() if self.column is None else decimal_arrow(self.column, self.dtype)
            if self.present is not None:
                host = pc.if_else(pa.array(self.present.get()), host, None)
            object.__setattr__(self, "host", host)  # made once, kept (the values themselves never change)
        return self.host


@dataclass
class GpuEncoder:
    """One fitted local transformation; its state matches upstream's ``FittedEncoder``."""

    identity: str
    config: Any
    dtype: DType
    entries: tuple[tuple[Any, int], ...] = ()
    boundaries: tuple[float, ...] = ()
    mean: float = 0.0
    scale: float = 1.0
    tokenizer_json: str | None = None  # Text: the tokenizer, as upstream's ``FittedEncoder`` holds it
    _keys: Any = field(default=None, repr=False)  # vocabulary: sorted keys (CuPy) or texts (pylibcudf)
    _ids: Any = field(default=None, repr=False)  # vocabulary: the ID of each key

    @property
    def roles(self) -> dict[str, int]:
        """Local IDs with a special meaning, as upstream's ``FittedEncoder.roles``."""

        from kumo.pytorch.contract.batch.encodings import (
            Buckets,
            Text,
            Vocabulary,
        )

        if isinstance(self.config, (Vocabulary, Text)):
            return {"missing": self.config.missing_id, "unknown": self.config.unknown_id}
        if isinstance(self.config, Buckets):
            return {"missing": 0}
        return {}

    @property
    def local_ids(self) -> tuple[int, ...]:
        """Every local ID this encoder can give, as upstream's ``FittedEncoder.local_ids``."""

        from kumo.pytorch.contract.batch.encodings import (
            Buckets,
            Text,
            Vocabulary,
        )

        if isinstance(self.config, Vocabulary):
            return tuple(sorted((*self.roles.values(), *(index for _, index in self.entries))))
        if isinstance(self.config, Buckets):
            return tuple(range(len(self.boundaries) + 2 + int(self.config.separate_zero)))
        if isinstance(self.config, Text):
            return tuple(sorted(set(self.tokenizer.get_vocab().values())))
        return ()

    @property
    def tokenizer(self) -> Any:
        """Text: the ``tokenizers`` tokenizer, loaded once."""

        if "_tokenizer" not in self.__dict__:
            from tokenizers import Tokenizer

            self.__dict__["_tokenizer"] = Tokenizer.from_str(self.tokenizer_json)
        return self.__dict__["_tokenizer"]

    def pieces(self, values: Values) -> tuple[Any, Any, Any]:
        """A Text field's pieces, as upstream's ``encode_arrow`` gives them: (every row's piece IDs one row after
        another, valid, ``piece_offsets``) as CuPy int64, bool and int64 arrays. A NULL is the missing ID; an
        empty text no piece, or the unknown ID if ``empty="unknown"``. Each distinct text is tokenized once, by
        upstream's own tokenizer on the host (`_piece_table`); rows then take theirs on the GPU."""

        import cupy

        valid = values.valid()
        count = int(valid.size)
        starts, table = self._piece_table(values.texts)
        if starts.size > 1:
            codes = cupy.where(valid, values.values, 0).astype(cupy.int64)
            lengths = cupy.where(valid, starts[codes + 1] - starts[codes], 1)
        else:  # no text at all: every row is NULL
            codes, lengths = cupy.zeros(count, dtype=cupy.int64), cupy.ones(count, dtype=cupy.int64)
        offsets = cupy.zeros(count + 1, dtype=cupy.int64)
        cupy.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])
        slot = cupy.arange(total, dtype=cupy.int64)
        row = cupy.searchsorted(offsets[1:], slot, side="right")
        if table.size:
            found = table[cupy.minimum(starts[codes[row]] + slot - offsets[row], table.size - 1)]
        else:
            found = cupy.zeros(total, dtype=cupy.int64)
        return cupy.where(valid[row], found, self.config.missing_id), valid, offsets

    def _piece_table(self, texts: Any) -> tuple[Any, Any]:
        """Each distinct text of `texts` as pieces: (where each starts, every piece ID), CuPy int64; built once
        per dictionary with upstream's tokenizer (``encode_batch``, as upstream encodes)."""

        import cupy

        cache = self.__dict__.setdefault("_piece_tables", {})
        key = id(texts)
        if key not in cache:
            strings = [] if texts is None else texts.to_arrow().to_pylist()
            lists = [list(item.ids) for item in self.tokenizer.encode_batch(strings)]
            if self.config.empty == "unknown":
                lists = [ids or [self.config.unknown_id] for ids in lists]
            starts = cupy.zeros(len(lists) + 1, dtype=cupy.int64)
            cupy.cumsum(cupy.asarray([len(ids) for ids in lists], dtype=cupy.int64), out=starts[1:])
            table = cupy.asarray([piece for ids in lists for piece in ids], dtype=cupy.int64)
            cache[key] = (texts, starts, table)  # keep `texts` alive so its id is not reused
        return cache[key][1], cache[key][2]

    def encode(self, values: Values, flags: list[tuple[str, Any]]) -> tuple[Any, Any]:
        """(encoded values, valid) as CuPy arrays; problems are appended to `flags` as (message, GPU bool)."""

        import cupy

        from kumo.pytorch.contract.batch.encodings import (
            Buckets,
            Vocabulary,
            ZScore,
        )

        valid = values.valid()
        config = self.config
        if isinstance(config, Vocabulary):
            return self._lookup(values, flags), valid
        if isinstance(config, (ZScore, Buckets)):
            source = values.numbers if values.values is None else values.values  # a decimal: its numbers
            if source is None:
                raise ValueError(f"encoder {self.identity}: test.kumo_pytorch.reference_engine.gpu does not encode {values.dtype.name} values "
                                 f"with {type(config).__name__} here yet")
            numeric = cupy.where(valid, source.astype(cupy.float64), 0.0)
            flags.append((f"encoder {self.identity}: numeric transforms require finite values",
                          (~cupy.isfinite(numeric) & valid).any()))
            if isinstance(config, ZScore):
                result = ((numeric - self.mean) / self.scale).astype(config.dtype)
                flags.append((f"encoder {self.identity}: normalized values overflow {config.dtype}",
                              (~cupy.isfinite(result) & valid).any()))
                return cupy.where(valid, result, result.dtype.type(0)), valid
            cuts = cupy.asarray(self.boundaries, dtype=cupy.float64)
            result = cupy.searchsorted(cuts, numeric, side="right").astype(cupy.int64) + 1 + int(config.separate_zero)
            if config.separate_zero:
                result = cupy.where(numeric == 0, 1, result)
            return cupy.where(valid, result, 0), valid
        raise ValueError(f"encoder {self.identity}: {type(config).__name__} is not built on the GPU yet")

    def _lookup(self, values: Values, flags: list[tuple[str, Any]]) -> Any:
        import cupy

        config = self.config
        valid = values.valid()
        if values.texts is not None:
            table = self._code_table(values.texts)
            found = table[cupy.where(valid, values.values, 0).astype(cupy.int64)] if table.size else \
                cupy.full(values.values.shape, config.unknown_id, dtype=cupy.int64)
        elif is_text(values.dtype):  # text without a dictionary holds no text: every value NULL
            found = cupy.full(values.values.shape, config.unknown_id, dtype=cupy.int64)
        else:
            keys = _normalized(values, flags, self.identity)
            if self._keys.size == 0:
                found = cupy.full(keys.shape, config.unknown_id, dtype=cupy.int64)
            else:
                position = cupy.minimum(cupy.searchsorted(self._keys, keys), self._keys.size - 1)
                found = cupy.where(self._keys[position] == keys, self._ids[position], config.unknown_id)
        return cupy.where(valid, found, config.missing_id).astype(cupy.int64)

    def _code_table(self, texts: Any) -> Any:
        """Each code of `texts` to its ID (unknown when not an entry): one int64 per distinct text, built once."""

        import cupy
        import pylibcudf as plc

        cache = self.__dict__.setdefault("_tables", {})
        key = id(texts)
        if key in cache:
            return cache[key][1]
        table = cupy.full(texts.size(), self.config.unknown_id, dtype=cupy.int64)
        if self._keys is not None and self._keys.size() and texts.size():
            # where each entry's text sits among `texts` (both sorted by bytes), and whether it is there
            where = plc.search.lower_bound(plc.Table([texts]), plc.Table([self._keys]),
                                           [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])
            position = _view(where).astype(cupy.int64)
            inside = position < texts.size()
            clipped = cupy.minimum(position, texts.size() - 1)
            at = plc.copying.gather(plc.Table([texts]), _column(clipped.astype(cupy.int32)),
                                    plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
            same = plc.binaryop.binary_operation(at, self._keys, plc.binaryop.BinaryOperator.EQUAL,
                                                 plc.DataType(plc.TypeId.BOOL8))
            hit = inside & _view(same).view(cupy.bool_)
            table[clipped[hit]] = self._ids[hit]
        cache[key] = (texts, table)  # keep `texts` alive so its id is not reused
        return table


def encoder_identity(input_name: str, use: Any) -> str:
    """An input field's encoder identity, spelled as upstream Kumo PyTorch spells it: a shared encoder's
    (``UseEncoding``) names the encoder, so every field that uses it shares one fit."""

    import json

    from kumo.pytorch.contract.batch.encodings import UseEncoding

    parts = ("shared", use.encoding.name) if isinstance(use.encoding, UseEncoding) else \
        ("inputs", input_name, "fields", use.field)
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def graph_identity(input_name: str, table: str, use: Any) -> str:
    """A graph input's node-field encoder identity, as upstream spells it (a shared encoder's: its name)."""

    import json

    from kumo.pytorch.contract.batch.encodings import UseEncoding

    parts = ("shared", use.encoding.name) if isinstance(use.encoding, UseEncoding) else \
        ("inputs", input_name, "graph", table, use.field)
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def shared_texts(parts: list[Values]) -> tuple[Any, list[Any]]:
    """One dictionary for text parts with different ones: (it, each part's code map, or None to keep codes).
    A part without a dictionary holds no text (every value NULL, as a label no anchor supplies): its codes
    are kept, masked by its validity."""

    import cupy
    import pylibcudf as plc

    dictionaries = [part.texts for part in parts]
    known = [texts for texts in dictionaries if texts is not None]
    if not known or all(texts is known[0] for texts in known):
        return (known[0] if known else None), [None] * len(parts)
    merged = plc.concatenate.concatenate([plc.Table([texts]) for texts in known])
    merged = plc.stream_compaction.distinct(merged, [0], plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
                                            plc.types.NullEquality.EQUAL, plc.types.NanEquality.ALL_EQUAL)
    merged = plc.sorting.sort(merged, [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])
    maps = [None if texts is None else _view(plc.search.lower_bound(
        merged, plc.Table([texts]), [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])).astype(cupy.int32)
        for texts in dictionaries]
    return merged.columns()[0], maps


def recoded(codes: Any, remap: Any) -> Any:
    """Text codes moved into a merged dictionary (`remap` from `shared_texts`; a NULL's arbitrary code is
    clipped, and stays masked by its validity)."""

    import cupy

    if remap is None:
        return codes
    return remap[cupy.clip(codes, 0, remap.size - 1)] if remap.size else cupy.zeros_like(codes)


def concatenated(parts: list[Values]) -> Values:
    """Several fields' values as one, for a shared encoder's fit (texts put into one dictionary)."""

    import cupy

    if len(parts) == 1:
        return parts[0]
    if parts[0].values is None:  # decimals: their numbers (a numeric transform's fit)
        return Values(parts[0].dtype, None, cupy.concatenate([part.valid() for part in parts]),
                      numbers=cupy.concatenate([part.numbers for part in parts]))
    texts, maps = shared_texts(parts)
    values = cupy.concatenate([recoded(part.values, remap) for part, remap in zip(parts, maps, strict=True)])
    present = cupy.concatenate([part.valid() for part in parts])
    return Values(parts[0].dtype, values, present, texts)


def fit_encoder(identity: str, config: Any, dtype: DType, values: Values) -> GpuEncoder:
    """Fit once from the fitting anchors' values, NULLs excluded (upstream's ``fit_encoder``)."""

    from kumo.pytorch.contract.batch.encodings import (
        Buckets,
        Raw,
        Text,
        Vocabulary,
        ZScore,
    )

    if isinstance(config, (ZScore, Buckets)) and not dtype.numeric:
        raise ValueError(f"encoder {identity}: {type(config).__name__} requires a numeric dtype")
    if isinstance(config, Text):
        return _fit_text(identity, config, dtype, values)
    if isinstance(config, Raw):
        return GpuEncoder(identity, config, dtype)
    if isinstance(config, Vocabulary):
        return _fit_vocabulary(identity, config, dtype, values)
    if isinstance(config, ZScore):
        return _fit_zscore(identity, config, dtype, values)
    if isinstance(config, Buckets):
        return GpuEncoder(identity, config, dtype, boundaries=_fit_boundaries(identity, config, values))
    raise ValueError(f"encoder {identity}: {type(config).__name__} is not built on the GPU yet")


# -- Text -------------------------------------------------------------------------------------

def _fit_text(identity: str, config: Any, dtype: DType, values: Values) -> GpuEncoder:
    """Upstream's Text fit, with its own library: a supplied tokenizer checked, or a byte-level BPE learned
    from the values in the order upstream reads them (NULLs left out)."""

    import json

    from tokenizers import Tokenizer

    if dtype.name != "string":
        raise ValueError(f"encoder {identity}: Text requires string dtype")
    if config.state is not None:
        tokenizer = Tokenizer.from_str(config.state)
        if not {config.missing_id, config.unknown_id} <= set(tokenizer.get_vocab().values()):
            raise ValueError(f"encoder {identity}: supplied Text state must contain its missing and unknown IDs")
        unknown = json.loads(config.state).get("model", {}).get("unk_token")
        if unknown is not None and tokenizer.token_to_id(unknown) != config.unknown_id:
            raise ValueError(f"encoder {identity}: Text unknown_id differs from the tokenizer model's unknown entry")
    else:
        texts = [] if values.texts is None else values.texts.to_arrow().to_pylist()
        codes = values.values[values.valid()].get().tolist() if texts else []
        tokenizer = learned_tokenizer(config, (texts[code] for code in codes))
    return GpuEncoder(identity, config, dtype, tokenizer_json=tokenizer.to_str())


def learned_tokenizer(config: Any, texts: Iterable[str]) -> Any:
    """Upstream's byte-level BPE learned from `texts`. The library learns it from how often each word occurs, so
    the order of `texts` does not change it (a split's partitions give theirs in another order)."""

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=config.vocabulary_size, min_frequency=config.min_frequency,
        special_tokens=["[MISSING]", "[UNK]"], show_progress=False))
    return tokenizer


# -- Vocabulary ---------------------------------------------------------------------------------

def _fit_vocabulary(identity: str, config: Any, dtype: DType, values: Values) -> GpuEncoder:
    import cupy

    from kumo.pytorch.encoding import typed_key

    if config.values is None:
        flags: list[tuple[str, Any]] = []
        observed = (_normalized(values, flags, identity) if values.texts is None else values.values)[values.valid()]
        _raise_flags(flags)
        distinct, counts = cupy.unique(observed, return_counts=True)
        return counted_vocabulary(identity, config, dtype, distinct, counts, values.texts)
    # supplied: exact, never enlarged by what is observed
    entries = tuple((dtype.normalize(value), index) for value, index in config.values)
    keys = [typed_key(dtype, value) for value, _ in entries]
    if any(value is None for value, _ in entries) or len(set(keys)) != len(keys):
        raise ValueError(f"encoder {identity}: supplied vocabulary values must be distinct non-NULL typed values")
    return _with_lookup(GpuEncoder(identity, config, dtype, entries=entries))


def counted_vocabulary(identity: str, config: Any, dtype: DType, distinct: Any, counts: Any,
                       texts: Any) -> GpuEncoder:
    """A Vocabulary fitted from each distinct key observed (CuPy: normalized values, or codes into `texts`, a
    sorted dictionary) and how often: the most frequent first, ties in canonical typed order."""

    from kumo.pytorch.encoding import typed_key

    keep = counts >= config.min_frequency
    distinct, counts = distinct[keep], counts[keep]
    if dtype.name.startswith("float"):  # canonical order is the hex spelling: decide on the host
        pairs = sorted(zip((-counts).get().tolist(), distinct.get().tolist(), strict=True),
                       key=lambda pair: (pair[0], typed_key(dtype, pair[1])))
        ordered = [value for _, value in pairs]
    else:  # canonical order is the value's (a text's code sorts as its bytes do)
        import pylibcudf as plc

        # Separate typed sort columns avoid uint64/int64 promotion to float64,
        # which would merge ordering keys above 2**53.
        order = _view(plc.sorting.sorted_order(plc.Table([_column(-counts), _column(distinct)]),
                                               [plc.types.Order.ASCENDING] * 2,
                                               [plc.types.NullOrder.AFTER] * 2))
        chosen = distinct[order]
        if config.max_size is not None:
            chosen = chosen[: config.max_size]
        ordered = (_texts_at(texts, chosen, dtype) if texts is not None else _host(chosen, dtype))
    if config.max_size is not None:
        ordered = ordered[: config.max_size]
    reserved = {config.missing_id, config.unknown_id}
    available = (index for index in range(len(ordered) + 2) if index not in reserved)
    entries = tuple((value, next(available)) for value in ordered)
    return _with_lookup(GpuEncoder(identity, config, dtype, entries=entries))


def _with_lookup(encoder: GpuEncoder) -> GpuEncoder:
    """`encoder` with its vocabulary's GPU lookup: its entries' keys (numbers sorted; texts as a ``pylibcudf``
    column, in any order, since each is looked up among a column's sorted texts), and each key's ID."""

    import cupy

    entries, dtype = encoder.entries, encoder.dtype
    ids = [index for _, index in entries]
    if is_text(dtype):
        encoder._keys = text_column([text for text, _ in entries])
        encoder._ids = cupy.asarray(ids, dtype=cupy.int64)
    else:
        raw = [_device_key(dtype, value) for value, _ in entries]
        order = sorted(range(len(raw)), key=raw.__getitem__)
        kind = _key_type(dtype)
        encoder._keys = cupy.asarray([raw[i] for i in order], dtype=kind)
        encoder._ids = cupy.asarray([ids[i] for i in order], dtype=cupy.int64)
    return encoder


def reused_encoder(fitted: Any) -> GpuEncoder:
    """Upstream's fitted encoder (``FittedEncoder``: a prior task's, or one read from an artifact) as this
    backend's, applied unchanged: its vocabulary entries, cuts, mean and scale, or tokenizer, as fitted."""

    from kumo.pytorch.contract.batch.encodings import Vocabulary

    encoder = GpuEncoder(fitted.identity, fitted.config, fitted.dtype, entries=tuple(fitted.entries),
                         boundaries=tuple(fitted.boundaries), mean=fitted.mean, scale=fitted.scale,
                         tokenizer_json=fitted.tokenizer_json)
    return _with_lookup(encoder) if isinstance(fitted.config, Vocabulary) else encoder


def _normalized(values: Values, flags: list[tuple[str, Any]], identity: str) -> Any:
    """Values as vocabulary keys: floats with -0.0 made 0.0 (one category), non-finite flagged."""

    import cupy

    keys = values.values
    if keys.dtype.kind == "f":
        from test.kumo_pytorch.reference_engine.gpu.data.keys import widened

        keys = widened(keys)
        flags.append((f"encoder {identity}: vocabulary values must be finite",
                      (~cupy.isfinite(keys) & values.valid()).any()))
        keys = cupy.where(keys == 0, keys.dtype.type(0), keys)
    elif keys.dtype == cupy.bool_:
        keys = keys.astype(cupy.uint8)
    return keys


def _key_type(dtype: DType) -> Any:
    import cupy

    if dtype.name == "float32":
        # Float32 arithmetic flushes subnormals; use exact widened lookup keys.
        return cupy.float64
    if dtype.name == "bool":
        return cupy.uint8
    if dtype.name == "timestamp":
        return cupy.int64
    if dtype.name == "date32":
        return cupy.int32
    return cupy.dtype(dtype.name)


def _device_key(dtype: DType, value: Any) -> Any:
    """A Python value as the number its column holds on the GPU."""

    import datetime as dt

    if dtype.name == "timestamp":
        return int(value.timestamp())
    if dtype.name == "date32":
        return (value - dt.date(1970, 1, 1)).days
    if dtype.name.startswith("float"):
        return 0.0 if value == 0 else float(value)
    return int(value) if dtype.name == "bool" else value


def _host(values: Any, dtype: DType) -> list[Any]:
    """GPU keys back to Python values of `dtype` (as upstream's entries hold them)."""

    import datetime as dt

    numbers = values.get().tolist()
    if dtype.name == "timestamp":
        return [dt.datetime.fromtimestamp(value, dt.UTC) for value in numbers]
    if dtype.name == "date32":
        return [dt.date(1970, 1, 1) + dt.timedelta(days=value) for value in numbers]
    if dtype.name == "bool":
        return [bool(value) for value in numbers]
    return numbers


def _texts_at(texts: Any, codes: Any, dtype: Any = None) -> list[Any]:
    """The texts at `codes`, as ``str`` (``bytes`` for a `bytes` dtype)."""

    import pylibcudf as plc

    picked = plc.copying.gather(plc.Table([texts]), _column(codes.astype("int32")),
                                plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    return python_texts(picked.columns()[0], dtype)


# -- ZScore and Buckets ---------------------------------------------------------------------------

def _numbers(identity: str, values: Values) -> Any:
    """The non-NULL values as float64 (CuPy), refused if any is not finite."""

    import cupy

    source = values.numbers if values.values is None else values.values  # a decimal: its rounded numbers
    numeric = source.astype(cupy.float64)[values.valid()]
    if not bool(cupy.isfinite(numeric).all()):
        raise ValueError(f"encoder {identity}: numeric transforms require finite values")
    return numeric


def _fit_zscore(identity: str, config: Any, dtype: DType, values: Values) -> GpuEncoder:
    if config.mean is not None:
        return GpuEncoder(identity, config, dtype, mean=config.mean, scale=config.scale)
    numeric = _numbers(identity, values)
    if numeric.size:
        mean, scale = float(numeric.mean()), float(numeric.std())
    else:
        mean, scale = 0.0, 1.0
    return zscore_encoder(identity, config, dtype, mean, scale)


def zscore_encoder(identity: str, config: Any, dtype: DType, mean: float, scale: float) -> GpuEncoder:
    """A fitted ZScore: its mean and (population) standard deviation, refused unless finite; 0 scales by 1."""

    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError(f"encoder {identity}: fitted statistics overflow")
    return GpuEncoder(identity, config, dtype, mean=mean, scale=scale or 1.0)


def _fit_boundaries(identity: str, config: Any, values: Values) -> tuple[float, ...]:
    import cupy

    if config.boundaries is not None:
        return tuple(float(value) for value in config.boundaries)
    numeric = bucket_numbers(identity, config, values)
    if numeric.size == 0:
        return ()
    ordered = cupy.sort(numeric)
    return bucket_cuts(identity, config, int(ordered.size), lambda at: ordered[cupy.asarray(at)].get())


def bucket_numbers(identity: str, config: Any, values: Values) -> Any:
    """The numbers a Buckets encoder's cuts are chosen among (CuPy float64): the non-NULL values, zeros left out
    with ``separate_zero``, and -0.0 made 0.0. A sort keeps the two zeros in no defined order (numpy's, upstream's,
    neither), so a cut at zero would otherwise take the sign of whichever lands there: the same number, spelled
    two ways in the fitted state."""

    import cupy

    numeric = _numbers(identity, values)
    if config.separate_zero:
        numeric = numeric[numeric != 0]
    return cupy.where(numeric == 0, 0.0, numeric)


def bucket_cuts(identity: str, config: Any, count: int, smallest: Callable[[Any], Any]) -> tuple[float, ...]:
    """A Buckets encoder's cuts over `count` values (non-zero ones with ``separate_zero``): numpy's 'linear'
    quantiles, from the values at given positions of their sorted order (`smallest`: host int positions to
    float64 values)."""

    import numpy as np

    if count == 0:
        return ()
    ranges = config.bins - int(config.separate_zero)
    quantiles = np.linspace(0, 1, ranges + 1)[1:-1]
    # numpy's 'linear' method: virtual index (n - 1) * q, then numpy's own _lerp on its two neighbours
    virtual = (count - 1) * quantiles
    previous = np.floor(virtual).astype(np.intp)
    following = np.minimum(previous + 1, count - 1)
    low = np.asarray(smallest(previous), dtype=np.float64)
    high = np.asarray(smallest(following), dtype=np.float64)
    gamma = np.asanyarray(virtual - previous, dtype=np.float64)
    difference = np.subtract(high, low)
    cuts = np.add(low, difference * gamma)
    np.subtract(high, difference * (1 - gamma), out=cuts, where=gamma >= 0.5)
    boundaries = tuple(float(value) for value in np.unique(cuts))
    if not all(math.isfinite(value) for value in boundaries):
        raise ValueError(f"encoder {identity}: fitted bucket boundaries overflow float64")
    return boundaries


# -- helpers -----------------------------------------------------------------------------------

def _view(column: Any) -> Any:
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

    return GpuTable("t", column.size(), {"c": column}, {"c": None}).values("c")


def _column(array: Any) -> Any:
    """A CuPy array as a pylibcudf column (no copy)."""

    import pylibcudf as plc

    return plc.Column.from_cuda_array_interface(array)


def _raise_flags(flags: list[tuple[Any, Any]]) -> None:
    """Read queued GPU checks once; raise the first that failed (a callable message is called then)."""

    import cupy

    if not flags:
        return
    failed = cupy.stack([flag for _, flag in flags]).get()
    for (message, _), bad in zip(flags, failed, strict=True):
        if bad:
            raise ValueError(message() if callable(message) else message)
