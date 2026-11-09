# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitted typed encoders used by columns, tokens, graph attributes, and targets."""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, localcontext
from functools import cached_property
from typing import Any

import numpy as np
import torch

from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.batch.encodings import (
    Buckets,
    Raw,
    ScalarEncoding,
    Text,
    Vocabulary,
    ZScore,
)
from kumo.pytorch.contract.dtypes import DType, arrow_float64


def typed_key(dtype: DType, value: Any) -> tuple[str, Any]:
    """Exact, sortable category identity in the declared scalar domain."""

    value = dtype.normalize(value)
    if value is None:
        return ("null", "")
    if dtype.name.startswith("float"):
        if not math.isfinite(value):
            raise ValueError(f"{dtype.name} vocabulary values must be finite")
        return (dtype.name, (0.0 if value == 0 else value).hex())
    if dtype.name.startswith("decimal"):
        with localcontext() as context:
            context.prec = 200
            return (str(dtype), str(int(value.scaleb(dtype.scale))))
    if dtype.name == "bytes":
        return ("bytes", value.hex())
    if dtype.name == "timestamp":
        return ("timestamp", int(value.timestamp()))
    if dtype.name == "date32":
        return ("date32", (value - dt.date(1970, 1, 1)).days)
    return (dtype.name, value)


def json_value(value: Any) -> Any:
    """Lossless readable scalar values in fitted-state documents."""

    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, dt.datetime):
        return {"$timestamp": value.isoformat()}
    if isinstance(value, dt.date):
        return {"$date": value.isoformat()}
    if isinstance(value, float):
        return {"$float": value.hex()}
    if isinstance(value, Mapping):
        return {str(name): json_value(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _finite(values: Sequence[Any], dtype: DType, identity: str) -> np.ndarray:
    import pyarrow as pa

    try:
        result = arrow_float64(pa.array(values, type=dtype.to_arrow())).to_numpy(
            zero_copy_only=False
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"encoder {identity}: numeric conversion overflow") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"encoder {identity}: numeric transforms require finite values")
    return result


def unjson_value(value: Any) -> Any:
    """Decode the tagged scalar form without interpreting ordinary strings."""

    if isinstance(value, list):
        return tuple(unjson_value(item) for item in value)
    if not isinstance(value, Mapping):
        return value
    decoders = {
        "$bytes": base64.b64decode,
        "$decimal": Decimal,
        "$timestamp": dt.datetime.fromisoformat,
        "$date": dt.date.fromisoformat,
        "$float": float.fromhex,
    }
    if len(value) == 1:
        name = next(iter(value))
        if name in decoders:
            return decoders[name](value[name])
    return {name: unjson_value(item) for name, item in value.items()}


def _torch_type(dtype: DType) -> torch.dtype | None:
    return (
        getattr(torch, dtype.name, None)
        if dtype.name not in {"string", "bytes", "date32", "timestamp", "decimal128", "decimal256"}
        else None
    )


@dataclass(frozen=True)
class FittedEncoder:
    """One local transformation. Final numbering is an output binding's job."""

    identity: str
    config: ScalarEncoding
    dtype: DType
    entries: tuple[tuple[Any, int], ...] = ()
    boundaries: tuple[float, ...] = ()
    mean: float = 0.0
    scale: float = 1.0
    tokenizer_json: str | None = None

    def __getstate__(self):
        return {name: value for name, value in self.__dict__.items() if name != "_lookup"}

    def _cache_lookup(self, factory) -> None:
        """Attach a backend's derived string lookup, outside portable fitted state."""

        if (
            isinstance(self.config, Vocabulary)
            and self.dtype.name == "string"
            and "_lookup" not in self.__dict__
        ):
            object.__setattr__(self, "_lookup", factory(self.arrow_keys))

    @cached_property
    def mapping(self) -> Mapping[tuple[str, Any], int]:
        return FrozenMapping({typed_key(self.dtype, value): index for value, index in self.entries})

    @cached_property
    def arrow_keys(self):
        import pyarrow as pa

        values = [
            0.0 if self.dtype.name.startswith("float") and value == 0 else value
            for value, _ in self.entries
        ]
        return pa.array(values, type=self.dtype.to_arrow())

    @cached_property
    def entry_ids(self) -> np.ndarray:
        values = np.asarray([index for _, index in self.entries], dtype=np.int64)
        values.flags.writeable = False
        return values

    @cached_property
    def tokenizer(self):
        from tokenizers import Tokenizer

        return Tokenizer.from_str(self.tokenizer_json)

    @property
    def roles(self) -> Mapping[str, int]:
        if isinstance(self.config, Vocabulary):
            return {"missing": self.config.missing_id, "unknown": self.config.unknown_id}
        if isinstance(self.config, Buckets):
            return {"missing": 0}
        if isinstance(self.config, Text):
            return {"missing": self.config.missing_id, "unknown": self.config.unknown_id}
        return {}

    @property
    def local_ids(self) -> tuple[int, ...]:
        if isinstance(self.config, Vocabulary):
            return tuple(sorted((*self.roles.values(), *(index for _, index in self.entries))))
        if isinstance(self.config, Buckets):
            return tuple(range(len(self.boundaries) + 2 + int(self.config.separate_zero)))
        if isinstance(self.config, Text):
            return tuple(sorted(set(self.tokenizer.get_vocab().values())))
        return ()

    @property
    def description(self) -> dict[str, Any]:
        result = {"kind": type(self.config).__name__.lower(), "dtype": asdict(self.dtype)}
        if self.local_ids:
            result["vocabulary_size"] = max(self.local_ids) + 1
            result["roles"] = dict(self.roles)
        if isinstance(self.config, Vocabulary):
            result["entries"] = [[json_value(value), index] for value, index in self.entries]
        elif isinstance(self.config, Buckets):
            result.update(boundaries=list(self.boundaries), separate_zero=self.config.separate_zero)
        elif isinstance(self.config, ZScore):
            result.update(mean=self.mean, scale=self.scale, output_dtype=self.config.dtype)
        return result

    @property
    def size(self) -> int:
        return max(self.local_ids, default=-1) + 1

    def state_dict(self) -> dict[str, Any]:
        result = self.description
        if self.tokenizer_json is not None:
            result["tokenizer_json"] = self.tokenizer_json
        return result

    def bound_config(self) -> ScalarEncoding:
        if isinstance(self.config, Vocabulary):
            return replace(self.config, values=self.entries)
        if isinstance(self.config, Buckets):
            return replace(self.config, bins=None, boundaries=self.boundaries)
        if isinstance(self.config, ZScore):
            return replace(self.config, mean=self.mean, scale=self.scale)
        if isinstance(self.config, Text):
            return replace(self.config, state=self.tokenizer_json)
        return self.config

    def encode(self, value: Any) -> Any:
        """Encode one scalar into a typed scalar, local ID, or piece-ID tuple."""

        value = self.dtype.normalize(value, path=f"encoder {self.identity}")
        if isinstance(self.config, Raw):
            return value
        if isinstance(self.config, Vocabulary):
            return (
                self.config.missing_id
                if value is None
                else self.mapping.get(typed_key(self.dtype, value), self.config.unknown_id)
            )
        if isinstance(self.config, ZScore):
            numeric = (
                np.asarray([0.0]) if value is None else _finite([value], self.dtype, self.identity)
            )
            return float(self._normalize(numeric, np.asarray([value is not None]))[0])
        if isinstance(self.config, Buckets):
            if value is None:
                return 0
            numeric = _finite([value], self.dtype, self.identity)[0]
            if self.config.separate_zero and numeric == 0:
                return 1
            return (
                int(np.searchsorted(self.boundaries, numeric, side="right"))
                + 1
                + int(self.config.separate_zero)
            )
        if value is None:
            return (self.roles["missing"],)
        result = tuple(self.tokenizer.encode(value).ids)
        return result or ((self.roles["unknown"],) if self.config.empty == "unknown" else ())

    def _normalize(self, numeric: np.ndarray, valid: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            result = ((numeric - self.mean) / self.scale).astype(self.config.dtype)
        if not np.isfinite(result[valid]).all():
            raise ValueError(
                f"encoder {self.identity}: normalized values overflow {self.config.dtype}"
            )
        result[~valid] = 0
        return result

    def encode_values(self, values: Sequence[Any], *, model: bool = False) -> dict[str, Any]:
        import pyarrow as pa

        # Arrow is the common implementation for numeric/category columns, including
        # readable records. Text expands one string into a separate piece sequence.
        return self.encode_arrow(pa.array(values, type=self.dtype.to_arrow()), model=model)

    def encode_arrow(self, values: Any, *, model: bool = False, lookup=None) -> dict[str, Any]:
        import pyarrow as pa
        import pyarrow.compute as pc

        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if DType.from_arrow(values.type) != self.dtype:
            raise ValueError(
                f"encoder {self.identity}: expected declared {self.dtype}, "
                f"received Arrow {values.type}"
            )
        if self.dtype.name == "timestamp":
            if values.type.unit != "s":
                raise ValueError(
                    f"encoder {self.identity}: sampled timestamps must already "
                    "use stored whole seconds"
                )
            values = pc.cast(values, self.dtype.to_arrow())
        valid = torch.from_numpy(pc.is_valid(values).to_numpy(zero_copy_only=False, writable=True))
        config = self.config
        if isinstance(config, Raw):
            dtype = _torch_type(self.dtype)
            if dtype is None:
                if model:
                    raise ValueError(
                        f"encoder {self.identity}: Raw {self.dtype.name} requires "
                        "inspectable output or an explicit encoding"
                    )
                return {"values": values, "valid": valid}
            filled = pc.fill_null(values, False if self.dtype.name == "bool" else 0)
            return {
                "values": torch.from_numpy(filled.to_numpy(zero_copy_only=False, writable=True)).to(
                    dtype
                ),
                "valid": valid,
            }
        if isinstance(config, Vocabulary):
            if self.dtype.name.startswith("float"):
                numeric = pc.drop_null(values).to_numpy(zero_copy_only=False)
                if not np.isfinite(numeric).all():
                    raise ValueError(f"encoder {self.identity}: vocabulary values must be finite")
                # Arrow hash lookup distinguishes signed-zero bit patterns, while
                # the typed category rule intentionally gives them one identity.
                values = pc.if_else(pc.equal(values, 0), pa.scalar(0.0, type=values.type), values)
            if lookup is None and pa.types.is_string(values.type):
                lookup = self.__dict__.get("_lookup")
            positions = (
                pc.index_in(values, value_set=self.arrow_keys) if lookup is None else lookup(values)
            )
            index = pc.fill_null(positions, -1).to_numpy(zero_copy_only=False)
            encoded = np.full(len(values), config.unknown_id, dtype=np.int64)
            found = index >= 0
            encoded[found] = self.entry_ids[index[found]]
            encoded[~valid.numpy()] = config.missing_id
            return {"values": torch.from_numpy(encoded), "valid": valid}
        if isinstance(config, (ZScore, Buckets)):
            numeric = pc.fill_null(arrow_float64(values), 0.0).to_numpy(zero_copy_only=False)
            if not np.isfinite(numeric[valid.numpy()]).all():
                raise ValueError(
                    f"encoder {self.identity}: numeric transforms require finite values"
                )
            if isinstance(config, ZScore):
                result = self._normalize(numeric, valid.numpy())
                return {
                    "values": torch.from_numpy(result),
                    "valid": valid,
                }
            result = (
                np.searchsorted(self.boundaries, numeric, side="right").astype(np.int64)
                + 1
                + int(config.separate_zero)
            )
            if config.separate_zero:
                result[numeric == 0] = 1
            result[~valid.numpy()] = 0
            return {"values": torch.from_numpy(result), "valid": valid}
        strings = pc.fill_null(values, "").to_pylist()
        pieces = self.tokenizer.encode_batch(strings)
        lists = [list(item.ids) for item in pieces]
        for index in range(len(lists)):
            if not valid[index]:
                lists[index] = [self.roles["missing"]]
            elif not lists[index] and config.empty == "unknown":
                lists[index] = [self.roles["unknown"]]
        offsets = np.concatenate(([0], np.cumsum([len(item) for item in lists], dtype=np.int64)))
        return {
            "values": torch.tensor([part for item in lists for part in item], dtype=torch.long),
            "valid": valid,
            "piece_offsets": torch.from_numpy(offsets),
        }


def fit_encoder(
    identity: str, config: ScalarEncoding, dtype: DType, values: Iterable[Any]
) -> FittedEncoder:
    """Fit once from permitted observations, excluding NULLs and padding."""

    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    observed = [
        dtype.normalize(value, path=f"encoder {identity}") for value in values if value is not None
    ]
    if isinstance(config, (ZScore, Buckets)) and not dtype.numeric:
        raise ValueError(f"encoder {identity}: {type(config).__name__} requires a numeric dtype")
    if isinstance(config, Text) and dtype.name != "string":
        raise ValueError(f"encoder {identity}: Text requires string dtype")
    if isinstance(config, Vocabulary):
        if config.values is None:
            counts = Counter(typed_key(dtype, value) for value in observed)
            actual = {typed_key(dtype, value): value for value in observed}
            ordered = sorted(
                (key for key in counts if counts[key] >= config.min_frequency),
                key=lambda key: (-counts[key], key),
            )
            if config.max_size is not None:
                ordered = ordered[: config.max_size]
            reserved = {config.missing_id, config.unknown_id}
            available = (index for index in range(len(ordered) + 2) if index not in reserved)
            entries = tuple((actual[key], next(available)) for key in ordered)
        else:
            entries = tuple((dtype.normalize(value), index) for value, index in config.values)
            keys = [typed_key(dtype, value) for value, _ in entries]
            if any(value is None for value, _ in entries) or len(set(keys)) != len(keys):
                raise ValueError(
                    f"encoder {identity}: supplied vocabulary values must be "
                    "distinct non-NULL typed values"
                )
        return FittedEncoder(identity=identity, config=config, dtype=dtype, entries=entries)
    if isinstance(config, (ZScore, Buckets)):
        numeric = _finite(observed, dtype, identity)
        if isinstance(config, ZScore):
            mean = (
                config.mean
                if config.mean is not None
                else (float(numeric.mean()) if len(numeric) else 0.0)
            )
            scale = (
                config.scale
                if config.scale is not None
                else (float(numeric.std()) if len(numeric) else 1.0)
            )
            if not math.isfinite(mean) or not math.isfinite(scale):
                raise ValueError(f"encoder {identity}: fitted statistics overflow")
            return FittedEncoder(
                identity=identity, config=config, dtype=dtype, mean=mean, scale=scale or 1.0
            )
        if config.separate_zero:
            numeric = numeric[numeric != 0]
        boundaries = config.boundaries
        if boundaries is None:
            ranges = config.bins - int(config.separate_zero)
            boundaries = (
                tuple(np.unique(np.quantile(numeric, np.linspace(0, 1, ranges + 1)[1:-1])))
                if len(numeric)
                else ()
            )
        if not all(math.isfinite(value) for value in boundaries):
            raise ValueError(f"encoder {identity}: fitted bucket boundaries overflow float64")
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            boundaries=tuple(float(value) for value in boundaries),
        )
    if isinstance(config, Text):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        if config.state is not None:
            tokenizer = Tokenizer.from_str(config.state)
            ids = set(tokenizer.get_vocab().values())
            if not {config.missing_id, config.unknown_id} <= ids:
                raise ValueError(
                    f"encoder {identity}: supplied Text state must contain "
                    "its missing and unknown IDs"
                )
            unknown_token = json.loads(config.state).get("model", {}).get("unk_token")
            if (
                unknown_token is not None
                and tokenizer.token_to_id(unknown_token) != config.unknown_id
            ):
                raise ValueError(
                    f"encoder {identity}: Text unknown_id differs from "
                    "the tokenizer model's unknown entry"
                )
        else:
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.train_from_iterator(
                observed,
                trainers.BpeTrainer(
                    vocab_size=config.vocabulary_size,
                    min_frequency=config.min_frequency,
                    special_tokens=["[MISSING]", "[UNK]"],
                    show_progress=False,
                ),
            )
        return FittedEncoder(
            identity=identity, config=config, dtype=dtype, tokenizer_json=tokenizer.to_str()
        )
    return FittedEncoder(identity=identity, config=config, dtype=dtype)


@dataclass(frozen=True)
class FittedSpace:
    """Sealed final numbering and local-to-final translations."""

    roles: Mapping[str, int]
    members: Mapping[str, Mapping[int, int]]
    size: int

    @cached_property
    def _remap_tables(self) -> dict:
        return {}

    def __getstate__(self):
        return {name: value for name, value in self.__dict__.items() if name != "_remap_tables"}

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", FrozenMapping(dict(self.roles)))
        object.__setattr__(
            self,
            "members",
            FrozenMapping(
                {name: FrozenMapping(dict(value)) for name, value in self.members.items()}
            ),
        )

    def special(self, *, role: str) -> int:
        if role not in self.roles:
            raise ValueError(f"ID space does not declare role {role!r}")
        return self.roles[role]

    def remap(self, identity: str, values: torch.Tensor) -> torch.Tensor:
        mapping = self.members[identity]
        if not mapping or values.numel() == 0:
            return values.clone()
        cache_key = identity, values.device
        if cache_key not in self._remap_tables:
            local_ids = sorted(mapping)
            self._remap_tables[cache_key] = (
                torch.tensor(local_ids, dtype=torch.long, device=values.device),
                torch.tensor(
                    [mapping[index] for index in local_ids], dtype=torch.long, device=values.device
                ),
            )
        keys, final = self._remap_tables[cache_key]
        positions = torch.searchsorted(keys, values)
        clipped = positions.clamp(max=len(keys) - 1)
        if not bool(((positions < len(keys)) & (keys[clipped] == values)).all()):
            raise ValueError(f"ID space has no allocation for encoder {identity!r} local value")
        return final[positions]

    def state_dict(self) -> dict[str, Any]:
        return {
            "roles": dict(self.roles),
            "size": self.size,
            "members": {
                name: [[local, final] for local, final in sorted(values.items())]
                for name, values in sorted(self.members.items())
            },
        }


@dataclass(frozen=True)
class OutputBinding:
    """A field's encoder plus its optional final destination."""

    encoder: FittedEncoder
    destination: FittedSpace | None = None
    id_space: str | None = None

    def encode(self, value: Any) -> Any:
        encoded = self.encoder.encode(value=value)
        if self.destination is None:
            return encoded
        mapping = self.destination.members[self.encoder.identity]
        return (
            tuple(mapping[item] for item in encoded)
            if isinstance(encoded, tuple)
            else mapping[encoded]
        )

    def encode_values(self, values: Sequence[Any], *, model: bool = False) -> dict[str, Any]:
        return self._remap(self.encoder.encode_values(values=values, model=model))

    def encode_arrow(self, values: Any, *, model: bool = False, lookup=None) -> dict[str, Any]:
        return self._remap(self.encoder.encode_arrow(values=values, model=model, lookup=lookup))

    def _remap(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.destination is not None:
            result["values"] = self.destination.remap(self.encoder.identity, result["values"])
        return result
