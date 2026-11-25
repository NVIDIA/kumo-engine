# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact parameter and FLOP accounting for a composed model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tfm.contracts.errors import SchemaError

BACKWARD_MULTIPLIER = 2.0
TRAIN_MULTIPLIER = 1.0 + BACKWARD_MULTIPLIER


@dataclass(frozen=True)
class ParamBreakdown:
    """Parameter counts by role (embedding / non-embedding params)"""

    embedding: int
    head: int
    attention: int
    ffn: int
    norm: int

    @property
    def non_embedding(self) -> int:
        return self.attention + self.ffn + self.norm

    @property
    def total(self) -> int:
        return self.embedding + self.head + self.non_embedding


@dataclass(frozen=True)
class FlopBreakdown:
    """Forward FLOPs per token, by term."""

    qkv_proj: float
    attention: float
    out_proj: float
    ffn: float
    head: float

    @property
    def gemm(self) -> float:
        """Terms whose cost is independent of sequence length."""
        return self.qkv_proj + self.out_proj + self.ffn + self.head

    @property
    def total(self) -> float:
        return self.gemm + self.attention

    @property
    def attention_share(self) -> float:
        return self.attention / self.total if self.total else 0.0


@dataclass(frozen=True)
class CostReport:
    """What the composed model costs, derived from the spec alone."""

    depth: int
    hidden_size: int
    num_heads: int
    ffn_size: int
    vocab_size: int
    seq_len: int
    tied_embeddings: bool
    params: ParamBreakdown
    forward: FlopBreakdown
    caveats: tuple[str, ...] = ()

    @property
    def train_flops_per_token(self) -> float:
        return TRAIN_MULTIPLIER * self.forward.total

    def six_n(self, *, non_embedding: bool = True) -> float:
        """The ``6N`` approximation this report exists to replace"""
        n = self.params.non_embedding if non_embedding else self.params.total
        return 6.0 * n

    def six_n_error(self, *, non_embedding: bool = True) -> float:
        """Ratio of exact training FLOPs to the ``6N`` estimate

        Greater than 1 means ``6N`` *understates* the cost, which inflates any
        MFU computed from it
        """
        approx = self.six_n(non_embedding=non_embedding)
        return self.train_flops_per_token / approx if approx else float("inf")

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": {
                "depth": self.depth,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "ffn_size": self.ffn_size,
                "vocab_size": self.vocab_size,
                "seq_len": self.seq_len,
                "tied_embeddings": self.tied_embeddings,
            },
            "params": {
                **asdict(self.params),
                "non_embedding": self.params.non_embedding,
                "total": self.params.total,
            },
            "forward_flops_per_token": {
                **asdict(self.forward),
                "gemm": self.forward.gemm,
                "total": self.forward.total,
                "attention_share": self.forward.attention_share,
            },
            "train_flops_per_token": self.train_flops_per_token,
            "six_n_non_embedding": self.six_n(non_embedding=True),
            "six_n_error_non_embedding": self.six_n_error(non_embedding=True),
            "six_n_all_params": self.six_n(non_embedding=False),
            "six_n_error_all_params": self.six_n_error(non_embedding=False),
            "caveats": list(self.caveats),
        }


def decoder_cost(
    *,
    depth: int,
    hidden_size: int,
    num_heads: int,
    vocab_size: int,
    seq_len: int,
    ffn_size: int | None = None,
    tied_embeddings: bool = False,
) -> CostReport:
    """Exact counts for the ``Flat(Decoder(...))`` stack as implemented

    N.B. the formulas track :class:`tfm.layers.decoder.DecoderBlock` term by term!
    """
    if depth < 1:
        raise SchemaError("depth must be at least 1.")
    if hidden_size % num_heads != 0:
        raise SchemaError("hidden_size must be divisible by num_heads.")
    if seq_len < 1:
        raise SchemaError("seq_len must be at least 1.")

    d = float(hidden_size)
    ffn = float(ffn_size if ffn_size is not None else 4 * hidden_size)
    v = float(vocab_size)
    length = float(seq_len)

    # --- parameters -----------------------------------------------------
    # qkv is Linear(d, 3d, bias=False); out is Linear(d, d, bias=False)
    attention_params = depth * (3 * d * d + d * d)
    # ff is Linear(d, ffn) -> GELU -> Linear(ffn, d), both with bias
    ffn_params = depth * (d * ffn + ffn + ffn * d + d)
    # Two LayerNorms per block (weight + bias), plus one final norm
    norm_params = depth * (2 * 2 * d) + 2 * d
    embedding_params = v * d
    head_params = 0.0 if tied_embeddings else v * d

    params = ParamBreakdown(
        embedding=int(embedding_params),
        head=int(head_params),
        attention=int(attention_params),
        ffn=int(ffn_params),
        norm=int(norm_params),
    )

    # --- forward FLOPs per token ---------------------------------------
    # A GEMM of shape (m, k) x (k, n) costs 2*m*k*n; per token m = 1
    qkv = depth * 2 * d * (3 * d)
    out_proj = depth * 2 * d * d
    ffn_flops = depth * (2 * d * ffn + 2 * ffn * d)
    # Scores Q@K^T then weights@V. Both are L x L per head; per query token that
    # is 2*L*d each. The full matrix is computed and then masked, so the causal
    # saving is NOT taken -- this is the executed cost, not the necessary one.
    attention_flops = depth * (2 * length * d + 2 * length * d)
    head_flops = 2 * d * v

    forward = FlopBreakdown(
        qkv_proj=qkv,
        attention=attention_flops,
        out_proj=out_proj,
        ffn=ffn_flops,
        head=head_flops,
    )

    caveats = [
        "Elementwise work (LayerNorm, GELU, softmax, residuals) is excluded; it is "
        "bandwidth-bound, and counting it as FLOPs would inflate MFU.",
        "Attention is counted as the eager kernel executes it: the full L x L score "
        "matrix is built and then masked, so roughly half of it is discarded work.",
    ]
    if not tied_embeddings:
        caveats.append(
            "Embedding and lm_head are separate weights on the flat-decoder path, so the "
            f"vocabulary costs 2 x |V| x d = {2 * v * d:,.0f} parameters. Tying them would "
            f"save {v * d:,.0f} parameters and zero FLOPs: the head GEMM has the same shape "
            "either way."
        )
    return CostReport(
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_size=int(ffn),
        vocab_size=vocab_size,
        seq_len=seq_len,
        tied_embeddings=tied_embeddings,
        params=params,
        forward=forward,
        caveats=tuple(caveats),
    )


def cost_from_composition(composition: dict, *, seq_len: int | None = None) -> CostReport:
    """Build a :class:`CostReport` from ``TrxFM.to_config()`` output

    This is the load-bearing link: the model describes itself, and the cost
    follows from that description without instantiating or running anything
    """
    gran = composition.get("granularity", {})
    gran_type = str(gran.get("type", "flat")).lower()
    if gran_type != "flat":
        raise SchemaError(
            f"cost_from_composition supports granularity 'flat'; got {gran_type!r}. "
            "The profile_history stack has a different block structure and needs its "
            "own accounting."
        )
    backbone = composition.get("backbone", "DecoderModule")
    if backbone != "DecoderModule":
        raise SchemaError(
            f"cost_from_composition counts the eager DecoderModule; this composition runs "
            f"{backbone}, whose blocks it does not model."
        )
    dec = gran.get("decoder", {})
    resolved_len = seq_len if seq_len is not None else int(dec.get("max_seq_len", 0))
    if resolved_len < 1:
        raise SchemaError(
            "seq_len is required: attention cost is quadratic in sequence length and "
            "cannot be inferred from the composition alone."
        )
    return decoder_cost(
        depth=int(dec["depth"]),
        hidden_size=int(dec["hidden_size"]),
        num_heads=int(dec["num_heads"]),
        vocab_size=int(dec["vocab_size"]),
        seq_len=resolved_len,
        ffn_size=dec.get("ffn_size"),
    )
