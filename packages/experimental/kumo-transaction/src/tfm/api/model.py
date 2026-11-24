# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public TrxFM composition model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from torch import Tensor, nn

from tfm.api.validation import validate_composition
from tfm.attention.eager import EagerAttention
from tfm.compositions.flat import Flat
from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.outputs import ModelOutput
from tfm.contracts.protocols import AttentionSpec
from tfm.heads.binary import Binary, BinaryHead
from tfm.layers.decoder import Decoder, DecoderModule
from tfm.objectives.next_event import NextEvent, NextEventLoss
from tfm.objectives.supervised import SupervisedHead, SupervisedHeadLoss
from tfm.pooling.last_token import LastTokenPool
from tfm.pooling.mean import MeanPool
from tfm.pooling.registry import readout_module
from tfm.position.none import NoPE
from tfm.position.rope import RoPE


@runtime_checkable
class _HistoryBackbone(Protocol):
    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        causal: bool = True,
        segment_ids: Tensor | None = ...,
    ) -> Tensor: ...

    def logits(self, hidden: Tensor) -> Tensor: ...


def _supports_packing(backbone: object) -> bool:
    """True when the backbone accepts `segment_ids` and can isolate packed rows."""
    import inspect

    call = getattr(backbone, "forward", None)
    if not callable(call):
        return False
    try:
        return "segment_ids" in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


class TrxFM(nn.Module):
    """Declarative transaction foundation model.

    The core export surface supports Flat + causal Decoder + NoPE + EagerAttention +
    LastTokenPool + NextEvent, with optional Binary heads. `readout` also accepts
    the experimental `MeanPool` from `tfm.experimental`.
    """

    def __init__(
        self,
        granularity: Flat | None = None,
        position: NoPE | RoPE | None = None,
        attention: AttentionSpec | None = None,
        objective: NextEvent | SupervisedHead | Sequence[NextEvent | SupervisedHead] | None = None,
        readout: LastTokenPool | MeanPool | None = None,
        heads: Mapping[str, Binary] | None = None,
        *,
        decoder: Decoder | None = None,
        backbone: _HistoryBackbone | None = None,
    ) -> None:
        super().__init__()
        if granularity is None:
            granularity = Flat(decoder if decoder is not None else Decoder())
        position = NoPE() if position is None else position
        attn: AttentionSpec = EagerAttention() if attention is None else attention
        objective = NextEvent() if objective is None else objective
        readout = LastTokenPool() if readout is None else readout
        validate_composition(granularity, position, attn, objective, readout, dict(heads or {}))
        specs = list(objective) if isinstance(objective, Sequence) else [objective]
        self.granularity = granularity
        self.position = position
        self.attention = attn
        self.objective_spec = next(s for s in specs if isinstance(s, NextEvent))
        self.head_objective_specs = tuple(s for s in specs if isinstance(s, SupervisedHead))
        self.readout = readout
        self.head_specs = dict(heads or {})
        if backbone is None:
            module = DecoderModule(granularity.model, attn)
            self.backbone: _HistoryBackbone = module
            self._hidden_size = granularity.model.hidden_size
            self._vocab_size = granularity.model.vocab_size
        else:
            hidden = getattr(backbone, "hidden_size", None)
            vocab = getattr(backbone, "vocab_size", None)
            if hidden is None or vocab is None:
                raise IncompatibleCompositionError(
                    "Optional backbone must expose hidden_size and vocab_size."
                )
            if not callable(getattr(backbone, "logits", None)):
                raise IncompatibleCompositionError(
                    "Optional backbone must implement logits(hidden)."
                )
            self.backbone = backbone
            self._hidden_size = int(hidden)
            self._vocab_size = int(vocab)
        declared = getattr(self.backbone, "position", "nope")
        if declared != position.name:
            raise IncompatibleCompositionError(
                f"position={type(position).__name__}() but the backbone "
                f"{type(self.backbone).__name__} applies {declared!r} positions. Declare "
                "what the backbone does: RoPE() for the HuggingFace Llama backbone, NoPE() "
                "for the eager decoder."
            )
        self._backbone_supports_packing = _supports_packing(self.backbone)
        self.pool = readout_module(readout)
        self.objective_loss = NextEventLoss(ignore_index=self.objective_spec.ignore_index)
        self.head_losses = nn.ModuleList(
            [SupervisedHeadLoss(spec) for spec in self.head_objective_specs]
        )
        self.task_heads = nn.ModuleDict(
            {name: BinaryHead(self._hidden_size) for name in self.head_specs}
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(self, batch: TransactionBatch, *, compute_loss: bool = True) -> ModelOutput:
        if batch.segment_ids is None:
            hidden = self.backbone(batch.input_ids, batch.attention_mask, causal=True)
        elif not self._backbone_supports_packing:
            raise IncompatibleCompositionError(
                f"Backbone {type(self.backbone).__name__} cannot honor packed sequence "
                "boundaries, so packed batches would leak across entities. Use the "
                "padded collate, or the eager DecoderModule backbone."
            )
        else:
            hidden = self.backbone(
                batch.input_ids,
                batch.attention_mask,
                causal=True,
                segment_ids=batch.segment_ids,
            )
        hidden = self.position.apply(hidden)
        embeddings = self.pool(hidden, batch.attention_mask)
        logits = self.backbone.logits(hidden)
        detach = any(spec.detach_backbone for spec in self.head_objective_specs)
        head_input = embeddings.detach() if detach else embeddings
        predictions: dict[str, Tensor] = {}
        for name, head in self.task_heads.items():
            predictions[name] = head(head_input)
        output = ModelOutput(
            hidden_states=hidden,
            embeddings=embeddings,
            logits=logits,
            predictions=predictions or None,
        )
        if compute_loss:
            losses = {self.objective_spec.name: self.objective_loss(output, batch)}
            for spec, loss_fn in zip(self.head_objective_specs, self.head_losses, strict=True):
                losses[f"{spec.name}:{spec.head}"] = spec.weight * loss_fn(output, batch)
            output.losses = losses
            output.loss = sum(losses.values(), start=logits.new_zeros(()))
        return output

    def to_config(self) -> dict[str, object]:
        """The `model.composition` that is running, in config vocabulary.

        With an optional backbone, the `Decoder` spec is only validated, never
        built, so the decoder block is the backbone's own `describe()`, or None
        when it has none.
        """
        if isinstance(self.backbone, DecoderModule):
            dec = self.granularity.model
            decoder: dict[str, object] | None = {
                "depth": dec.depth,
                "mask": dec.mask,
                "hidden_size": dec.hidden_size,
                "num_heads": dec.num_heads,
                "dropout": dec.dropout,
                "vocab_size": dec.vocab_size,
                "max_seq_len": dec.max_seq_len,
                "ffn_size": dec.ffn_size,
            }
        else:
            describe = getattr(self.backbone, "describe", None)
            described = describe() if callable(describe) else None
            if described is not None and not isinstance(described, dict):
                raise IncompatibleCompositionError(
                    f"{type(self.backbone).__name__}.describe() must return a dict, "
                    f"got {type(described).__name__}."
                )
            decoder = described
        return {
            "granularity": {"type": "flat", "decoder": decoder},
            "position": self.position.name,
            "attention": self.attention.name,
            # TrxFM always trains one NextEvent; supervised heads are listed apart.
            "objective": "next_event",
            "head_objectives": [
                {"head": spec.head, "weight": spec.weight, "detach_backbone": spec.detach_backbone}
                for spec in self.head_objective_specs
            ],
            "readout": self.readout.name,
            "heads": {name: "binary" for name in self.head_specs},
            "backbone": type(self.backbone).__name__,
        }
