# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace causal LM as a HistoryModel backbone (experimental extra)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from tfm.config import RecipeConfig
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.extras.capability import require_extra


@dataclass(frozen=True)
class HuggingFaceLM:
    """Composition token: load or randomly initialize a causal LM."""

    pretrained_path: str | None = None
    name: str = "huggingface_lm"
    support_tier: str = "experimental"


class HuggingFaceCausalBackbone(nn.Module):
    """Wraps AutoModelForCausalLM to the HistoryModel + logits contract.

    Llama rotates queries and keys by token index, so the composition declaring
    this backbone must say `RoPE()`; `TrxFM` checks `position` against it.
    """

    position = "rope"

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        config = model.config
        self._hidden_size = int(config.hidden_size)
        self._vocab_size = int(config.vocab_size)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, causal: bool = True) -> Tensor:
        del causal
        inner = getattr(self.model, "model", None)
        if inner is None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = outputs.last_hidden_state
        else:
            outputs = inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state
        if hidden is None:
            raise IncompatibleCompositionError("HuggingFace backbone did not return hidden states.")
        return hidden

    def logits(self, hidden: Tensor) -> Tensor:
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise IncompatibleCompositionError("HuggingFace backbone is missing lm_head.")
        return lm_head(hidden)

    def describe(self) -> dict[str, object]:
        """The Llama's dimensions, as `TrxFM.to_config()` reports them."""
        cfg = self.model.config
        return {
            "depth": int(cfg.num_hidden_layers),
            "mask": "causal",
            "hidden_size": int(cfg.hidden_size),
            "num_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "dropout": float(cfg.attention_dropout),
            "vocab_size": int(cfg.vocab_size),
            "max_seq_len": int(cfg.max_position_embeddings),
            "ffn_size": int(cfg.intermediate_size),
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        }


ATTENTION_IMPLEMENTATIONS = {"eager": "eager", "sdpa": "sdpa"}
"""`model.composition.attention` to Llama's `attn_implementation`."""


def _attn_implementation(config: RecipeConfig) -> str:
    name = config.model.composition.attention
    if name not in ATTENTION_IMPLEMENTATIONS:
        raise IncompatibleCompositionError(
            f"The HuggingFace backbone runs attention {sorted(ATTENTION_IMPLEMENTATIONS)}; "
            f"model.composition.attention is {name!r}."
        )
    return ATTENTION_IMPLEMENTATIONS[name]


def build_huggingface_backbone(
    config: RecipeConfig, *, vocab_size: int
) -> HuggingFaceCausalBackbone:
    require_extra("huggingface", "transformers")
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
    )

    spec = HuggingFaceLM(pretrained_path=config.model.huggingface.pretrained_path)
    attn_implementation = _attn_implementation(config)
    if spec.pretrained_path:
        path = Path(spec.pretrained_path).expanduser()
        if not path.exists():
            raise IncompatibleCompositionError(
                f"HuggingFace pretrained path does not exist: {path}"
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(path), attn_implementation=attn_implementation
        )
        ckpt_vocab = int(model.config.vocab_size)
        if ckpt_vocab != vocab_size:
            raise IncompatibleCompositionError(
                f"Checkpoint vocab_size={ckpt_vocab} does not match tokenizer vocab_size={vocab_size}."
            )
        return HuggingFaceCausalBackbone(model)

    decoder = config.model.composition.granularity.decoder
    hf = config.model.huggingface
    kv_heads = hf.num_key_value_heads
    if kv_heads is None:
        kv_heads = max(1, int(decoder.num_heads) // 4) if decoder.num_heads >= 4 else 1
    if int(decoder.num_heads) % int(kv_heads) != 0:
        raise IncompatibleCompositionError(
            f"num_attention_heads={decoder.num_heads} must be divisible by "
            f"num_key_value_heads={kv_heads}."
        )
    intermediate = (
        max(32, 4 * int(decoder.hidden_size)) if decoder.ffn_size is None else int(decoder.ffn_size)
    )
    extra: dict[str, Any] = {"attn_implementation": attn_implementation}
    if hf.rope_theta is not None:
        extra["rope_theta"] = float(hf.rope_theta)
    llama_cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=int(decoder.hidden_size),
        num_hidden_layers=int(decoder.depth),
        num_attention_heads=int(decoder.num_heads),
        num_key_value_heads=int(kv_heads),
        intermediate_size=intermediate,
        max_position_embeddings=max(int(decoder.max_seq_len), int(config.data.max_tokens)),
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_dropout=float(decoder.dropout),
        tie_word_embeddings=bool(hf.tie_word_embeddings),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        **extra,
    )
    return HuggingFaceCausalBackbone(LlamaForCausalLM(llama_cfg))
