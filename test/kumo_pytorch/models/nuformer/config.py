"""Paper-based decoder sizes and explicit reference implementation choices."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class NuFormerConfig:
    vocab_size: int
    dim: int = 256
    depth: int = 24
    n_heads: int = 16
    ffn_multiplier: int = 4
    dropout: float = 0.0
    bias: bool = True
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    max_seq_len: int = 2048
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "dim", "depth", "n_heads", "ffn_multiplier", "max_seq_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.layer_norm_eps > 0:
            raise ValueError("layer_norm_eps must be positive")
        if not isinstance(self.pad_token_id, int) or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be an integer in the vocabulary")

    @classmethod
    def preset(cls, name: str, vocab_size: int, overrides: dict | None = None) -> "NuFormerConfig":
        """tiny is for smoke checks; small/large use the paper's decoder dimensions.

        Parameter counts depend on vocabulary size and the documented architectural
        assumptions. These presets do not guarantee exactly 24M or 330M parameters.
        """
        sizes = {
            "tiny": dict(dim=64, depth=2, n_heads=4, max_seq_len=128),
            "small": dict(dim=256, depth=24, n_heads=16),
            "large": dict(dim=1024, depth=24, n_heads=16),
        }
        if name not in sizes:
            raise ValueError(f"Unknown preset {name!r}; choose from {tuple(sizes)}")
        if overrides and "vocab_size" in overrides:
            raise ValueError("Pass vocab_size explicitly, not through overrides")
        config = cls(vocab_size=vocab_size, **sizes[name])
        return replace(config, **(overrides or {}))
