# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokenization diagnostics on a token cache, and floors for next-event loss.

For a seeded sample of users, from the cache's own arrays (no model):

- tokens per event: quantiles and the share of events at the per-event cap;
- per key: its kind, vocabulary, share of events carrying it, tokens per
  occurrence, `[UNK]` rate on the train side (`ts < train_end`) and after it,
  and the population stability index (PSI) of its value distribution between
  the two sides;
- two floors for a model's per-key next-event cross-entropy, on each user's
  last `sequence_length` events before `validation_end`:
  - `unigram_ce`: the train-side marginal distribution of the key's values;
  - `copy_ce`: the user's own earlier values in the window, smoothed toward the
    marginal (single-token keys only). A model that does not beat it has
    learned little beyond "the user repeats themselves".

`--proxy` merges a `proxy_eval.py` result's per-key cross-entropy into the
table. Output is JSON plus a Markdown table on stdout.

    uv run python benchmarks/diagnostics_tokens.py \\
        --cache ~/data/tfm_cache/synth_100k_gen2_m31_seed0_lab \\
        --out outputs/proxy/diagnostics_tokens.json \\
        [--proxy outputs/proxy/stage0/one_row_per_user_1ep/proxy.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tfm.data.token_cache import TokenCache
from tfm.features.tokenizers.key_value_time import UNK


def _us(ts: str) -> int:
    return int(datetime.fromisoformat(ts).replace(tzinfo=UTC).timestamp() * 1_000_000)


def _entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    return -sum(c / total * math.log(c / total) for c in counts.values() if c)


def _psi(a: Counter[int], b: Counter[int], eps: float = 1e-4) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    if not ta or not tb:
        return float("nan")
    out = 0.0
    for v in set(a) | set(b):
        pa, pb = max(a[v] / ta, eps), max(b[v] / tb, eps)
        out += (pa - pb) * math.log(pa / pb)
    return out


def diagnose(
    cache: TokenCache,
    *,
    train_end: str,
    validation_end: str,
    sequence_length: int,
    max_tokens: int,
    n_users: int,
    seed: int,
) -> dict[str, Any]:
    state = cache.tokenizer_state
    key_name = {int(i): k for k, i in state["key_vocab"].items()}
    kinds: dict[str, str] = state["field_kind"]
    t_train, t_val = _us(train_end), _us(validation_end)
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(cache.n_users, size=min(n_users, cache.n_users), replace=False))

    per_event_tokens: list[int] = []
    occurrences: Counter[str] = Counter()
    tokens_of: Counter[str] = Counter()
    unk = {"train": Counter[str](), "after": Counter[str]()}
    seen = {"train": Counter[str](), "after": Counter[str]()}
    values = {"train": {}, "after": {}}  # side -> key -> Counter[value id]
    windows: list[tuple[int, int]] = []
    n_events = {"train": 0, "after": 0}
    offsets = cache.token_offsets
    for u in users.tolist():
        a, b = int(cache.user_start[u]), int(cache.user_stop[u])
        ts = np.asarray(cache.ts_us[a:b])
        stop = a + int(np.searchsorted(ts, t_val, side="left"))
        windows.append((max(a, stop - sequence_length), stop))
        lo, hi = int(offsets[a]), int(offsets[b])
        keys = np.asarray(cache.key_ids[lo:hi])
        vals = np.asarray(cache.value_ids[lo:hi])
        bounds = np.asarray(offsets[a : b + 1]) - lo
        for e in range(b - a):
            side = "train" if ts[e] < t_train else "after"
            n_events[side] += 1
            s, t = int(bounds[e]), int(bounds[e + 1])
            per_event_tokens.append(t - s)
            present: set[str] = set()
            for k, v in zip(keys[s:t].tolist(), vals[s:t].tolist(), strict=True):
                name = key_name.get(k, "<unk key>")
                tokens_of[name] += 1
                seen[side][name] += 1
                if v == UNK:
                    unk[side][name] += 1
                values[side].setdefault(name, Counter())[v] += 1
                present.add(name)
            for name in present:
                occurrences[name] += 1

    # Next-event floors on each user's window before validation_end.
    unigram_ce: dict[str, list[float]] = {}
    copy_ce: dict[str, list[float]] = {}
    marginal = {k: (c, sum(c.values()), len(c) + 1) for k, c in values["train"].items()}
    for a, b in windows:
        if b <= a:
            continue
        lo = int(offsets[a])
        keys = np.asarray(cache.key_ids[lo : int(offsets[b])])
        vals = np.asarray(cache.value_ids[lo : int(offsets[b])])
        bounds = np.asarray(offsets[a : b + 1]) - lo
        history: dict[str, Counter[int]] = {}
        for e in range(b - a):
            s, t = int(bounds[e]), int(bounds[e + 1])
            fields: dict[str, list[int]] = {}
            for k, v in zip(keys[s:t].tolist(), vals[s:t].tolist(), strict=True):
                fields.setdefault(key_name.get(k, "<unk key>"), []).append(v)
            for name, vs in fields.items():
                counts, total, support = marginal.get(name, (Counter(), 0, 1))
                for v in vs:
                    p_g = (counts[v] + 1) / (total + support)
                    unigram_ce.setdefault(name, []).append(-math.log(p_g))
                if len(vs) == 1 and kinds.get(name) != "text":
                    v = vs[0]
                    own = history.setdefault(name, Counter())
                    n_own = sum(own.values())
                    p_g = (counts[v] + 1) / (total + support)
                    alpha = 1.0
                    p = (own[v] + alpha * p_g) / (n_own + alpha)
                    copy_ce.setdefault(name, []).append(-math.log(p))
                    own[v] += 1

    tokens = np.asarray(per_event_tokens)
    keys_out: dict[str, Any] = {}
    total_events = sum(n_events.values())
    for name in sorted(tokens_of, key=lambda k: -occurrences[k]):
        entry = {
            "kind": kinds.get(name, "?"),
            "event_share": occurrences[name] / max(total_events, 1),
            "tokens_per_occurrence": tokens_of[name] / max(occurrences[name], 1),
            "unk_rate_train": unk["train"][name] / max(seen["train"][name], 1),
            "unk_rate_after": unk["after"][name] / max(seen["after"][name], 1),
            "distinct_value_tokens_train": len(values["train"].get(name, {})),
            "value_entropy_train": _entropy(values["train"].get(name, Counter()))
            if values["train"].get(name)
            else float("nan"),
            "psi_train_vs_after": _psi(
                values["train"].get(name, Counter()), values["after"].get(name, Counter())
            ),
            "unigram_ce": float(np.mean(unigram_ce[name])) if name in unigram_ce else float("nan"),
            "copy_ce": float(np.mean(copy_ce[name])) if name in copy_ce else float("nan"),
        }
        keys_out[name] = entry
    return {
        "users": int(len(users)),
        "events": n_events,
        "tokens_per_event": {
            "p50": float(np.quantile(tokens, 0.5)),
            "p90": float(np.quantile(tokens, 0.9)),
            "p99": float(np.quantile(tokens, 0.99)),
            "max": int(tokens.max()),
            "share_at_or_over_cap": float((tokens >= max_tokens).mean()),
            "cap": max_tokens,
        },
        "cache_truncated_events": cache.manifest.get("n_truncated_events"),
        "vocab_size": cache.manifest.get("vocab_size"),
        "keys": keys_out,
    }


def markdown(result: dict[str, Any], proxy: dict[str, float] | None) -> str:
    lines = [
        "| key | kind | events | tok/occ | UNK train | UNK after | PSI | H(train) | unigram CE | copy CE |"
        + (" model CE |" if proxy else ""),
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if proxy else ""),
    ]
    for name, e in result["keys"].items():
        row = (
            f"| `{name}` | {e['kind']} | {e['event_share']:.1%} | {e['tokens_per_occurrence']:.1f} "
            f"| {e['unk_rate_train']:.2%} | {e['unk_rate_after']:.2%} | {e['psi_train_vs_after']:.3f} "
            f"| {e['value_entropy_train']:.2f} | {e['unigram_ce']:.2f} | {e['copy_ce']:.2f} |"
        )
        if proxy:
            row += f" {proxy.get(name, float('nan')):.2f} |"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-end", default="2024-06-01T00:00:00")
    parser.add_argument("--validation-end", default="2024-09-01T00:00:00")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--users", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy", type=Path, default=None, help="A proxy_eval.py result.")
    args = parser.parse_args()
    cache = TokenCache(args.cache.expanduser())
    result = diagnose(
        cache,
        train_end=args.train_end,
        validation_end=args.validation_end,
        sequence_length=args.sequence_length,
        max_tokens=args.max_tokens,
        n_users=args.users,
        seed=args.seed,
    )
    proxy = None
    if args.proxy is not None:
        proxy = json.loads(args.proxy.read_text())["heldout"]["next_event_ce_by_key"]
        result["model_ce"] = proxy
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("keys", "model_ce")}, indent=1))
    print(markdown(result, proxy))


if __name__ == "__main__":
    main()
