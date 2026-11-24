# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a tiny synthetic event dump (JSON) for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfm.data.synthetic.generator import generate_synthetic_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/synthetic_events.json"))
    parser.add_argument("--n-entities", type=int, default=4)
    parser.add_argument("--events-per-entity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    events = generate_synthetic_events(
        seed=args.seed, n_entities=args.n_entities, events_per_entity=args.events_per_entity
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "entity_id": event.entity_id,
            "event_id": event.event_id,
            "event_timestamp": event.event_timestamp.isoformat(),
            "numeric": dict(event.numeric),
            "categorical": dict(event.categorical),
            "labels": {
                name: {
                    "value": label.value,
                    "available_at": label.available_at.isoformat(),
                    "level": label.level,
                }
                for name, label in event.labels.items()
            },
        }
        for event in events
    ]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} n={len(payload)}")


if __name__ == "__main__":
    main()
