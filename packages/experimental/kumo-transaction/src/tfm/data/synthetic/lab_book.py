# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small pragmatiq-shaped book with enough structure to splice and to probe.

The generated fixture (`generate_pragmatiq_fixture`) is a few rows per user,
enough to check wiring. The lab notebooks' smoke mode needs more: users with
habits, several channels, home countries to match donors on, a history long
enough for the spliced identity-theft task (ADR 0015), and forecast labels
issued after the lab config's cutoffs. This book has those, at a size that
prepares on a CPU in seconds. It is synthetic throughout; no value describes
a real person, card, or merchant.

Each user has a persona (spending categories, merchants, amount scale, active
hours, channel mix), a home country and currency, one device, a monthly
salary, app sessions, and occasional peer-to-peer transfers. About one user in
seven churns: activity fades from a user-specific date in late 2024, and
`churn_6m` (issued 2025-01-01) is 1 for them. `default_12m` (issued
2024-07-01) leans on persona and income. `ltv_positive` (issued 2025-01-01)
is 1 when a noisy six-month profit, zero for churners and rising with income,
is positive. These labels are simple functions of
the generator and exist to exercise the probes, not to benchmark them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

LAB_BOOK_START = datetime(2023, 10, 1, tzinfo=UTC)
LAB_BOOK_END = datetime(2025, 2, 1, tzinfo=UTC)
CHURN_ISSUED = datetime(2025, 1, 1, tzinfo=UTC)
DEFAULT_ISSUED = datetime(2024, 7, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _Persona:
    name: str
    mccs: tuple[str, ...]
    merchants: tuple[str, ...]
    amount_median: float
    card_per_day: float
    app_per_day: float
    hours: tuple[int, int]
    online_share: float
    default_rate: float


_PERSONAS: tuple[_Persona, ...] = (
    _Persona(
        "commuter",
        ("5411", "5812", "4111", "5541"),
        ("CITY GROCER", "RAIL PASS", "CORNER CAFE", "FUEL POINT", "LUNCH BAR", "METRO KIOSK"),
        18.0,
        1.4,
        0.5,
        (7, 20),
        0.15,
        0.03,
    ),
    _Persona(
        "student",
        ("5814", "5815", "5411", "5942"),
        ("BURGER HUT", "STREAMFLIX", "CAMPUS SHOP", "BOOK NOOK", "NOODLE BOX", "GAME STORE"),
        11.0,
        1.1,
        1.1,
        (11, 24),
        0.45,
        0.12,
    ),
    _Persona(
        "family",
        ("5411", "5651", "5912", "5311"),
        ("HYPERMART", "KIDS WEAR", "PHARMA PLUS", "DEPT STORE", "HOME DIY", "PET WORLD"),
        46.0,
        1.0,
        0.4,
        (8, 21),
        0.25,
        0.05,
    ),
    _Persona(
        "traveller",
        ("4511", "7011", "5812", "4121"),
        ("SKY AIR", "GRAND HOTEL", "BISTRO NOVA", "RIDE CAB", "DUTY FREE", "CAR HIRE"),
        85.0,
        0.8,
        0.6,
        (6, 23),
        0.55,
        0.04,
    ),
)
_COUNTRIES: tuple[tuple[str, str], ...] = (("GB", "GBP"), ("PL", "PLN"), ("DE", "EUR"))
_SCREENS = ("home", "payments", "cards", "transfers", "insights")
_ACTIONS = ("view", "tap")


def _ts(day: datetime, hour: float) -> datetime:
    return day + timedelta(hours=float(hour))


def lab_book_records(n_users: int = 120, *, seed: int = 0) -> dict[str, Any]:
    """The book as records for `write_pragmatiq_dataset`: events, profiles, transfers, labels."""
    rng = np.random.default_rng(seed)
    days = (LAB_BOOK_END - LAB_BOOK_START).days
    events: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    churn: list[dict[str, Any]] = []
    default: list[dict[str, Any]] = []
    ltv: list[dict[str, Any]] = []
    users = [f"u_{i:05d}" for i in range(n_users)]
    for i, user in enumerate(users):
        persona = _PERSONAS[i % len(_PERSONAS)]
        country, currency = _COUNTRIES[int(rng.integers(len(_COUNTRIES)))]
        income = float(rng.lognormal(np.log(2600.0), 0.35))
        favourites = rng.choice(len(persona.merchants), size=4, replace=False)
        stores = [f"{persona.merchants[j]} {int(rng.integers(1000, 9999))}" for j in favourites]
        device = f"dev_{int(rng.integers(10**9, 10**10))}"
        os_name = "ios_18" if rng.random() < 0.5 else "android_15"
        version = f"10.{int(rng.integers(4, 9))}"
        scale = persona.amount_median * float(rng.lognormal(0.0, 0.25))
        opened = LAB_BOOK_START + timedelta(days=int(rng.integers(0, 60)))
        churns = rng.random() < 1 / 7
        fade = CHURN_ISSUED - timedelta(days=int(rng.integers(40, 120))) if churns else None
        profiles.append(
            {
                "user_id": user,
                "as_of": LAB_BOOK_END,
                "attributes": {
                    "country": country,
                    "persona": persona.name,
                    "income_band": "4k+" if income > 4000 else "2k-4k" if income > 2000 else "<2k",
                },
                "lifelong": [("account_opened", opened)],
            }
        )
        churn.append({"user_id": user, "eval_ts": CHURN_ISSUED, "label": int(churns)})
        risk = persona.default_rate * (2.0 if income < 2000 else 1.0)
        default.append(
            {"user_id": user, "eval_ts": DEFAULT_ISSUED, "label": int(rng.random() < risk)}
        )
        profit = (0.0 if churns else 0.18 * income) - 450.0 + float(rng.normal(0.0, 100.0))
        ltv.append(
            {
                "user_id": user,
                "eval_ts": CHURN_ISSUED,
                "label": int(profit > 0),
                "profit_6m": round(profit, 2),
            }
        )
        for d in range((opened - LAB_BOOK_START).days, days):
            day = LAB_BOOK_START + timedelta(days=d)
            activity = 1.0
            if fade is not None and day > fade:
                activity = max(0.0, 1.0 - (day - fade).days / 30.0)
            if day.day == 1:
                events.append(
                    {
                        "user_id": user,
                        "ts": _ts(day, 6.0),
                        "source": "transaction",
                        "fields": {
                            "amount": f"{income:.2f}",
                            "currency": currency,
                            "mcc": "0000",
                            "merchant": "EMPLOYER PAYROLL",
                            "txn_type": "credit_transfer",
                            "channel": "transfer",
                            "country": country,
                        },
                    }
                )
            lo, hi = persona.hours
            for _ in range(int(rng.poisson(persona.card_per_day * activity))):
                j = int(rng.integers(len(stores))) if rng.random() < 0.85 else -1
                merchant = stores[j] if j >= 0 else f"MARKET {int(rng.integers(100, 999))}"
                atm = rng.random() < 0.05
                events.append(
                    {
                        "user_id": user,
                        "ts": _ts(day, rng.uniform(lo, hi)),
                        "source": "transaction",
                        "fields": {
                            "amount": f"{scale * float(rng.lognormal(0.0, 0.6)):.2f}",
                            "currency": currency,
                            "mcc": "6011" if atm else persona.mccs[int(rng.integers(4))],
                            "merchant": "ATM" if atm else merchant,
                            "txn_type": "atm_withdrawal" if atm else "card_payment",
                            "channel": "online"
                            if rng.random() < persona.online_share
                            else "contactless",
                            "country": country,
                        },
                    }
                )
            for _ in range(int(rng.poisson(persona.app_per_day * activity))):
                start = rng.uniform(lo, hi)
                for k in range(int(rng.integers(2, 5))):
                    events.append(
                        {
                            "user_id": user,
                            "ts": _ts(day, start + k / 60.0),
                            "source": "app",
                            "fields": {
                                "screen": _SCREENS[int(rng.integers(len(_SCREENS)))],
                                "action": _ACTIONS[int(rng.integers(2))],
                                "device_id": device,
                                "os": os_name,
                                "app_version": version,
                                "geo_country": country,
                            },
                        }
                    )
            if rng.random() < 0.03 * activity:
                friend = users[int(rng.integers(n_users))]
                amount = float(rng.uniform(10, 150))
                ts = _ts(day, rng.uniform(lo, hi))
                transfers.append({"from_user": user, "to_user": friend, "ts": ts, "amount": amount})
                events.append(
                    {
                        "user_id": user,
                        "ts": ts,
                        "source": "transaction",
                        "fields": {
                            "amount": f"{amount:.2f}",
                            "currency": currency,
                            "mcc": "4829",
                            "merchant": "P2P",
                            "counterparty": friend,
                            "txn_type": "p2p_transfer",
                            "channel": "transfer",
                            "country": country,
                        },
                    }
                )
    events.sort(key=lambda e: (e["user_id"], e["ts"]))
    return {
        "events": events,
        "profiles": profiles,
        "transfers": transfers,
        "labels": {"churn_6m": churn, "default_12m": default, "ltv_positive": ltv},
    }


def generate_lab_book(path: Path | str, n_users: int = 120, *, seed: int = 0) -> Path:
    """Write `lab_book_records` as a pragmatiq book under `path`; returns `path`."""
    from tfm.data.adapters.pragmatiq import write_pragmatiq_dataset

    return write_pragmatiq_dataset(path, lab_book_records(n_users, seed=seed))


__all__ = ["generate_lab_book", "lab_book_records"]
