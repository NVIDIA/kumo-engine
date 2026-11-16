# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Iterator, Sequence

from .errors import SourceError, TransientSourceError
from .types import ReadMode

Classify = Callable[[BaseException], type[SourceError] | None]
Summarize = Callable[[BaseException], str]
Rule = tuple[Any, "type[SourceError] | Classify"]
Rules = Callable[[], Sequence[Rule]]
Translator = Callable[[str, ReadMode], ContextManager[None]]


def safe_summary(error: BaseException, **codes: Any) -> str:
    """Name a failure by its type and codes, never by its text: vendor clients
    put the statement, the response body and the presigned link in there."""
    named = " ".join(f"{key}={value}" for key, value in codes.items()
                     if value not in (None, ""))
    return f"{type(error).__name__} {named}".strip()


@contextmanager
def translated(
    where: str,
    *,
    table: str,
    rules: Sequence[Rule],
    summarize: Summarize,
) -> Iterator[None]:
    """Nothing is chained: a chained cause prints the vendor text this
    withholds."""
    try:
        yield
    except SourceError:
        raise
    except BaseException as error:
        reported = _classify(error, rules)
        if reported is None:
            raise
        raise reported(f"{where} failed: {summarize(error)}",
                       table=table) from None


def source_translator(vendor: str, *, rules: Rules,
                      summarize: Summarize) -> Translator:
    """``rules`` is a callable so the driver import it needs stays inside a
    read: a connector wheel must import without the vendor's driver
    installed."""
    @contextmanager
    def translate(table: str, mode: ReadMode) -> Iterator[None]:
        with translated(f"{vendor} {mode.value} read of {table}", table=table,
                        rules=rules(), summarize=summarize):
            yield

    return translate


def transient(error: BaseException, rules: Sequence[Rule]) -> bool:
    kind = (type(error)
            if isinstance(error, SourceError) else _classify(error, rules))
    return kind is not None and issubclass(kind, TransientSourceError)


def _classify(error: BaseException,
              rules: Sequence[Rule]) -> type[SourceError] | None:
    for matches, reported in rules:
        if not isinstance(error, matches):
            continue
        if isinstance(reported, type):
            return reported
        chosen = reported(error)
        if chosen is not None:
            return chosen
    return None
