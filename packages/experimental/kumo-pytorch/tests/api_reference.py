# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the API reference in ``docs/api/`` from the public symbols' docstrings.

``python -m tests.api_reference --write`` regenerates the pages;
``python -m tests.api_reference`` (or ``--check``) exits nonzero when they are stale.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
import re
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "docs" / "api"


@dataclass(frozen=True)
class Page:
    file: str
    title: str
    intro: str
    module: str
    symbols: tuple[str, ...]
    groups: dict[str, tuple[str, str]] = field(default_factory=dict)


API_PAGES: tuple[Page, ...] = (
    Page(
        file="schema.md",
        title="Schema and types",
        intro="Stored tables, their scalar column types, keys, relationships and clocks.",
        module="kumo.pytorch.api",
        symbols=("DataSchema", "TableSpec", "RelationSpec", "NON_TEMPORAL", "NonTemporal", "DType"),
    ),
    Page(
        file="samples.md",
        title="Samples",
        intro=(
            "What one example may read at its prediction time: rows, values, calculations, "
            "histories, graph neighborhoods, candidates and generated targets."
        ),
        module="kumo.pytorch.api",
        symbols=(
            "SampleSpec",
            "EntitySpec",
            "Rows",
            "TimeWindow",
            "RowFilter",
            "ActiveInterval",
            "Field",
            "Constant",
            "Derived",
            "CalendarPart",
            "MapValues",
            "EventTimeDelta",
            "Aggregate",
            "GroupBy",
            "Reduce",
            "Compare",
            "DatedEntityValue",
            "EventSpec",
            "CandidateSetSpec",
            "NeighborhoodSpec",
            "GraphLayer",
            "GraphSample",
            "ExternalTarget",
            "RowTarget",
            "SampleRequest",
            "CandidateRequest",
            "TargetValue",
            "BuiltSample",
        ),
    ),
    Page(
        file="batches.md",
        title="Batches",
        intro=(
            "Named model inputs: sources, ordered fields, row and history arrangements, "
            "layouts, batching limits and the emitted `Batch`."
        ),
        module="kumo.pytorch.api",
        symbols=(
            "BatchSpec",
            "Input",
            "InputSpec",
            "InputSource",
            "EntitySource",
            "EventSource",
            "EventsOf",
            "OccurrencesOf",
            "CandidateSource",
            "NeighborhoodSource",
            "Representation",
            "Fields",
            "Encode",
            "Graph",
            "Columns",
            "Tokens",
            "Composite",
            "Grouped",
            "Flatten",
            "SequenceLayout",
            "Padded",
            "Packed",
            "Nested",
            "Separate",
            "Batching",
            "Batch",
        ),
    ),
    Page(
        file="encodings.md",
        title="Encodings and ID spaces",
        intro="Explicit value encoders, shared encoders, final ID spaces and field-name IDs.",
        module="kumo.pytorch.api",
        symbols=(
            "ValueEncoding",
            "ScalarEncoding",
            "Raw",
            "ZScore",
            "Buckets",
            "Vocabulary",
            "Text",
            "UseEncoding",
            "IdSpace",
            "FieldNames",
        ),
    ),
    Page(
        file="targets.md",
        title="Targets",
        intro="Explicitly encoded labels and input-derived objectives.",
        module="kumo.pytorch.api",
        symbols=(
            "TargetSpec",
            "LabelTarget",
            "StructuredTarget",
            "CandidateTarget",
            "AlignedEventTarget",
            "ThresholdEventLabel",
            "MaskedValuesTarget",
            "NextTokenTarget",
            "MatchingGroupTarget",
            "InputExposureWarning",
        ),
    ),
    Page(
        file="lifecycle.md",
        title="Lifecycle",
        intro=(
            "The contract, validation, preparation, anchors, fitting, batch iteration and previews."
        ),
        module="kumo.pytorch.api",
        symbols=(
            "DataLoaderSpec",
            "check",
            "prepare",
            "preview",
            "Plan",
            "Task",
            "TaskBatches",
            "PreviewResult",
            "PreparedBatchPlan",
            "prepare_batch_spec",
            "AnchorTable",
            "CandidateColumn",
            "ALL_ANCHORS",
            "AllAnchors",
            "shard_positions",
            "padded_positions",
            "ValidationError",
            "ValidationIssue",
        ),
    ),
    Page(
        file="engines.md",
        title="Datasets and engines",
        intro="Where table bytes live, which engine executes the contract, and what it supports.",
        module="kumo.pytorch.api",
        symbols=(
            "Dataset",
            "InMemoryRows",
            "Parquet",
            "Engine",
            "SampleSource",
            "AnchorTask",
            "DiskGraphEngine",
            "available_engines",
            "engine_from_name",
            "register_engine",
            "Unsupported",
            "InMemoryEngine",
        ),
        groups={
            "InMemoryEngine": (
                "For trying things out",
                "`InMemoryEngine` runs a contract over a handful of rows you hold in memory. "
                "It's for playing with the library: print a batch or a contract to see what "
                "it looks like, or train a small model on a toy dataset. The tests use it "
                "too. It isn't meant for real data or production use; use `DiskGraphEngine` "
                "for that.",
            ),
        },
    ),
    Page(
        file="artifacts-and-serving.md",
        title="Artifacts and serving",
        intro=(
            "Saving and loading fitted state, model weights, and serving from caller-supplied rows."
        ),
        module="kumo.pytorch.api",
        symbols=(
            "Artifact",
            "FittedState",
            "FitRecord",
            "save_weights",
            "load_weights",
            "ServingAssembler",
            "ServingRowsSpec",
        ),
    ),
)

AUTHORING_PAGE = Page(
    file="authoring.md",
    title="Object-linked authoring (experimental)",
    intro=(
        "`kumo.pytorch.build` links declarations through Python objects and lowers them to the "
        "ordinary `DataLoaderSpec`. It is experimental: names and arguments may change."
    ),
    module="kumo.pytorch.build",
    symbols=(),
)

DATASETS_PAGE = Page(
    file="datasets.md",
    title="Example datasets",
    intro="Seeded synthetic datasets with planted signals, used by the example notebooks.",
    module="kumo.pytorch.datasets",
    symbols=(),
)

VALUE_DOCS: dict[str, str] = {
    "NON_TEMPORAL": (
        "The `NonTemporal` marker: pass it as `TableSpec(time_column=NON_TEMPORAL)` for a table "
        "whose rows have no time semantics."
    ),
    "ALL_ANCHORS": (
        "The `AllAnchors` selector: pass it as `fit_split` or `split` to select every anchor, "
        "whatever its split."
    ),
}

_ROLE = re.compile(r":(?:class|data|func|meth|attr|mod|obj|exc):`~?([^`]+)`")
_FIELD = re.compile(r"^:(param|raises|returns?)\s*([^:]*):\s*(.*)$")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_QUALIFIED = re.compile(r"\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])")
_SIGNATURE_WIDTH = 88


def _module_or_none(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _pages() -> tuple[Page, ...]:
    build = importlib.import_module("kumo.pytorch.build")
    api = importlib.import_module("kumo.pytorch.api")
    authoring = tuple(
        name for name in build.__all__ if getattr(api, name, None) is not getattr(build, name)
    )
    pages = [*API_PAGES, dataclasses.replace(AUTHORING_PAGE, symbols=authoring)]
    datasets = _module_or_none("kumo.pytorch.datasets")
    exported = tuple(getattr(datasets, "__all__", ()))
    if exported:
        pages.append(dataclasses.replace(DATASETS_PAGE, symbols=exported))
    return tuple(pages)


def _check_coverage(pages: tuple[Page, ...]) -> None:
    api = importlib.import_module("kumo.pytorch.api")
    placed = [name for page in pages if page.module == "kumo.pytorch.api" for name in page.symbols]
    duplicated = sorted({name for name in placed if placed.count(name) > 1})
    unplaced = sorted(set(api.__all__) - set(placed))
    unknown = sorted(set(placed) - set(api.__all__))
    problems = []
    if unplaced:
        problems.append(f"public symbols not mapped to a page: {unplaced}")
    if duplicated:
        problems.append(f"symbols mapped to more than one page: {duplicated}")
    if unknown:
        problems.append(f"mapped symbols missing from kumo.pytorch.api.__all__: {unknown}")
    if problems:
        raise SystemExit("tests/api_reference.py API_PAGES: " + "; ".join(problems))


def anchor(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to hyphens."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _clean(text: str) -> str:
    text = _ADDRESS.sub("", text)
    return _QUALIFIED.sub("", text)


def _is_auto_doc(obj: object, doc: str | None) -> bool:
    name = getattr(obj, "__name__", "")
    return doc is None or not doc.strip() or (inspect.isclass(obj) and doc.startswith(f"{name}("))


def own_doc(obj: object) -> str | None:
    """The object's own docstring, ignoring dataclass-generated signatures."""
    doc = obj.__doc__ if inspect.isclass(obj) else inspect.getdoc(obj)
    if _is_auto_doc(obj, doc):
        return None
    return inspect.cleandoc(doc)


class _Text:
    """Renders as its text, so signatures print annotations and defaults unquoted."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _annotation_text(annotation: object) -> str:
    rendered = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return _clean(rendered)


def _default_text(owner: object, name: str, default: object) -> str:
    if dataclasses.is_dataclass(owner) and repr(default) == "<factory>":
        field = {f.name: f for f in dataclasses.fields(owner)}.get(name)
        if field is not None and field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
    return _clean(repr(default))


def _format_signature(
    prefix: str, signature: inspect.Signature, returns: bool, owner: object = None
) -> str:
    params: list[str] = []
    star_needed = not any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()
    )
    previous: inspect._ParameterKind | None = None
    for param in signature.parameters.values():
        if previous is inspect.Parameter.POSITIONAL_ONLY and param.kind is not previous:
            params.append("/")
        if param.kind is inspect.Parameter.KEYWORD_ONLY and star_needed:
            params.append("*")
            star_needed = False
        changes: dict[str, object] = {}
        if param.annotation is not inspect.Parameter.empty:
            changes["annotation"] = _Text(_annotation_text(param.annotation))
        if param.default is not inspect.Parameter.empty:
            changes["default"] = _Text(_default_text(owner, param.name, param.default))
        params.append(str(param.replace(**changes)))
        previous = param.kind
    if previous is inspect.Parameter.POSITIONAL_ONLY:
        params.append("/")
    suffix = ""
    if returns and signature.return_annotation is not inspect.Signature.empty:
        suffix = f" -> {_annotation_text(signature.return_annotation)}"
    one_line = f"{prefix}({', '.join(params)}){suffix}"
    if len(one_line) <= _SIGNATURE_WIDTH:
        return one_line
    body = "".join(f"    {param},\n" for param in params)
    return f"{prefix}(\n{body}){suffix}"


def _signature(obj: object, prefix: str, drop_self: bool = False) -> str | None:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    if drop_self:
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name in ("self", "cls"):
            signature = signature.replace(parameters=parameters[1:])
    return _format_signature(prefix, signature, returns=not prefix.startswith("class "), owner=obj)


def _inline(text: str) -> str:
    return _ROLE.sub(r"`\1`", text).replace("``", "`")


def render_doc(doc: str) -> str:
    """Render a reST-flavoured docstring as Markdown."""
    lines = doc.splitlines()
    body: list[str] = []
    fields: dict[str, list[list[str]]] = {"param": [], "raises": [], "returns": []}
    current: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        match = _FIELD.match(line)
        if match:
            kind, name, text = match.groups()
            kind = "returns" if kind.startswith("return") else kind
            current = [name.strip(), text.strip()]
            fields[kind].append(current)
            continue
        if current is not None and line.startswith((" ", "\t")) and line.strip():
            current[1] += " " + line.strip()
            continue
        current = None
        if not line.rstrip().endswith("::"):
            body.append(_inline(line))
            continue
        lead = line.rstrip()[:-1]
        if lead.strip() != ":":
            body.append(_inline(lead))
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index == len(lines):
            continue
        indent = len(lines[index]) - len(lines[index].lstrip())
        block: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) >= indent
        ):
            block.append(lines[index][indent:])
            index += 1
        while block and not block[-1].strip():
            block.pop()
        body.extend(["", "```python", *block, "```", ""])
    text = "\n".join(body).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    sections = [text] if text else []
    labels = {"param": "Parameters", "raises": "Raises", "returns": "Returns"}
    for kind, entries in fields.items():
        if not entries:
            continue
        items = []
        for name, description in entries:
            description = _inline(description)
            items.append(f"- `{name}`: {description}" if name else f"- {description}")
        sections.append(f"**{labels[kind]}**\n\n" + "\n".join(items))
    return "\n\n".join(sections)


def _kind(obj: object) -> str:
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception"
        if issubclass(obj, enum.Enum):
            return "enum"
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if isinstance(obj, types.UnionType):
        return "type alias"
    return "constant"


def _union_members(obj: types.UnionType) -> list[str]:
    return [getattr(member, "__name__", repr(member)) for member in obj.__args__]


def members(cls: type) -> Iterator[tuple[str, object, str]]:
    """Public methods and properties a class adds, as ``(name, raw, kind)``."""
    seen: set[str] = set()
    for owner in cls.__mro__:
        if not owner.__module__.startswith("kumo.pytorch"):
            continue
        for name, raw in owner.__dict__.items():
            if name.startswith("_") or name in seen:
                continue
            seen.add(name)
            if isinstance(raw, property):
                kind = "property"
            elif isinstance(raw, classmethod):
                kind = "classmethod"
            elif isinstance(raw, staticmethod):
                kind = "staticmethod"
            elif inspect.isfunction(raw):
                kind = "method"
            else:
                continue
            if owner is not cls and _documented_elsewhere(owner):
                continue
            if owner is cls and _overrides_documented(cls, name, raw):
                continue
            yield name, raw, kind


_PUBLIC_CLASSES: set[type] = set()


def _documented_elsewhere(owner: type) -> bool:
    return owner in _PUBLIC_CLASSES


def _overrides_documented(cls: type, name: str, raw: object) -> bool:
    if member_doc(raw) is not None:
        return False
    if any(base in _PUBLIC_CLASSES and name in base.__dict__ for base in cls.__mro__[1:]):
        return True
    return any(
        getattr(protocol, "_is_protocol", False) and name in protocol.__dict__
        for protocol in _PUBLIC_CLASSES
        if protocol is not cls
    )


def member_doc(raw: object) -> str | None:
    target = raw.fget if isinstance(raw, property) else raw
    if isinstance(target, classmethod | staticmethod):
        target = target.__func__
    doc = getattr(target, "__doc__", None)
    return inspect.cleandoc(doc) if doc and doc.strip() else None


def _member_section(cls: type, name: str, raw: object, kind: str) -> str:
    heading = f"#### `{cls.__name__}.{name}`"
    parts = [heading, ""]
    if kind == "property":
        parts.append("*property*")
    else:
        function = raw.__func__ if isinstance(raw, classmethod | staticmethod) else raw
        signature = _signature(
            function, f"{cls.__name__}.{name}", drop_self=kind in ("method", "classmethod")
        )
        if signature:
            label = "" if kind == "method" else f"*{kind}*\n\n"
            parts.append(f"{label}```python\n{signature}\n```")
    doc = member_doc(raw)
    if doc:
        parts.extend(["", render_doc(doc)])
    return "\n".join(parts).rstrip() + "\n"


def _symbol_section(name: str, obj: object, module: str) -> str:
    kind = _kind(obj)
    parts = [f"### `{name}`", "", f"*{kind}* · `{module}.{name}`", ""]
    if kind in ("class", "exception", "enum"):
        signature = None if kind != "class" else _signature(obj, f"class {name}")
        if signature:
            parts.extend(["```python", signature, "```", ""])
        doc = own_doc(obj)
        if doc:
            parts.extend([render_doc(doc), ""])
        if kind == "enum":
            items = [f"- `{member.name}` = `{member.value!r}`" for member in obj]
            parts.extend(["**Members**", "", *items, ""])
        for member, raw, member_kind in members(obj):
            parts.append(_member_section(obj, member, raw, member_kind))
    elif kind == "function":
        signature = _signature(obj, f"def {name}")
        if signature:
            parts.extend(["```python", signature, "```", ""])
        doc = own_doc(obj)
        if doc:
            parts.extend([render_doc(doc), ""])
    elif kind == "type alias":
        union = " | ".join(_union_members(obj))
        parts.extend(["```python", f"{name} = {union}", "```", ""])
        parts.append("One of: " + ", ".join(f"`{m}`" for m in _union_members(obj)) + ".")
        parts.append("")
    else:
        parts.extend(["```python", f"{name}: {type(obj).__name__}", "```", ""])
        parts.append(VALUE_DOCS[name])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _page_text(page: Page, api: ModuleType) -> str:
    module = importlib.import_module(page.module)
    header = [
        f"# {page.title}",
        "",
        "<!-- Generated by `python -m tests.api_reference --write`; do not edit. -->",
        "",
        page.intro,
        "",
        "[API reference index](README.md)",
        "",
    ]
    if page.module == "kumo.pytorch.build":
        build = importlib.import_module("kumo.pytorch.build")
        shared = [
            name for name in build.__all__ if getattr(api, name, None) is getattr(build, name)
        ]
        if shared:
            links = ", ".join(f"[`{name}`]({_page_of(name)}#{anchor(name)})" for name in shared)
            header.extend([f"Also re-exported from `kumo.pytorch.api`: {links}.", ""])
    sections = []
    for name in page.symbols:
        if name in page.groups:
            heading, note = page.groups[name]
            sections.append(f"## {heading}\n\n{note}\n")
        sections.append(_symbol_section(name, getattr(module, name), page.module))
    return "\n".join(header) + "\n" + "\n".join(sections)


def _page_of(name: str) -> str:
    for page in API_PAGES:
        if name in page.symbols:
            return page.file
    raise KeyError(name)


def _summary(obj: object, name: str) -> str:
    if name in VALUE_DOCS:
        doc = VALUE_DOCS[name]
    elif isinstance(obj, types.UnionType):
        doc = "One of: " + ", ".join(f"`{m}`" for m in _union_members(obj)) + "."
    else:
        doc = own_doc(obj) or ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ")
    first = _inline(first)
    return first.replace("|", "\\|")


def _index_text(pages: tuple[Page, ...]) -> str:
    lines = [
        "# API reference",
        "",
        "<!-- Generated by `python -m tests.api_reference --write`; do not edit. -->",
        "",
        "Every public symbol of `kumo.pytorch.api`, `kumo.pytorch.build` and "
        "`kumo.pytorch.datasets`, with its signature, parameters, behavior and refusals, "
        "generated from the docstrings. "
        "Import declarations and lifecycle functions from `kumo.pytorch.api`.",
        "",
        "The [guide](../README.md) explains the concepts in order. The behavior spec, with "
        "tested golden examples, is in "
        "[`conformance/spec/guarantees/`](../../conformance/spec/guarantees/).",
        "",
        "| Page | Covers |",
        "| --- | --- |",
    ]
    for page in pages:
        intro = page.intro.replace("|", r"\|")
        lines.append(f"| [{page.title}]({page.file}) | {intro} |")
    lines.extend(["", "## Symbols", "", "| Symbol | Module | Summary |", "| --- | --- | --- |"])
    for page in pages:
        module = importlib.import_module(page.module)
        for name in page.symbols:
            obj = getattr(module, name)
            link = f"[`{name}`]({page.file}#{anchor(name)})"
            lines.append(f"| {link} | `{page.module}` | {_summary(obj, name)} |")
    return "\n".join(lines) + "\n"


def generate() -> dict[str, str]:
    """The ``docs/api/`` files, by name, as the docstrings currently define them."""
    pages = _pages()
    _check_coverage(pages)
    api = importlib.import_module("kumo.pytorch.api")
    _PUBLIC_CLASSES.clear()
    for page in pages:
        module = importlib.import_module(page.module)
        for name in page.symbols:
            obj = getattr(module, name)
            if inspect.isclass(obj):
                _PUBLIC_CLASSES.add(obj)
    files = {page.file: _page_text(page, api) for page in pages}
    files["README.md"] = _index_text(pages)
    return files


def public_objects() -> Iterator[tuple[str, object]]:
    """Every documented public symbol as ``(qualified name, object)``."""
    for page in _pages():
        module = importlib.import_module(page.module)
        for name in page.symbols:
            yield f"{page.module}.{name}", getattr(module, name)


def stale_files(files: dict[str, str]) -> list[str]:
    on_disk = {path.name for path in API_DIR.glob("*.md")} if API_DIR.exists() else set()
    stale = sorted(name for name in on_disk if name not in files)
    for name, text in files.items():
        path = API_DIR / name
        if not path.exists() or path.read_text() != text:
            stale.append(name)
    return sorted(set(stale))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate docs/api/")
    parser.add_argument("--check", action="store_true", help="fail when docs/api/ is stale")
    args = parser.parse_args(argv)
    files = generate()
    if args.write:
        API_DIR.mkdir(parents=True, exist_ok=True)
        for path in API_DIR.glob("*.md"):
            if path.name not in files:
                path.unlink()
        for name, text in files.items():
            (API_DIR / name).write_text(text)
        return 0
    stale = stale_files(files)
    if stale:
        print(f"docs/api/ is stale: {stale}. Run: python -m tests.api_reference --write")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
