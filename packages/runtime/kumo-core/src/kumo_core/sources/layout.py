# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from pathlib import Path
from typing import Any, Sequence, Union
from urllib.parse import urlsplit

from .errors import PermanentSourceError, SourceNotFoundError

PARQUET = "parquet"
CSV = "csv"

#: A flat listing caps top-level entries, a recursive scan caps objects, and
#: a connector caps tables. The hosted file connector imports all three.
MAX_SUPPORTED_FILES = 200
MAX_SUPPORTED_TABLES = 200
MAX_SCANNED_ENTRIES = 100_000

#: What a CSV table may be, on either backend. A graph the cloud would refuse
#: must not be buildable locally, so both enforce these through the two
#: functions below rather than through a limit of their own.
MAX_CSV_PARTITIONS = 128
MAX_CSV_DATASIZE = 10 * 1024 * 1024 * 1024


def csv_partitions_refusal(count: int) -> str | None:
    if count <= MAX_CSV_PARTITIONS:
        return None
    return (f"Kumo supports {MAX_CSV_PARTITIONS} partitions for a csv table. "
            f"Instead, Kumo got {count} partitions.")


def csv_size_refusal(total: int) -> str | None:
    if total <= MAX_CSV_DATASIZE:
        return None
    return (f"Kumo supports {MAX_CSV_DATASIZE} bytes for a csv  file. "
            f"Instead, Kumo got {total} bytes.")


_HIVE_PARTITION_SEGMENT = re.compile(r"^[^/=]+=[^/]+$")
_MAX_SUGGESTED_TABLES = 5


def file_type(path: str) -> str | None:
    lowered = path.lower()
    if lowered.endswith(f".{PARQUET}"):
        return PARQUET
    if lowered.endswith(f".{CSV}"):
        return CSV
    return None


def is_hidden(path: str) -> bool:
    """A Spark ``_SUCCESS`` marker sits beside the parts it describes, so a
    table that ships one is otherwise read as mixing two formats."""
    return any(part.startswith((".", "_")) for part in path.split("/"))


def nested(relative_paths: Sequence[str]) -> bool:
    """Whether any object sits below the root rather than directly in it."""
    return any("/" in relative for relative in relative_paths)


#: Components a table name may not contain: ``..`` walks out of the connector
#: root, and the root is the only fence a local read has.
_UNSAFE_COMPONENTS = frozenset({"", ".", ".."})


def confined(name: str) -> str:
    """``name`` as one or more safe path components below a root, or a
    refusal."""
    trimmed = name.strip("/")
    components = trimmed.split("/") if trimmed else []
    if "\\" in name or any(part in _UNSAFE_COMPONENTS for part in components):
        raise PermanentSourceError(
            f"Table name {name!r} is not a path below the connector root. "
            f"Name the table by its key relative to that root.", table=name)
    return trimmed


def within_root(output_path: str, root: str) -> None:
    """Require an object output to name a destination under ``root``.

    The write direction of :func:`confined`, and the same fence: a read names
    a table below a connector's root, a write names a destination below it,
    and neither may leave. Shared with the hosted connector, which imports it
    from here, so a destination the cloud refuses is refused locally too.

    Raises:
        ValueError: If ``root`` is empty, if the path carries a query or
            fragment, or if it addresses another provider, bucket, or prefix.
    """
    if not root:
        raise ValueError(
            'Any-URI object outputs require an explicit full destination')
    output = urlsplit(output_path)
    parsed_root = urlsplit(root)
    if output.query or output.fragment:
        raise ValueError(
            'Object-store output paths cannot contain a query or fragment')
    if (output.scheme.casefold(),
            output.netloc.casefold()) != (parsed_root.scheme.casefold(),
                                          parsed_root.netloc.casefold()):
        raise ValueError('Output path is outside the selected connector target')

    def normalized(path: str) -> str:
        value = posixpath.normpath('/' + path.lstrip('/'))
        return '/' if value == '/.' else value

    output_value = normalized(output.path)
    root_value = normalized(parsed_root.path)
    if (output_value != root_value
            and not output_value.startswith(root_value.rstrip('/') + '/')):
        raise ValueError('Output path is outside the selected connector target')


def _join(root: str, name: str) -> str:
    root = root.rstrip("/")
    name = confined(name)
    if not root:
        return name
    return f"{root}/{name}" if name else root


#: A name that is already exactly one harmless path component: it starts with
#: an alphanumeric, so neither ``.`` nor ``..`` nor an absolute path can match
#: it, and it is bounded well under the 255-byte per-component filesystem
#: limit.
_ONE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _fingerprint(value: str) -> str:
    r"""The digest a staging directory is named by.

    Spelled exactly as the SDK spelled it before this moved, because the name
    it returns is a directory on disk: changing the digest would re-stage every
    table whose name is not already one safe component into a new directory and
    orphan the old one. ``json.dumps`` rather than ``str.encode`` for the same
    reason -- and because it tolerates a lone surrogate, which ``encode`` does
    not.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def staging_dir(root: Union[str, Path], name: str) -> Path:
    r"""Returns where :obj:`name` stages beneath :obj:`root`.

    Staging *empties* the directory it is handed, so whatever names it decides
    what gets deleted -- and ``Path('/scratch') / '/etc/kumo'`` is
    ``/etc/kumo``. A name that is not already a single harmless component is
    replaced by a fingerprint of itself rather than scrubbed into one, since
    scrubbing maps distinct names onto the same directory.

    The containment check is redundant against that rule, and is kept as the
    last thing between an edit to it and an ``rmtree`` outside the tree; it
    resolves, so a symlink planted at the destination is caught too.

    Here rather than in a backend because every source that stages to local
    disk needs it, and a second copy of this rule is a second thing to get
    right.
    """
    root = Path(root)
    label = str(name)
    if not _ONE_COMPONENT.fullmatch(label):
        label = _fingerprint(str(name))
    destination = root / label
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            f"Staging {name!r} would write to {destination.resolve()}, which "
            f"is outside {root}. Nothing was read or deleted.")
    return destination


def discover_tables(relative_paths: Sequence[str]) -> list[str]:
    names: set[str] = set()
    directory_types: dict[str, set[str]] = {}
    for relative in relative_paths:
        kind = file_type(relative)
        if kind is None or is_hidden(relative):
            continue
        directory = posixpath.dirname(relative)
        if not directory:
            names.add(relative.removesuffix(f".{kind}"))
            continue
        if kind == PARQUET:
            while (directory and _HIVE_PARTITION_SEGMENT.fullmatch(
                    posixpath.basename(directory))):
                directory = posixpath.dirname(directory)
        if directory:
            directory_types.setdefault(directory, set()).add(kind)

    candidates = {
        directory
        for directory, kinds in directory_types.items() if len(kinds) == 1
    }
    ancestors: set[str] = set()
    for directory in candidates:
        parent = posixpath.dirname(directory)
        while parent and parent not in ancestors:
            ancestors.add(parent)
            parent = posixpath.dirname(parent)
    names.update(candidates - ancestors)
    if len(names) > MAX_SUPPORTED_TABLES:
        raise PermanentSourceError(
            f"The connector root holds {len(names)} tables, more than the "
            f"{MAX_SUPPORTED_TABLES} Kumo supports. Set the connector prefix "
            f"to a narrower parent of the tables and retry.")
    return sorted(names)


def scan(
    filesystem: Any,
    path: str,
    *,
    max_entries: int | None = None,
) -> list[str]:
    """Paths of the readable objects under ``path``, relative to it. A root
    that does not exist raises, since an empty listing cannot be told from a
    root that genuinely holds nothing.
    """
    from pyarrow.fs import FileSelector, FileType, LocalFileSystem

    base = path.rstrip("/")
    if filesystem.get_file_info(base).type is FileType.NotFound:
        if isinstance(filesystem, LocalFileSystem):
            advice = ("Check that the local directory exists and that you can "
                      "read and traverse it.")
        else:
            advice = "Check the bucket and prefix."
        raise SourceNotFoundError(f"Nothing exists at {base}. {advice}",
                                  table=base)
    selector = FileSelector(base, recursive=True, allow_not_found=True)
    entries = 0
    relative_paths = []
    for info in filesystem.get_file_info(selector):
        entries += 1
        if max_entries is not None and entries > max_entries:
            raise PermanentSourceError(
                f"{path} holds more than {max_entries} objects to scan. Set "
                f"the connector prefix to a narrower parent of the tables and "
                f"retry.")
        if info.type is not FileType.File:
            continue
        relative = info.path[len(base) + 1:]
        if file_type(relative) is not None:
            relative_paths.append(relative)
    return sorted(relative_paths)


def objects(filesystem: Any, path: str, *, suffix: str | None = None,
            max_entries: int | None = None,
            include_hidden: bool = False) -> list[Any]:
    from pyarrow.fs import FileSelector, FileType

    info = filesystem.get_file_info(path.rstrip("/"))
    if info.type is FileType.File:
        if max_entries is not None and max_entries < 1:
            raise PermanentSourceError("Dataset exceeds the entry limit")
        return [info]
    base = path.rstrip("/")
    selector = FileSelector(base, recursive=True, allow_not_found=True)
    found = []
    for count, entry in enumerate(filesystem.get_file_info(selector), start=1):
        if max_entries is not None and count > max_entries:
            raise PermanentSourceError(
                f"Dataset exceeds the {max_entries} entry limit")
        if not entry.path.startswith(base + "/"):
            raise PermanentSourceError(
                "Dataset listing escapes the source root")
        relative = entry.path[len(base) + 1:]
        if not relative or relative != relative.strip("/"):
            raise PermanentSourceError(
                "Dataset listing returned an unsafe relative path")
        confined(relative)
        supported = (
            file_type(entry.path) is not None if suffix is None else
            entry.path.lower().endswith(f".{suffix.lstrip('.').lower()}"))
        if entry.type is FileType.File and supported and (
                include_hidden or not is_hidden(relative)):
            found.append(entry)
    return sorted(found, key=lambda entry: entry.path)


def _hive_layout(relative_paths: Sequence[str]) -> bool:
    """Whether every directory a part sits under is a ``key=value`` partition,
    which is the only nesting a Parquet table is read through."""
    parents = [posixpath.dirname(relative) for relative in relative_paths]
    segments = [
        part for parent in parents if parent for part in parent.split("/")
    ]
    return all(_HIVE_PARTITION_SEGMENT.fullmatch(part) for part in segments)


def classify(path: str, visible: Sequence[str],
             table_name: str = "") -> str | None:
    """The single format the objects under ``path`` are stored in, or ``None``
    where none of them is readable. ``visible`` is relative to ``path`` and
    already filtered through :func:`is_hidden`.
    """
    kinds = {file_type(relative) for relative in visible} - {None}
    if len(kinds) > 1:
        raise PermanentSourceError(
            f"Files under {path} mix CSV and Parquet formats.")
    if not kinds:
        return None
    kind = kinds.pop()
    if kind == CSV and nested(visible):
        raise PermanentSourceError(
            f"CSV table {path} contains nested sub-directories, which "
            f"are not supported." + nested_suggestion(visible, table_name))
    if kind == PARQUET and not _hive_layout(visible):
        raise PermanentSourceError(
            f"Nested files under {path} are not a Parquet Hive table "
            f"with key=value partition directories." +
            nested_suggestion(visible, table_name))
    return kind


def _path_type(filesystem: Any, path: str, table_name: str = "") -> str:
    from pyarrow.fs import FileType

    info = filesystem.get_file_info(path.rstrip("/"))
    if info.type is FileType.Directory:
        visible = [
            relative for relative in scan(filesystem, path)
            if not is_hidden(relative)
        ]
        kind = classify(path, visible, table_name)
    else:
        kind = file_type(path)
    if kind is not None:
        return kind

    raise SourceNotFoundError(
        f"No files directly under {path} have a .csv or .parquet suffix.",
        table=table_name or path)


def locate(filesystem: Any, root: str, table_name: str, *,
           exact: bool = False) -> Any:
    from pyarrow.fs import FileType

    path = _join(root, table_name)
    info = filesystem.get_file_info(path)
    if info.type is not FileType.NotFound:
        return info
    if not exact:
        for kind in (PARQUET, CSV):
            info = filesystem.get_file_info(f"{path}.{kind}")
            if info.type is FileType.File:
                return info
    raise SourceNotFoundError(
        f"Table {table_name} does not exist in path {path}. Ensure the object "
        f"exists and that the connector root is set up correctly." +
        _suffix_hint(table_name), table=table_name)


def resolve(filesystem: Any, root: str, table_name: str) -> tuple[str, str]:
    """Parquet is probed before CSV, matching hosted path resolution."""
    from pyarrow.fs import FileType

    info = locate(filesystem, root, table_name)
    path = info.path
    if info.type is FileType.Directory:
        return f"{path.rstrip('/')}/", _path_type(filesystem, path, table_name)
    kind = file_type(path)
    if kind is None:
        raise PermanentSourceError(
            f"Table {table_name} does not have a supported file type "
            f"(.csv or .parquet).", table=table_name)
    return path, kind


def hive_partitioned(path: str, object_paths: Sequence[str]) -> bool:
    base = path.rstrip("/")
    return nested([entry[len(base) + 1:] for entry in object_paths])


def nested_suggestion(relative_paths: Sequence[str], table_name: str) -> str:
    selectable = [
        posixpath.join(table_name, name) if table_name else name
        for name in discover_tables(relative_paths)
    ]
    if not selectable:
        return ""
    listed = ", ".join(selectable[:_MAX_SUGGESTED_TABLES])
    remainder = len(selectable) - _MAX_SUGGESTED_TABLES
    if remainder > 0:
        listed += f", and {remainder} more"
    return f" Select a table below it, such as: {listed}."


def _suffix_hint(table_name: str) -> str:
    for kind in (PARQUET, CSV):
        if table_name.endswith(kind):
            return f" Or remove the suffix {kind}."
    return ""
