# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable

from pyarrow.fs import FileSystem, FileSystemHandler, PyFileSystem

# A decoded transfer returns its local size; None preserves listed size checks.
ObjectFetch = Callable[[str, str], int | None]
ObjectIdentity = Callable[[str], tuple[str, ...]]


@dataclass(frozen=True)
class ObjectStoreAccess:
    filesystem: FileSystem
    fetch: ObjectFetch | None = None
    object_identity: ObjectIdentity | None = None


def refreshing(current: Callable[[], FileSystem]) -> FileSystem:
    """A filesystem that asks ``current`` for the signed one on every call,
    so a listing, footer or file opened after credentials rotate is signed
    with the new ones. Opened files are the delegate's own, read natively."""
    return PyFileSystem(_Refreshing(current))


class _Refreshing(FileSystemHandler):
    def __init__(self, current: Callable[[], FileSystem]) -> None:
        self.current = current

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, _Refreshing)
                and other.current is self.current)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def get_type_name(self) -> str:
        return f"refreshing-{self.current().type_name}"

    def normalize_path(self, path):
        return self.current().normalize_path(path)

    def get_file_info(self, paths):
        return self.current().get_file_info(paths)

    def get_file_info_selector(self, selector):
        return self.current().get_file_info(selector)

    def create_dir(self, path, recursive):
        self.current().create_dir(path, recursive=recursive)

    def delete_dir(self, path):
        self.current().delete_dir(path)

    def delete_dir_contents(self, path, missing_dir_ok=False):
        self.current().delete_dir_contents(path, missing_dir_ok=missing_dir_ok)

    def delete_root_dir_contents(self):
        self.current().delete_dir_contents("/", accept_root_dir=True)

    def delete_file(self, path):
        self.current().delete_file(path)

    def move(self, src, dest):
        self.current().move(src, dest)

    def copy_file(self, src, dest):
        self.current().copy_file(src, dest)

    def open_input_stream(self, path):
        return self.current().open_input_stream(path)

    def open_input_file(self, path):
        return self.current().open_input_file(path)

    def open_output_stream(self, path, metadata):
        return self.current().open_output_stream(path, metadata=metadata)

    def open_append_stream(self, path, metadata):
        return self.current().open_append_stream(path, metadata=metadata)
