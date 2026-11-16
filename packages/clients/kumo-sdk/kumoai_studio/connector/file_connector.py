# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from kumoapi.source_table import SourceTableValidateResponse
from typing_extensions import override

from kumoai_studio.connector.identity_backed import IdentityBackedConnector
from kumoai_studio.connector.source_table import SourceTable

_PATH_SAFE = "/-._~!$&'()*+,;=:@"


class FileURI:
    """Strict parser for a connector's object-store root URI."""

    _schemes: tuple[str, ...] = ()

    def __init__(self, uri: str):
        self.uri = uri

    @property
    def is_valid(self) -> bool:
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def validate(self):
        parsed = urlsplit(self.uri)
        if parsed.scheme.lower() not in self._schemes:
            raise ValueError(
                f"Expected one of {self._schemes}, got {self.uri!r}.")
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"Invalid object-store URI {self.uri!r}.")
        if (parsed.path.startswith('//')
                or re.search(r'%(?:2f|5c)', parsed.path, flags=re.IGNORECASE)):
            raise ValueError(
                'Object-store paths cannot contain ambiguous or encoded '
                'separators.')
        return self

    @staticmethod
    def _decode_path(path: str) -> str:
        if re.search(r'%(?![0-9a-fA-F]{2})', path):
            raise ValueError(f'Malformed percent escape in path {path!r}.')
        try:
            return unquote(path, errors='strict')
        except UnicodeDecodeError as error:
            raise ValueError(f'Invalid UTF-8 escape in path {path!r}.') \
                from error

    @staticmethod
    def _unsafe_decoded_character(path: str) -> bool:
        return ('\\' in path
                or any(ord(char) < 32 or ord(char) == 127 for char in path))

    @classmethod
    def _normalize_decoded_relative_table_name(cls, name: str,
                                               original: str) -> str:
        has_trailing_slash = name.endswith('/')
        normalized = name[:-1] if has_trailing_slash else name
        if (not normalized or normalized.startswith('/')
                or cls._unsafe_decoded_character(normalized)):
            raise ValueError('Table path must be safe and connector-relative: '
                             f'{original!r}.')
        parts = normalized.split('/')
        if any(part in {'', '.', '..'} for part in parts):
            raise ValueError(
                f'Table path contains an empty or unsafe component: '
                f'{original!r}.')
        result = '/'.join(parts)
        return f'{result}/' if has_trailing_slash else result

    @classmethod
    def normalize_relative_table_name(cls, name: str) -> str:
        r"""Return one canonical table coordinate relative to a connector.

        Percent escapes are decoded and one trailing slash is preserved to
        distinguish a dataset directory from a same-named file. A final
        ``.csv`` or ``.parquet`` suffix is also preserved. Absolute paths,
        URIs, empty components, and dot segments are rejected.
        """
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError('Table path must be a non-empty string without '
                             'surrounding whitespace.')
        parsed = urlsplit(name)
        if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment):
            raise ValueError(
                f'Table path must be relative without a URI scheme, query, '
                f'or fragment: {name!r}.')
        decoded = cls._decode_path(parsed.path)
        return cls._normalize_decoded_relative_table_name(decoded, name)

    @classmethod
    def canonical_uri(cls, scheme: str, authority: str, path: str = '') -> str:
        """Build a canonical URI from a decoded provider path."""
        encoded_path = quote(path.strip('/'), safe=_PATH_SAFE)
        path_component = f'/{encoded_path}' if encoded_path else ''
        return urlunsplit((scheme.lower(), authority, path_component, '', ''))

    @classmethod
    def decoded_prefix(cls, path: str) -> str:
        """Decode and validate a connector prefix from a URI path."""
        decoded = cls._decode_path(path).strip('/')
        if (cls._unsafe_decoded_character(decoded)
                or (decoded and any(part in {'', '.', '..'}
                                    for part in decoded.split('/')))):
            raise ValueError('Connector prefix contains an empty, dot, or '
                             'unsafe path component.')
        return decoded

    def relative_table_name(self, root_uri: str) -> str:
        r"""Convert this full URI to a table coordinate under ``root_uri``.

        Provider scheme and authority must match exactly, and the configured
        root path must be a complete path-component prefix. This prevents
        prefix confusion such as treating ``root2`` as a child of ``root``.
        """
        full = urlsplit(self.validate().uri)
        root_parser = type(self)(root_uri)
        root = urlsplit(root_parser.validate().uri)
        if (full.scheme.lower() != root.scheme.lower()
                or full.netloc.casefold() != root.netloc.casefold()):
            raise ValueError(
                f'URI {self.uri!r} is outside connector root {root_uri!r}.')

        decoded_full_path = self._decode_path(full.path)
        has_trailing_slash = decoded_full_path.endswith('/')
        full_path = (decoded_full_path[:-1]
                     if has_trailing_slash else decoded_full_path)
        root_path = self._decode_path(root.path).strip('/')
        full_parts = full_path.lstrip('/').split('/') if full_path else []
        root_parts = root_path.split('/') if root_path else []
        if (self._unsafe_decoded_character(full_path)
                or self._unsafe_decoded_character(root_path)
                or any(part in {'', '.', '..'} for part in full_parts)
                or any(part in {'', '.', '..'} for part in root_parts)):
            raise ValueError('Object-store paths cannot contain empty, dot, '
                             'or parent path components.')
        if (len(full_parts) <= len(root_parts)
                or full_parts[:len(root_parts)] != root_parts):
            raise ValueError(
                f'URI {self.uri!r} is outside connector root {root_uri!r}.')
        relative = '/'.join(full_parts[len(root_parts):])
        if has_trailing_slash:
            relative = f'{relative}/'
        return self._normalize_decoded_relative_table_name(relative, self.uri)

    @property
    def object_name(self) -> str:
        return unquote(urlsplit(self.uri).path).lstrip('/')

    @property
    def root_dir(self) -> str:
        parsed = urlsplit(self.uri)
        return f'{parsed.scheme}://{parsed.netloc}'


class ObjectStoreConnector(IdentityBackedConnector):
    r"""Base class for identity-backed object-store connectors.

    Exact table lookup is separate from shallow :meth:`table_names`
    discovery. It accepts arbitrary-depth coordinates below the configured
    root and asks the project-scoped source-table configuration endpoint to
    validate the exact file or dataset directory with the persisted identity.
    """

    _uri_class: type[FileURI]
    root_dir: Optional[str]
    any_uri: bool = False

    def _normalize_table_reference(self, name: str) -> str:
        if self.any_uri:
            raise NotImplementedError(
                f'{type(self).__name__} must define any-URI normalization')
        parsed = urlsplit(name)
        if parsed.scheme or parsed.netloc:
            assert self.root_dir is not None
            return self._uri_class(name).relative_table_name(self.root_dir)
        return self._uri_class.normalize_relative_table_name(name)

    def _rooted_destination(self, path: str) -> str:
        """Normalize an output coordinate for this connector mode."""
        if self.any_uri:
            return self._normalize_table_reference(path)
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc:
            assert self.root_dir is not None
            path = self._uri_class(path).relative_table_name(self.root_dir)
        return self._uri_class.normalize_relative_table_name(path)

    def _full_uri_for_relative(self, path: str) -> str:
        """Render a canonical full URI for one rooted relative coordinate."""
        assert self.root_dir is not None
        relative = self._uri_class.normalize_relative_table_name(path)
        is_directory = relative.endswith('/')
        root = urlsplit(self.root_dir)
        root_path = self._uri_class._decode_path(root.path).strip('/')
        full_path = '/'.join(part for part in (root_path, relative) if part)
        uri = self._uri_class.canonical_uri(root.scheme, root.netloc,
                                            full_path)
        return f'{uri}/' if is_directory else uri

    def _validate_normalized_table(
            self, normalized: str) -> SourceTableValidateResponse:
        return super()._validate_table(normalized)

    @override
    def _validate_table(self, table_name: str) -> SourceTableValidateResponse:
        normalized = self._normalize_table_reference(table_name)
        return self._validate_normalized_table(normalized)

    @override
    def has_table(self, name: str) -> bool:
        r"""Return whether an exact file or dataset path is readable.

        Unlike :meth:`table_names`, this method supports paths at any depth
        under the connector root. ``name`` may be connector-relative or a full
        same-root provider URI. The exact target is validated using the
        connector's persisted identity on every call, so authorization changes
        are observed immediately.
        """
        try:
            normalized = self._normalize_table_reference(name)
        except ValueError:
            return False
        return self._validate_normalized_table(normalized).is_valid

    @override
    def table(self, name: str) -> SourceTable:
        r"""Return a source table at any path under this connector's root.

        Args:
            name: Connector-relative file or dataset-directory path, or a full
                same-root provider URI. File suffixes and one explicit
                directory slash are preserved, including for directories
                whose names end in ``.csv`` or ``.parquet``.

        Raises:
            :class:`ValueError`: If the path is unsafe, outside the root, or
                does not identify a readable table.
        """
        normalized = self._normalize_table_reference(name)
        if not self._validate_normalized_table(normalized).is_valid:
            raise ValueError(f'The table {normalized!r} does not exist in '
                             f'{self}.')
        return SourceTable(name=normalized, connector=self)

    def table_from_uri(self, uri: str) -> SourceTable:
        r"""Return a source table addressed by a full object-store URI.

        In rooted mode, the URI must use this connector's provider and
        authority and lie strictly below its configured root; it is stored as
        the same canonical relative coordinate accepted by :meth:`table`. In
        any-URI mode, the canonical full provider URI is retained and may
        address any authority permitted by the connector's identity.

        Args:
            uri: Full provider URI for a CSV/Parquet file or sharded dataset
                directory.

        Raises:
            :class:`ValueError`: If the URI has the wrong provider, violates
                the active mode's rooted or any-URI boundary, contains unsafe
                components, or does not identify a readable table.
        """
        return self.table(uri)

    @override
    def table_names(self) -> list[str]:
        """List rooted tables; any-URI connectors have no discovery root."""
        if self.any_uri:
            scheme = self._uri_class._schemes[0]
            raise NotImplementedError(
                f'Any-URI {type(self).__name__} connectors cannot list '
                f"tables; pass a full '{scheme}://' URI to table().")
        return super().table_names()
