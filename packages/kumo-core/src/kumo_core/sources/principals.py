# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from kumoapi.data_source import (
    CreateIdentityArgs,
    IdentityConfig,
    IdentityCredentials,
    IdentityType,
    validate_identity_credentials_match,
)

__all__ = [
    "IdentityRef",
    "Named",
    "Principal",
    "Registration",
    "PrincipalStore",
    "identity_id",
    "resolve",
    "store",
]


@runtime_checkable
class Named(Protocol):
    """Anything carrying the slug an identity was created with.

    The SDK's credential-free identity object satisfies this, and is the
    shape a caller most often holds. Matched structurally rather than by type
    because that class is defined above this layer, in a package a plugin
    resolving an identity must not import.
    """
    @property
    def slug(self) -> str:
        ...


#: Anything that names an identity: the object the SDK handed back, the ID a
#: bound connector recorded, the slug, or the display name it was created
#: under. All four resolve to one store name.
IdentityRef = Union[str, Named]


def identity_id(name: str) -> str:
    r"""Derives the ID of an identity created in this process.

    A local identity is not assigned an ID by a server, so it derives one from
    the name it was created under: the same name yields the same ID in every
    process that creates it, which is what lets a connector bound in one
    process resolve in another. Connectors and saved graphs record this ID, so
    the derivation is a compatibility surface -- changing it orphans them.
    """
    digest = hashlib.sha256(json.dumps(name).encode("utf-8")).hexdigest()
    return f"identity-{digest[:16]}"


@dataclass(frozen=True)
class Principal:
    r"""The principal a source plugin opens its connections as.

    Four things, and only the things a plugin reads. :attr:`config` names the
    account, host or workspace; :attr:`credentials` holds material where the
    authentication kind stores any, and is ``None`` where it does not -- a
    Snowflake ``oauth_token_file`` identity reads a token the container
    runtime mounts, and a Google workload identity uses the environment's own
    principal. Both are fully configured and store nothing, so ``None`` is not
    an error.

    :attr:`name` travels because a plugin that caches connections keys them by
    it, and because it is what an error should say when a connection cannot be
    opened. It is also what a saved graph records, so a reload in a fresh
    process finds the same principal behind it.

    The slug and description an identity was created with are deliberately
    *not* here. They serve resolution and the credential-free summary the SDK
    hands a user; no plugin reads either, and the narrower this is, the less
    an implementation can come to depend on.
    """
    name: str
    config: IdentityConfig
    credentials: Optional[IdentityCredentials]

    @property
    def type(self) -> IdentityType:
        return self.config.type


@dataclass(frozen=True)
class Registration:
    r"""One identity as the store holds it: a principal, and how it is named.

    The slug is matched when a caller names an identity by something other
    than its display name; the description is reported back in the
    credential-free summary. Neither crosses into a plugin.
    """
    principal: Principal
    slug: str
    description: str

    @property
    def name(self) -> str:
        return self.principal.name


class PrincipalStore:
    r"""The identities this process holds, by name.

    Safe to use from several threads. The module-level :data:`store` is the
    one every caller shares, so the identity area and the data sources cannot
    disagree about what a name means; this class is instantiable so a test can
    work against an isolated one.

    Nothing here is serialized or written to an artifact. A saved graph
    records the identity *reference*; the material never leaves this process,
    and credential types protect themselves -- ``SnowflakePasswordCredentials``
    and its siblings hold ``SecretStr``, whose representation is redacted
    unless a caller asks for plaintext explicitly.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_name: Dict[str, Registration] = {}

    @staticmethod
    def _check_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("An identity name must be a non-empty string.")
        if name != name.strip():
            raise ValueError(
                f"Identity name {name!r} must not have leading or trailing "
                f"whitespace; names are matched exactly and are recorded in "
                f"saved graphs.")
        return name

    def create(self, args: CreateIdentityArgs) -> Registration:
        name = self._check_name(args.display_name)
        # Re-checked here even though `CreateIdentityArgs` validates on
        # construction, because a caller may have built the args before this
        # store existed and mutated them since.
        validate_identity_credentials_match(args.config, args.credentials)
        with self._lock:
            if name in self._by_name:
                raise ValueError(
                    f"Identity {name!r} already exists in this process. "
                    f"Use a different name, or delete the existing identity or "
                    f"rotate its credentials.")
            registration = Registration(
                principal=Principal(name=name, config=args.config,
                                    credentials=args.credentials),
                slug=args.slug or name,
                description=args.description or "",
            )
            self._by_name[name] = registration
            return registration

    def get(self, name: str) -> Principal:
        r"""Returns the principal behind :obj:`name`, for opening a connection.
        """
        with self._lock:
            return self._by_name[name].principal

    def registration(self, name: str) -> Registration:
        r"""Returns the full record, for the SDK's own credential-free view."""
        with self._lock:
            return self._by_name[name]

    def names(self) -> List[str]:
        with self._lock:
            return sorted(self._by_name)

    def remove(self, name: str) -> None:
        with self._lock:
            self._by_name.pop(name, None)

    def rotate(self, name: str,
               credentials: Optional[IdentityCredentials]) -> Registration:
        r"""Replaces the material behind :obj:`name`.

        The config is untouched: rotation replaces a secret, it does not move
        the identity to a different principal or authentication kind. That is
        also why it does not invalidate connections by itself -- a session
        already open authenticates as the same principal either way. What a
        caller *does* have to do is close cached connections, so that a
        rotation prompted by a failing one is not answered from the cache; the
        identity area does that, because only it knows which plugin to ask.
        """
        with self._lock:
            current = self._by_name[name]
            validate_identity_credentials_match(current.principal.config,
                                                credentials)
            rotated = replace(
                current,
                principal=replace(current.principal, credentials=credentials),
            )
            self._by_name[name] = rotated
            return rotated

    def resolve(self, reference: Optional[IdentityRef]) -> Optional[str]:
        r"""Maps anything that names an identity to its store name.

        A caller may hold the credential-free identity object the SDK handed
        back, the ID a bound connector recorded, or the name the identity was
        created under, and all three reach here. Resolving them in one place
        is what keeps the identity area and the data sources from drifting.

        The identity object is matched by shape rather than by type, because
        this layer is below the SDK that defines it: anything carrying a
        ``slug`` is taken to mean that slug.

        Returns ``None`` for anything unknown, because a caller who supplied a
        default identity instead is not in error. A caller that does require
        one raises on ``None`` itself, where it can say what it wanted.
        """
        if reference is None:
            return None
        if isinstance(reference, str):
            wanted = reference
        elif isinstance(reference, Named):
            wanted = reference.slug
        else:
            # Unknown shape: `None`, as the docstring promises, so a caller
            # who also supplied a default gets it instead of an AttributeError
            # raised from inside resolution.
            return None
        with self._lock:
            if wanted in self._by_name:
                return wanted
            for name, registration in self._by_name.items():
                if wanted in (registration.slug, identity_id(name)):
                    return name
        return None

    def clear(self) -> None:
        with self._lock:
            self._by_name.clear()


#: The identities this process holds. One store, so the identity area and the
#: data sources cannot disagree about what a name means.
store = PrincipalStore()


def resolve(
    candidates: Sequence[Optional[IdentityRef]],
    *,
    connector: object,
    config_example: str,
    fallback: str,
    connector_example: Optional[str] = None,
    optional: bool = False,
) -> Optional[Principal]:
    r"""Finds the principal behind whatever a source backend has.

    One resolver for every provider, because the fallthrough is the same in
    all of them: a bound connector records the identity *ID* the identity area
    issued, an unbound target may carry a name or the identity object itself,
    and a backend may hold a default of any of those shapes. They differ only
    in where they look, not in what they mean.

    A reference this process cannot resolve falls through to the next
    candidate rather than failing, because a caller who also supplied a
    default meant it to be used. If nothing resolves, what *was* named is
    reported: an identity that was never created and a name misspelled at
    creation are the two ways to arrive here, and the caller can only tell
    them apart by seeing the name.

    Args:
        candidates: References to try in order. ``None`` and empty entries are
            skipped, so a caller passes the same three slots every time.
        connector: Whatever named the identity, for the error message.
        config_example: How this provider's identity is configured, spelled as
            the argument to ``CreateIdentityArgs``.
        fallback: What leaving the identity unset does. The main UX of a
            misconfigured identity, and the one part that is genuinely
            per-backend rather than per-provider.
        connector_example: How the identity is then named on a connector,
            where the provider has one to name.
        optional: Whether naming *no* identity at all is allowed. It is for an
            object store, where an unset identity selects the host's ambient
            credential chain. A name that fails to resolve is still an error:
            falling back to a different principal silently is not a service to
            anyone.
    """
    named = [reference for reference in candidates if reference]
    for reference in named:
        name = store.resolve(reference)
        if name is not None:
            return store.get(name)
    if optional and not named:
        return None
    wanted = ", ".join(repr(reference) for reference in named) or "no"
    known = ", ".join(repr(name) for name in store.names()) or "none"
    bind = f", as in {connector_example}" if connector_example else ""
    raise ValueError(
        f"Connector {getattr(connector, 'id', connector)!r} names {wanted} "
        f"identity, and none exists in this process. Create the identity "
        f"first (config={config_example}), or pass an Identity object on the "
        f"connector{bind}. {fallback} Existing: {known}.")
