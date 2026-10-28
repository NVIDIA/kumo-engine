# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The in-process contract a warehouse plugin implements.

Everything else in :mod:`kumoapi` describes the wire: what a client sends to a
Kumo server and what comes back. This module describes something different --
the boundary between the SDK and a plugin loaded into the *same process*, such
as the Snowflake, Databricks, BigQuery and object-store readers.

The two are kept apart deliberately, because they are versioned against
different things. The wire schema must stay compatible with servers already
deployed; this ABI must stay compatible with plugins already installed. A
change that is safe for one is not automatically safe for the other, and a
reader who cannot see which contract a type belongs to cannot judge either.

Nothing here is serialized. These are structural protocols and the plain data
that crosses them, so an implementation satisfies the contract by shape and
never imports the SDK: that is what lets a plugin ship in its own
distribution, be consumed by the Kumo platform, and still be loaded by
:class:`~kumoai_studio.backend.LocalBackend`.
"""

import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import pandas as pd

import kumoapi.source_table as source_table_api
from kumoapi.data_source import ConnectorResourceConfig, DataSourceType
from kumoapi.jobs import WriteMode
from kumoapi.typing import Dtype

__all__ = [
    'BUNDLE_ID_KEY',
    'ENGINE_MARKER',
    'REGISTRY_KEY_PREFIX',
    'ArtifactLocation',
    'ArtifactSink',
    'FileStoreSink',
    'IngestResult',
    'ModelAliasingSink',
    'ModelFetchingSink',
    'ModelListingSink',
    'ModelPublishingSink',
    'ModelRegistryLocation',
    'PrepareIngestResult',
    'PublishedModel',
    'SnapshotStore',
    'SourceBackend',
    'SourceConnector',
    'SourceCredentials',
    'check_model_alias',
    'registry_lineage',
]

#: Prefix of the keys an engine export records in registry metadata.
REGISTRY_KEY_PREFIX = 'kumo_'
#: Registry metadata key marking a version as a Kumo engine bundle.
ENGINE_MARKER = 'kumo_engine'
#: Registry metadata key holding the version's bundle id; the bundle's own
#: provenance.json holds its lineage.
BUNDLE_ID_KEY = 'kumo_bundle_id'
_ALIAS = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,254}')
_RESERVED_ALIAS = re.compile(r'DEFAULT|FIRST|LAST|LATEST|V[0-9]+', re.I)


@dataclass(frozen=True)
class SourceCredentials:
    r"""Access material for one authentication profile.

    Args:
        profile: The name the ``source`` descriptor references. Two tables
            reaching the same location report the same profile, so DiskGraph
            builds one client and refreshes one credential for both.
        values: The material itself, in whatever form the profile's provider
            expects.
    """
    profile: str
    values: Mapping[str, str]


@dataclass(frozen=True)
class PrepareIngestResult:
    r"""What one staged source table hands to DiskGraph ingest.

    Two things travel, on deliberately different channels. ``source`` is the
    durable description of *where* the data is; it is serialized into the
    ingest config, which is written beside the snapshot and re-read on every
    retry. ``credentials`` is *how to reach it*, which must not be: it is
    short-lived, and a persisted config is the wrong place for a secret.

    A backend uses one or the other. A pre-signed URL carries its own access,
    so ``source`` is self-sufficient and ``credentials`` stays ``None``. A
    source reached with scoped credentials names an ``auth_profile`` in
    ``source`` and supplies that profile's material here, deferring the
    connection to DiskGraph.

    Args:
        source: A DiskGraph ``DataSource``, tagged by ``"type"``. Which type it
            is decides how ingest reaches the data: ``parquet`` for local
            files, ``object_store_parquet`` for a cloud manifest,
            ``https_parquet`` for pre-signed URLs, ``databricks`` for a live
            query.
        source_size: Bytes ingest should expect to read. This feeds DiskGraph's
            shard-count estimation, so an estimate is acceptable but a wild one
            is not.
        version: An identity for the data as staged. Two calls over unchanged
            data must produce the same value, since it is what decides whether
            an existing snapshot can be reused rather than rebuilt.
        credentials: Access material, for a ``source`` that is not
            self-sufficient.
        mode_used: How the data was moved, for a backend that had a choice.
            Reported rather than acted on, so a run can be told which path it
            took; ``None`` where the question does not arise.
    """
    source: Mapping[str, Any]
    source_size: int
    version: str
    credentials: Optional[SourceCredentials] = None
    mode_used: Optional[str] = None


@dataclass(frozen=True)
class ModelRegistryLocation:
    r"""A named, versioned model in a registry.

    The registry-addressed counterpart of a rooted connector's path: a version
    is *named*, not placed, so there is nothing here to normalize and no
    container to state -- the database and schema, or catalog and schema,
    holding the registry are the connector's own.

    Here rather than in the SDK because it crosses this boundary in both
    directions: :meth:`ArtifactSink.publish_model` and
    :meth:`ArtifactSink.fetch_model` are handed one, and a sink ships in its
    own distribution and never imports the SDK. Re-exported as
    ``kumoai_studio.ModelRegistryLocation``, which is the name a caller writes.

    .. code-block:: python

        import kumoai_studio

        registry = kumoai_studio.SnowflakeConnector(
            name="ml-registry",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ML",
        )
        config = kumoai_studio.ModelExportConfig(
            training_job_id="training-job-...",
            output_connector=registry,
            output_location=kumoai_studio.ModelRegistryLocation("CHURN_GNN"),
        )

    Args:
        model_name: Registry-local model name.
        version: Exact version to write or read. ``None`` lets the registry
            assign the next one on export; on import it selects Snowflake's
            default or Databricks' latest Kumo version.
        comment: Optional description recorded on the version.
        alias: Set on the new version after export; on import, the version
            it points at. Exclusive with :obj:`version`.
    """
    model_name: str
    version: Optional[str] = None
    comment: Optional[str] = None
    alias: Optional[str] = None

    def __post_init__(self) -> None:
        if (not isinstance(self.model_name, str) or not self.model_name
                or self.model_name != self.model_name.strip()):
            raise ValueError('model_name must be a non-empty name without '
                             'surrounding whitespace')
        if self.version is not None and not self.version:
            raise ValueError(
                'version must be a non-empty version, or None to let the '
                'registry choose')
        if self.alias is not None and (not self.alias
                                       or self.version is not None):
            raise ValueError('alias must be a non-empty alias, and is '
                             'mutually exclusive with version')


@dataclass(frozen=True)
class PublishedModel:
    r"""What a destination calls a model bundle once it holds one.

    The same triple ``kumo_core`` returns from its own publisher, because it is
    what every target can answer: a registry assigns a version, an object store
    treats the path as one, and both can say where the result now lives.

    Args:
        name: The model's name in its destination.
        version: The version written. Assigned by a registry when the caller
            named none, which is the reason this type exists -- otherwise a
            caller could not learn what was created.
        uri: Where the bundle now is, in the destination's own terms.
        metadata: Searchable Kumo metadata recorded with the version. Empty
            when a destination cannot return metadata without downloading the
            bundle.
        metrics: Metrics recorded with the version. Empty when a destination
            does not expose them during discovery.
    """
    name: str
    version: str
    uri: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class SourceConnector(Protocol):
    r"""What a :class:`SourceBackend` may rely on any connector to expose.

    Declared structurally so this contract does not import any connector
    implementation. A reader for one specific system may of course read richer
    fields off the concrete connector it handles, such as an identity name,
    database, or schema.
    """
    @property
    def id(self) -> str:
        ...

    @property
    def kumo_project_id(self) -> str:
        ...

    @property
    def source_type(self) -> DataSourceType:
        ...

    @property
    def config(self) -> ConnectorResourceConfig:
        r"""What this connector points at, in its provider's own terms.

        The union, not one provider's member: a reader narrows it to its own
        and says what it found otherwise. Narrowing inside the read rather
        than in the signature is what lets an implementation actually satisfy
        :class:`SourceBackend` -- a method declared to take
        ``SnowflakeConnectorResourceConfig`` would not.

        In practice a reader is only ever handed connectors of its own type,
        because the host routes on :attr:`source_type`. The check is what
        turns a host that got that wrong into a sentence rather than an
        attribute error from somewhere further in.
        """
        ...


@runtime_checkable
class SourceBackend(Protocol):
    r"""Read-only access to the tables behind a connector.

    A connector answers four questions about the data it points at: which
    tables exist, what a table's shape is, what its columns and sample rows
    are, and where its Parquet data can be read from.

    Which connector types an implementation serves is named by whoever
    registers it, not declared here. That is what lets one instance serve
    every type -- as a reader that forwards to a Kumo server does -- while a
    warehouse-native reader serves exactly one, without the two needing
    different contracts.

    Every method receives the **connector**, not just its ID. That is what
    lets a single implementation serve connectors backed by different systems:
    a cloud reader forwards ``connector.id`` to the Kumo service, which holds
    the target and its credentials, while a warehouse-native reader reads the
    connector's own target fields and resolves its identity to open a
    connection of its own.

    A table this backend cannot find is reported by raising
    ``kumo_core.sources.errors.SourceNotFoundError``, not by returning an
    empty answer and not by raising anything HTTP-shaped: an in-process plugin
    has no status code to speak in, and the SDK translates at its own edge.

    ``project_id`` defaults to the connector's own project. Callers that
    already hold an exact project pass it explicitly.

``sql`` reaches the three methods that can answer over a query rather than
    a named table. A host passes it only when a caller actually holds one, so
    a reader whose system cannot run SQL will never see a non-``None`` value
    -- but it must still accept the parameter, because that is what makes one
    reader substitutable for another.

    Implementations take :class:`SourceConnector` itself, and narrow
    :attr:`SourceConnector.config` inside the read rather than in the
    signature -- which is what lets them satisfy this protocol structurally
    rather than only in spirit.
    """
    def list_tables(
        self,
        connector: SourceConnector,
        project_id: Optional[str] = None,
    ) -> source_table_api.SourceTableListResponse:
        r"""Returns the source tables visible through a connector."""
        ...

    def get_table_config(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> source_table_api.SourceTableConfigResponse:
        r"""Returns one source table's provider-native configuration."""
        ...

    def get_columns(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> source_table_api.SourceTableDataResponse:
        r"""Returns column metadata for one source table, or for a query."""
        ...

    def get_sample(
        self,
        connector: SourceConnector,
        table_name: str,
        sample_rows: int,
        project_id: Optional[str] = None,
        sql: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> source_table_api.SourceTableDataResponse:
        r"""Returns column metadata and up to ``sample_rows`` rows.

        Built-in warehouse readers use bounded LIMIT queries, not random
        source sampling. Clustered samples can understate distinct counts used in semantic
        inference.

        Args:
            connector: The connector whose data is being read.
            table_name: The provider-native table to read.
            sample_rows: How many rows to return.
            project_id: Authorizing project, where one applies.
            sql: A read-only query to sample instead of a named table.
            timeout: Seconds the sample may run for. Rides with ``sql`` and
                only with it; physical samples use provider-specific row bounds.
        """
        ...

    def get_download_urls(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> List[str]:
        r"""Returns readable references to one source table's Parquet data."""
        ...

    def get_data_version(
        self,
        connector: SourceConnector,
        table_name: str,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        r"""Returns an identity for one source table's data, without staging.

        The same value :attr:`PrepareIngestResult.version` reports, obtained
        without exporting a row. Two calls over unchanged data must agree,
        since this is what decides whether an existing snapshot can be reused
        rather than rebuilt.

        Splitting this out of :meth:`prepare_ingest` is what lets an unchanged
        graph restore its index cheaply. A snapshot is identified by its
        inputs, so their versions must be known *before* it can be looked up --
        but establishing a version is a catalog read, while staging is a bulk
        unload. A caller that asks for the cheap half first pays the unload
        only when the index actually has to be rebuilt.

        Establishing it must not read rows: a warehouse answers from its
        catalog, an object store from its listing. A backend that can only
        learn the version by staging returns ``None`` instead, and its caller
        falls back to staging.

        Args:
            connector: The connector whose data is being identified.
            table_name: The provider-native table to identify.
            project_id: Authorizing project, where one applies.
        """
        ...

    def prepare_ingest(
        self,
        connector: SourceConnector,
        table_name: str,
        dest_dir: str,
        dtypes: Optional[Mapping[str, Dtype]] = None,
        project_id: Optional[str] = None,
        data_version: Optional[str] = None,
        sql: Optional[str] = None,
    ) -> PrepareIngestResult:
        r"""Stages one source table and describes how ingest should read it.

        Everything that must happen before ingest happens here: a warehouse
        backend creates its stage, unloads, and obtains access; a backend whose
        data is already files only describes them. What comes back says which
        of DiskGraph's source types to use, so a table read from a warehouse
        and a table read from an object store can sit in one graph.

        Saying *how* rather than handing back a local Parquet path is what
        keeps a warehouse source from being funnelled through this process: a
        table that a warehouse can export in parallel is described, not
        materialized here.

        Args:
            connector: The connector whose data is being staged.
            table_name: The provider-native table to read.
            dest_dir: Local scratch a backend *may* write to. A backend that
                stages inside its own warehouse ignores it.
            dtypes: Column name to :class:`~kumoapi.typing.Dtype`, as the
                *graph* declares them. These win over what the catalog reports,
                because a caller may knowingly override it -- reading a text
                column as a date is the common case, and ingest is entitled to
                find the type the graph promised.
            project_id: Authorizing project, where one applies.
            data_version: The version to report back, when the caller already
                established one. A caller that read :meth:`get_data_version`
                keyed its output on that stamp, so what is staged must be
                published under the same one rather than under a second stamp
                derived here. ``None`` asks the backend for its own.
            sql: A read-only query to stage instead of a named table.
        """
        ...


#: One snapshot's identity, carried across this contract without being read.
#: A type variable rather than a protocol on purpose: nothing here inspects a
#: handle, and a protocol would force every store to accept *any* handle --
#: which ``kumo_core``'s ``LocalSnapshotStore``, typed against its own concrete
#: ``DiskgraphSnapshotHandle``, correctly does not.
_Handle = TypeVar('_Handle', contravariant=True)


class IngestResult(Protocol):
    r"""What a completed ingest reports about the index it built.

    Satisfied by ``kumo_core``'s ``DiskgraphIngestResult``, which is also what
    a snapshot's ``_SUCCESS`` marker serialises.
    """
    index_path: str

    def missing_paths(self) -> List[str]:
        r"""The files this result promises that are not actually present."""


class SnapshotStore(Protocol[_Handle]):
    r"""The snapshot lifecycle an :class:`ArtifactLocation` drives.

    Satisfied by ``kumo_core``'s ``LocalSnapshotStore``. Its filesystem
    operations are injected rather than called directly, which is what lets one
    lifecycle serve local disk, a Volume, or an object store -- and why the
    callbacks appear in these signatures.
    """
    def path(self, handle: _Handle) -> str:
        r"""Where ingest built, or would build, this snapshot's index."""

    def durable_path(self, handle: _Handle, durable_root: str) -> str:
        r"""Where this snapshot is published under :obj:`durable_root`."""

    def verify(self, handle: _Handle) -> IngestResult:
        r"""Checks the index's integrity and reads its marker back."""

    def restore(
        self,
        handle: _Handle,
        durable_root: str,
        *,
        exists: Callable[[str], bool],
        copy_tree: Callable[[str, str], None],
    ) -> Optional[IngestResult]:
        r"""Copies a published snapshot back, or ``None`` if there is none."""

    def checkpoint(
        self,
        handle: _Handle,
        durable_root: str,
        *,
        exists: Callable[[str], bool],
        copy_tree: Callable[[str, str], None],
        replace_tree: Callable[[str, str], None],
    ) -> str:
        r"""Publishes a verified index, returning where it landed."""


@runtime_checkable
class ArtifactLocation(Protocol):
    r"""Where a verified DiskGraph snapshot is published, and read back from.

    Ingest builds an index under ``scratch``, which is reconstructable and so
    may not outlive the process. Publishing a verified copy somewhere durable
    is what lets the next run restore instead of spending the ingest again.
    Whether that is worth doing depends on the deployment, not on the code
    around it, which is why it is a seam rather than a branch.

    All three methods take the snapshot store and the handle rather than
    holding them, so one location serves every snapshot a run makes. The
    :attr:`root` they publish under is the location's own, which is what lets
    :class:`Storage` name a durable destination and a publication policy
    in one field rather than two that could disagree.
    """
    @property
    def root(self) -> str:
        r"""The durable snapshot root, which a location only ever reads. A
        plain attribute satisfies this; the protocol just does not ask for a
        setter. Local producers may use a separate filesystem working root
        when this is a remote URI; snapshots are published through the methods
        below.
        """

    def restore(self, store: SnapshotStore[_Handle],
                handle: _Handle) -> Optional[IngestResult]:
        r"""Returns the ingest result for :obj:`handle`, or ``None`` to build.

        Returning ``None`` is not an error: it is how a location says it has
        no usable copy, and the caller answers by ingesting.
        """

    def checkpoint(self, store: SnapshotStore[_Handle],
                   handle: _Handle) -> Optional[str]:
        r"""Publishes the freshly built index, returning where it landed."""

    def durable_path(self, store: SnapshotStore[_Handle],
                     handle: _Handle) -> Optional[str]:
        r"""Where a published snapshot lives, without publishing one."""


@runtime_checkable
class ArtifactSink(Protocol):
    r"""Writing a produced artifact into the system behind one connector.

    The mirror of :class:`SourceBackend`, and the second half of the reason
    those two areas take a connector rather than a path: what a connector
    points at decides how a table is *written* to it exactly as it decides how
    one is read. A sink is therefore selected the same way a reader is, by the
    connector's source type, and a backend registers the two side by side.

    Which type that is belongs to the registration, not to this contract. A
    host maps a type to an implementation -- in a mapping it was handed, or by
    the name of an entry point -- so the mapping is complete before any of
    these modules loads, one implementation may serve several types, and none
    can claim a type it does not serve.

    One method is required, because an export has one shape -- a frame, a
    destination table, and what to do about a table already sitting there.
    Anything a particular system needs beyond that it reads off the concrete
    connector it handles, as a reader does. A sink that also handles model
    bundles satisfies :class:`ModelPublishingSink`, :class:`ModelFetchingSink`
    or :class:`ModelListingSink` besides, each of which is a separate contract
    for exactly that reason: naming them here would make every table-only sink
    fail to be a sink.
    """
    def write_table(
        self,
        connector: SourceConnector,
        table_name: str,
        frame: pd.DataFrame,
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
    ) -> str:
        r"""Writes :obj:`frame` to :obj:`table_name`, and says where it landed.

        Args:
            connector: The connector naming the system and the container --
                database and schema, catalog and namespace -- to write into.
                The destination is the connector's, so a sink never invents
                one: writing somewhere else means binding another connector.
            table_name: The provider-native table to write, unqualified.
            frame: The artifact to write.
            write_mode: :attr:`~kumoapi.jobs.WriteMode.OVERWRITE` to replace
                the table's contents, :attr:`~kumoapi.jobs.WriteMode.APPEND`
                to add to them. Appending to a table that does not exist
                creates it.
            project_id: Authorizing project, where one applies.
            export_id: What identifies the artifact being written, for a
                destination that can tell a retry of this export from another
                one landing on it. Named as the hosted writer names it, whose
                ``write_artifact_to_s3`` takes the same argument for the same
                reason. A sink whose destination is a table ignores it: a
                warehouse write is one statement, and there is no partial
                result for a rerun to recognise.

                Added after this protocol first shipped, so a sink that omits
                it is still called -- the export area passes it only to a
                ``write_table`` that accepts it, by name or through
                ``**kwargs``. Such a sink is told nothing about what it is
                writing, which for a directory destination means it cannot
                recognise its own earlier attempt at the same export.

        Returns:
            The fully qualified name of the table written, for a caller to
            report and for a test to assert on.
        """
        ...


@runtime_checkable
class FileArtifactSink(Protocol):
    r"""Optional bounded-file output capability for prediction artifacts.

    Dataframe sinks remain valid ArtifactSink implementations. Batch output
    requires this capability instead of silently collecting all output rows.
    """
    def write_files(
        self,
        connector: SourceConnector,
        table_name: str,
        files: List[str],
        write_mode: WriteMode = WriteMode.OVERWRITE,
        project_id: Optional[str] = None,
        *,
        export_id: Optional[str] = None,
        extra_columns: Optional[Mapping[str, Any]] = None,
    ) -> str:
        r"""Write exactly these Parquet files, stamping only requested columns."""
        ...


@runtime_checkable
class FileStoreSink(Protocol):
    r"""A sink that also holds files: snapshot exports, model bundles and
    export manifests go through its filesystem."""
    def holds_files(self, connector: SourceConnector, path: str) -> bool:
        r"""Whether ``path`` is a file location of this sink rather than a
        table name; decided from the path alone, without I/O."""
        ...

    def filesystem(self, connector: SourceConnector,
                   path: str) -> Tuple[Any, str]:
        r"""A ``pyarrow.fs.FileSystem`` and ``path`` resolved on it, for a
        path :meth:`holds_files` accepts."""
        ...


@runtime_checkable
class ModelPublishingSink(ArtifactSink, Protocol):
    r"""A sink that can also receive a trained model bundle.

    Separate from :class:`ArtifactSink` rather than an optional method on it,
    because "optional" is not something a protocol can say: a sink that writes
    tables and nothing else -- which most do -- would fail to satisfy a
    contract that named this. A host that has a bundle to publish asks for
    this one, and :func:`isinstance` answers.
    """
    def publish_model(
        self,
        connector: SourceConnector,
        bundle_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        metadata: Mapping[str, str] = {},
        metrics: Mapping[str, float] = {},
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        r"""Publishes a trained model bundle, and says where it landed.

        Optional. A sink that does not define this one cannot receive a model,
        and the export area reports that by name rather than letting the call
        fail as a missing attribute.

        Separate from :meth:`write_table` rather than a mode of it, because a
        frame going to a table and a directory going to a registry share only
        the connection: different destination, different addressing, different
        result.

        Args:
            connector: The connector naming the system to publish into. As with
                a table, the container is the connector's own.
            bundle_dir: Local directory holding the published model bundle,
                including the manifest an importer verifies it against.
            path: Destination for a path-addressed target. Mutually exclusive
                with :obj:`location`; a sink handles whichever its connector
                addresses by and refuses the other.
            location: A :class:`ModelRegistryLocation` for a
                registry-addressed target.
            metadata: Flat tags to record on the published model, so it can be
                found without being downloaded. Derived from the bundle, and
                never read back: the bundle stays the one source of truth.
            metrics: Numeric metrics to record natively, where a target has a
                notion of them.
            project_id: Authorizing project, where one applies.

        Returns:
            The name, version, and URI the destination assigned.
        """
        ...


@runtime_checkable
class ModelFetchingSink(ArtifactSink, Protocol):
    r"""A sink a published bundle can be read back out of.

    The read direction of :class:`ModelPublishingSink`, and separate from it:
    publishing where nothing imports is useful, and it keeps a write-only
    target honest about what it serves.
    """
    def fetch_model(
        self,
        connector: SourceConnector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        r"""Downloads a published model bundle into :obj:`local_dir`.

        Optional, and the read direction of :meth:`publish_model`. A sink may
        define one without the other: publishing where nothing imports is
        useful, and it keeps a write-only target honest about what it serves.

        Args:
            connector: The connector naming the system to read from.
            local_dir: Directory to place the bundle in. Created if absent.
            path: Source for a path-addressed target.
            location: A registry location. A version of ``None`` selects the
                registry's choice (see :class:`ModelRegistryLocation`), which
                is resolved here and reported back.
            project_id: Authorizing project, where one applies.

        Returns:
            The name, version, and URI actually read, with any ``None`` version
            resolved to the one that was fetched.
        """
        ...


@runtime_checkable
class ModelAliasingSink(ArtifactSink, Protocol):
    r"""A registry sink that sets aliases and fetches by
    :attr:`ModelRegistryLocation.alias`."""
    def set_model_alias(
        self,
        connector: SourceConnector,
        location: ModelRegistryLocation,
        alias: str,
    ) -> None:
        r"""Points :obj:`alias` at ``location.version``, moving it if held."""
        ...


def check_model_alias(alias: str) -> None:
    r"""Raises :class:`ValueError` unless every model registry can set
    :obj:`alias`: a letter or underscore, then letters, digits or underscores
    (at most 255), and not ``DEFAULT``, ``FIRST``, ``LAST``, ``LATEST`` or
    ``v<digits>`` in any case."""
    if (not isinstance(alias, str) or not _ALIAS.fullmatch(alias)
            or _RESERVED_ALIAS.fullmatch(alias)):
        raise ValueError(
            f'{alias!r} cannot be set as a model alias: use letters, digits '
            f'and underscores, starting with a letter or underscore, and '
            f'avoid the reserved DEFAULT, FIRST, LAST, LATEST and v<digits> '
            f'(for DEFAULT use ALTER MODEL ... SET DEFAULT_VERSION).')


def registry_lineage(recorded: Mapping[str, Any]) -> Dict[str, str]:
    r"""The ``kumo_*`` entries of a version's registry metadata, such as
    :data:`BUNDLE_ID_KEY`, as strings and without :data:`ENGINE_MARKER`."""
    return {
        key: str(value)
        for key, value in recorded.items()
        if key.startswith(REGISTRY_KEY_PREFIX) and key != ENGINE_MARKER
    }


@runtime_checkable
class ModelListingSink(ArtifactSink, Protocol):
    r"""A sink that can enumerate the models published through it.

    Separate again, because a catalog is what makes this answerable: a
    registry has one, and a path-addressed store does not.
    """
    def list_models(
        self,
        connector: SourceConnector,
        *,
        project_id: Optional[str] = None,
    ) -> List[PublishedModel]:
        r"""Lists model versions this SDK published through the connector.

        Optional. A registry sink implements this when it can identify Kumo
        versions from catalog metadata without downloading every bundle.
        Path-addressed stores have no catalog and therefore do not.
        """
        ...
