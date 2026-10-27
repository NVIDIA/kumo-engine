# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Publishing a trained model bundle to the Snowflake Model Registry.

The third thing this package does with one session, after reading a table and
writing one back. It is a separate module rather than more of ``sink`` because
it is the only part that needs ``snowflake-ml-python``, a far larger tree than
the two client libraries the rest of the package uses. That library is
imported inside the two methods below, so this module loading with the sink
costs nothing: it is reached only when a model actually moves.

With kumo-engine installed (it registers a builder under
:data:`MODEL_BUILDER_GROUP`), a version is a runnable Kumo model: it carries the vital
bundle and scores entity ids against an exported graph snapshot in Snowpark
Container Services, e.g. with ``ModelVersion.run_batch`` on a GPU compute
pool. Without kumo-engine (a hosted export), the bundle travels in
:class:`_BundleModel`, a carrier that refuses to score. Either way the
registry also buys versioning, lineage, discovery, and the account's own
access control.

The carrier declares its signature explicitly rather than supplying sample
data, because supplying sample data makes the packager infer a signature by
*calling* the model -- and a carrier that answered with plausible-looking
numbers would be worse than one that refuses.

An import downloads files and hands them to ``kumo_core``, so the bundle is
found by :func:`_locate_bundle` looking for the spec the export wrote, rather
than at the path the packager stores artifacts under today. That path is
snowflake-ml's private layout; the marker is ours.
"""

import json
import logging
import math
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from kumo_snowflake.session import (
    SnowflakeSession,
    _require_snowpark,
    _target_config,
    resolved,
)
from kumoapi.plugin import (
    BUNDLE_ID_KEY,
    ENGINE_MARKER,
    ModelRegistryLocation,
    PublishedModel,
    SourceConnector,
    check_model_alias,
    registry_lineage,
)

if TYPE_CHECKING:
    # ModelVersion is used by quoted annotations, which Vulture cannot resolve.
    from snowflake.ml.model import ModelVersion  # noqa: F401
    from snowflake.ml.model import custom_model, model_signature
    from snowflake.ml.registry import Registry

#: The file an export writes into every bundle, and so the marker that says a
#: directory in a downloaded tree *is* the bundle. Kept in step with
#: :data:`~kumoai_studio.backend.local.export._SPEC_FILE`, which writes it.
_SPEC_FILE = 'training-spec.json'

#: A second marker, checked alongside the first. ``kumo_core`` writes it, so a
#: directory carrying both is a published bundle rather than something that
#: merely happens to be named alike.
_MANIFEST_FILE = 'publication-manifest.json'

_SYSTEM_ALIASES = frozenset({'DEFAULT', 'FIRST', 'LAST'})
_IDENTIFIER = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:[^"]|"")+")'
_MODEL_NAME = re.compile(rf'(?:{_IDENTIFIER}\.){{0,2}}{_IDENTIFIER}')

_LOG = logging.getLogger(__name__)

#: Entry-point group of the callable that turns a bundle into ``log_model``
#: arguments for a runnable version; kumo-engine registers one.
MODEL_BUILDER_GROUP = 'kumo_snowflake.registry_models'

#: Where a fetched version is unpacked before its bundle is lifted out. Named
#: rather than a temporary directory so a half-finished import leaves something
#: a reader can look at.
_EXPORT_DIR = '_registry-export'

#: How a version comment's line of holdout metrics begins.
_METRICS_LINE = 'Holdout: '


def _require_registry() -> type["Registry"]:
    r"""Returns the ``Registry`` class, or explains how to install it."""
    try:
        from snowflake.ml.registry import Registry
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "Publishing a model to the Snowflake Model Registry requires "
            "'snowflake-ml-python', which is not installed. Install the "
            "registry extra with `pip install 'kumo-snowflake[sdk]'`, or "
            "`pip install 'kumoai-studio[snowflake]'`, which pulls it."
        ) from error
    return Registry


def _bundle_model(bundle_dir: str) -> "custom_model.CustomModel":
    r"""Builds the carrier a bundle travels into the registry as.

    Defined inside the function so importing this module does not import
    ``snowflake.ml``: the class subclasses the library's own, so it cannot
    exist at module scope without it.
    """
    import pandas as pd
    from snowflake.ml.model import custom_model

    class _BundleModel(custom_model.CustomModel):
        r"""A Kumo model bundle, in the shape the registry can hold one.

        Deliberately not runnable. Scoring a Kumo model needs a materialized
        DiskGraph task over the graph being scored, built where the data is, so
        this refuses rather than answering. The version exists to carry,
        version and expose the bundle. Load it with
        ``Trainer.load_from_connector(SnowflakeConnector(...),
        location=ModelRegistryLocation(...))`` and predict against a snapshot
        (kumo-engine), or with :func:`kumoai_studio.import_model` (hosted
        service).
        """
        # Unquoted, and evaluated here rather than deferred: the registry
        # compares this annotation against `pandas.DataFrame` itself, so a
        # string it never resolves is rejected.
        @custom_model.inference_api
        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError(
                "This model version carries a Kumo model bundle; it does not "
                "score in Snowflake. Load it with Trainer.load_from_connector("
                "SnowflakeConnector(...), location=ModelRegistryLocation(...)) "
                "and predict against a snapshot (kumo-engine), or "
                "kumoai_studio.import_model() (hosted service).")

    return _BundleModel(
        custom_model.ModelContext(artifacts={'bundle': bundle_dir}))


#: Snowflake ML task types by Kumo task type; Model Monitor needs one.
_TASKS = {
    'binary_classification': 'TABULAR_BINARY_CLASSIFICATION',
    'multiclass_classification': 'TABULAR_MULTI_CLASSIFICATION',
    'regression': 'TABULAR_REGRESSION',
}


def _task(bundle: Path) -> Dict[str, Any]:
    r"""``task=`` for ``log_model`` from the bundle's prediction schema;
    empty for other tasks, or when it cannot be read."""
    try:
        from snowflake.ml.model import type_hints

        schema = json.loads((bundle / 'prediction-schema.json').read_text())
        name = _TASKS.get(schema.get('task_type'))
        return {} if name is None else {'task': getattr(type_hints.Task, name)}
    except Exception:
        return {}


def _carrier_signature() -> Dict[str, "model_signature.ModelSignature"]:
    r"""Declares the carrier's signature instead of inferring one.

    Inference needs sample data, and the packager infers by calling the model
    with it -- which this carrier refuses to answer. A declared signature takes
    the packager down the branch that validates rather than predicts, so
    nothing ever invokes it.
    """
    from snowflake.ml.model import model_signature

    return {
        'predict':
        model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(
                    name='entity', dtype=model_signature.DataType.STRING)
            ],
            outputs=[
                model_signature.FeatureSpec(
                    name='score', dtype=model_signature.DataType.DOUBLE)
            ],
        )
    }


def _model_args(bundle: Path, metadata: Mapping[str, str],
                connector: SourceConnector, workdir: str) -> Dict[str, Any]:
    r"""``log_model`` arguments for a runnable model when an installed host
    provides one (kumo-engine does), else for the carrier."""
    try:
        build = _model_builder()
        if build is not None:
            return build(str(bundle), metadata, _serving_target(connector),
                         workdir)
    except Exception as error:
        _LOG.warning(
            'Registering the bundle as a version that does not score in '
            'Snowflake: %s', error)
    return _carrier_args(bundle)


def _model_builder() -> Optional[Callable[..., Dict[str, Any]]]:
    from importlib.metadata import entry_points

    for entry in entry_points(group=MODEL_BUILDER_GROUP):
        return entry.load()
    return None


def _carrier_args(bundle: Path) -> Dict[str, Any]:
    return {
        'model': _bundle_model(str(bundle)),
        'signatures': _carrier_signature()
    }


def _serving_target(connector: SourceConnector) -> Dict[str, Optional[str]]:
    config = _target_config(connector)
    return {
        'warehouse': config.warehouse,
        'database': config.database,
        'schema': config.schema_name,
    }


def _version_holding(registry: "Registry", model_name: str,
                     bundle_id: Optional[str]) -> Optional["ModelVersion"]:
    r"""The Kumo version of ``model_name`` that records ``bundle_id``."""
    if not bundle_id:
        return None
    try:
        versions = registry.get_model(model_name).versions()
    except Exception:
        return None
    for version in versions:
        try:
            recorded = dict(version.show_metrics() or {})
        except Exception as error:
            _LOG.warning('Could not read registry metadata: %s', error)
            continue
        if (recorded.get(ENGINE_MARKER) == '1'
                and recorded.get(BUNDLE_ID_KEY) == bundle_id):
            return version
    return None


def _locate_bundle(root: Path) -> Path:
    r"""Finds the published bundle inside a downloaded model tree.

    By marker rather than by path. snowflake-ml stores a custom model's
    artifacts under a private layout it is free to change; the two files a
    bundle always carries are ours, so searching for them keeps an import
    working across a library upgrade.

    Raises:
        ValueError: If no bundle is there, which means this version was not
            published by this SDK.
    """
    for spec in sorted(root.rglob(_SPEC_FILE)):
        if (spec.parent / _MANIFEST_FILE).is_file():
            return spec.parent
    raise ValueError(
        f"No Kumo model bundle under {root}: no directory holds both "
        f"{_SPEC_FILE} and {_MANIFEST_FILE}. This model version was not "
        f"published by kumoai_studio.export_model().")


def _model_uri(connector: SourceConnector, version: "ModelVersion") -> str:
    r"""Returns the fully qualified name of one model version."""
    name = getattr(version, 'fully_qualified_model_name', None)
    if not name:
        name = (f'{resolved(_target_config(connector).database)}.'
                f'{resolved(_target_config(connector).schema_name)}.'
                f'{version.model_name}')
    return f'{name}/{version.version_name}'


def _published(
    connector: SourceConnector,
    version: "ModelVersion",
    *,
    metadata: Mapping[str, str] = {},
    metrics: Mapping[str, float] = {},
) -> PublishedModel:
    return PublishedModel(
        name=str(version.model_name),
        version=str(version.version_name),
        uri=_model_uri(connector, version),
        metadata=dict(metadata),
        metrics=dict(metrics),
    )


def _refuse_path(path: Optional[str], action: str) -> None:
    if path is not None:
        raise ValueError(
            f'A Snowflake connector names a registry, which addresses a model '
            f'by name, so {action} takes a ModelRegistryLocation rather than '
            f'a path.')


def _version_record(metadata: Mapping[str, str]) -> Dict[str, str]:
    record = {ENGINE_MARKER: '1'}
    if metadata.get(BUNDLE_ID_KEY):
        record[BUNDLE_ID_KEY] = str(metadata[BUNDLE_ID_KEY])
    return record


def _finite(metrics: Mapping[str, float]) -> Dict[str, float]:
    return {name: float(value) for name, value in metrics.items()
            if math.isfinite(value)}


def _comment(user: Optional[str],
             metrics: Mapping[str, float]) -> Optional[str]:
    r"""The user's comment, then a line of the model's holdout metrics."""
    line = ' · '.join(f'{name} {value:.3g}'
                      for name, value in _finite(metrics).items())
    return '\n'.join(text for text in (user, line and _METRICS_LINE + line)
                     if text) or None


def _backfill(version: "ModelVersion", metrics: Mapping[str, float],
              comment: Optional[str]) -> None:
    r"""Records ``metrics`` on a reused ``version``, in its metadata and as
    its comment line after ``comment`` (`None`: the version's own comment),
    leaving what it already holds; a refused change warns, since it needs
    OWNERSHIP of the model."""
    try:
        recorded = dict(version.show_metrics() or {})
        for name, value in _finite(metrics).items():
            if recorded.get(name) != value:
                version.set_metric(name, value)
        current = version.comment or None
        if comment is None:
            comment = '\n'.join(
                line for line in (current or '').split('\n')
                if not line.startswith(_METRICS_LINE))
        wanted = _comment(comment, metrics)
        if wanted is not None and wanted != current:
            version.comment = wanted
    except Exception as error:
        _LOG.warning(
            'Could not record the holdout metrics on %s version %s, which '
            'needs OWNERSHIP of the model: %s', version.model_name,
            version.version_name, error)


def _recorded_lineage(version: "ModelVersion") -> Dict[str, str]:
    try:
        return registry_lineage(dict(version.show_metrics() or {}))
    except Exception as error:
        _LOG.warning('Could not read registry metadata: %s', error)
        return {}


def _alias_holders(model: Any) -> Dict[str, str]:
    holders = {}
    for row in model.show_versions().to_dict('records'):
        for alias in json.loads(row.get('aliases') or '[]'):
            holders[alias] = str(row['name'])
    return holders


def _restore_alias(model: Any, version: str, alias: str) -> None:
    try:
        model.version(version).set_alias(alias)
    except Exception as error:
        _LOG.warning('Alias %s is now unset: restoring it on version %s '
                     'failed: %s', alias, version, error)
    else:
        _LOG.warning('Could not move alias %s; it stays on version %s.',
                     alias, version)


class SnowflakeModelRegistryMixin:
    r"""The model half of :class:`SnowflakeSinkBackend`.

    Mixed in rather than written inline so the module holding it -- and the
    ``snowflake.ml`` import it needs -- is loaded only when a model actually
    moves. It reaches the warehouse through the same
    :class:`~kumo_snowflake.session._SnowflakeSessions`
    resolution the table path uses, so a model is published under the identity
    its connector names and not some other one.
    """
    if TYPE_CHECKING:
        # The one thing this mixin needs from its host, declared rather than
        # reached for blindly: every class that mixes it in also inherits
        # `_SnowflakeSessions`, which supplies the implementation.
        def _session(
            self,
            connector: SourceConnector,
        ) -> ContextManager[SnowflakeSession]:
            ...

    def check_model_location(self, connector: SourceConnector,
                             location: ModelRegistryLocation) -> None:
        r"""Raises :class:`ValueError` unless ``location.model_name`` is a
        Snowflake identifier."""
        del connector
        if not _MODEL_NAME.fullmatch(location.model_name):
            raise ValueError(
                f'Snowflake cannot register a model named '
                f'{location.model_name!r}: start with a letter or underscore '
                f'and use letters, digits, underscores and $, or double-quote '
                f"the name, e.g. 'USERS_CHURN'.")

    @contextmanager
    def _registry(self, connector: SourceConnector) -> Iterator["Registry"]:
        r"""A registry statement is a session statement, so the caller keeps
        the warehouse for as long as it is asking the registry anything.
        """
        with self._session(connector) as session:
            _require_snowpark(session)
            yield _require_registry()(
                session=session,
                database_name=resolved(_target_config(connector).database),
                schema_name=resolved(_target_config(connector).schema_name),
            )

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
        r"""Registers :obj:`bundle_dir` as a version of a registry model.

        The database and schema are the connector's, as they are for a table,
        so only the model name is given here. A Kumo version of the model
        that already records this ``kumo_bundle_id`` is returned instead of
        registering the bundle again.

        Args:
            connector: The Snowflake target whose schema holds the registry.
            bundle_dir: The published model bundle to carry into the version.
            path: Refused. A registry addresses a model by name.
            location: The
                :class:`~kumoapi.plugin.ModelRegistryLocation` naming
                the model, and optionally the version. A version of ``None``
                lets the registry assign one -- the usual case, and the reason
                this reports back what it created.
            metadata: Returned with the publication result; its
                ``kumo_bundle_id`` is also recorded in the version's metrics.
                ``kumo.serving_snapshot`` and ``kumo.serving.<option>``
                configure a runnable version.
            metrics: The model's holdout metrics, recorded in the version's
                metrics and as a line after ``location.comment``; also
                returned with the publication result.
            project_id: Ignored; a local runtime has no projects.

        Returns:
            The model name, the version written or reused, and where it
            lives.
        """
        del project_id
        _refuse_path(path, 'publishing')
        if location is None:
            raise ValueError(
                'Publishing to a registry needs a model name; pass '
                'output_location=ModelRegistryLocation(...).')
        bundle = Path(bundle_dir)
        if not (bundle / _SPEC_FILE).is_file():
            raise ValueError(
                f"{bundle} does not look like a published model bundle: it "
                f"has no {_SPEC_FILE}, which an export writes into every one.")

        record = {**_finite(metrics), **_version_record(metadata)}
        with tempfile.TemporaryDirectory() as workdir, \
                self._registry(connector) as registry:
            version = _version_holding(registry, location.model_name,
                                       metadata.get(BUNDLE_ID_KEY))
            if version is not None:
                _LOG.info('%s version %s already holds bundle %s; reused it',
                          location.model_name, version.version_name,
                          metadata.get(BUNDLE_ID_KEY))
                _backfill(version, metrics, location.comment)
            else:
                version = registry.log_model(
                    model_name=location.model_name,
                    version_name=location.version,
                    comment=_comment(location.comment, metrics),
                    metrics=record,
                    **_model_args(bundle, metadata, connector, workdir),
                    **_task(bundle),
                )
        return _published(connector, version, metadata=metadata,
                          metrics=metrics)

    def list_models(
        self,
        connector: SourceConnector,
        *,
        project_id: Optional[str] = None,
    ) -> List[PublishedModel]:
        r"""Lists Kumo model versions visible in the connector's registry.

        A Kumo export records ``kumo_engine=1`` in registry metadata.
        That marker makes discovery both cheap and precise:
        versions logged by another tool are omitted, and no bundle is
        downloaded merely to decide whether it belongs in this list.
        ``metadata`` carries the recorded ``kumo_bundle_id``.
        """
        del project_id
        found: List[PublishedModel] = []
        with self._registry(connector) as registry:
            for model in registry.models():
                for version in model.versions():
                    recorded = dict(version.show_metrics() or {})
                    if recorded.get(ENGINE_MARKER) != '1':
                        continue
                    found.append(
                        _published(connector, version,
                                   metadata=registry_lineage(recorded)))
        return sorted(found, key=lambda item: (item.name, item.version))

    def fetch_model(
        self,
        connector: SourceConnector,
        local_dir: str,
        *,
        path: Optional[str] = None,
        location: Optional[ModelRegistryLocation] = None,
        project_id: Optional[str] = None,
    ) -> PublishedModel:
        r"""Downloads a published bundle, leaving it at :obj:`local_dir`.

        What lands is the bundle itself, lifted out of the tree the registry
        exports, so a caller gets back the directory that was published rather
        than a library layout wrapped around it.

        Args:
            connector: The Snowflake target whose schema holds the registry.
            local_dir: Where the bundle should end up. Replaced if it exists,
                though the caller always hands this a fresh, empty
                ``mkdtemp``.
            path: Refused, as in :meth:`publish_model`.
            location: The model, and optionally the version or an alias. With
                neither, takes the registry's default version, which is what a
                caller who names none means.
            project_id: Ignored.

        Returns:
            The name, the version actually read -- resolved, when the caller
            named none -- where it lives, metrics read from metrics.json, and
            the version's recorded ``kumo_bundle_id`` as ``metadata``.
        """
        del project_id
        _refuse_path(path, 'fetching')
        if location is None:
            raise ValueError(
                'Fetching from a registry needs a model name; pass a '
                'ModelRegistryLocation.')

        destination = Path(local_dir)
        with self._registry(connector) as registry:
            # Resolved before anything on disk is touched: a name the registry
            # does not know must not cost the caller the bundle already there.
            model = registry.get_model(location.model_name)
            alias = getattr(location, 'alias', None)
            if alias is not None:
                if resolved(alias) not in _alias_holders(model):
                    raise ValueError(
                        f'No version of {location.model_name} holds alias '
                        f'{alias}.')
                version = model.version(alias)
            elif location.version is None:
                version = model.default
            else:
                version = model.version(location.version)
            recorded = _recorded_lineage(version)

            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.parent / f'{destination.name}{_EXPORT_DIR}'
            # Re-create export staging so an import cannot read stale files.
            if staged.exists():
                shutil.rmtree(staged)
            staged.mkdir(parents=True)
            version.export(str(staged))

        bundle = _locate_bundle(staged)
        metrics_path = bundle / 'metrics.json'
        metrics = json.loads(
            metrics_path.read_text()) if metrics_path.is_file() else {}
        if (not isinstance(metrics, dict) or not all(
                isinstance(key, str) and isinstance(value, (int, float))
                for key, value in metrics.items())):
            raise ValueError('Invalid training metrics in model bundle.')
        published = _published(connector, version, metadata=recorded,
                               metrics=metrics)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(bundle), str(destination))
        shutil.rmtree(staged, ignore_errors=True)
        return published

    def set_model_alias(
        self,
        connector: SourceConnector,
        location: ModelRegistryLocation,
        alias: str,
    ) -> None:
        r"""Points :obj:`alias` at ``location.version``, moving it if held.

        Snowflake allows one alias per version. A failed move puts the alias
        back on the version that held it.
        """
        if location.version is None:
            raise ValueError('Setting an alias needs a ModelRegistryLocation '
                             'naming an exact version.')
        check_model_alias(alias)
        wanted = resolved(alias)
        with self._registry(connector) as registry:
            model = registry.get_model(location.model_name)
            target = model.version(location.version)
            name = resolved(str(target.version_name))
            holders = {
                held: version
                for held, version in _alias_holders(model).items()
                if held not in _SYSTEM_ALIASES
            }
            if holders.get(wanted) == name:
                return
            held = sorted(a for a, v in holders.items() if v == name)
            if held:
                raise ValueError(
                    f'Version {name} of {location.model_name} already holds '
                    f'alias {held[0]}; unset it first.')
            previous = holders.get(wanted)
            if previous is not None:
                model.version(previous).unset_alias(alias)
            try:
                target.set_alias(alias)
            except Exception:
                if previous is not None:
                    _restore_alias(model, previous, alias)
                raise
