# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Importing a published model back into a process that never trained it.

The mirror of :func:`~kumoai_studio.artifact_export.job.export_model`, and the reason
a published bundle carries its own spec: the process doing the importing has
none of what training left behind -- no job, no query, no model plan -- so
everything a model needs to be usable again has to have travelled with it.

Local only, deliberately. The Kumo service holds every training job itself and
addresses one by ID, so it has nothing to import *into*; a hosted backend says
so rather than pretending.

It sits at the top level rather than in :mod:`kumoai_studio.artifact_export` because
of what it returns. Exporting hands back a job; importing hands back a
:class:`~kumoai_studio.trainer.Trainer` and a
:class:`~kumoai_studio.pquery.PredictiveQuery`, which are the two things every layer
below is written in terms of. A function producing them belongs above them
rather than beside the configs it takes -- and putting it beside them made
``artifact_export`` import ``pquery``, which imports ``trainer``, which imports
``artifact_export``.
"""

from typing import Any, List, Optional, Tuple, cast

from kumoapi.plugin import PublishedModel

from kumoai_studio.artifact_export.config import ModelImportConfig
from kumoai_studio.backend import cloud_backend
from kumoai_studio.connector.base import Connector
from kumoai_studio.pquery.predictive_query import PredictiveQuery
from kumoai_studio.trainer.trainer import Trainer


def list_published_models(
    connector: Connector,
    project_id: Optional[str] = None,
) -> List[PublishedModel]:
    r"""List durable Kumo model versions available through a registry.

    Unlike local training and export job IDs, registry coordinates survive the
    process. The returned name and version can be passed to
    :func:`import_model` through :class:`~kumoai_studio.ModelRegistryLocation`.

    Only versions carrying Kumo's catalog projection are returned; unrelated
    models in the same registry are omitted without downloading them.
    """
    connector_project_id = connector.kumo_project_id
    if project_id is None:
        project_id = connector_project_id
    elif project_id != connector_project_id:
        raise ValueError(f'Registry connector belongs to project '
                         f'{connector_project_id!r}, not {project_id!r}.')
    return cloud_backend().artifact_export.list_models(connector.id,
                                                       project_id=project_id)


def import_model(
    config: ModelImportConfig,
    project_id: Optional[str] = None,
) -> Tuple[Trainer, PredictiveQuery]:
    r"""Import a published model, and the query it answers.

    The counterpart of :func:`~kumoai_studio.export_model`, and the same pair a model
    trained on the Kumo platform is reached by --
    :meth:`~kumoai_studio.trainer.Trainer.load` beside
    :meth:`~kumoai_studio.pquery.PredictiveQuery.load_from_training_job`. An imported
    model is therefore used exactly like one that was never exported.

    The query carries the graph, rebuilt from the definition in the bundle over
    the connector named in the config. What the bundle records is what the
    graph *is* -- tables, columns, keys and links; where that data lives now is
    the caller's to say, which is why ``graph_connector`` is required.

    .. note::
        On the local backend
        :meth:`~kumoai_studio.pquery.PredictiveQuery.generate_prediction_table` is not
        available: the entity set is materialized inside the prediction run
        from the query's ``FOR EACH`` clause. Pass no prediction table to
        :meth:`~kumoai_studio.trainer.Trainer.predict` and let the run build one.

    .. code-block:: python

        import kumoai_studio

        registry = kumoai_studio.SnowflakeConnector(
            name="ml-registry",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ML",
        )
        warehouse = kumoai_studio.SnowflakeConnector(
            name="analytics",
            warehouse="COMPUTE_WH",
            database="PROD",
            schema_name="ANALYTICS",
        )
        trainer, pquery = kumoai_studio.import_model(kumoai_studio.ModelImportConfig(
            connector=registry,
            location=kumoai_studio.ModelRegistryLocation("CHURN_GNN", version="V3"),
            graph_connector=warehouse,
        ))

        job = trainer.predict(pquery.graph, prediction_time=datetime.now())

    Args:
        config: A :class:`~kumoai_studio.artifact_export.ModelImportConfig` naming the
            connector and the model to read.
        project_id: Exact project owning the connector. ``None`` uses the
            connector's own, which never changes global SDK state.

    Returns:
        The trainer holding the imported model, and the predictive query it
        answers, over a graph bound to ``config.graph_connector``.

    Raises:
        :class:`NotImplementedError`: On a backend with no import path, such as
            the Kumo service.
        :class:`ValueError`: If the bundle did not survive the transfer, was
            not published by this SDK, or carries a spec version this SDK
            cannot read.
    """
    from kumoai_studio.graph import Graph

    connector_project_id = config.kumo_project_id
    if project_id is None:
        project_id = connector_project_id
    elif project_id != connector_project_id:
        raise ValueError(
            f'Source connector belongs to project {connector_project_id!r}, '
            f'not {project_id!r}.')

    api = cloud_backend().artifact_export
    job_id = api.import_model(config.connector.id, path=config.path,
                              location=config.location, project_id=project_id)
    # `graph_definition`/`query_for_job` are reconstruction helpers the
    # TrainingBackend contract does not expose, since a hosted job otherwise
    # never needs to be adopted back into a process.
    training = cast(Any, cloud_backend().training)
    assert config.graph_connector is not None  # ModelImportConfig validates it
    graph = Graph.from_definition(training.graph_definition(job_id),
                                  config.graph_connector,
                                  project_id=project_id)
    pquery = PredictiveQuery(graph=graph, query=training.query_for_job(job_id))
    return Trainer.load(job_id, project_id=project_id), pquery
