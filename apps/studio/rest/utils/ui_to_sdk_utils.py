# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

import kumoapi.graph as graph_api
import kumoapi.jobs as jobs_api
from kumo_core.types import TaskType

from io_utils import logging
from rest.config.data_backend import ConnectorSpec
from rest.config.prediction_output import PredictionOutputConfig
from rest.config.table_v2 import DataSourceType, TableReferenceMetadata
from statedb.config.common import GeneratePredictionTableJobID

logger = logging.getLogger(__name__)

DISKGRAPH_FORECASTING_BP_UNSUPPORTED_MESSAGE = (
    "DiskGraph forecasting batch prediction is not yet supported by SDK replay. "
    "Skipping before training to avoid a known unsupported prediction-table "
    "generation failure.")


def is_diskgraph_forecasting_batch_prediction_sdk_unsupported(
    is_diskgraph: bool,
    task_type: Union[TaskType, str],
    forecast_length: Optional[int] = None,
) -> bool:
    task_type_value = getattr(task_type, "value", task_type)
    is_forecasting_plan = (forecast_length is not None and forecast_length > 1)
    return is_diskgraph and (task_type_value == TaskType.FORECASTING.value
                             or is_forecasting_plan)


# ----------- Generate script methods -----------
def generate_conn_graph_or_table_sdk_script(
    connector_configs: List[Union[ConnectorSpec, Dict[str, Any]]],
    table_configs: Optional[List[TableReferenceMetadata]] = None,
    edge_configs: Optional[List[graph_api.Edge]] = None,
    *,
    project_id: str,
    include_header: bool = True,
) -> str:
    """
    Based on the connector, table, and edge configs
    generate the final, fully executable script to load
    a connector, a table, or a graph in the SDK.

    Args:
        connector_configs: List[Union[ConnectorSpec, Dict[str, Any]]]
            Type union to support S3 prediction export paths that are
            not persisted as ConnectorDBEntry in the DB.
        table_configs: List[TableReferenceMetadata]
        edge_configs: Optional[List[graph_api.Edge]]:
            Only used to generate graph script.

    Response:
        str: The final python script.
    """
    final_script = ""
    if include_header:
        header = generate_header(project_id)
        final_script = header
    tables = {}
    project_upload_connector_ids = set()

    # Generate connector(s) script
    for connector in connector_configs:
        if isinstance(connector, ConnectorSpec):
            connector_config = connector.connector_config
        else:
            connector_config = connector

        if connector_config.get("is_project_upload"):
            project_upload_connector_ids.add(connector_config["connector_id"])
            continue
        connector_script = generate_connector_script(connector_config)
        final_script += connector_script

    if table_configs:
        for table in table_configs:
            tables[
                table.source_table_name] = "\\" + table.source_table_name + "\\"
            final_script += generate_table_script(
                table.table_config,
                is_project_upload=(table.table_config["connector_id"]
                                   in project_upload_connector_ids),
            )

    if edge_configs:
        edge_confs_as_dicts = [
            recursively_convert_dataclass_to_dict(edge) for edge in edge_configs
        ]
        # Generate graph portion
        graph_script = f"""graph = kumo.Graph(
    tables={tables},
    edges={edge_confs_as_dicts},
)

graph.validate()
"""
        return final_script + graph_script

    return final_script


def generate_header(project_id: str):
    script = f"""import kumoai_studio as kumo
from kumoai_studio.artifact_export import OutputConfig
from kumoai_studio.graph import Column
from kumoapi.typing import Stype, Dtype

# Please update your target URL and kumo API key.
API_URL = ""
API_KEY = ""
kumo.init(url=API_URL, api_key=API_KEY, project={json.dumps(str(project_id))})

"""
    return script


def generate_connector_script(connector_config: Dict[str, Any]) -> str:
    """Load an existing project connector using its safe display name.

    Generated variable references remain keyed by immutable connector ID, while
    the SDK lookup uses the user-facing name accepted by ``get_by_name``.
    """
    connector_id = connector_config["connector_id"]
    lookup_name = connector_config["lookup_name"]
    connector_classes = {
        DataSourceType.S3: "S3Connector",
        DataSourceType.GCS: "GCSConnector",
        DataSourceType.ADLS: "ADLSConnector",
        DataSourceType.SNOWFLAKE: "SnowflakeConnector",
        DataSourceType.DATABRICKS: "DatabricksConnector",
        DataSourceType.BIGQUERY: "BigQueryConnector",
    }
    try:
        connector_class = connector_classes[
            connector_config["data_source_type"]]
    except KeyError as exc:
        raise ValueError(
            f"Invalid data source type; {connector_config['data_source_type']}"
        ) from exc
    return (f"{connector_id}_connector = kumo.{connector_class}.get_by_name("
            f"{json.dumps(lookup_name)})\n\n")


def generate_table_script(
    table_config: Dict[str, Any],
    *,
    is_project_upload: bool = False,
):
    connector_id = table_config["connector_id"]
    if "sql" in table_config:
        # Custom SQL table: reconstruct via Table.from_sql (mirrors the SDK
        # surface) rather than a physical, name-keyed source table. The backing
        # connector is emitted under its own warehouse type by the connector
        # script; only the table body differs here.
        return f"""{table_config["table_name"]}_config = {table_config}

{table_config["table_name"]} = kumo.Table.from_sql(
    connector={connector_id}_connector,
    sql={table_config["table_name"]}_config["sql"],
    columns={table_config["table_name"]}_config["cols"],
    primary_key={table_config["table_name"]}_config['pkey'],
    time_column={table_config["table_name"]}_config['time_col'],
    end_time_column={table_config["table_name"]}_config["end_time_col"],
)

"""
    if is_project_upload:
        source_table = (
            f'kumo.FileUploadConnector(file_type={table_config["table_name"]}'
            f'_config["file_type"].lower()).table('
            f'{table_config["table_name"]}_config["table_name"])')
    else:
        source_table = (f'{connector_id}_connector.table('
                        f'{table_config["table_name"]}_config["table_name"])')

    script = f"""{table_config["table_name"]}_config = {table_config}

{table_config["table_name"]} = kumo.Table(
    source_table={source_table},
    columns={table_config["table_name"]}_config["cols"],
    primary_key={table_config["table_name"]}_config['pkey'],
    time_column={table_config["table_name"]}_config['time_col'],
    end_time_column={table_config["table_name"]}_config["end_time_col"]
)

"""
    return script


def generate_training_job_sdk_script(
    pq_config: Dict[str, str],
    model_plan_config: Dict[str, Any],
    training_table_gen_config: Dict[str, Any],
    train_table_job_id: Optional[str] = None,
    encoders: Optional[Dict[str, Any]] = None,
) -> str:
    """
    A method that puts all the supplied configs together
    into a final SDK Python script.

    Args:
        pq_config(Dict[str, Union[str, Graph]])
        model_plan_config(Dict[str, Any])
        training_table_gen_config(Dict[str, Any])
        train_table_job_id(Optional[str]): Optional
            because it's optional in the TrainingJobResponse.

    Returns:
        Fully executable Python SDK script.
    """

    script = f"""
# Generate predictive query
pquery_config = {pq_config}

pquery = kumo.PredictiveQuery(
    graph=pquery_config["graph"],
    query=pquery_config["query_string"],
)

# Generate training table
train_table_config = {training_table_gen_config}

# The following command will generate a new training table generation job
train_table = pquery.generate_training_table(plan=train_table_config)

"""
    if train_table_job_id:
        script += f"""# If you instead prefer to load an existing training table job, uncomment the next line
# train_table = kumo.TrainingTableJob(job_id=\"{train_table_job_id}\")

"""

    script += f"""# Generate model plan and training job
model_plan = {model_plan_config}

"""
    if encoders:
        script += f"""# Encoders
encoder_overrides={encoders}

# Enable this to override current encoders
#model_plan["column_processing"]["encoder_overrides"] = encoder_overrides

"""

    script += """trainer = kumo.Trainer(model_plan)
training_job = trainer.fit(
    graph=graph,
    train_table=train_table,
    non_blocking=False,
    custom_tags={"sdk_source": "generated_training_script"},
)

"""
    return script


def generate_prediction_only_sdk_script(training_job_id: str, ) -> str:
    """
    Generates SDK code to load an existing trained model instead of
    training from scratch. Used for batch prediction SDK scripts where
    the user already has a trained model.

    Args:
        training_job_id: The ID of the existing training job.

    Returns:
        Python SDK script that loads the trained model, its predictive
        query, graph, and training job reference.
    """
    script = f"""# Load existing trained model
pquery = kumo.PredictiveQuery.load_from_training_job("{training_job_id}")
graph = pquery.graph
trainer = kumo.Trainer.load("{training_job_id}")
training_job = kumo.TrainingJob("{training_job_id}")

"""
    return script


def generate_prediction_job_sdk_script(
    pred_table_plan: Dict[str, Any],
    predict_options_dict: Dict[str, Any],
    output_config: Dict[str, Any],
    export_connector_conf: Optional[Dict[str, str]] = None,
    pred_table_job_id: Optional[GeneratePredictionTableJobID] = None,
    pq_config: Optional[Dict[str, str]] = None,
    *,
    project_id: str,
) -> str:
    """
    A method that puts all the supplied configs together
    into a final SDK Python script.
    Args:
        pred_table_plan: Dict[str, Any],
        predict_options_dict: Dict[str, Any]
        output_config: Dict[str, str]
        export_connector_conf: Optional[Dict[str, str]]
        pred_table_job_id: Optional[GeneratePredictionTableJobID]
        pq_config: Optional[Dict[str, str]]
    Returns:
        Fully executable Python SDK script.
    """
    script = ""
    query_name = "pquery"

    if pq_config:
        query_name = "prediction_pquery"
        script += f"""# Generate new predictive query
{query_name}_config = {pq_config}
{query_name} = kumo.PredictiveQuery(
    graph=prediction_pquery_config["graph"],
    query=prediction_pquery_config["query_string"],
)
"""

    script += f"""# Generate prediction table
prediction_table_config = {pred_table_plan}

# The following command will generate a new prediction table generation job
pred_table = {query_name}.generate_prediction_table(plan=prediction_table_config)

"""
    if pred_table_job_id:
        script += f"""# If you instead prefer to load an existing prediction table job, uncomment the next line
# pred_table = kumo.PredictionTableJob(job_id=\"{pred_table_job_id}\")

"""
    if export_connector_conf:
        script += generate_conn_graph_or_table_sdk_script(
            [export_connector_conf],
            project_id=project_id,
            include_header=False,
        )

    script += f"""# Generate predictions
batch_prediction_options = {predict_options_dict}
output_configurations = OutputConfig(**{output_config})

prediction_job = trainer.predict(
    graph=graph,
    prediction_table=pred_table,
    output_config=output_configurations,
    training_job_id=training_job.job_id,
    binary_classification_threshold=batch_prediction_options["binary_classification_threshold"],
    num_classes_to_return=batch_prediction_options["num_classes_to_return"],
    num_workers=batch_prediction_options["num_workers"],
    non_blocking=False,
    custom_tags={{"sdk_source": "generated_prediction_script"}},
)
"""

    return script


def generate_unsupported_diskgraph_forecasting_bp_sdk_script() -> str:
    """Generate an executable SDK replay script for a known unsupported flow."""
    # Use json.dumps() to safely embed the message as a Python string literal,
    # so future edits to the message constant cannot inject quotes/newlines
    # that would break the emitted script.
    message_literal = json.dumps("SKIPPED_UNSUPPORTED: " +
                                 DISKGRAPH_FORECASTING_BP_UNSUPPORTED_MESSAGE)
    return f"""import sys

print({message_literal})
sys.exit(0)

"""


# ----------- Helper methods -----------
def recursively_convert_dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert a dataclass instance to a dictionary.
    This makes formatting the code between kumo and SDK easier.

    Args:
        obj: The object to convert (can be a dataclass, list,
            dict, or primitive).

    Returns:
        A dictionary representation of the dataclass.
    """
    if not dataclasses.is_dataclass(obj):
        # If it's a list, recursively convert each element
        if isinstance(obj, list):
            return [recursively_convert_dataclass_to_dict(item) for item in obj]

        # If it's a dict, recursively convert its values
        if isinstance(obj, dict):
            return {
                k: recursively_convert_dataclass_to_dict(v)
                for k, v in obj.items()
            }

        return obj

    # Convert dataclass to dictionary, including any pydantic extra fields
    # (e.g. per-edge-type overrides in HopConfig).
    result = {
        f.name: recursively_convert_dataclass_to_dict(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
    }

    extra = getattr(obj, '__pydantic_extra__', None)
    if extra:
        result.update({
            k: recursively_convert_dataclass_to_dict(v)
            for k, v in extra.items()
        })

    return result


def render_str_vals_as_strings(data: Dict[str, Any]):
    """
    Recursively convert any values that should be of type
    str to str. Examples are ??? or step in model plan.
    This is because through REST, these values get converted
    to objects, but we want the final model plan dictionary
    to have primitive types only.
    This makes formatting the code between kumo and SDK easier.

    Args:
        obj: The dictionary to parse.

    Returns:
        Parsed dictionary with correct data types.
    """
    if isinstance(data, dict):
        return {
            key: render_str_vals_as_strings(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [render_str_vals_as_strings(item) for item in data]
    elif isinstance(data, (bool, int, float, type(None))):
        return data  # Return primitive types as is
    else:
        if str(data):
            return str(data)  # Convert non-primitive types to string


def clean_up_pquery_string(
    pquery_str: str,
    table_refs: List[TableReferenceMetadata],
) -> str:
    """
    The pquery string retains some formatting artifacts
    (ex. \\n or many empty spaces). This method cleans up
    those formatting artifacts and creates a continous string.

    Args:
        pquery_str(str): The pquery string to parse.

    Returns:
        Parsed pquery string.
    """
    replacements = {
        r'\s+': ' ',  # Replace multiple spaces with a single space
        r'\\n': ' '  # Replace \\n dots with a single space
    }

    # Ensure that table is reffered by source table name in pquery
    table_names_map = {}
    for table in table_refs:
        if table.table_alias:
            table_names_map[table.table_alias] = table.source_table_name

        if table.table_alias_in_graph:
            table_names_map[
                table.table_alias_in_graph] = table.source_table_name

    replacements.update(table_names_map)
    pattern = re.compile('|'.join(f'({key})' for key in replacements.keys()))

    def replace(match):
        for i, key in enumerate(replacements.keys()):
            if match.group(i + 1) is not None:
                return replacements[key]
        return match.group(0)

    cleaned_pquery_str = pattern.sub(replace, pquery_str)

    return cleaned_pquery_str.strip()


def make_stype_and_dtype_sdk_compat(script: str) -> str:
    """
    Replaces any occurance of `stype=` -> `stype=Stype.`
    and `dtype=` -> `dtype=Dtype.` to make it SDK compatible
     in scripts generated by `generate_sdk_file`.


    Args:
        script (str): Complete SDK script.
    Returns:
        str: In-out method that returns the input string
        with stype and dtype formatting chages.
    """
    replacements = {
        "stype=": "stype=Stype.",
        "dtype=": "dtype=Dtype.",
        "\\\\'": "",
        "'\\\\": "",
    }

    for old, new in replacements.items():
        script = script.replace(old, new)

    return script


def bp_job_ouput_conf(
    outputs: List[PredictionOutputConfig],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    r"""
    Used to generate the SDK-compatible output config and export connector config
    for a prediction job.
    This method is part of the UI to SDK feature and will be called in the UI.

    Args:
        outputs (List[PredictionOutputConfig])
    Response:
        Tuple[Dict[str, str], Dict[str, str]] --> Tuple of output config
            dictionary and export connector config as dictionary.
    """
    output_types = []
    output_table_names = []
    connector_config: Dict[str, str] = {}
    connector_id: Optional[str] = None
    connector_type: Optional[str] = None
    extra_fields: Optional[List[Any]] = None
    staging_table_name: Optional[str] = None
    staging_table_name_set = False
    write_mode: Optional[str] = None

    file_output_types = (jobs_api.S3PredictionOutput,
                         jobs_api.GCSPredictionOutput,
                         jobs_api.ADLSPredictionOutput)
    for artifact in outputs:
        output_types.append(artifact.artifact_type.value)
        if isinstance(artifact, jobs_api.LocalPredictionOutput):
            artifact_extra_fields = list(artifact.extra_fields)
            if (extra_fields is not None
                    and artifact_extra_fields != extra_fields):
                raise ValueError(
                    "SDK replay cannot represent outputs with different "
                    "extra_fields.")
            extra_fields = artifact_extra_fields
            continue

        artifact_connector_id = artifact.connector_id
        artifact_connector_type = artifact.storage_type.value
        if (connector_id is not None and artifact_connector_id != connector_id):
            raise ValueError(
                "SDK replay supports only one output connector per batch "
                "prediction job.")
        if (connector_type is not None
                and artifact_connector_type != connector_type):
            raise ValueError(
                "SDK replay supports only one output connector type per "
                "batch prediction job.")
        connector_id = artifact_connector_id
        connector_type = artifact_connector_type

        artifact_extra_fields = list(artifact.extra_fields)
        if (extra_fields is not None and artifact_extra_fields != extra_fields):
            raise ValueError(
                "SDK replay cannot represent outputs with different "
                "extra_fields.")
        extra_fields = artifact_extra_fields

        if isinstance(artifact, file_output_types):
            output_path = artifact.file_path
            _, output_table_name = output_path.rstrip("/").rsplit(
                "/", maxsplit=1)
            output_table_name = output_table_name.removesuffix(".parquet")
            output_table_name += "_sdk"
            if urlsplit(output_path).scheme:
                parent = output_path.rstrip('/').rsplit('/', maxsplit=1)[0]
                output_table_name = f'{parent}/{output_table_name}'
            output_table_names.append(output_table_name)

        else:
            output_table_name = artifact.table_name + "_sdk"
            output_table_names.append(output_table_name)
            if isinstance(artifact, jobs_api.SnowflakePredictionOutput):
                artifact_write_mode = artifact.write_mode.value
                if (write_mode is not None
                        and artifact_write_mode != write_mode):
                    raise ValueError(
                        "SDK replay cannot represent outputs with different "
                        "write modes.")
                write_mode = artifact_write_mode
            elif isinstance(artifact, jobs_api.BigQueryPredictionOutput):
                artifact_write_mode = artifact.write_mode.value
                if (write_mode is not None
                        and artifact_write_mode != write_mode):
                    raise ValueError(
                        "SDK replay cannot represent outputs with different "
                        "write modes.")
                if (staging_table_name_set
                        and artifact.staging_table_name != staging_table_name):
                    raise ValueError(
                        "SDK replay cannot represent outputs with different "
                        "staging tables.")
                staging_table_name = artifact.staging_table_name
                staging_table_name_set = True
                write_mode = artifact_write_mode

    output_connector = ""
    if connector_id is not None and connector_type is not None:
        connector_config = {
            "connector_id": connector_id,
            "data_source_type": connector_type,
        }
        output_connector = "\\" + connector_id + "_connector\\"

    output_config: Dict[str, Any] = {}
    output_types = set(output_types)
    if output_types:
        output_config["output_types"] = set(output_types)
    if output_connector:
        output_config["output_connector"] = output_connector
    if extra_fields:
        output_config["extra_fields"] = extra_fields
    if staging_table_name:
        output_config["staging_table_name"] = staging_table_name
    if write_mode:
        output_config["write_mode"] = write_mode

    if len(output_table_names) > 1:
        output_config["output_table_name"] = tuple(output_table_names)
    elif len(output_table_names) == 1:
        output_config["output_table_name"] = output_table_names[0]

    # NOTE: Output config is empty for local download, but the SDK expects
    # a non-empty set for output_types and output_table_name.
    # Therefore, add a minimum of PREDICTIONS to output_types and predictions_output
    # to output_table_name.
    if "output_types" not in output_config:
        output_config["output_types"] = {"PREDICTIONS"}
    if "output_table_name" not in output_config:
        output_config["output_table_name"] = "predictions_output"

    return (output_config, connector_config)
