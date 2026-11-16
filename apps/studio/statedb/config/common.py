# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import uuid
from dataclasses import field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import kumo_core.types.schema as core_schema
import kumoapi.jobs as jobs_api
import kumoapi.source_table as api
from kumoapi.data_snapshot import GraphSnapshotID, TableSnapshotID
from temporalio.client import WorkflowExecutionStatus

from common_utils.enum import extends_enum
from common_utils.string import (
    validate_id,
    validate_max_length,
    validate_non_empty,
    validate_source_name,
)
from io_utils import kumo_dataclass
from io_utils.dataclasses import ValidatedStr
from statedb.config.timestamp import ResourceTimestamps, now_utc

# Separator for fingperint suffix in v2 TableID and GraphID
_FP_SUFFIX_SEPARATOR = '-'


class ID(ValidatedStr):
    """A String ID restricted to a "safe" set of characters.

    For now, identifiers should be at most 256 characters long, must begin with
    a letter, and may only contain alphanumeric characters and underscores.
    """
    def __new__(cls, name: str, entity: str = ""):
        validate_id(name, entity or cls.__name__)
        # In Kumo v1 legacy experience, this was 64 characters long. We
        # changed it to 256 characters to support the scenarios where same
        # table names are distinguished by prepending with connector name.
        validate_max_length(name, 256)
        return super().__new__(cls, name)


class DBID:
    pass


class TableName(ID):
    pass


class SourceTableName(ValidatedStr):
    """A String ID restricted to a "safe" set of characters for source table.

    For now, identifiers should be at most 64 characters long, and may only
    contain alphanumeric characters and underscores.
    """
    def __new__(cls, name: str):
        validate_source_name(name)
        validate_max_length(name, 256)
        return super().__new__(cls, name)


class PredictiveQueryID(ValidatedStr, DBID):
    """
    PQuery ID can come in two forms:
    1) <pquery_name> which must be a valid ID
    2) <pquery_name>-<core_schema.FPID> that contains a fingerprint ID
       separated by "-"
    """
    def __new__(cls, contents: str, entity: str = 'Predictive Query'):
        validate_non_empty(contents)
        if _FP_SUFFIX_SEPARATOR not in contents:
            return cls.compose(contents)

        pquery_name, fpid_str = contents.split(_FP_SUFFIX_SEPARATOR, maxsplit=1)
        if not core_schema.FPID.is_valid(fpid_str):
            raise ValueError(f'{entity} ID has invalid version suffix')
        return cls.compose(pquery_name, core_schema.FPID(fpid_str))

    @classmethod
    def compose(
        cls,
        pquery_name: Optional[str] = None,
        fpid: Optional[core_schema.FPID] = None,
    ) -> 'PredictiveQueryID':
        pquery_name = pquery_name or 'NA'
        name_id = ID(pquery_name, entity='Predictive Query')
        if not fpid:
            return super().__new__(cls, name_id)
        else:
            return super().__new__(cls,
                                   _FP_SUFFIX_SEPARATOR.join([name_id, fpid]))

    @classmethod
    def parse(
            cls,
            id: str) -> Tuple['PredictiveQueryID', Optional[core_schema.FPID]]:
        if _FP_SUFFIX_SEPARATOR not in id:
            return cls(id), None
        pqid, fpid = id.split(_FP_SUFFIX_SEPARATOR, maxsplit=1)
        return cls(pqid), core_schema.FPID(fpid)

    @classmethod
    def has_fpid(cls, id: str):
        return cls.parse(id)[1] is not None

    @property
    def name(self) -> 'PredictiveQueryID':
        if _FP_SUFFIX_SEPARATOR not in self:
            return self
        return PredictiveQueryID(self.split(_FP_SUFFIX_SEPARATOR)[0])

    @property
    def fpid(self) -> Optional[core_schema.FPID]:
        if _FP_SUFFIX_SEPARATOR not in self:
            return None
        return core_schema.FPID(self.split(_FP_SUFFIX_SEPARATOR, maxsplit=1)[1])

    @classmethod
    def public(cls, fpid: core_schema.FPID) -> 'PredictiveQueryID':
        """
        Compose a public-facing PQ resource identifer, formatted as:
          `pquery-<fingerprint id hex>`
        This is just for cosmetic purpose to use "pquery" instead of "NA" as
        placeholder value for absent "pquery_id" component.
        """
        return PredictiveQueryID.compose(PredictiveQueryID('pquery'), fpid)

    @classmethod
    def is_public(cls, id: str) -> bool:
        name, fpid = cls.parse(id)
        return name == 'pquery' and fpid is not None

    @classmethod
    def from_public(cls, id: Union[str,
                                   'PredictiveQueryID']) -> 'PredictiveQueryID':
        name, fpid = cls.parse(id)
        if not fpid:
            raise ValueError(f"Invalid predictive query ID: {id}")
        if name == "pquery":
            return PredictiveQueryID.compose(pquery_name=None, fpid=fpid)
        return PredictiveQueryID.compose(pquery_name=name, fpid=fpid)


class GraphIDv1(ID, DBID):
    """
    GraphID v1 is the name of graph, must be valid identifiers, as defined by
    :class:`ID`.
    """
    def __new__(cls, contents: str, entity: str = ""):
        return super().__new__(cls, contents, entity)


_PUBLIC_GRAPH_ID_V2_PREFIX = PredictiveQueryID('graph')


class GraphIDv2(ValidatedStr, DBID):
    r"""A Graph (V2) identifier is composed of two components: the pquery ID
    that the graph is part of (if present) and the fingerprint of the graph,
    separated by exactly one ``-``.

    If the predictive query ID is missing, it will be replaced with "NA"."""
    def __new__(cls, contents: str):
        validate_non_empty(contents)
        pquery_id_str, fpid_str = cls._split(contents)
        try:
            pquery_id = (None if pquery_id_str == "NA" else
                         PredictiveQueryID(pquery_id_str))
            fpid = None if fpid_str == "NA" else core_schema.FPID(fpid_str)
            return cls.compose(pquery_id, fpid)
        except ValueError as err:
            raise ValueError(
                f'Graph ID v2 must be valid identifier, got: {contents}. '
                f'err: {err}') from err

    @staticmethod
    def _split(id: str) -> Tuple[str, str]:
        """Split a canonical graph ID or raise a clean shape error."""
        parts = id.split(_FP_SUFFIX_SEPARATOR)
        if len(parts) != 2:
            raise ValueError(f'Invalid Graph ID: {id}')
        return parts[0], parts[1]

    @property
    def pquery_id(self) -> Optional[PredictiveQueryID]:
        return GraphIDv2.parse(self)[0]

    @property
    def fpid(self) -> Optional[core_schema.FPID]:
        return GraphIDv2.parse(self)[1]

    @staticmethod
    def parse(
        id: Union[str, 'GraphIDv2'],
    ) -> Tuple[Optional[PredictiveQueryID], Optional[core_schema.FPID]]:
        pqid, fpid = GraphIDv2._split(id)
        if pqid in (_PUBLIC_GRAPH_ID_V2_PREFIX, 'NA'):
            pqid = None
        else:
            pqid = PredictiveQueryID(pqid)
        return (pqid, core_schema.FPID(fpid) if fpid != 'NA' else None)

    @classmethod
    def compose(
        cls,
        pquery_id: Optional[PredictiveQueryID] = None,
        fpid: Optional[core_schema.FPID] = None,
    ) -> 'GraphIDv2':
        if not pquery_id and not fpid:
            raise ValueError('Unable to compose empty GraphID.')
        pquery_id_str = pquery_id or 'NA'
        fpid_str = fpid or 'NA'
        graph_id_v2_str = _FP_SUFFIX_SEPARATOR.join([pquery_id_str, fpid_str])
        return super().__new__(cls, graph_id_v2_str)

    @classmethod
    def public(cls, fpid: core_schema.FPID) -> 'GraphIDv2':
        """
        Compose a public-facing Graph resource identifer, formatted as:
          `graph-<fingerprint id hex>`
        This is just for cosmetic purpose to use "graph" instead of "NA" as
        placeholder value for absent "pquery_id" component.
        """
        return GraphIDv2.compose(_PUBLIC_GRAPH_ID_V2_PREFIX, fpid)

    @classmethod
    def from_public(cls, id: Union[str, 'GraphIDv2']) -> 'GraphIDv2':
        pqid, fpid = cls.parse(id)
        if not fpid:
            raise ValueError(f"Invalid Graph ID: {id}")
        return GraphIDv2.compose(pquery_id=pqid, fpid=fpid)


GraphID = Union[GraphIDv1, GraphIDv2]


def to_graph_id(graph_id_str: str) -> GraphID:
    if _FP_SUFFIX_SEPARATOR in graph_id_str:
        return GraphIDv2(graph_id_str)
    return GraphIDv1(graph_id_str)


class ConnectorID(ID, DBID):
    def __new__(cls, name: str):
        if not re.fullmatch(r"^connector_[a-f0-9]{32}$", name):
            raise ValueError(f"Invalid ConnectorID '{name}': must match "
                             "'connector_<32 hex chars>'.")
        return str.__new__(cls, name)

    @classmethod
    def generate(cls) -> "ConnectorID":
        return cls(f"connector_{uuid.uuid4().hex}")


class UserID(str):
    """External or local authentication user identifier."""
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema())


class APIKeyID(ValidatedStr, DBID):
    """Identifier for API key metadata/secret records."""
    def __new__(cls, value: str):
        validate_non_empty(value)
        validate_max_length(value, 256)
        return super().__new__(cls, value)


class OrganizationID(str):
    """Authentication organization identifier."""
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema())


class JobID(ValidatedStr, DBID):
    """Job IDs may be at most 64 characters long, and may only contain
    alphanumeric, and dash characters. This makes them save to use as
    kubernetes job identifiers, filenames, and URL fragments. Note that
    kubernetes identifiers do not permit underscore characters.
    """
    def __new__(cls, name: str):

        validate_non_empty(name)
        validate_max_length(name, 64)
        if not re.fullmatch(r'[a-zA-Z0-9\-]*', name):
            raise ValueError(
                f"{cls.__name__} identifier may only contain alphanumeric "
                f"characters and dashes. Got: {name}")
        return super().__new__(cls, name)


class TrainingJobID(JobID, DBID):
    @staticmethod
    def make_random_id() -> 'TrainingJobID':
        return TrainingJobID(f"trainingjob-{uuid.uuid4().hex}")

    @staticmethod
    def is_valid_id(job_id: str) -> bool:
        return job_id.startswith("trainingjob-")

    @staticmethod
    def as_parent_of(
            child_job_id: 'GenerateTrainTableJobID') -> 'TrainingJobID':
        hex = child_job_id.split('-')[-1]
        return TrainingJobID(f"trainingjob-{hex}")


class LLMEmbeddingJobID(JobID, DBID):
    @staticmethod
    def make_random_id() -> 'LLMEmbeddingJobID':
        return LLMEmbeddingJobID(f"llm-embedding-job-{uuid.uuid4().hex}")


class GenerateTrainTableJobID(JobID, DBID):
    @staticmethod
    def as_child_of(
        parent_job_id: TrainingJobID, ) -> 'GenerateTrainTableJobID':
        """
        Compose ID of a GenerateTrainTableJob as direct child of a parent
        TrainingJob, the returned job ID will contain the same hex suffix.

        This is useful to distinguish the relationship between a TrainingJob
        and a GenerateTrainTableJob, i.e., either a strict parent-child
        relationship or decoupled/composite relationship.
        """
        hex = parent_job_id.split('-')[-1]
        return GenerateTrainTableJobID(f'gen-traintable-job-{hex}')

    @staticmethod
    def make_random_id() -> 'GenerateTrainTableJobID':
        return GenerateTrainTableJobID(f"gen-traintable-job-{uuid.uuid4().hex}")

    @staticmethod
    def is_valid_id(job_id: str) -> bool:
        return job_id.startswith('gen-traintable-job-')

    def is_child_of(
        self,
        parent_job_id: TrainingJobID,
    ) -> bool:
        """Determine (by id) whether this job is a direct child of the given
        training job."""
        return (self == GenerateTrainTableJobID.as_child_of(parent_job_id)
                and parent_job_id == TrainingJobID.as_parent_of(self))

    @property
    def wf_id(self) -> str:  # Temporal workflow id for this job.
        return f'{self}-wf'


class SplitJobID(JobID, DBID):
    @staticmethod
    def as_child_of(parent_job_id: GenerateTrainTableJobID,
                    split_fpid: core_schema.FPID) -> 'SplitJobID':
        """Compose ID of a SplitJob as direct child of a parent
        GenerateTrainTableJob, the returned job ID will contain the same
        hex suffix.

        This is useful to distinguish the relationship between a
        GenerateTrainTableJob and a SplitJob, i.e., either a strict
        parent-child relationship or decoupled/composite relationship.

        Args:
            parent_job_id (GenerateTrainTableJobID): The parent generate train
                table job.
            split_fpid (core_schema.FPID): The fingerprint of the split.

        Returns:
            SplitJobID: The split job ID.
        """
        hex = parent_job_id.split('-')[-1]
        short_fpid = split_fpid[:8]
        return SplitJobID(f'split-job-{hex}-{short_fpid}')

    @staticmethod
    def make_random_id() -> 'SplitJobID':
        return SplitJobID(f"split-job-{uuid.uuid4().hex}")

    def is_child_of(
        self,
        generate_train_table_job_id: GenerateTrainTableJobID,
        split_fpid: core_schema.FPID,
    ) -> bool:
        """Determine (by id) whether this job is a direct child of the given
        generate train table job and split fingerprint.

        Args:
            generate_train_table_job_id (GenerateTrainTableJobID): The parent
                generate train table job.
            split_fpid (core_schema.FPID): The fingerprint of the split.

        Returns:
            bool: Whether this job is a direct child of the given generate
                train table job and split fingerprint.
        """
        return self == SplitJobID.as_child_of(
            generate_train_table_job_id,
            split_fpid,
        )

    @property
    def wf_id(self) -> str:  # Temporal workflow id for this job.
        return f'{self}-wf'


class BatchPredictionJobID(JobID, DBID):
    @staticmethod
    def make_random_id() -> 'BatchPredictionJobID':
        return BatchPredictionJobID(f"bp-job-{uuid.uuid4().hex}")

    @staticmethod
    def is_valid_id(job_id: str) -> bool:
        return job_id.startswith('bp-job-')

    @staticmethod
    def as_parent_of(
            child_job_id: 'GeneratePredictionTableJobID'
    ) -> 'BatchPredictionJobID':
        hex = child_job_id.split('-')[-1]
        return BatchPredictionJobID(f"bp-job-{hex}")


class ArtifactExportJobID(JobID, DBID):
    @staticmethod
    def make_random_id() -> 'ArtifactExportJobID':
        # TODO(dong) make id based on job id and output type / config.
        return ArtifactExportJobID(f"ae-job-{uuid.uuid4().hex}")


class GeneratePredictionTableJobID(JobID, DBID):
    @staticmethod
    def as_child_of(
        parent_job_id: BatchPredictionJobID,
    ) -> 'GeneratePredictionTableJobID':
        """
        Compose ID of a GeneratePredictionTableJob as direct child of a parent
        BatchPredictionJob, the returned job ID will contain the same hex
        suffix.

        This is useful to distinguish the relationship between a
        BatchPredictionJob and a GeneratePredictionTableJobID, i.e., either a
        strict parent-child relationship or decoupled/composite relationship.
        """
        hex = parent_job_id.split('-')[-1]
        return GeneratePredictionTableJobID(f'gen-predtable-job-{hex}')

    @staticmethod
    def make_random_id() -> 'GeneratePredictionTableJobID':
        return GeneratePredictionTableJobID(
            f"gen-predtable-job-{uuid.uuid4().hex}")

    @staticmethod
    def is_valid_id(job_id: str) -> bool:
        return job_id.startswith('gen-predtable-job-')

    def is_child_of(self, parent_job_id: BatchPredictionJobID) -> bool:
        """Determine (by id) whether this job is a direct child of the given
        batch prediction job."""
        return (self == GeneratePredictionTableJobID.as_child_of(parent_job_id)
                and parent_job_id == BatchPredictionJobID.as_parent_of(self))

    @property
    def wf_id(self) -> str:  # Temporal workflow id for this job.
        return f'{self}-wf'


class BatchPredictionWorkflowID(ID, DBID):
    pass


@extends_enum(jobs_api.JobType)
class JobType(str, Enum):
    GENERATE_TRAIN_TABLE_JOB = "GENERATE_TRAIN_TABLE_JOB"
    GENERATE_PREDICTION_TABLE_JOB = "GENERATE_PREDICTION_TABLE_JOB"
    TRAINING_JOB = "TRAINING_JOB"
    BATCH_PREDICTION_JOB = "BATCH_PREDICTION_JOB"

    @staticmethod
    def from_id(id: JobID) -> 'JobType':
        if TrainingJobID.is_valid_id(id):
            return JobType.TRAINING_JOB
        if BatchPredictionJobID.is_valid_id(id):
            return JobType.BATCH_PREDICTION_JOB
        if GenerateTrainTableJobID.is_valid_id(id):
            return JobType.GENERATE_TRAIN_TABLE_JOB
        if GeneratePredictionTableJobID.is_valid_id(id):
            return JobType.GENERATE_PREDICTION_TABLE_JOB
        raise AssertionError(f'Unsuported job type from {id}')


class JobSource(str, Enum):
    SDK = "SDK"
    UI = "UI"
    REST = "REST"
    UNKNOWN = "UNKNOWN"


class TableIDv1(ValidatedStr, DBID):
    """
    V1 table ID has format of `<connector_id>.<table_name>`

    Both `connector_id` and `table_name` must be valid identifiers,
    as defined by :class:`ID`.
    """
    def __new__(cls, contents: str):
        try:
            validate_non_empty(contents)
            connector_id, table_name = contents.split('.')
            return TableIDv1.compose(ConnectorID(connector_id),
                                     TableName(table_name))
        except ValueError as err:
            raise ValueError(
                "TableID must have the format <connector_id>.<table_name>, "
                "or <pquery_id>.<table_name>:<fpid>, "
                f"got {contents}, err: {err}")

    @classmethod
    def compose(cls, connector_id: ConnectorID,
                table_name: TableName) -> 'TableIDv1':
        return super().__new__(cls, f"{connector_id}.{table_name}")

    @staticmethod
    def parse(id: Union[str, 'TableIDv1']) -> Tuple[ConnectorID, TableName]:
        connector_id, table_name = id.split('.')
        return (ConnectorID(connector_id), TableName(table_name))

    @property
    def table_name(self) -> TableName:
        return TableIDv1.parse(self)[1]

    @property
    def connector_id(self) -> ConnectorID:
        return TableIDv1.parse(self)[0]


class TableIDv2(ValidatedStr, DBID):
    """
    V2 table ID is composed of 3 componenents formatted as
    `pquery_id.table_name:fpid`. `hex_fp` is expected to be 32-char hex
    fingerprint. If a component is missing, we use "NA" as placeholder.

    DEPRECATION INFO: All calls right now across the UI and SDK only use the
    NA.NA-<hex_fp> format. Note that <pquery>.<table_name-><hex_fp> format
    is legacy and associated with v1 object model which is completely
    deprecated now. There might be traces of data in the DB with this format,
    but this is not actively used in any API anymore.
    """
    def __new__(cls, contents: str):
        validate_non_empty(contents)
        try:
            # Accept both `<pquery>.<table>-<fpid>` and the legacy
            # `<pquery>.<table>` shape (no separator). Per the class
            # docstring, missing components are normalized to "NA".
            if _FP_SUFFIX_SEPARATOR in contents:
                table_id, fpid = contents.split(_FP_SUFFIX_SEPARATOR,
                                                maxsplit=1)
            else:
                table_id, fpid = contents, 'NA'
            pquery_id, table_name = table_id.split('.', maxsplit=1)
            return TableIDv2.compose(
                PredictiveQueryID(pquery_id), TableName(table_name),
                core_schema.FPID(fpid) if fpid != "NA" else None)
        except ValueError as err:
            raise ValueError(
                "TableID must have the format <connector_id>.<table_name>, "
                "or <pquery_id>.<table_name>:<fpid>, "
                f"got {contents}, err: {err}")

    @staticmethod
    def parse(
        id: Union[str, 'TableIDv2']
    ) -> Tuple[Optional[PredictiveQueryID], TableName,
               Optional[core_schema.FPID]]:
        # Mirror ``__new__`` leniency: ids without the ``-fpid`` suffix
        # parse as fpid='NA'. Per the class docstring, missing components
        # are normalized to "NA".
        if _FP_SUFFIX_SEPARATOR in id:
            pq_and_name, fpid = id.split(_FP_SUFFIX_SEPARATOR, maxsplit=1)
        else:
            pq_and_name, fpid = id, 'NA'
        pq_id, table_name = pq_and_name.split('.', maxsplit=1)
        return (
            PredictiveQueryID(pq_id) if pq_id != 'NA' else None,
            TableName(table_name),
            core_schema.FPID(fpid) if fpid != 'NA' else None,
        )

    @classmethod
    def compose(
        cls,
        pquery_id: Optional[PredictiveQueryID],
        table_name: TableName,
        fpid: Optional[core_schema.FPID],
    ) -> 'TableIDv2':
        pquery_id = pquery_id or PredictiveQueryID('NA')
        fpid_str = fpid or 'NA'
        id = f"{pquery_id}.{table_name}{_FP_SUFFIX_SEPARATOR}{fpid_str}"
        return super().__new__(cls, id)

    @classmethod
    def public(
        cls,
        table_name: TableName,
        fpid: core_schema.FPID,
        pquery_id: Optional[PredictiveQueryID],
    ) -> 'TableIDv2':
        """
        Compose a public-facing Table resource identifer, formatted as:
          `pquery_id.table_name-<fingerprint id hex>`
        Use "NA" as placeholder value for absent "pquery_id" component.
        """
        pquery_id = pquery_id or PredictiveQueryID('NA')
        return TableIDv2.compose(pquery_id, table_name, fpid)

    @classmethod
    def from_public(cls, id: Union[str, 'TableIDv2']) -> 'TableIDv2':
        pqid, table_name, fpid = cls.parse(id)
        if not fpid:
            raise ValueError(f"Invalid Table ID: {id}")
        if table_name == TableName('table'):
            return TableIDv2.compose(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=fpid,
            )
        return TableIDv2.compose(
            pquery_id=pqid,
            table_name=table_name,
            fpid=fpid,
        )

    @staticmethod
    def update(id: 'TableIDv2', new_pquery_id: Optional[PredictiveQueryID],
               include_fpid: bool):
        _, table_name, fpid = TableIDv2.parse(id)
        fpid = fpid if include_fpid else None
        return TableIDv2.compose(new_pquery_id, table_name, fpid)

    @property
    def table_name(self) -> TableName:
        return TableIDv2.parse(self)[1]


TableID = Union[TableIDv1, TableIDv2]


def to_table_id(table_id_str: str) -> TableID:
    if _FP_SUFFIX_SEPARATOR in table_id_str:
        return TableIDv2(table_id_str)
    return TableIDv1(table_id_str)


def to_table_ids(table_ids: List[str]) -> List[TableID]:
    return [to_table_id(table_id) for table_id in table_ids]


def to_table_ids_dict(tabls_id_dict: Dict[str, Any]) -> Dict[TableID, Any]:
    return {
        to_table_id(table_id): value
        for table_id, value in tabls_id_dict.items()
    }


@kumo_dataclass(frozen=True, eq=True, order=True)
class ColumnID:
    """A struct that globally uniquely identifies a column within a table.

    It is unique, because table_id is guaranteed to be unique across
    all tables, while all columns within a table are guaranteed to be
    unique within a table. This will be used as part of the public API
    that constructs graphs.
    """
    table_id: TableID
    col_name: str

    def __str__(self) -> str:
        return f"{self.table_id}.{self.col_name}"

    @staticmethod
    def from_fqn(full_qualified_name: str) -> 'ColumnID':
        fq_table_name, col_name = full_qualified_name.rsplit('.', 1)
        return ColumnID(to_table_id(fq_table_name), col_name)


SourceColumn = api.SourceColumn


@kumo_dataclass
class GraphLink:
    """GraphLink contains information needed to render a URL to a graph.

    It can be used to render a link on the UI, with the link text = name
    and the href generated from the given graph ID. This is initially
    intended to be used in the Table webpage, enabling the user to navigate
    to the graphs that include the given table.
    """
    name: str
    id: Optional[GraphID] = None
    snapshot_id: Optional[GraphSnapshotID] = None


@kumo_dataclass
class ConnectorLink:
    """ConnectorLink contains information needed to render a URL to a
    connector.
    """
    name: str
    id: ConnectorID


@kumo_dataclass
class PredictiveQueryLink:
    """PredictiveQueryLink contains information needed to render a URL to a
    predictive query.
    """
    name: str
    id: Optional[PredictiveQueryID] = None
    training_job_id: Optional[TrainingJobID] = None


@kumo_dataclass
class BatchPredictionWorkflowLink:
    """BatchPredictionWorkflowLink contains information needed to render a
    URL to a Batch Prediction Workflow.
    """
    name: str
    id: BatchPredictionWorkflowID


@kumo_dataclass
class TableLink:
    """TableLink contains information needed to render a URL to a table.
    """
    name: str
    id: TableID


@kumo_dataclass
class JobFailureInfo:
    message: str
    stacktrace: str

    def remove_employee_only_fields(self) -> None:
        """Delete fields that should not be shown to customers.
        """
        self.stacktrace = ""


@kumo_dataclass
class EntityValidationMessage:
    """ Represents a message tied to a DB entity (such as table or graph). """
    # ID that can be used to generate the URL to the entity
    id: str

    # Human readable name of the entity
    name: str

    message: str


class CallToActionLevel(Enum):
    # Indicates that the CTA is non-blocking, but recommended to fix. For
    # example, it could indicate that a table has many invalid values, which
    # could degrade model performance, but will not block training.
    WARNING = "WARNING"

    # Indicates that the CTA is blocking a core product flow. For example, it
    # could indicate that a table is missing underlying data, which would block
    # training from occurring.
    CRITICAL = "CRITICAL"


@kumo_dataclass
class CallToAction:
    """Represents a message that gets shown in the UI, to indicate that the
    user needs to perform a particular action, such as retraining a Predictive
    Query.
    """
    level: CallToActionLevel
    message: str

    @staticmethod
    def crit(message: str) -> 'CallToAction':
        return CallToAction(level=CallToActionLevel.CRITICAL, message=message)

    @staticmethod
    def warn(message: str) -> 'CallToAction':
        return CallToAction(level=CallToActionLevel.WARNING, message=message)


@kumo_dataclass
class WarningsResponse:
    warnings: List[str] = field(default_factory=list)


@kumo_dataclass
class DBEntry:
    timestamps: ResourceTimestamps


@kumo_dataclass
class ErrWarnCTA:
    name: str  # To be shown on the CTA button
    url: str  # This is a string that is mapped to a proper URL in the UI.


@kumo_dataclass
class ErrWarnDetails:
    is_error: bool  # For internal use; true => Error ; false => Warning
    title: str  # Title string of error / warning
    description: str  # Description string of error / warning
    is_dismissible: bool  # Used for Warnings only. ( in future )
    # CTA , if a detected error has a call-to-action associated with it.
    cta: Optional[ErrWarnCTA] = None
    id: Optional[str] = None


@kumo_dataclass
class ErrWarnResponse:
    errors: List[ErrWarnDetails]
    warnings: List[ErrWarnDetails]
    info_items: List[ErrWarnDetails]


@kumo_dataclass
class ErrWarnArgs:
    ids: List[str]


@kumo_dataclass
class LastIngestedTimestamp:
    timestamp: Optional[datetime] = None


@kumo_dataclass
class JobProgress:
    iterations_completed: int = 0
    iterations_remaining: int = 0
    start_time: datetime = field(default_factory=now_utc)
    elapsed_time: timedelta = timedelta(seconds=0)
    estimated_end_time: Optional[datetime] = None


class JobStatus(str, Enum):
    # Default status (no training job can be found)
    UNTRAINED = 'UNTRAINED'

    # Job has been submitted and is in the job queue.
    QUEUED = 'QUEUED'

    # Job has been submitted and is currently running.
    RUNNING = 'RUNNING'

    # Terminal status:
    DONE = 'DONE'  # Job has completed successfully
    FAILED = 'FAILED'  # Job has failed due to error.
    CANCELLED = 'CANCELLED'  # Job has been aborted/cancelLed by the customer.

    # Job is missing / deleted / not filed:
    MISSING = 'MISSING'

    @property
    def is_terminal(self) -> bool:
        return self in [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED]

    @property
    def is_submitted(self) -> bool:
        return self in [JobStatus.QUEUED, JobStatus.RUNNING]


# An event log entry could carry information about either
# 1) an internal state transition event, e.g., job failed due to a crash; job
#    got canceled by user, etc.
# 2) a span of internal processing step, e.g., ingesting table ABC, or
# running # AutoML experiement #12.
#
# A job event log entry may carry additional message, which may be updated
# later on, only the latest update will be returned per event_name. E.g.,
# a "Ingest table ABC" event could carry a progress percentage message that
# gets updated until it reaches "100%".  For FAILED/CANCELED job status, the
# message could be additional detailed reason or other debug info.
@kumo_dataclass
class JobEventLogEntry(jobs_api.JobEventLogEntry):
    @staticmethod
    def _timestamp(curr_datetime: Optional[datetime]) -> datetime:
        return curr_datetime or now_utc()

    @classmethod
    def untrained(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Not yet trained",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This query has been created but no training job "
                    "can be found. Please try training again."),
        )

    @classmethod
    def not_started(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Not Started",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job has been queued and is waiting to start "
            "processing.",
        )

    @classmethod
    def starting(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Ingesting Data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is processing data from your warehouse.",
        )

    @classmethod
    def generating_training_table(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Generating Training Data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is currently generating all training examples.",
        )

    @classmethod
    def generating_pred_table(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Generating Prediction Table data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This job is currently generating the prediction table."),
        )

    @classmethod
    def waiting_training_table(
            cls, job_id: GenerateTrainTableJobID,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Waiting for Training Data to be generated",
            last_updated_at=cls._timestamp(curr_datetime), detail=(
                "This training job is currently waiting for training data to "
                f"be generated by job={job_id}"))

    @classmethod
    def waiting_pred_table(
            cls, job_id: GeneratePredictionTableJobID,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Waiting for Prediction Data to be generated",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This batch prediction job is currently waiting for pred "
                    f"table to be generated by job={job_id}"))

    @classmethod
    def materializing(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Data Materialization",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is currently materializing the Kumo graph.",
        )

    @classmethod
    def provisioning_resources(
        cls,
        curr_datetime: Optional[datetime] = None,
    ) -> 'JobEventLogEntry':
        return cls(
            stage_name="Provisioning Resources",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This job is currently provisioning required resources. "
                    "This may take up to 10 minutes."),
        )

    @classmethod
    def loading_materialized_data(
        cls,
        curr_datetime: Optional[datetime] = None,
    ) -> 'JobEventLogEntry':
        return cls(
            stage_name="Loading Materialized Data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is currently loading materialized data.",
        )

    @classmethod
    def configuring_prediction_job(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Configuring prediction job",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This job is currently being configured. "
                    "Batch Prediction will start soon! "
                    "This may take up to 10 minutes."),
        )

    @classmethod
    def training(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Training",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is currently training your model.",
        )

    @classmethod
    def refit_trainval(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Refitting on Validation Data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This job is currently refitting the best model "
                    "found using Auto ML on both training and validation "
                    "data for most optimal performance on the holdout data"),
        )

    @classmethod
    def refit_full(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Refitting on Full Data",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=("This job is currently refitting the best model found "
                    "using Auto ML on all available data (training, "
                    "validation and holdout) for most optimal performance at "
                    "prediction time"),
        )

    @classmethod
    def predicting(
            cls,
            curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Predicting",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job is currently generating your predictions.",
        )

    @classmethod
    def done(cls,
             curr_datetime: Optional[datetime] = None) -> 'JobEventLogEntry':
        return cls(
            stage_name="Done",
            last_updated_at=cls._timestamp(curr_datetime),
            detail="This job successfully completed!",
        )

    @classmethod
    def failed(cls, curr_datetime: Optional[datetime] = None,
               message: str = "This job has failed.") -> 'JobEventLogEntry':
        return cls(
            stage_name="Failed",
            last_updated_at=cls._timestamp(curr_datetime),
            detail=message,
        )

    @classmethod
    def cancelled(
        cls,
        curr_datetime: Optional[datetime] = None,
        message: str = 'This job has been cancelled.',
    ) -> 'JobEventLogEntry':
        return cls(
            stage_name='Canceled',
            last_updated_at=cls._timestamp(curr_datetime),
            detail=message,
        )


@kumo_dataclass
class JobExecution:
    """
    Generic base dataclass to persist execution states/metedata into DB for
    a long-running job (executed as a Temporal workflow).
    """
    status: JobStatus
    log: JobEventLogEntry
    # Set when a queue-managed job is admitted to execution, so it is unset
    # while the job is still QUEUED.  Every concrete subclass declares the
    # same optionality; keeping the base in sync matters because generic
    # surfaces (e.g. ``GenericJob.execution``) validate against this type.
    start_time: Optional[datetime] = None
    last_updated_at: datetime = field(default_factory=now_utc)

    # Record the terminal state of the workflow on Temporal. This is the
    # source-of-truth terminal state and freezes further execution updates.
    workflow_end_state: Optional[WorkflowExecutionStatus] = None

    # present iff status=FAILED
    failure_info: Optional[JobFailureInfo] = None


# Table or Graph snapshot enum class for formatting and parsing snapshot id.
class DataSnapshot(Enum):
    TABLE = ('table-', TableSnapshotID)
    GRAPH = ('graph-', GraphSnapshotID)

    def __init__(self, prefix: str, id_cls: type) -> None:
        self.prefix = prefix
        self.id_cls = id_cls

    def format_id(self, fpid: core_schema.FPID, data_version: str):
        return self.id_cls(f'{self.prefix}{fpid}@{data_version}')

    def parse_id(self, snapshot_id: str) -> Tuple[core_schema.FPID, str]:
        """Parse a snapshot ID with its prefix and exactly one ``@``."""
        if not snapshot_id.startswith(self.prefix):
            raise ValueError(
                f'{self.name} snapshot ID is invalid: {snapshot_id}')
        fpid_data_ver = snapshot_id[len(self.prefix):]
        parts = fpid_data_ver.split('@')
        if len(parts) != 2:
            raise ValueError(
                f'{self.name} snapshot ID is invalid: {snapshot_id}')
        fpid, data_ver = parts
        return core_schema.FPID(fpid), data_ver

    def canonicalize(self, snapshot_id: str):
        """Return ``snapshot_id`` in canonical
        ``<prefix><fpid>@<data_version>`` form.

        Idempotent: passing an already-canonical id round-trips
        unchanged. Useful at API/RPC boundaries where the caller may
        have constructed the id differently (e.g., padded whitespace,
        an alternate fpid form) — callers normalize once and then use
        the result as the single identity for locks, lease tracking,
        log lines, etc.

        Raises ``ValueError`` if ``snapshot_id`` doesn't match the
        expected ``xxxx-<fpid>@<version>`` shape.
        """
        graph_id, data_version = snapshot_id.split('@', 1)
        _, fpid = GraphIDv2.parse(graph_id)
        if not fpid:
            raise ValueError(f"Invalid graph snapshot ID: {snapshot_id}")
        return self.format_id(fpid, data_version)


class IdentityID(ID, DBID):
    def __new__(cls, name: str):
        if not re.fullmatch(r"^identity_[a-f0-9]{32}$", name):
            raise ValueError(f"Invalid IdentityID '{name}': must match "
                             "'identity_<32 hex chars>'.")
        return str.__new__(cls, name)

    @classmethod
    def generate(cls) -> "IdentityID":
        return cls(f"identity_{uuid.uuid4().hex}")


class IdentityGroupID(ID, DBID):
    def __new__(cls, name: str):
        if not re.fullmatch(r"^identity_group_[a-f0-9]{32}$", name):
            raise ValueError(f"Invalid IdentityGroupID '{name}': must match "
                             "'identity_group_<32 hex chars>'.")
        return str.__new__(cls, name)

    @classmethod
    def generate(cls) -> "IdentityGroupID":
        return cls(f"identity_group_{uuid.uuid4().hex}")


class ProjectID(ID, DBID):
    def __new__(cls, name: str):
        if name != 'default' and not re.fullmatch(r"^project_[a-f0-9]{32}$",
                                                  name):
            raise ValueError(
                f"Invalid ProjectID '{name}': must be 'default' or match "
                "'project_<32 hex chars>'.")
        return str.__new__(cls, name)

    @classmethod
    def generate(cls) -> "ProjectID":
        return cls(f"project_{uuid.uuid4().hex}")


class GroupID(ID, DBID):
    def __new__(cls, name: str):
        if not re.fullmatch(r"^group_[a-f0-9]{32}$", name):
            raise ValueError(
                f"Invalid GroupID '{name}': must match 'group_<32 hex chars>'.")
        return str.__new__(cls, name)

    @classmethod
    def generate(cls) -> "GroupID":
        return cls(f"group_{uuid.uuid4().hex}")


class GroupMemberID(ID, DBID):
    @classmethod
    def generate(cls) -> "GroupMemberID":
        return cls(f"group_member_{uuid.uuid4().hex}")


class ExternalGroupMappingID(ID, DBID):
    @classmethod
    def generate(cls) -> "ExternalGroupMappingID":
        return cls(f"external_group_mapping_{uuid.uuid4().hex}")


class GroupMembershipRuleID(ID, DBID):
    @classmethod
    def generate(cls) -> "GroupMembershipRuleID":
        return cls(f"group_membership_rule_{uuid.uuid4().hex}")


class UploadedFileID(ID, DBID):
    @classmethod
    def generate(cls) -> "UploadedFileID":
        return cls(f"uploaded_file_{uuid.uuid4().hex}")


class ConnectorAuthorizationID(ID, DBID):
    @classmethod
    def generate(cls) -> "ConnectorAuthorizationID":
        return cls(f"connector_authorization_{uuid.uuid4().hex}")


class ConnectorAuthorizationUseID(ID, DBID):
    @classmethod
    def generate(cls) -> "ConnectorAuthorizationUseID":
        return cls(f"connector_authorization_use_{uuid.uuid4().hex}")
