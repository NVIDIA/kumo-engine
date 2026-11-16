# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Tuple

from kumo_core.types.schema import (
    BaseColumnConfig,
    IngestedTableFingerprint,
    TableFingerprint,
    TrainingTableFingerprint,
    source_name_path_token,
)
from kumoapi.typing import Dtype, Stype

from io_utils.json import from_json, to_json
from kumo.config.graph import ColumnConfig, TableConfig


def test_ingested_fingerprint():
    time_col_config = BaseColumnConfig(name="TIME_COL")
    end_time_col_config = BaseColumnConfig(name="END_TIME_COL")

    # test human-readable prefix
    fp1 = IngestedTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
        cols=dict([
            ("COL1", Dtype.float32),
            ("COL2", Dtype.time),
        ]),
        pkey=None,
        time_col=time_col_config,
        end_time_col=end_time_col_config,
    )
    # prefix is human-friendly
    table_prefix = f"fpv3.CONN.{source_name_path_token('TABLE_NAME')[:12]}."
    assert fp1.fpid().startswith(table_prefix)
    # suffix is an md5 sum
    assert len(fp1.fpid().split('.')[3]) == 32

    # test column order invariant
    fp2 = IngestedTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
        cols=dict([
            ("COL2", Dtype.time),
            ("COL1", Dtype.float32),
        ]),
        pkey=None,
        time_col=time_col_config,
        end_time_col=end_time_col_config,
    )
    assert fp1.cols == fp2.cols
    assert fp1.fpid() == fp2.fpid()
    fp2.cols = dict(reversed(fp1.cols.items()))
    assert fp1.fpid() == fp2.fpid()

    # test connector name mismatch
    fp3 = deepcopy(fp1)
    fp3.connector_name = "CONN_MISMATCH"
    assert fp1.connector_name != fp3.connector_name
    assert fp1.fpid() != fp3.fpid()
    assert fp3.fpid().startswith(table_prefix)

    # test table name mismatch
    fp4 = deepcopy(fp1)
    fp4.source_name = "TABLE_MISMATCH"
    assert fp1.source_name != fp4.source_name
    assert fp1.fpid() != fp4.fpid()
    assert fp4.fpid().startswith(
        f"fpv3.CONN.{source_name_path_token('TABLE_MISMATCH')[:12]}.")

    # test missing time column mismatch
    fp5 = deepcopy(fp1)
    fp5.time_col = None
    assert fp1.time_col != fp5.time_col
    assert fp1.fpid() != fp5.fpid()
    assert fp5.fpid().startswith(table_prefix)

    # test short names
    fp6 = IngestedTableFingerprint(
        connector_name="C",
        source_name="TA",
        cols=dict([
            ("COL1", Dtype.float32),
            ("COL2", Dtype.time),
        ]),
        pkey=None,
        time_col=time_col_config,
        end_time_col=end_time_col_config,
    )
    # prefix is human-friendly
    assert fp6.fpid().startswith(f"fpv3.C.{source_name_path_token('TA')[:12]}.")
    # suffix is an md5 sum
    assert len(fp6.fpid().split('.')[3]) == 32

    # test dtype mismatch
    fp7 = deepcopy(fp1)
    fp7.cols["COL1"] = Dtype.float64
    assert fp1.cols != fp7.cols
    assert fp1.fpid() != fp7.fpid()
    assert fp7.fpid().startswith(table_prefix)

    # test end time col mismatch
    fp8 = deepcopy(fp1)
    fp8.end_time_col = None
    assert fp1.fpid() != fp8.fpid()

    # Durable scalar parsing semantics must partition every stats,
    # materialization, and backend cache keyed by this fingerprint. The empty
    # default is omitted from the serialized hash input so existing tables
    # retain their byte-identical FPID.
    legacy_fpid = fp1.fpid()
    no_delimiter = deepcopy(fp1)
    no_delimiter.multicategorical_delimiters = {}
    comma = deepcopy(fp1)
    comma.multicategorical_delimiters = {'COL1': ','}
    pipe = deepcopy(fp1)
    pipe.multicategorical_delimiters = {'COL1': '|'}

    assert no_delimiter.fpid() == legacy_fpid
    assert comma.fpid() != legacy_fpid
    assert comma.fpid() != pipe.fpid()
    # Raw source ingestion remains shared; only interpretation-dependent
    # downstream cache identities change.
    assert comma.ingest_path_fpid == pipe.ingest_path_fpid


def test_timestamp_metadata_fingerprint():
    time_col_config = BaseColumnConfig(name="TIME_COL",
                                       metadata={"TS_FORMAT": "%Y-%m-%d"})
    end_time_col_config = BaseColumnConfig(name="END_TIME_COL")

    # Table with time column and metadata fingerprint.
    fp1 = IngestedTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
        cols=dict([
            ("COL1", Dtype.float32),
            ("COL2", Dtype.time),
        ]),
        pkey=None,
        time_col=time_col_config,
        end_time_col=end_time_col_config,
    )
    prefix = f"fpv3.CONN.{source_name_path_token('TABLE_NAME')[:12]}."
    assert fp1.fpid() == prefix + "07f6305b86ce03c0769d5d78bb1f02ca"
    assert len(fp1.fpid().split('.')[3]) == 32

    time_col_config2 = BaseColumnConfig(name="TIME_COL")

    # Table with time column and no metadata fingerprint.
    fp2 = IngestedTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
        cols=dict([
            ("COL1", Dtype.float32),
            ("COL2", Dtype.time),
        ]),
        pkey=None,
        time_col=time_col_config2,
        end_time_col=end_time_col_config,
    )
    assert fp2.fpid() == prefix + "dc48c6a559805b262bc27f84208c29b3"
    assert len(fp2.fpid().split('.')[3]) == 32


def test_de_serialize():
    time_col_config = BaseColumnConfig(name="TIME_COL")
    end_time_col_config = BaseColumnConfig(name="END_TIME_COL")

    fp1 = IngestedTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
        cols=dict([
            ("COL1", Dtype.float32),
            ("COL2", Dtype.time),
        ]),
        pkey=None,
        time_col=time_col_config,
        end_time_col=end_time_col_config,
    )
    # Note: in serialized json, Tuple and List are the same
    #   to_json requires List, from_json requires Tuple
    fp1_rt, = from_json(to_json([fp1]), Tuple[TableFingerprint])
    assert type(fp1) is type(fp1_rt)
    assert fp1 == fp1_rt
    assert fp1.fpid() == fp1_rt.fpid()

    fp2 = deepcopy(fp1)
    fp2.multicategorical_delimiters = {'COL1': '|'}
    fp2_rt, = from_json(to_json([fp2]), Tuple[TableFingerprint])
    assert fp2_rt.multicategorical_delimiters == {'COL1': '|'}
    assert fp2 == fp2_rt
    assert fp2.fpid() == fp2_rt.fpid()
    assert fp2.fpid() != fp1.fpid()

    fp3 = TrainingTableFingerprint(
        connector_name="CONNECTOR_NAME",
        source_name="TABLE_NAME",
    )
    fp3_rt, = from_json(to_json([fp3]), Tuple[TableFingerprint])
    assert type(fp3) is type(fp3_rt)
    assert fp3 == fp3_rt
    assert fp3.fpid() == fp3_rt.fpid()


def test_table_config():
    table_config = TableConfig(
        table_name="abc", data_source="def", data_version="123", cols=[
            ColumnConfig(name="a1", stype=Stype.ID, dtype=Dtype.int),
            ColumnConfig(name="a2", stype=Stype.ID, dtype=Dtype.int, stats={}),
        ])
    table_metadata_config = table_config.as_metadata_config()
    assert table_metadata_config.table_name == table_config.table_name
    assert isinstance(table_metadata_config.cols[0], BaseColumnConfig)
    assert isinstance(table_metadata_config.cols[1], BaseColumnConfig)
