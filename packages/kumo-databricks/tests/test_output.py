# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The destination rules the hosted connector and the local sink share."""

from __future__ import annotations

import pytest
from kumo_databricks import output
from kumo_databricks.errors import ArtifactValidationError


def test_an_output_names_its_schema_and_its_table():
    assert output.within_schema("features__kumo__churn",
                                "features") == ("features", "churn")


def test_unity_catalog_resolves_a_schema_without_regard_to_case():
    assert output.within_schema("FEATURES__kumo__churn",
                                "features") == ("FEATURES", "churn")


@pytest.mark.parametrize(
    "name",
    ["churn", "a__kumo__b__kumo__c", "__kumo__churn", "features__kumo__"])
def test_a_name_that_is_not_one_schema_and_one_table_is_refused(name):
    with pytest.raises(ValueError, match="exactly one schema"):
        output.within_schema(name, "features")


def test_a_table_outside_the_connector_schema_is_refused():
    with pytest.raises(ValueError, match="does not match the connector"):
        output.within_schema("reporting__kumo__churn", "features")


def test_a_connector_with_no_schema_addresses_nothing():
    with pytest.raises(ValueError, match="does not match the connector"):
        output.within_schema("features__kumo__churn", "")


def test_a_stage_sits_under_the_one_volume_a_grant_names():
    stage = output.staging_stage("main", "features")
    assert stage.startswith("/Volumes/main/features/_kumo_staging/"
                            "artifact-export/")


def test_the_same_export_stages_in_the_same_place_twice():
    first = output.staging_stage("main", "features", "export-1")
    assert first == output.staging_stage("main", "features", "export-1")
    assert first != output.staging_stage("main", "features", "export-2")
    assert first != output.staging_stage("main", "features")


def test_an_existing_volume_is_described_rather_than_created():
    run = []
    output.ensure_staging_volume(run.append, "main", "features")
    assert run == ["DESCRIBE VOLUME `main`.`features`.`_kumo_staging`"]


def test_a_volume_that_is_not_there_is_created():
    run = []

    def execute(statement):
        run.append(statement)
        if statement.startswith("DESCRIBE"):
            raise RuntimeError("[VOLUME_NOT_FOUND] no such volume")

    output.ensure_staging_volume(execute, "main", "features")
    assert run[-1] == ("CREATE VOLUME IF NOT EXISTS "
                       "`main`.`features`.`_kumo_staging`")


def test_a_volume_that_cannot_be_described_names_the_grants_it_needs():
    def execute(statement):
        raise RuntimeError("PERMISSION_DENIED")

    with pytest.raises(ArtifactValidationError,
                       match="READ VOLUME, and WRITE VOLUME"):
        output.ensure_staging_volume(execute, "main", "features")


def test_a_volume_that_cannot_be_created_asks_an_administrator():
    def execute(statement):
        if statement.startswith("DESCRIBE"):
            raise RuntimeError("[VOLUME_NOT_FOUND] no such volume")
        raise RuntimeError("PERMISSION_DENIED")

    with pytest.raises(ArtifactValidationError, match="administrator"):
        output.ensure_staging_volume(execute, "main", "features")


def test_a_warehouse_failure_that_is_neither_is_left_alone():
    def execute(statement):
        raise RuntimeError("the warehouse is asleep")

    with pytest.raises(RuntimeError, match="asleep"):
        output.ensure_staging_volume(execute, "main", "features")
