# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import kumoapi.source_table as source_table_api

from common_utils.error import PublicException
from io_utils import logging
from kumo.connector.connector import ConnectorError
from kumo.pquery.utils import validate_weight_table_num_rows
from kumo.validation import ValidationException, ValidationResponse
from kumo.validation.validation import ValidationError
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.activities import custom_train_table
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.training_job import CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH
from rest.driver.table_logic import _get_source_table_metadata
from rest.utils.driver_common import (
    async_connector_lifecycle,
    extract_source_table_type,
    get_connector_from_spec,
    get_connector_from_spec_async,
)
from statedb.utils.errors import sql_query_error_message

log = logging.getLogger(__name__)


@connector_activity_registry.connector_authorized_activity
async def validate_custom_train_table_activity(
    reference: ConnectorExecutionRef,
    custom_table: source_table_api.SourceTableType,
    weight_col: Optional[str],
    orig_train_table_path: str,
    extensive_validation: bool,
    split_col: Optional[str] = None,
) -> ValidationResponse:
    source_table, source_table_name, _ = extract_source_table_type(custom_table)
    source_type = getattr(source_table, 'data_source_type', None)
    source_type = getattr(source_type, 'value', source_type)
    is_custom_sql = (isinstance(source_table,
                                source_table_api.CustomSQLSourceTable)
                     or (isinstance(source_type, str)
                         and source_type.casefold() == 'custom_sql'))
    connector = await get_connector_from_spec_async(
        reference, resolved_factory=get_connector_from_spec)
    async with async_connector_lifecycle(connector):
        try:
            assert source_table_name is not None
            connector_type_name = connector._get_connector_type()
            connector_type_name = getattr(connector_type_name, 'value',
                                          connector_type_name)
            connector_type_name = str(connector_type_name).casefold()
            if is_custom_sql:
                from kumo_core.types import ConnectorType

                from async_utils.sync import sync_to_async
                from kumo.data.custom_sql import (
                    advisory_read_only_check,
                    subquery_safe_sql,
                    validate_connector_sql_scope,
                )

                connector_type = ConnectorType(connector_type_name)
                if connector_type != ConnectorType.DATABRICKS:
                    raise ValueError(
                        'Custom SQL training sources are currently supported '
                        'only for Databricks connectors.')
                advisory_read_only_check(source_table.sql, connector_type)
                validate_connector_sql_scope(source_table.sql, connector_type,
                                             connector)
                sql = subquery_safe_sql(source_table.sql, connector_type)
                table_metadata = await sync_to_async(
                    connector.get_query_metadata,
                    executor=None,
                )(sql)
            else:
                table_metadata = await _get_source_table_metadata.coroutine(
                    source_table_name, connector)
        except ConnectorError as error:
            return ValidationResponse(errors=[
                ValidationError(
                    title=('Warehouse query failed' if is_custom_sql else
                           'Source table validation failed'),
                    message=(
                        sql_query_error_message(error) if is_custom_sql else
                        'The warehouse could not validate this source table. '
                        'Verify that it exists and that the connector can '
                        'read it.'),
                )
            ])
        except (AssertionError, ValueError, PublicException) as error:
            public_error = ValidationException(message=str(error))
            return ValidationResponse(errors=[
                ValidationError(
                    title=('Invalid custom SQL'
                           if is_custom_sql else 'Source table not found'),
                    message=str(public_error),
                )
            ])

        columns_by_name = {
            column.name: column
            for column in table_metadata.cols
        }
        if weight_col is not None and weight_col not in columns_by_name:
            return ValidationResponse(errors=[
                ValidationError(
                    title='Weight column not found',
                    message=(f"Weight column '{weight_col}' not found in "
                             f"source table '{source_table_name}'."),
                )
            ])
        if split_col is not None and split_col not in columns_by_name:
            return ValidationResponse(errors=[
                ValidationError(
                    title='Split column not found',
                    message=(f"Split column '{split_col}' not found in "
                             f"source table '{source_table_name}'."),
                )
            ])

        # Names only, and the same rule on every vendor. What the columns
        # have to be *typed* as is checked on the parquet that arrives, by
        # `validate_weighted_train_table`, against the contract DiskGraph
        # exports -- a warehouse's own type metadata is a different vocabulary
        # and answering with it would be a second contract. Refusing a
        # misnamed table here is worth the one read because it saves paying
        # for the export first.
        missing = custom_train_table.missing_source_column_errors(
            [column.name for column in table_metadata.cols],
            orig_train_table_path,
            weight_col,
            split_col,
        )
        if missing:
            return ValidationResponse(errors=missing)

        response = ValidationResponse()
        if extensive_validation:
            if CUSTOM_TT_ALLOW_ROW_COUNT_MISMATCH.parse_value():
                log.warning(
                    'Skipping custom training table row-count validation')
            else:
                response.merge(await validate_weight_table_num_rows(
                    connector,
                    source_table,
                    train_table_path=orig_train_table_path,
                ))
        return response
