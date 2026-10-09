# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Query Language Validation
UNSUPPORTED_QUERY_DIALECT_TITLE = "Invalid Predictive Query"
UNSUPPORTED_QUERY_DIALECT_MESSAGE = "Query language syntax {} is not supported"

# Batch Prediction Workflow
BATCH_PREDICTION_WORKFLOW_DELETE_TITLE = "Cannot delete this query"

# Batch Prediction Jobs
BATCH_PREDICTION_JOBS_DELETE_TITLE = "Cannot delete this query"
BATCH_PREDICTION_JOBS_DELETE_MESSAGE = (
    "Cannot delete this query used in batch prediction jobs:{}")

# Invalid Predictive Query
INVALID_PREDICTIVE_QUERY_TITLE = "Invalid Predictive Query"
INVALID_PREDICTIVE_QUERY_MESSAGE = "{}"  # Takes error message as input

# Training Job Validation
TRAINING_JOB_NOT_FOUND_TITLE = "Invalid Training Job"
TRAINING_JOB_NOT_FOUND_MESSAGE = (
    "Training job {} was not found. Please ensure you are referencing a valid "
    "training job ID.")

TRAINING_JOB_INCOMPLETE_TITLE = "Invalid Training Job"
TRAINING_JOB_INCOMPLETE_MESSAGE = (
    "Training job {} has not completed successfully, so it cannot be used for "
    "batch prediction. Status is {}.")

TRAINING_JOB_NO_PQUERY_TITLE = "Internal Error"
TRAINING_JOB_NO_PQUERY_MESSAGE = (
    "Training job {} is not associated with a predictive query. Please "
    "contact Kumo for assistance.")

# Prediction Table Validation
PREDICTION_TABLE_JOB_FAILED_TITLE = "Prediction Table Unavailable"
PREDICTION_TABLE_JOB_FAILED_MESSAGE = (
    "Prediction table job {} did not finish successfully (status is {}), so "
    "it cannot be used for batch prediction. Please regenerate the prediction "
    "table and try again.")

# Graph Validation
INCOMPATIBLE_GRAPH_TITLE = "Incompatible Graph"
INCOMPATIBLE_GRAPH_MESSAGE = (
    "Training job {} was trained with table names {}, which are incompatible "
    "with the prediction job table names {}")

INCOMPATIBLE_EDGES_TITLE = "Incompatible Edges"
INCOMPATIBLE_EDGES_MESSAGE = (
    "Training job {} was trained with edges {}, which are incompatible with "
    "the prediction job edges {}")

INCOMPATIBLE_PRIMARY_KEY_TITLE = "Incompatible Primary Key"
INCOMPATIBLE_PRIMARY_KEY_MESSAGE = (
    "Training job {} was trained with table {} primary key {}, which is "
    "incompatible with the prediction job primary key {}")

INCOMPATIBLE_TIME_COLUMN_TITLE = "Incompatible Time Column"
INCOMPATIBLE_TIME_COLUMN_MESSAGE = (
    "Training job {} was trained with table {} time column {}, which is "
    "incompatible with the prediction job time column {}")

INCOMPATIBLE_END_TIME_COLUMN_TITLE = "Incompatible End Time Column"
INCOMPATIBLE_END_TIME_COLUMN_MESSAGE = (
    "Training job {} was trained with table {} end time column {}, which is "
    "incompatible with the prediction job end time column {}")

INCOMPATIBLE_COLUMNS_TITLE = "Incompatible Columns"
INCOMPATIBLE_COLUMNS_MESSAGE = (
    "Training job {} was trained with table {} columns {}, which are "
    "incompatible with the prediction job columns {}")

INCOMPATIBLE_COLUMN_TYPES_TITLE = "Incompatible Columns"
INCOMPATIBLE_COLUMN_TYPES_MESSAGE = (
    "Training job {} was trained with table {} column {} with type {} and "
    "stype {}, which is incompatible with the prediction job column {} with "
    "type {} and stype {}")

# Prediction Output Size
PREDICTION_OUTPUT_SIZE_TITLE = "Output Size Limit Exceeded"
PREDICTION_OUTPUT_SIZE_MESSAGE = (
    "The size of the output table is {} bytes, please reduce it to be under "
    "1GB by limiting the number of predictions generated using the Entity "
    "filters. If this is not possible, contact Kumo for assistance.")

# General validation errors
INVALID_PREDICTIVE_QUERY_TITLE = "Invalid Predictive Query"
INVALID_PREDICTIVE_QUERY_MESSAGE = "{}"  # Takes error message as input

# Creation errors
PREDICTIVE_QUERY_CREATION_ERROR_TITLE = "Error in creating Predictive Query"
PREDICTIVE_QUERY_CREATION_ERROR_MESSAGE = "{}"  # Takes error message as input

# Filter override errors
DEPRECATED_TEMPORAL_FILTER_OVERRIDE_TITLE = (
    "Error in creating Predictive Query with Filter Overrides")
DEPRECATED_TEMPORAL_FILTER_OVERRIDE_MESSAGE = (
    "The temporal_entity_filter_override field has been deprecated. Please "
    "use the entity_filter_override field instead.")

NON_EDITABLE_TARGET_FILTER_TITLE = (
    "Error in creating Predictive Query with Filter Overrides")
NON_EDITABLE_TARGET_FILTER_MESSAGE = (
    "The target filter field is not editable for this query type.")

CONNECTOR_TYPE_NOT_SUPPORTED_TITLE = "Unsupported Connector Type"
CONNECTOR_TYPE_NOT_SUPPORTED_MESSAGE = (
    "{} connector is not supported in this deployment.")

S3_PATH_INVALID_TITLE = "Invalid S3 Path"
S3_PATH_INVALID_MESSAGE = (
    "S3 path should have the format `s3://<bucket>/<key prefix>`; got {}")

S3_PATH_RESTRICTED_TITLE = "Restricted S3 Path"
S3_PATH_RESTRICTED_MESSAGE = (
    "You do not have permission to create a connector in this bucket: {}")

S3_PATH_NOT_FOUND_TITLE = "S3 Path Not Found"
S3_PATH_NOT_FOUND_MESSAGE = (
    "S3 path `{}` is either empty or you have insufficient permissions "
    "to access this resource. First, make sure that the directory exists. "
    "After that, refer to our user guide to give Kumo access to your "
    "S3 directory.")

S3_PATH_WRONG_AWS_REGION_MESSAGE = (
    "S3 path {} is in a different AWS region, {}. It must be in the AWS "
    "region of your Kumo instance, {}.")

S3_PATH_VALIDATION_ERROR_MESSAGE = (
    "Unable to validate S3 path. Please check that the path is correct "
    "and accessible. First, make sure that the directory exists. "
    "After that, refer to our user guide to give Kumo access to your "
    "S3 directory.")

# GCS connector validation
GCS_PATH_INVALID_TITLE = "Invalid GCS Path"
GCS_PATH_INVALID_MESSAGE = (
    "GCS path should have the format `gs://<bucket>/<prefix>`; got {}")

GCS_PATH_NOT_FOUND_TITLE = "GCS Path Not Found"
GCS_PATH_NOT_FOUND_MESSAGE = (
    "GCS path `{}` is either empty or you have insufficient permissions "
    "to access this resource. First, make sure that the directory exists. "
    "After that, ensure your GKE workload has the necessary IAM permissions "
    "to access your GCS bucket.")

GCS_PATH_VALIDATION_ERROR_MESSAGE = (
    "Unable to validate GCS path. Please check that the path is correct "
    "and accessible. First, make sure that the directory exists. "
    "After that, ensure your GKE workload has the necessary IAM permissions "
    "to access your GCS bucket.")

GCS_PATH_RESTRICTED_TITLE = "Restricted GCS Path"
GCS_PATH_RESTRICTED_MESSAGE = (
    "GCS bucket `{}` is restricted and cannot be used as a connector root "
    "directory. Please use a different GCS bucket.")

# ADLS connector validation
ADLS_PATH_INVALID_TITLE = "Invalid ADLS Path"
ADLS_PATH_INVALID_MESSAGE = (
    "ADLS path should have the format "
    "`abfss://<container>@<account>.dfs.core.windows.net/<prefix>`; got {}")

ADLS_PATH_NOT_FOUND_TITLE = "ADLS Path Not Found"
ADLS_PATH_NOT_FOUND_MESSAGE = (
    "ADLS path `{}` is either empty or you have insufficient permissions "
    "to access this resource. First, make sure that the directory exists. "
    "After that, ensure the Azure workload identity has permission to access "
    "the ADLS container.")

ADLS_PATH_VALIDATION_ERROR_MESSAGE = (
    "Unable to validate ADLS path. Please check that the path is correct "
    "and accessible. First, make sure that the directory exists. "
    "After that, ensure the Azure workload identity has permission to access "
    "the ADLS container.")

ADLS_PATH_RESTRICTED_TITLE = "Restricted ADLS Path"
ADLS_PATH_RESTRICTED_MESSAGE = (
    "ADLS container `{}` is restricted and cannot be used as a connector root "
    "directory. Please use a different ADLS container.")

# File upload validation
FILE_UPLOAD_NOT_SUPPORTED_TITLE = "File Upload Not Supported"
FILE_UPLOAD_NOT_SUPPORTED_MESSAGE = (
    "File upload requires a local filesystem data plane or a data backend "
    "that supports presigned upload URLs (S3 or GCS). The configured "
    "data_plane_storage_root ({root_dir!r}) does not. Reconfigure the "
    "deployment to use local, S3, or GCS storage, or verify that the service "
    "has the required IAM/storage permissions if it is already S3/GCS.")

FILE_UPLOAD_CONNECTOR_INVALID_TITLE = "Invalid Upload Connector"
FILE_UPLOAD_CONNECTOR_INVALID_MESSAGE = (
    "File upload operation is not supported for connector {}")

FILE_SIZE_EXCEEDED_TITLE = "File Too Large"
FILE_SIZE_EXCEEDED_MESSAGE = (
    "The uploaded file is too large. Maximum file size supported by Kumo "
    "is {}")

# Graph validation
GRAPH_NAME_EXISTS_TITLE = "Graph Already Exists"
GRAPH_NAME_EXISTS_MESSAGE = (
    "Graph with name={} already exists and has a different configuration than "
    "the inserted graph.")

GRAPH_RENAME_TITLE = "Cannot Rename Graph"
GRAPH_RENAME_MESSAGE = (
    "This graph definition already exists with name {}; cannot create another "
    "graph with the same definition under a different name.")

# Batch prediction validation
CONNECTOR_DELETE_TITLE = "Cannot Delete Connector"
CONNECTOR_DELETE_MESSAGE = (
    "You cannot delete this connector before deleting all associated "
    "tables:\n{}")

BATCH_PREDICTION_V2_DELETE_TITLE = "Cannot Delete Connector"
BATCH_PREDICTION_V2_DELETE_MESSAGE = (
    "You cannot delete this connector because it is used to connect named "
    "tables {}")

SYNTHESIZED_CONNECTOR_DELETE_TITLE = "Cannot Delete System Connector"
SYNTHESIZED_CONNECTOR_DELETE_MESSAGE = (
    "Unable to delete a system-defined connector.")

CONNECTOR_NAME_CHANGE_TITLE = "Cannot Change Connector Name"
CONNECTOR_NAME_CHANGE_MESSAGE = "Unable to change connector name."

CONNECTOR_TYPE_CHANGE_TITLE = "Cannot Change Connector Type"
CONNECTOR_TYPE_CHANGE_MESSAGE = (
    "Connector type cannot be changed from {} to {}. Create new connector if "
    "needed.")

# Graph validation errors
COLUMN_GROUP_NO_PKEY_TITLE = "Invalid Column Group"
COLUMN_GROUP_NO_PKEY_MESSAGE = (
    "Column Group needs to have exactly one primary key. Got no primary keys "
    "in: {}")

COLUMN_GROUP_MULTIPLE_PKEYS_TITLE = "Invalid Column Group"
COLUMN_GROUP_MULTIPLE_PKEYS_MESSAGE = (
    "Column Group may not have more than one primary key. Got multiple "
    "primary keys: '{}' and '{}'")

COLUMN_GROUP_MISSING_COLUMN_TITLE = "Invalid Column Group"
COLUMN_GROUP_MISSING_COLUMN_MESSAGE = (
    "Column Group contains a column that does not exist: '{}'. If this "
    "column is expected to be present in the table, consider editing table "
    "'{}` to include this column")

# Add the new error constant here
CANNOT_INFER_LINKS_WITH_EDGES_TITLE = "Cannot Infer Links"
CANNOT_INFER_LINKS_WITH_EDGES_MESSAGE = (
    "Cannot infer links if edges are already defined")

# Graph editing errors
GRAPH_NAME_EDIT_TITLE = "Invalid Graph Edit"
GRAPH_NAME_EDIT_MESSAGE = "Editing the 'name' field is not permitted"

# Table validation errors
TABLE_NOT_FOUND_TITLE = "Table Not Found"
TABLE_NOT_FOUND_MESSAGE = "Table IDs do not exist: {}"

TABLE_COLUMN_NOT_FOUND_TITLE = "Column Not Found"
SOURCE_TABLE_NOT_FOUND_TITLE = "Source Table Not Found"
DUPLICATE_COLUMN_NAMES_TITLE = "Duplicate Column Names"
INVALID_PRIMARY_KEY_TITLE = "Invalid Primary Key"
INVALID_TIME_COLUMN_TITLE = "Invalid Time Column"
INVALID_MULTICATEGORICAL_COLUMN_TITLE = "Invalid Multicategorical Column"
MISSING_COLUMN_DATA_TYPE_TITLE = "Missing Column Data Type"
INVALID_COLUMN_DATA_TYPE_TITLE = "Invalid Column Data Type"

TABLE_QUERY_ERROR_TITLE = "Table Query Error"
TABLE_QUERY_ERROR_MESSAGE = "Error querying table: {}"

TABLE_TYPE_ERROR_TITLE = "Invalid Table Type"
TABLE_TYPE_ERROR_MESSAGE = "Retrieved unsupported table type: {}"

TABLE_READ_ERROR_MESSAGE = (
    "Failed to read source table {} from connector {} due to the following "
    "error: {}")
TABLE_READ_PROVIDER_ERROR = (
    "The data provider rejected or could not complete the read. Verify the "
    "source path and connector identity permissions.")

TABLE_SIZE_EXCEEDS_LIMIT_TITLE = "Large Data Volume Detected"
TABLE_SIZE_EXCEEDS_LIMIT_MESSAGE = (
    "This operation on source table {} from {} connector {} will scan over {} "
    "of data which may lead to significant costs. To proceed, please "
    "contact Kumo for assistance.")

# Graph validation errors
INVALID_GRAPH_PARAMETERS_TITLE = "Invalid Graph Parameters"
INVALID_GRAPH_PARAMETERS_MESSAGE = "Invalid Graph Parameters: {}"

GRAPH_SNAPSHOT_NOT_FOUND_MESSAGE = (
    "Graph snapshot {} was not found. Please ensure you are referencing a "
    "valid graph snapshot ID.")

BATCH_PREDICTION_JOB_NOT_FOUND_MESSAGE = (
    "Batch prediction job {} was not found. Please ensure you are referencing "
    "a valid batch prediction job ID.")

PREDICTION_ARTIFACT_TYPE_NOT_FOUND_MESSAGE = (
    "Artifact type {} was not found. Please ensure you are referencing one of "
    "the originally specified artifact types for the batch prediction job.")

UNSUPPORTED_LOCAL_ARTIFACTS_MESSAGE = (
    "Getting artifact URLs is not supported for local outputs.")
