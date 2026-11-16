# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from io_utils import kumo_dataclass


class RestStatusCode(Enum):
    INVALID = "Reserved (Invalid) status code"
    OK = "Successful"
    CONNECTOR_TYPE_NOT_SUPPORTED = "Failed. Connector type not supported"
    CONNECTOR_ALREADY_EXISTS = "Failed. Connector already exists"
    CONNECTOR_DOES_NOT_EXIST = "Failed. Connector does not exist"
    CONNECTOR_DELETE_IS_INVALID = "Failed. Connector delete is invalid"
    TABLE_DOES_NOT_EXIST = "Failed. Table does not exist"
    TABLE_NOT_PROPERLY_CONFIGURED = "Failed. Table is not properly configured."
    GRAPH_DOES_NOT_EXIST = "Failed. Graph does not exist"
    PREDICTIVE_QUERY_DOES_NOT_EXIST = "Failed. Predictive query does not exist"
    TRAINING_JOB_DOES_NOT_EXIST = "Failed. Training Job does not exist"
    PREDICTION_JOB_DOES_NOT_EXIST = "Failed. Prediction Job does not exist"
    PREDICTION_WORKFLOW_DOES_NOT_EXIST = "Failed. Prediction Workflow does " \
        "not exist"
    EVALUATION_NOT_YET_COMPLETED = 'Failed. Training job test set ' \
        'evaluation not done.'
    EVALUATION_NOT_AVAILABLE = 'Failed. Training job test set evaluation ' \
        'done but eval results not available'
    ENCODERS_NOT_AVAILABLE = 'Failed. Encoders are not available: the ' \
        'training job did not complete.'
    CREDENTIALS_NOT_PROVIDED = "Failed. Credentials have not been supplied"
    AUTHTYPE_NOT_SUPPORTED = "Failed. Only username/password supported"
    NOT_IMPLEMENTED = "Failed. Functionality not implemented"
    QUERY_ALREADY_EXIST = "Failed. Query already exist"
    QUERY_DOES_NOT_EXIST = "Failed. Query does not exist"
    QUERY_DELETE_IS_INVALID = "Failed. Query delete is not valid"
    QUERY_CREATION_ERROR = "Failed. Query creation error"
    QUERY_PLAN_CREATION_ERROR = "Failed. Unable to create task for the query"
    EXPERIMENT_ALREADY_EXISTS = "Failed. Query already exists in the system."
    EXPERIMENT_CREATION_ERROR = "Failed. Experiment creation error"
    EXPERIMENT_DOES_NOT_EXIST = "Failed. Experiment does not exist"
    EXPERIMENT_DELETE_FAILED = "Failed. Experiment was unable to be deleted."
    AUTH_WRONG_CREDENTIALS = "Incorrect email or password."
    AUTH_LOGIN_RATE_LIMITED = "Too many sign-in attempts. Please wait and " \
        "try again."
    AUTH_NOT_ADMIN = "Failed. This operation can only be performed " \
        "by an Organization Administrator."
    AUTH_INVALID_API_KEY = "Failed. API Key is not valid."
    AUTH_INVALID_TOKEN = "Failed. Access Token is not valid or expired."
    AUTH_MISSING_TOKEN = "Failed. Access Token was not provided."
    ORG_ALREADY_EXISTS = "Failed. This Organization already exists."
    INVALID_PERMISSIONS = "Failed. You do not have permission to access " \
        "this resource"
    INVALID_ISSUER = "Failed. Error code: INV_ISS. Please contact Kumo for " \
        "assistance"
    IO_ERROR = "Failed. Error with I/O."
    MISCELLANEOUS_INTERNAL_SERVER_ERROR = "Failed. Internal error."
    UPLOADED_FILE_DOES_NOT_EXIST = "Failed. Uploaded file does not exist."


@kumo_dataclass
class IDResponse:
    id: str


@kumo_dataclass
class EmptyResponse:
    pass


@kumo_dataclass
class PatchResponse:
    resource_updated: bool


PQ_NEEDS_RETRAIN: str = (
    "This Predictive Query, or the Graph/Tables belonging to this "
    "Predictive Query have recently been edited. Please retrain this "
    "Predictive Query before performing any Batch Predictions.")
