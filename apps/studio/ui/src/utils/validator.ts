// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { validateProjectsName } from "gen/v0/api/projects/projects";
import { validateV1GraphsName } from "gen/v1/api/graphs/graphs";
import { validateV1TablesNameByTableName } from "gen/v1/api/tables/tables";

import { CustomError } from "types/common";

import { cancelAPICall, inputState, kumoResource } from "./Constants";
import { parseErrorMessage } from "./errorHandler";

/**
 *  This method invokes REST API call to validate the name of kumo resource
 *  which is passed as a string argument 'nameValidatorResoure'
 *  to this function.
 *
 * @param self class instance
 * @param nameValidatorResource kumo resource for which name to be validated
 */
export async function validateResourceName(
    nameValidatorResource: string,
    name = "",
    entityId: string = "",
    signal: AbortSignal | undefined,
    projectId?: string | null
) {
    const err = emptyNameCheck(nameValidatorResource, name);
    if (err) {
        return {
            validationResponse: {
                isValid: false,
                errMsg: err
            },
            inputState: inputState.None,
            invalidName: false
        };
    }
    const nameStr = checkAgainstProtectedNames(name);
    if (nameStr !== "") {
        return {
            validationResponse: {
                isValid: false,
                errMsg:
                    'Unable to use the name "' +
                    nameStr +
                    '" since it is a restricted word. Please choose a different name.'
            },
            inputState: inputState.None,
            invalidName: false
        };
    }
    const underscoreErr = consecutiveUnderscoreCheck(
        nameValidatorResource,
        name
    );
    if (underscoreErr) {
        return {
            validationResponse: {
                isValid: false,
                errMsg: underscoreErr
            },
            inputState: inputState.None,
            invalidName: false
        };
    }

    try {
        let response = null;
        switch (nameValidatorResource) {
            case kumoResource.Table:
                if (!entityId || !projectId) {
                    return {
                        validationResponse: {
                            isValid: false,
                            errMsg: !entityId
                                ? "Please select Connector"
                                : "Please select Project"
                        },
                        inputState: inputState.None
                    };
                }
                response = await validateV1TablesNameByTableName(
                    name,
                    { project_id: projectId },
                    undefined,
                    signal
                );
                break;

            case kumoResource.Graph:
                if (!projectId) {
                    return {
                        validationResponse: {
                            isValid: false,
                            errMsg: "Please select Project"
                        },
                        inputState: inputState.None
                    };
                }
                response = await validateV1GraphsName(
                    { name, project_id: projectId },
                    undefined,
                    signal
                );

                break;

            case kumoResource.Project:
                response = await validateProjectsName(
                    { name },
                    undefined,
                    signal
                );
                break;

            default:
                break;
        }

        if (response && response.data) {
            return {
                validationResponse: {
                    isValid: response.data.is_valid,
                    errMsg: response.data.error_msg
                },
                inputState: response.data.is_valid
                    ? inputState.Valid
                    : inputState.None,
                invalidName: !response.data.is_valid
            };
        } else {
            return {
                validationResponse: { isValid: false, errMsg: "" },
                inputState: inputState.None,
                invalidName: false
            };
        }
    } catch (error) {
        const response = {
            isValid: false,
            errMsg: parseErrorMessage(error as CustomError)
        };
        if (response?.errMsg !== cancelAPICall) {
            return {
                validationResponse: response,
                inputState: inputState.None,
                invalidName: true
            };
        } else {
            return {
                validationResponse: {
                    isValid: true,
                    errMsg: false
                },
                inputState: inputState.None,
                invalidName: false
            };
        }
    }
}

/**
 *  This method detects whether the name of kumo resource 'resource'
 *  is empty. Returns the error if empty.
 *
 * @param resource kumo resource name.
 */
function emptyNameCheck(resource: string, name: string | null = null) {
    let err = "";
    switch (resource) {
        case kumoResource.Connector:
            if (name?.trim() === "")
                err = "Please enter a valid Connector name.";
            break;

        case kumoResource.Table:
            if (name?.trim() === "") err = "Please enter name of your Table";
            break;

        case kumoResource.Graph:
            if (name?.trim() === "") err = "Please enter Graph Name";
            break;

        case kumoResource.Project:
            if (name?.trim() === "") err = "Please enter Project Name";
            break;

        default:
            break;
    }
    return err;
}

/**
 *  This method detects whether the name of kumo resource 'resource'
 *  is one of the kumo protected name strings. Returns the resource
 *  name if it is one of the restricted 'protected' names, "" otherwise.
 *
 * @param resource kumo resource.
 * @param resName kumo resource name.
 */
function checkAgainstProtectedNames(name: string | null) {
    const protectedNames = ["table", "graph"];

    name = name?.trim() as string;

    if (!name) {
        name = "";
    }
    if (!(protectedNames.indexOf(name.toLowerCase()) > -1)) {
        //Reset name if name is NOT one of protected names
        name = "";
    }
    return name;
}

/**
 * This method checks whether the given name contains
 * consecutive underscores (`__`). If it does, it returns a resource-specific
 * error message. Otherwise, it returns an empty string.
 *
 * @param resource kumo resource.
 * @param name kumo resource name.
 */
function consecutiveUnderscoreCheck(resource: string, name: string) {
    if (/__+/.test(name)) {
        switch (resource) {
            case kumoResource.Connector:
                return "Connector name contains consecutive underscores. Please replace them with a single underscore.";
            case kumoResource.Table:
                return "Table name contains consecutive underscores. Please replace them with a single underscore.";
            case kumoResource.Graph:
                return "Graph name contains consecutive underscores. Please replace them with a single underscore.";
            case kumoResource.Project:
                return "Project name contains consecutive underscores. Please replace them with a single underscore.";
            default:
                return "Name contains consecutive underscores. Please replace them with a single underscore.";
        }
    }
    return "";
}
