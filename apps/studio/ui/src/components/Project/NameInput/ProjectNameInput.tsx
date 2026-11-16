// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useState } from "react";

import { TextInput } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import {
    useValidatedName,
    ValidationResponse
} from "hooks/validation/validatedName";

import { InputState } from "types/inputs";

import { kumoResource } from "utils/Constants";

/**
 * ProjectNameInput Component
 *
 * A text input component for projects with validation.
 *
 */
export interface ProjectNameInputProps {
    name?: string;
    setName?: (name: string) => void;
    readOnly?: boolean;
    setError?: (err: string) => void;
    showState?: boolean;
    shouldValidate?: boolean;
    placeholder?: string;
}

const ProjectNameInput = ({
    name = "",
    setName,
    readOnly = false,
    setError,
    showState = false,
    shouldValidate = true,
    placeholder = "Please enter project name"
}: ProjectNameInputProps) => {
    const [invalidName, setInvalidName] = useState<boolean>(false);

    const [nameInputState, setInputState] = useState(InputState.None);

    const [validationResponse, setValidationResponse] =
        useState<ValidationResponse>({
            "isValid": true,
            "errMsg": ""
        });

    const onValidationStart = useCallback(
        () => setInputState(InputState.Validating),
        []
    );
    const onValidationEnd = useCallback(
        (valResponse: {
            validationResponse: ValidationResponse;
            inputState?: string;
        }) => {
            setInputState(valResponse?.inputState as InputState);
            setInvalidName(!valResponse?.validationResponse?.isValid);
            setValidationResponse(valResponse?.validationResponse);
            if (!valResponse?.validationResponse?.isValid)
                setError?.(valResponse.validationResponse.errMsg);
        },
        [setError]
    );
    const onShouldValidateChange = useCallback(() => {
        setValidationResponse({ isValid: true, errMsg: "" });
    }, []);

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [validatedName, setValName] = useValidatedName(
        name,
        /* onValidationStart */ onValidationStart,
        /* onValidationEnd */ onValidationEnd,
        /* onShouldValidateChange */ onShouldValidateChange,
        kumoResource.Project,
        /* shouldValidate */ shouldValidate && !readOnly && name !== "",
        /* validateOnMount */ false,
        /* entityID */ ""
    );

    const onTextChange = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            if (setName) {
                setName(e?.target?.value);
            }
            if (setError) {
                setError("");
            }
            setValName(e?.target?.value);
            setValidationResponse({
                "isValid": true,
                "errMsg": ""
            });
        },
        [setValName, setName, setError]
    );

    return (
        <TextInput
            data-testid="project-name-input"
            id={"project-name"}
            label={"Project name"}
            labelClassname="font-medium text-title-secondary leading-5"
            name={"project_name"}
            placeholder={placeholder}
            readOnly={readOnly}
            value={name}
            isInvalid={invalidName && !validationResponse.isValid}
            errorMsg={validationResponse.errMsg}
            onChange={onTextChange}
            state={
                name?.trim() === "" || readOnly
                    ? InputState.None
                    : nameInputState
            }
            showState={showState}
            className="px-4 py-2 h-10 font-medium placeholder:font-medium placeholder:text-base !text-base"
        />
    );
};

export default WithErrorBoundary(ProjectNameInput);
