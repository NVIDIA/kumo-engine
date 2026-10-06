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
 * GraphNameInput Component
 *
 * A text input component for graphs with async server-side name validation.
 *
 * `setError` reports the server validation result to the parent: it is called
 * with the server's error message when the name is invalid and reset to ""
 * on every keystroke, so parents can gate submit/Next buttons on it (see
 * pages/Graphs/NewGraph/NewGraph.tsx).
 */
export interface GraphNameInputProps {
    name?: string;
    setName?: (name: string) => void;
    readOnly?: boolean;
    setError?: (err: string) => void;
    showState?: boolean;
    shouldValidate?: boolean;
    placeholder?: string;
}

const GraphNameInput = ({
    name = "",
    setName,
    readOnly = false,
    setError,
    showState = false,
    shouldValidate = true,
    placeholder = "Please enter graph name"
}: GraphNameInputProps) => {
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
        kumoResource.Graph,
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
            data-testid="graph-name-input"
            id={"graph-name"}
            label={"Name"}
            name={"graph_name"}
            placeholder={placeholder}
            className="!text-base font-medium placeholder:font-medium placeholder:!text-base"
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
        />
    );
};

export default WithErrorBoundary(GraphNameInput);
