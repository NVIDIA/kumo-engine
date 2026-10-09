// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { forwardRef, useCallback, useImperativeHandle } from "react";

import { TextInput, usePrevious } from "@kumo-ai/kumo-ui-catalog";

import { WithErrorBoundaryForwardRef } from "components/@hoc/WithErrorBoundary";

import {
    useValidatedName,
    ValidationResponse
} from "hooks/validation/validatedName";

import { InputState } from "types/inputs";

import { inputState, kumoResource } from "utils/Constants";
import { isValid } from "utils/Helpers/typeGuards";

export interface ValidatedNameInputProps {
    readOnly: boolean;
    value: string;
    setName: (newvalue: string) => void;
    nameInputState: InputState;
    setNameInputState: (newval: InputState) => void;
    nameValidationResponse: ValidationResponse;
    setNameValidationResponse: (newval: ValidationResponse) => void;
    connectorId: string;
    updateLocalFileTableName: (name: string) => void;
    fileConnectors: string[];
    /** Applied to the field label so callers can match a page's form layout. */
    labelClassName?: string;
}

const ValidatedNameInput = forwardRef(
    (
        {
            readOnly,
            value,
            setName,
            nameInputState,
            setNameInputState,
            nameValidationResponse,
            setNameValidationResponse,
            connectorId,
            updateLocalFileTableName,
            fileConnectors,
            labelClassName
        }: ValidatedNameInputProps,
        ref
    ) => {
        const prevValue = usePrevious(value);

        const onValidationStart = useCallback(() => {
            setNameInputState(InputState.Validating);
        }, [setNameInputState]);

        const onValidationEnd = useCallback(
            (valResponse: {
                validationResponse: ValidationResponse;
                inputState?: string;
            }) => {
                if (valResponse) {
                    setNameInputState(valResponse.inputState as InputState);
                    setNameValidationResponse(valResponse.validationResponse);
                }
            },
            [setNameInputState, setNameValidationResponse]
        );

        const onShouldValidateChange = useCallback(() => {
            setNameValidationResponse({ isValid: true, errMsg: "" });
        }, [setNameValidationResponse]);

        const [validatedName, setValidatedName] = useValidatedName(
            /* initialValue */ value,
            /* onValidationStart */ onValidationStart,
            /* onValidationEnd */ onValidationEnd,
            /* onShouldValidateChange */ onShouldValidateChange,
            kumoResource.Table,
            /* shouldValidate */ value !== "" &&
                value !== prevValue &&
                !readOnly,
            /* validateOnMount */ true,
            /* entityId */ connectorId
        );

        useImperativeHandle(ref, () => ({
            updateName(name: string) {
                setValidatedName(name);
                setName(name);
            }
        }));

        return (
            <TextInput
                id={"table name"}
                label={"Table name"}
                labelVariant="secondary"
                labelClassname={labelClassName}
                className="flex-1 font-medium placeholder:font-medium"
                value={validatedName}
                placeholder="Use default"
                onChange={(e) => {
                    if (fileConnectors?.includes(connectorId)) {
                        updateLocalFileTableName(e?.target?.value);
                    } else {
                        setValidatedName(e?.target?.value);
                        // Set the name in the outer component even before validation
                        // completes
                        setName(e?.target?.value);
                    }
                }}
                readOnly={readOnly}
                isInvalid={
                    validatedName?.trim() !== ""
                        ? !nameValidationResponse?.isValid
                        : false
                }
                errorMsg={nameValidationResponse?.errMsg}
                state={
                    (isValid(validatedName)
                        ? nameInputState
                        : inputState.None) as InputState
                }
                showState={true}
            />
        );
    }
);

export default WithErrorBoundaryForwardRef(ValidatedNameInput);
