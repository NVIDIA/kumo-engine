// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import { useIsMounted } from "@kumo-ai/kumo-ui-catalog";
import { useQuery } from "@tanstack/react-query";

import { useProjectId } from "hooks/project/projects";

import { kumoResource, splitKey } from "../../utils/Constants";

import { useQueryOptions } from "../misc/queryOptions";

export type ValidationResponse = {
    isValid: boolean;
    errMsg: string;
};

export const useValidatedName = (
    initialValue: string,
    onValidationStart: () => void,
    onValidationEnd: (s: {
        validationResponse: ValidationResponse;
        inputState?: string;
    }) => void,
    onShouldValidateChange: () => void,
    resource: kumoResource,
    shouldValidate: boolean = true,
    validateOnMount: boolean = true,
    entityId: string = ""
): [string, (s: string) => void] => {
    const [input, setInput] = useState(initialValue);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const isMounted = useIsMounted();
    const [debouncedInput, setDebouncedInput] = useState("");
    const { queryOption } = useQueryOptions();
    const prevDataRef = useRef<typeof data | null>(null);

    const projectId = useProjectId();

    // If shouldValidate changes, make sure to remove any validation errors
    // that were triggered before
    useEffect(() => {
        onShouldValidateChange();
    }, [shouldValidate, onShouldValidateChange]);

    useEffect(() => {
        return () => {
            clearTimeout(timerRef.current as NodeJS.Timeout);
        };
    }, []);

    const restartTimer = useCallback(() => {
        // Reset any existing timer
        clearTimeout(timerRef.current as NodeJS.Timeout);

        // Start a timer to validate name method in 1 seconds
        const seconds = 1;
        timerRef.current = setTimeout(() => {
            setDebouncedInput(input + splitKey + new Date().getTime());
        }, seconds * 1000);
    }, [input]);

    const { data, isLoading } = useQuery(
        queryOption.nameValidation(
            debouncedInput,
            resource,
            entityId,
            projectId
        )
    );

    useEffect(() => {
        if (data?.inputState && data !== prevDataRef.current && !isLoading) {
            prevDataRef.current = data;
            const newData = {
                validationResponse:
                    data.validationResponse as ValidationResponse,
                inputState: data.inputState
            };
            onValidationEnd(newData);
        }
    }, [data, onValidationEnd, isLoading]);

    useEffect(() => {
        if (
            shouldValidate &&
            (validateOnMount || isMounted?.current === true)
        ) {
            onValidationStart();
            restartTimer();
        }
    }, [
        isMounted,
        input,
        restartTimer,
        onValidationStart,
        shouldValidate,
        validateOnMount
    ]);

    return [input, setInput];
};
