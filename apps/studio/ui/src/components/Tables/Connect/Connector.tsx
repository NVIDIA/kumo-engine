// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo } from "react";
import { useLocation } from "react-router";

import {
    cn,
    Form,
    FormField,
    FormWrapper,
    getFromLocalStorage,
    Label,
    Select
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

import { ConnectorDropdownOption, InputState } from "types/inputs";
import { FlowType } from "types/tables";

import { storageKeys } from "utils/Constants";

import { formLabelClassName, formRowClassName } from "./formRow";

export interface ConnectorSelectionProps {
    connectors: Array<ConnectorDropdownOption>;
    onConnectorChange: (e: ConnectorDropdownOption) => void;
    nameValidationState: InputState;
    fileConnectors: string[];
    flowType: FlowType;
    connectorId?: string;
    hideLabel?: boolean;
    placeHolder?: string;
    disableDefaultSelection?: boolean;
    /** Renders the label beside the control instead of above it. */
    horizontal?: boolean;
}

const ConnectorSelection = ({
    connectors,
    onConnectorChange,
    nameValidationState,
    fileConnectors,
    flowType,
    connectorId,
    hideLabel,
    placeHolder = "Select Connector",
    disableDefaultSelection = false,
    horizontal = false
}: ConnectorSelectionProps) => {
    const isFileConnector = fileConnectors?.includes(connectorId as string);

    const filteredConnectors = useMemo(() => {
        const listConnectors = connectors?.filter(
            (c) => !fileConnectors?.includes(c?.value as string)
        );
        if (flowType === FlowType.EDIT && connectorId !== "") {
            const checkConId = listConnectors.findIndex(
                (c) => c.value === connectorId
            );
            if (checkConId === -1) {
                listConnectors.push({ value: connectorId, label: connectorId });
            }
        }
        return listConnectors;
    }, [connectorId, connectors, fileConnectors, flowType]);

    const location = useLocation();

    const setDefaultConnector = useCallback(() => {
        const lastUsed = getFromLocalStorage<{ value: string; type: string }>(
            storageKeys.lastUsedConn
        );

        const clist = filteredConnectors?.map((c) => c?.value);
        let connector = filteredConnectors[0];
        if (location?.state?.value) {
            if (clist?.includes(location?.state?.value)) {
                connector = filteredConnectors.find(
                    (c) => c?.value === location?.state?.value
                ) as ConnectorDropdownOption;
            }
        } else if (lastUsed) {
            if (clist?.includes(lastUsed?.value)) {
                connector = filteredConnectors.find(
                    (c) => c?.value === lastUsed?.value
                ) as ConnectorDropdownOption;
            }
        }
        onConnectorChange(connector);
    }, [filteredConnectors, location, onConnectorChange]);

    useEffect(() => {
        if (
            filteredConnectors?.length > 0 &&
            connectorId === "" &&
            !disableDefaultSelection
        ) {
            setDefaultConnector();
        }
    }, [
        filteredConnectors,
        connectorId,
        onConnectorChange,
        setDefaultConnector,
        disableDefaultSelection
    ]);

    const control = (
        <div className={horizontal ? "w-full" : "w-72.5"}>
            <Select
                id="select-connector"
                placeholder={placeHolder}
                search={true}
                emptyText="No connector found."
                mode="single"
                selectedValues={
                    isFileConnector || connectorId === ""
                        ? undefined
                        : [connectorId as string]
                }
                options={filteredConnectors.map((item) => ({
                    value: item.value as string,
                    label: item.label as string
                }))}
                onValuesChange={(newValue) => {
                    const currSel = filteredConnectors.find((item) =>
                        newValue.includes(item.value as string)
                    );
                    onConnectorChange(currSel as ConnectorDropdownOption);
                }}
                disabled={
                    flowType !== FlowType.CREATE ||
                    nameValidationState === InputState.Validating
                }
            />
        </div>
    );

    const label = !hideLabel && (
        <Label
            htmlFor="Connector"
            className={cn(
                "text-title-secondary",
                horizontal && formLabelClassName
            )}>
            Connector
        </Label>
    );

    if (horizontal) {
        return (
            <div className={formRowClassName}>
                {label}
                {control}
            </div>
        );
    }

    return (
        <Form>
            <FormWrapper>
                <FormField className="gap-0">
                    {label}
                    {control}
                </FormField>
            </FormWrapper>
        </Form>
    );
};

export default WithErrorBoundary(ConnectorSelection);
