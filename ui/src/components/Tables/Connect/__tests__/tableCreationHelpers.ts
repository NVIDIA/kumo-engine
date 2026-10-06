// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";

import { CreateTableArgs } from "types/tables";

import { setCreateTableResponseType } from "./MockFunction/CreateTable.mock";
import { setValidateTableResponseType } from "./MockFunction/ValidateTable.mock";

export const setupValidateResponse = (
    type: "success" | "failure" | "invalid"
) => {
    if (type === "failure") {
        setValidateTableResponseType(testResponseType.ERROR);
    } else if (type === "invalid") {
        setValidateTableResponseType(testResponseType.INVALID);
    } else {
        setValidateTableResponseType(testResponseType.ALL);
    }
};

export const setupCreateResponse = (
    type: "success" | "failure" | "invalid"
) => {
    if (type === "failure") {
        setCreateTableResponseType(testResponseType.ERROR);
    } else if (type === "invalid") {
        setCreateTableResponseType(testResponseType.INVALID);
    } else {
        setCreateTableResponseType(testResponseType.ALL);
    }
};

export const assertValidateApiCalled = async (args: CreateTableArgs) => {
    await waitFor(
        () => {
            const url = `v1/tables/validate`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                "headers": {
                    "Content-Type": "application/json"
                },
                "params": { "project_id": "project-tests" }
            });
        },
        { timeout: 2000 }
    );
};

export const assertCreateApiCalled = async (
    args: CreateTableArgs,
    newTableName: string,
    forceRename: boolean = false
) => {
    await waitFor(() => {
        const url = `v1/tables/`;
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
            headers: { "Content-Type": "application/json" },
            params: {
                force_rename: forceRename,
                name_alias: newTableName,
                project_id: "project-tests"
            }
        });
    });
};

export const closeFailureAlert = async () => {
    const failureText = await screen.findByText(
        "Failed to create table!",
        {},
        { timeout: 5000 }
    );
    expect(failureText).toBeInTheDocument();

    const alertElement = failureText.closest('[role="alert"]');
    expect(alertElement).toBeInTheDocument();

    const closeButton = alertElement!.querySelector(
        '[data-testid="alert-close"]'
    );
    expect(closeButton).toBeInTheDocument();
    await userEvent.click(closeButton!);

    await waitFor(() => {
        expect(
            screen.queryByText("Failed to create table!")
        ).not.toBeInTheDocument();
    });
};

export const assertAndCloseInternalErrorBanner = async () => {
    const errorText = await screen.findByText(
        "Internal error",
        {},
        { timeout: 5000 }
    );
    expect(errorText).toBeInTheDocument();

    const alertElement = errorText.closest('[role="alert"]');
    expect(alertElement).toBeInTheDocument();

    const closeButton = alertElement!.querySelector(
        '[data-testid="alert-close"]'
    );
    expect(closeButton).toBeInTheDocument();
    await userEvent.click(closeButton!);

    await waitFor(() => {
        expect(screen.queryByText("Internal error")).not.toBeInTheDocument();
    });
};

export const assertAndCloseFailureAlert = async () => {
    await closeFailureAlert();
};

export const assertNavigatedToTable = async (newTableName: string) => {
    await waitFor(() => {
        expect(mockUseNavigate).toHaveBeenCalledWith(
            `/project/project-tests/tables/${newTableName}`
        );
    });
};

interface TableCreationTestOptions {
    validateWithFailure?: boolean;
    validateWithInvalid?: boolean;
    createWithFailure?: boolean;
    createWithInvalid?: boolean;
    newTableName?: string;
    forceRename?: boolean;
}

export const executeTableCreationFlow = async (
    args: CreateTableArgs,
    clickSubmitButton: () => Promise<void>,
    {
        validateWithFailure = false,
        validateWithInvalid = false,
        createWithFailure = false,
        createWithInvalid = false,
        newTableName = "transaction_ltv1",
        forceRename = false
    }: TableCreationTestOptions
) => {
    setupValidateResponse(
        validateWithFailure
            ? "failure"
            : validateWithInvalid
              ? "invalid"
              : "success"
    );
    setupCreateResponse(
        createWithFailure
            ? "failure"
            : createWithInvalid
              ? "invalid"
              : "success"
    );

    await clickSubmitButton();
    await assertValidateApiCalled(args);

    if (validateWithFailure) {
        await assertAndCloseInternalErrorBanner();
        return;
    }

    if (validateWithInvalid) {
        await assertAndCloseFailureAlert();
        return;
    }

    await assertCreateApiCalled(args, newTableName, forceRename);

    if (createWithFailure) {
        await assertAndCloseInternalErrorBanner();
        return;
    }

    if (createWithInvalid) {
        await assertAndCloseFailureAlert();
        return;
    }

    await assertNavigatedToTable(newTableName);
};
