// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";

import { GraphDefinition } from "types/graphs";

import { getProjectPath } from "utils/Constants";

import { setGraphCreateResponseType } from "./MockFunction/GraphCreate.test.mock";
import { setGraphValidateResponseType } from "./MockFunction/GraphValidate.test.mock";

export const setupValidateResponse = (
    type: "success" | "failure" | "invalid" | "invalidResponse"
) => {
    if (type === "failure") {
        setGraphValidateResponseType(testResponseType.ERROR);
    } else if (type === "invalid") {
        setGraphValidateResponseType(testResponseType.INVALID);
    } else if (type === "invalidResponse") {
        setGraphValidateResponseType(testResponseType.INVALID_RESPONSE);
    } else {
        setGraphValidateResponseType(testResponseType.ALL);
    }
};

export const setupCreateResponse = (
    type: "success" | "failure" | "invalid"
) => {
    if (type === "failure") {
        setGraphCreateResponseType(testResponseType.ERROR);
    } else if (type === "invalid") {
        setGraphCreateResponseType(testResponseType.INVALID);
    } else {
        setGraphCreateResponseType(testResponseType.ALL);
    }
};

export const assertValidateApiCalled = async (
    args: GraphDefinition | Record<string, unknown>
) => {
    await waitFor(() => {
        const url = `v1/graphs/validate`;
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
            headers: { "Content-Type": "application/json" },
            params: { project_id: "project-tests" }
        });
    });
};

export const assertCreateApiCalled = async (
    args: GraphDefinition | Record<string, unknown>,
    graphName: string,
    forceRename: boolean = false
) => {
    await waitFor(() => {
        const url = `v1/graphs/`;
        expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
            headers: { "Content-Type": "application/json" },
            params: {
                name_alias: graphName,
                force_rename: forceRename,
                project_id: "project-tests"
            }
        });
    });
};

export const closeValidationFailureDialog = async () => {
    const closeButtons = screen.getAllByRole("button", {
        name: "Cancel"
    });
    const closeButton = closeButtons[0];
    await userEvent.click(closeButton);

    await waitFor(() => {
        expect(
            screen.queryByTestId("alert-dialog-content")
        ).not.toBeInTheDocument();
    });
};

export const closeValidationFailureAlert = async () => {
    const closeButton = screen.getByTestId("alert-close");
    expect(closeButton).toBeInTheDocument();
    await userEvent.click(closeButton);

    await waitFor(() => {
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
};

export const closeCreationFailureDialog = async () => {
    const closeButtons = screen.getAllByRole("button", {
        name: "Cancel"
    });
    const closeButton = closeButtons[0];
    await userEvent.click(closeButton);

    await waitFor(() => {
        expect(
            screen.queryByTestId("alert-dialog-content")
        ).not.toBeInTheDocument();
    });
};

export const closeCreationFailureAlert = async () => {
    const closeButton = screen.getByTestId("alert-close");
    expect(closeButton).toBeInTheDocument();
    await userEvent.click(closeButton);

    await waitFor(() => {
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
};

export const assertAndCloseValidationFailureDialog = async () => {
    const dialog = screen.getByTestId("alert-dialog-content");
    await waitFor(() => {
        expect(dialog).toBeInTheDocument();
    });
    expect(dialog).toHaveTextContent("Graph Validation failed!");

    await closeValidationFailureDialog();
};

export const assertAndCloseValidationFailureAlert = async () => {
    const modalheader = screen.getByText("Graph Validation failed!");
    await waitFor(() => {
        expect(modalheader).toBeInTheDocument();
    });

    await closeValidationFailureAlert();
};

export const assertAndCloseCreationFailureDialog = async () => {
    const dialog = screen.getByTestId("alert-dialog-content");
    await waitFor(() => {
        expect(dialog).toBeInTheDocument();
    });
    expect(dialog).toHaveTextContent("Graph Creation failed!");

    await closeCreationFailureDialog();
};

export const assertAndCloseCreationFailureAlert = async () => {
    const modalheader = screen.getByText("Graph Creation failed!");
    await waitFor(() => {
        expect(modalheader).toBeInTheDocument();
    });

    await closeCreationFailureAlert();
};

export const assertNavigatedToGraph = async (graphName: string) => {
    await waitFor(() => {
        expect(mockUseNavigate).toHaveBeenCalledWith(
            `${getProjectPath("project-tests").Graphs}${graphName}`
        );
    });
};

interface GraphCreationTestOptions {
    validateWithFailure?: boolean;
    validateWithInvalid?: boolean;
    validateWithInvalidSuccess?: boolean;
    createWithFailure?: boolean;
    createWithInvalid?: boolean;
    validName?: string;
    forceRename?: boolean;
}

export const executeGraphCreationFlow = async (
    args: GraphDefinition,
    clickSubmitButton: () => Promise<void>,
    {
        validateWithFailure = false,
        validateWithInvalid = false,
        validateWithInvalidSuccess = false,
        createWithFailure = false,
        createWithInvalid = false,
        validName = "test_graph",
        forceRename = false
    }: GraphCreationTestOptions
) => {
    setupValidateResponse(
        validateWithFailure
            ? "failure"
            : validateWithInvalid
              ? "invalid"
              : validateWithInvalidSuccess
                ? "invalidResponse"
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
        await assertAndCloseValidationFailureDialog();
        return;
    }

    if (validateWithInvalid) {
        await assertAndCloseValidationFailureAlert();
        return;
    }

    if (validateWithInvalidSuccess) {
        await assertAndCloseValidationFailureAlert();
        return;
    }

    await assertCreateApiCalled(args, validName, forceRename);

    if (createWithFailure) {
        await assertAndCloseCreationFailureDialog();
        return;
    }

    if (createWithInvalid) {
        await assertAndCloseCreationFailureAlert();
        return;
    }

    await assertNavigatedToGraph(validName);
};
