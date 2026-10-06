// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { mockLocationStateValue } from "tests/mockRouter";
import { fireEventChange } from "tests/testHelpers";
import { createQueryClient, createWrapper } from "tests/testUtil";

import ProjectNameInput, { ProjectNameInputProps } from "../ProjectNameInput";

import "tests/mockAxios";

const setName = vi.fn();
const setError = vi.fn();
const defaultProps: ProjectNameInputProps = {
    name: "",
    setName,
    readOnly: false,
    setError,
    showState: false,
    shouldValidate: true
};

describe("ProjectNameInput", () => {
    let queryClient: QueryClient;

    const renderProjectNameInput = async (
        props: ProjectNameInputProps = defaultProps
    ) => {
        await act(async () => {
            render(<ProjectNameInput {...props} />, {
                wrapper: createWrapper(queryClient)
            });
        });
    };

    const getInput = () => screen.getByRole("textbox");

    const changeInput = (value: string) => {
        fireEventChange(getInput(), value);
    };

    beforeEach(() => {
        queryClient = createQueryClient();
        vi.clearAllMocks();
        mockLocationStateValue(null);
    });

    afterEach(() => {
        queryClient.clear();
    });

    it("Should render the default placeholder when none is provided", async () => {
        await renderProjectNameInput();

        expect(screen.getByText("Project name")).toBeInTheDocument();
        expect(
            screen.getByPlaceholderText("Please enter project name")
        ).toBeInTheDocument();

        const input = getInput();
        expect(input).toHaveAttribute("id", "project-name");
        expect(input).toHaveAttribute("name", "project_name");
    });

    it.each([
        {
            description: "custom placeholder",
            placeholder: "Enter your project name",
            expectedPlaceholder: "Enter your project name"
        },
        {
            description: "empty placeholder",
            placeholder: "",
            expectedPlaceholder: ""
        }
    ])(
        "Should render $description correctly",
        async ({ placeholder, expectedPlaceholder }) => {
            await renderProjectNameInput({ ...defaultProps, placeholder });
            expect(getInput()).toHaveAttribute(
                "placeholder",
                expectedPlaceholder
            );
        }
    );

    it("Should render a read-only input when readOnly is true", async () => {
        await renderProjectNameInput({ ...defaultProps, readOnly: true });
        expect(getInput()).toHaveAttribute("readonly");
    });

    it("Should display the current name value", async () => {
        await renderProjectNameInput({
            ...defaultProps,
            name: "existing_project"
        });
        expect(getInput()).toHaveValue("existing_project");
    });

    it.each([
        {
            description: "setName with typed value",
            fn: setName,
            expectedArg: "my_project"
        },
        {
            description: "setError with empty string",
            fn: setError,
            expectedArg: ""
        }
    ])(
        "Should call $description on input change",
        async ({ fn, expectedArg }) => {
            await renderProjectNameInput();
            changeInput("my_project");
            expect(fn).toHaveBeenCalledWith(expectedArg);
        }
    );

    it("Should not throw when setName is undefined and input changes", async () => {
        await renderProjectNameInput({ ...defaultProps, setName: undefined });
        expect(() => changeInput("test")).not.toThrow();
    });

    it("Should not show an error message initially", async () => {
        await renderProjectNameInput();
        expect(
            screen.queryByText(/consecutive underscores/)
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText(/Please enter Project Name/)
        ).not.toBeInTheDocument();
    });
});
