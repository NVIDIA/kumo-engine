// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RenderResult } from "@testing-library/react";
import { act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { setupGraphTests } from "components/Graphs/__tests__/graphTestUtils";

import { useValidatedName } from "hooks/validation/validatedName";

import { InputState } from "types/inputs";

vi.mock("hooks/validation/validatedName", () => ({
    useValidatedName: vi.fn(() => ["", vi.fn()])
}));

const mockedUseValidatedName = vi.mocked(useValidatedName);

describe("GraphNameInput", () => {
    const { graphNameInputMocks, renderGraphNameInput } = setupGraphTests();
    const PLACEHOLDER = "Please enter graph name";
    const INVALID_CHARS_ERROR = (name: string) =>
        `Table identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "${name}"`;
    const LENGTH_ERROR = (name: string) =>
        `Identifier must be fewer than 64 characters long. Got: ${name}`;

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
    });

    const getInput = (
        getByPlaceholderText: RenderResult["getByPlaceholderText"]
    ) => getByPlaceholderText(PLACEHOLDER);

    const mockValidation = (
        isValid: boolean,
        errorMsg: string,
        returnName: string,
        onValidationEndSpy: ReturnType<typeof vi.fn>
    ) => {
        mockedUseValidatedName.mockImplementation(
            (_name, _onValidationStart, onValidationEnd) => {
                onValidationEndSpy.mockImplementation(() => {
                    act(() => {
                        onValidationEnd({
                            validationResponse: { isValid, errMsg: errorMsg },
                            inputState: isValid
                                ? InputState.Valid
                                : InputState.InValid
                        });
                    });
                });
                return [returnName, graphNameInputMocks.setName];
            }
        );
    };

    const mockSkippedValidation = (
        returnName: string,
        setNameFn: (s: string) => void = vi.fn<(s: string) => void>()
    ) => {
        const { validationStartSpy, validationEndSpy } = graphNameInputMocks;

        mockedUseValidatedName.mockImplementationOnce(
            (_name, onValidationStart, onValidationEnd) => {
                validationStartSpy.mockImplementation(() =>
                    onValidationStart()
                );
                validationEndSpy.mockImplementation(() =>
                    onValidationEnd({
                        validationResponse: { isValid: true, errMsg: "" },
                        inputState: InputState.Valid
                    })
                );
                return [returnName, setNameFn];
            }
        );
    };

    it("Should render NameInput with initial value", async () => {
        const { getByText, getByRole } = await renderGraphNameInput();

        expect(getByText("Name")).toBeInTheDocument();
        const inputElement = getByRole("textbox");
        expect(inputElement).toHaveAttribute("name", "graph_name");
    });

    it("Should display empty input field when no name is provided", async () => {
        const { getByPlaceholderText } = await renderGraphNameInput();

        const input = getInput(getByPlaceholderText);
        expect(input).toHaveValue("");
    });

    it("Should call setName and setError on input change", async () => {
        const { setName, setError } = graphNameInputMocks;

        const { getByPlaceholderText } = await renderGraphNameInput({
            setName,
            setError
        });

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, "NewGraph");

        expect(setName).toHaveBeenCalled();
        expect(setError).toHaveBeenCalledWith("");
    });

    it("Should disable input when readOnly is true", async () => {
        const { getByPlaceholderText } = await renderGraphNameInput({
            name: "InitialName",
            readOnly: true
        });

        const input = getInput(getByPlaceholderText);
        expect(input).toHaveAttribute("readOnly");
    });

    it("Should show error message for invalid name", async () => {
        const { setError, onValidationEndSpy } = graphNameInputMocks;
        const invalidName = "InvalidName!";
        const errorMsg = INVALID_CHARS_ERROR(invalidName);

        mockValidation(false, errorMsg, invalidName, onValidationEndSpy);

        const { getByPlaceholderText, findByText } = await renderGraphNameInput(
            { setError }
        );

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, "!");

        onValidationEndSpy();

        expect(await findByText(errorMsg)).toBeInTheDocument();
        expect(setError).toHaveBeenCalledWith(errorMsg);
    });

    it("Should clear error when a valid name is entered", async () => {
        const { setError, setName, onValidationEndSpy } = graphNameInputMocks;
        const validName = "ValidName";

        mockValidation(true, "", validName, onValidationEndSpy);

        const { getByPlaceholderText, queryByText } =
            await renderGraphNameInput({
                name: "InvalidName!",
                setName,
                setError
            });

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, validName);

        onValidationEndSpy();
        expect(onValidationEndSpy).toHaveBeenCalled();

        // Check that error message is not shown
        await waitFor(() => {
            expect(queryByText(INVALID_CHARS_ERROR("InvalidName!"))).toBeNull();
        });

        expect(setError).toHaveBeenCalledWith("");
    });

    it("Should not validate when readOnly is true", async () => {
        const { setError, setName, validationStartSpy, validationEndSpy } =
            graphNameInputMocks;

        mockSkippedValidation("ReadOnlyName");

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: "ReadOnlyName",
            setName,
            readOnly: true,
            setError
        });

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, "NewName");

        expect(input).toHaveValue("ReadOnlyName");
        expect(input).toHaveAttribute("readOnly");
        expect(setName).not.toHaveBeenCalled();
        expect(validationStartSpy).not.toHaveBeenCalled();
        expect(validationEndSpy).not.toHaveBeenCalled();
    });

    it("Should show error message when name exceeds character limit", async () => {
        const { setError, setName } = graphNameInputMocks;
        const longName = "A".repeat(1000);
        const errorMsg = LENGTH_ERROR(longName);

        mockedUseValidatedName.mockImplementationOnce(
            (_name, _onValidationStart, onValidationEnd) => {
                onValidationEnd({
                    validationResponse: {
                        isValid: false,
                        errMsg: errorMsg
                    },
                    inputState: InputState.InValid
                });
                return [longName, setName];
            }
        );

        const { findByText } = await renderGraphNameInput({
            name: longName,
            setName,
            setError
        });

        expect(await findByText(errorMsg)).toBeInTheDocument();
        expect(setError).toHaveBeenCalledWith(errorMsg);
    });

    it("Should use an empty string as the default value for name", async () => {
        const { setError, setName } = graphNameInputMocks;

        const { getByPlaceholderText } = await renderGraphNameInput({
            setName,
            setError
        });

        const input = getInput(getByPlaceholderText);
        expect(input).toHaveValue("");
    });

    it("Should allow input when readOnly is false", async () => {
        const { setError, setName } = graphNameInputMocks;

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: "SomeName",
            setName,
            setError
        });

        const input = getInput(getByPlaceholderText);
        expect(input).not.toHaveAttribute("readonly");

        await userEvent.type(input, "NewGraph");

        expect(setName).toHaveBeenCalledTimes(8);
    });

    it("Should not validate when shouldValidate is false", async () => {
        const { setError, setName, validationStartSpy, validationEndSpy } =
            graphNameInputMocks;

        mockSkippedValidation("TestName", setName);

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: "TestName",
            setName,
            setError,
            shouldValidate: false
        });

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, "A");

        expect(validationStartSpy).not.toHaveBeenCalled();
        expect(validationEndSpy).not.toHaveBeenCalled();
    });

    it("Should validate when shouldValidate is true and name is not empty", async () => {
        const { setError, setName, onValidationEndSpy } = graphNameInputMocks;
        const validName = "ValidGraphName";

        mockValidation(true, "", validName, onValidationEndSpy);

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: validName,
            setName,
            setError,
            shouldValidate: true
        });

        const input = getInput(getByPlaceholderText);
        await userEvent.type(input, "A");

        onValidationEndSpy();

        expect(onValidationEndSpy).toHaveBeenCalled();
    });

    it("Should display custom placeholder text", async () => {
        const customPlaceholder = "Enter your custom graph name";

        const { getByPlaceholderText } = await renderGraphNameInput({
            placeholder: customPlaceholder
        });

        const input = getByPlaceholderText(customPlaceholder);
        expect(input).toBeInTheDocument();
        expect(input).toHaveAttribute("placeholder", customPlaceholder);
    });

    it("Should not validate when name is empty string", async () => {
        const { setError, setName, validationStartSpy, validationEndSpy } =
            graphNameInputMocks;

        mockSkippedValidation("", setName);

        await renderGraphNameInput({
            name: "",
            setName,
            setError,
            shouldValidate: true
        });

        expect(validationStartSpy).not.toHaveBeenCalled();
        expect(validationEndSpy).not.toHaveBeenCalled();
    });

    it("Should handle whitespace-only names correctly", async () => {
        const { setError, setName } = graphNameInputMocks;
        const whitespaceName = "   ";

        mockedUseValidatedName.mockImplementationOnce(() => [
            whitespaceName,
            setName
        ]);

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: whitespaceName,
            setName,
            setError
        });

        const input = getInput(getByPlaceholderText);
        expect(input).toHaveValue(whitespaceName);
    });

    it("Should show validating state during validation", async () => {
        const { setError, setName, validationStartSpy } = graphNameInputMocks;
        const testName = "TestGraph";

        mockedUseValidatedName.mockImplementationOnce(
            (_name, onValidationStart) => {
                validationStartSpy.mockImplementation(() => {
                    act(() => {
                        onValidationStart();
                    });
                });
                return [testName, setName];
            }
        );

        await renderGraphNameInput({
            name: testName,
            setName,
            setError,
            showState: true
        });

        validationStartSpy();

        expect(validationStartSpy).toHaveBeenCalled();
    });

    it("Should render with correct label and name attributes", async () => {
        const { getByText, getByRole } = await renderGraphNameInput();

        const label = getByText("Name");
        expect(label).toBeInTheDocument();

        const input = getByRole("textbox");
        expect(input).toHaveAttribute("name", "graph_name");
    });

    it("Should clear validation error when text changes", async () => {
        const { setError, setName } = graphNameInputMocks;
        const initialName = "InvalidName!";
        const errorMsg = INVALID_CHARS_ERROR(initialName);

        mockedUseValidatedName.mockImplementationOnce(
            (_name, _onValidationStart, onValidationEnd) => {
                onValidationEnd({
                    validationResponse: {
                        isValid: false,
                        errMsg: errorMsg
                    },
                    inputState: InputState.InValid
                });
                return [initialName, setName];
            }
        );

        const { getByPlaceholderText } = await renderGraphNameInput({
            name: initialName,
            setName,
            setError
        });

        const input = getInput(getByPlaceholderText);

        await userEvent.type(input, "A");

        expect(setError).toHaveBeenCalledWith("");
    });
});
