// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockNavigateCall } from "tests/mockRouter";

import { DropdownOption } from "types/inputs";

import {
    ModelPlannerInput,
    ModelPlannerInputProps,
    RunModeSelection,
    RunModeSelectionProps
} from "../AdvancedOptionsSection";

const defaultRunModeProps: RunModeSelectionProps = {
    selectedRunMode: [] as DropdownOption,
    onRunModeChange: vi.fn()
};

describe("RunModeSelection", () => {
    const renderRunModeSelection = async (props: RunModeSelectionProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <RunModeSelection {...props} />
                </MemoryRouter>
            );
        });
    };

    const verifyRunModeLabel = () => {
        expect(screen.getByText("Run Mode")).toBeInTheDocument();
    };

    const verifyRunModeDescription = () => {
        expect(
            screen.getByText(
                /The Run Mode field is used to determine the default values for your model's Model Plan based on common usage patterns./i
            )
        ).toBeInTheDocument();
    };

    const getDropdown = () => {
        return screen.getByTestId("testid-run-mode-select");
    };

    const clickDropdown = async () => {
        const dropdown = getDropdown();
        await userEvent.click(dropdown);
        return dropdown;
    };

    const verifyDropdownOptions = async (
        options: Array<{ index: number; text: string }>
    ) => {
        await waitFor(() => {
            options.forEach(({ index, text }) => {
                expect(
                    screen.getByTestId(`run-mode-select-option-${index}`)
                ).toHaveTextContent(text);
            });
        });
    };

    it("Should render `RunModeSelection`", async () => {
        await renderRunModeSelection(defaultRunModeProps);

        verifyRunModeLabel();
        verifyRunModeDescription();
    });

    it("Should display `FAST, BEST, NORMAL` options when the dropdown is clicked", async () => {
        await renderRunModeSelection(defaultRunModeProps);

        await clickDropdown();

        await verifyDropdownOptions([
            { index: 1, text: "FAST" },
            { index: 2, text: "NORMAL" },
            { index: 0, text: "BEST" }
        ]);
    });

    it("Should display default option when the `DropdownOption` is set", async () => {
        const onRunModeChange = vi.fn();
        await renderRunModeSelection({
            ...defaultRunModeProps,
            selectedRunMode: { value: "fast", label: "fast" } as DropdownOption,
            onRunModeChange
        });

        verifyRunModeLabel();

        const dropdown = getDropdown();
        expect(dropdown).toHaveTextContent("FAST");

        await clickDropdown();

        await verifyDropdownOptions([{ index: 0, text: "FAST" }]);
    });

    it("Should render correctly in non-V2 mode with tooltip", async () => {
        await renderRunModeSelection({
            ...defaultRunModeProps,
            selectedRunMode: { value: "normal", label: "normal" }
        });

        verifyRunModeLabel();
    });

    it("Should call `onRunModeChange` when the option is changed", async () => {
        const onRunModeChange = vi.fn();
        await renderRunModeSelection({
            ...defaultRunModeProps,
            selectedRunMode: { value: "fast", label: "fast" } as DropdownOption,
            onRunModeChange
        });

        verifyRunModeLabel();

        const dropdown = getDropdown();
        expect(dropdown).toHaveTextContent("FAST");

        await clickDropdown();

        const optionToSelect = await screen.findByTestId(
            "run-mode-select-option-1"
        );
        expect(optionToSelect).toHaveTextContent("BEST");
        await userEvent.click(optionToSelect);

        expect(onRunModeChange).toHaveBeenCalledWith({
            value: "BEST",
            label: "BEST"
        });
    });
});

const defaultModelPlannerProps: ModelPlannerInputProps = {
    title: "Model Planner",
    optionsYamlWarning: "",
    value: "",
    onChange: vi.fn()
};

describe("ModelPlannerInput", () => {
    mockNavigateCall();

    const renderModelPlannerInput = async (props: ModelPlannerInputProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ModelPlannerInput {...props} />
                </MemoryRouter>
            );
        });
    };

    const verifyModelPlannerTitle = (title: string = "Model Planner") => {
        expect(screen.getByText(title)).toBeInTheDocument();
    };

    const verifyModelPlannerDescription = () => {
        expect(
            screen.getByText(
                /Provide the config for training data generation, hyperparameters, and advanced evaluation options./i
            )
        ).toBeInTheDocument();
    };

    const verifyDocsLink = () => {
        expect(
            screen.getByText((content, node) => {
                const hasText = node?.textContent?.includes(
                    "See docs for more details."
                );
                const isParagraph = node?.tagName.toLowerCase() === "p";
                return Boolean(hasText && isParagraph);
            })
        ).toBeInTheDocument();

        const docsLink = screen.getByTestId("testid-query_suggestion_doc");
        expect(docsLink).toBeInTheDocument();
        expect(docsLink).toHaveAttribute(
            "href",
            "https://kumo.ai/docs/model-planner"
        );
        expect(docsLink).toHaveAttribute("target", "_blank");
        return docsLink;
    };

    const getEditor = () => {
        return screen.getByTestId("pql-editor");
    };

    const verifyEditorExists = () => {
        const editor = getEditor();
        expect(editor).toBeInTheDocument();
        return editor;
    };

    it("Should render `ModelPlannerInput` with title, description, docs link, and editor", async () => {
        await renderModelPlannerInput(defaultModelPlannerProps);

        verifyModelPlannerTitle();
        verifyModelPlannerDescription();
        verifyDocsLink();
        verifyEditorExists();
    });

    it("Should populate the editor with the given initial YAML value", async () => {
        const initialValue = "training:\n  epochs: 10";
        await renderModelPlannerInput({
            ...defaultModelPlannerProps,
            value: initialValue
        });

        const editor = verifyEditorExists();
        expect(editor).toHaveTextContent("training:");
        expect(editor).toHaveTextContent("epochs: 10");
    });

    it("Should display validation warning when `optionsYamlWarning` is provided", async () => {
        const warningMessage = "Invalid YAML format.";
        await renderModelPlannerInput({
            ...defaultModelPlannerProps,
            optionsYamlWarning: warningMessage,
            value: "bad: yaml: example"
        });

        expect(screen.getByText(warningMessage)).toBeInTheDocument();
    });

    it("Should call `onChange` when editor content is changed", async () => {
        const onChange = vi.fn();

        await renderModelPlannerInput({
            ...defaultModelPlannerProps,
            onChange
        });

        const editor = screen.getByRole("textbox");
        await userEvent.type(editor, "foo: bar");
        expect(onChange).toHaveBeenCalled();
        expect(onChange).toHaveBeenLastCalledWith("foo: bar");
    });
});
