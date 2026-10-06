// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClient } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createQueryClient, createWrapper } from "tests/testUtil";

import ModelPlannerEditor, {
    ModelPlannerEditorProps
} from "../ModelPlannerEditor";

import "tests/mockAxios";

const FIXTURE_YAML = `training_table_generation:
  split: ApproxDateOffsetSplit(ratio=[0.8, 0.1, 0.1])
training_job:
  num_experiments: 2
  metrics:
    - mae
    - mse
  tune_metric: mae
column_processing:
  encoder_overrides:
    age: numerical
neighbor_sampling:
  num_neighbors:
    - hop1:
        default: 12
      hop2:
        default: 6
model_architecture:
  model:
    - name: GNN
`;

const closeWarning = vi.fn();
const defaultProps: ModelPlannerEditorProps = {
    title: "Model Planner",
    optionsYamlWarning: "",
    value: FIXTURE_YAML,
    onChange: vi.fn(),
    showOverwrittenWarning: false,
    closeWarning: closeWarning
};

describe("ModelPlannerEditor", () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        queryClient = createQueryClient();
        vi.clearAllMocks();
    });

    afterEach(() => {
        queryClient.clear();
    });

    const renderEditor = (props = defaultProps) =>
        render(
            <MemoryRouter>
                <ModelPlannerEditor {...props} />
            </MemoryRouter>,
            { wrapper: createWrapper(queryClient) }
        );

    // Test 1: Renders Form and YAML tab triggers
    it("renders Form and YAML tab triggers", async () => {
        await act(async () => {
            renderEditor();
        });
        expect(screen.getByRole("tab", { name: "Form" })).toBeInTheDocument();
        expect(screen.getByRole("tab", { name: "YAML" })).toBeInTheDocument();
    });

    // Test 2: Form tab is default; all 5 section headers visible
    it("shows Form tab by default with all 5 section headers", async () => {
        await act(async () => {
            renderEditor();
        });
        await waitFor(() => {
            expect(
                screen.getByText("Training Table Generation")
            ).toBeInTheDocument();
            expect(screen.getByText("Training Job")).toBeInTheDocument();
            expect(screen.getByText("Column Processing")).toBeInTheDocument();
            expect(screen.getByText("Neighbor Sampling")).toBeInTheDocument();
            expect(screen.getByText("Model Architecture")).toBeInTheDocument();
        });
    });

    // Test 3: Empty state when value=""
    it("shows empty state when value is empty", async () => {
        await act(async () => {
            renderEditor({ ...defaultProps, value: "" });
        });
        expect(screen.getByText(/No valid YAML found/i)).toBeInTheDocument();
    });

    // Test 4: Empty state button switches to YAML tab
    it("empty state button switches to YAML tab", async () => {
        await act(async () => {
            renderEditor({ ...defaultProps, value: "" });
        });
        const goToYamlBtn = screen.getByRole("button", {
            name: /go to yaml tab/i
        });
        await userEvent.click(goToYamlBtn);
        await waitFor(() => {
            expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
                "aria-selected",
                "true"
            );
        });
    });

    // Test 5: YAML tab shows code editor
    it("YAML tab shows code editor", async () => {
        await act(async () => {
            renderEditor();
        });
        const yamlTab = screen.getByRole("tab", { name: "YAML" });
        await userEvent.click(yamlTab);
        await waitFor(() => {
            expect(screen.getByRole("tab", { name: "YAML" })).toHaveAttribute(
                "aria-selected",
                "true"
            );
            // CodeMirror renders a .cm-editor element
            expect(document.querySelector(".cm-editor")).toBeInTheDocument();
        });
    });

    // Test 6: Branch dropdown change calls onChange with training_table_generation in YAML
    it("branch dropdown change calls onChange with updated YAML", async () => {
        const onChangeMock = vi.fn();
        await act(async () => {
            renderEditor({ ...defaultProps, onChange: onChangeMock });
        });
        // Open the split type select
        const comboboxes = screen.getAllByTestId("testid-combobox");
        // First combobox should be the split type selector
        await userEvent.click(comboboxes[0]);
        await waitFor(() => {
            expect(screen.getByText("Random Split")).toBeInTheDocument();
        });
        await userEvent.click(screen.getByText("Random Split"));
        await waitFor(() => {
            expect(onChangeMock).toHaveBeenCalled();
            const lastCall =
                onChangeMock.mock.calls[onChangeMock.mock.calls.length - 1][0];
            expect(lastCall).toContain("training_table_generation");
        });
    });

    // Test 7: "+ Add Hop" adds a hop row
    it("Add Hop button adds a hop row", async () => {
        await act(async () => {
            renderEditor();
        });
        const addHopBtn = screen.getByRole("button", { name: /add hop/i });
        const initialHopRows = screen.getAllByText(/Hop \d/i).length;
        await userEvent.click(addHopBtn);
        await waitFor(() => {
            expect(screen.getAllByText(/Hop \d/i).length).toBeGreaterThan(
                initialHopRows
            );
        });
    });

    // Test 8: "+ Add rule" adds a Column Processing row
    it("Add rule button adds a column processing row", async () => {
        await act(async () => {
            renderEditor();
        });
        const addRuleBtn = screen.getByRole("button", { name: /add rule/i });
        await userEvent.click(addRuleBtn);
        // After clicking, a new rule row should appear (2 total: existing + new)
        await waitFor(() => {
            expect(
                screen.getAllByText(/Rule \d/).length
            ).toBeGreaterThanOrEqual(2);
        });
    });

    // Test 9: Model Architecture shows architecture dropdown only for link prediction
    it("Model Architecture section shows Model architecture only for link prediction tasks", async () => {
        // Without taskType: no architecture dropdown
        await act(async () => {
            renderEditor();
        });
        await waitFor(() => {
            expect(screen.getByText("Model type")).toBeInTheDocument();
        });
        expect(
            screen.queryByText("Model architecture")
        ).not.toBeInTheDocument();

        // With link prediction taskType: architecture dropdown visible
        cleanup();
        await act(async () => {
            renderEditor({
                ...defaultProps,
                taskType: "temporal_link_prediction"
            });
        });
        await waitFor(() => {
            expect(screen.getByText("Model type")).toBeInTheDocument();
            expect(screen.getByText("Model architecture")).toBeInTheDocument();
        });
    });

    // Test 10: Info icon hover shows HoverCard popover content
    it("info icon hover shows popover content", async () => {
        await act(async () => {
            renderEditor();
        });
        // Find hover card triggers (data-slot="hover-card-trigger")
        const infoTriggers = document.querySelectorAll(
            "[data-slot='hover-card-trigger']"
        );
        expect(infoTriggers.length).toBeGreaterThan(0);
        // Hover over first trigger
        await userEvent.hover(infoTriggers[0] as HTMLElement);
        await waitFor(() => {
            expect(
                screen.getByText(/Defines how Kumo splits your data/i)
            ).toBeInTheDocument();
        });
    });

    it("offers only the default text embedding model, and keeps another one a plan names", async () => {
        await act(async () => {
            renderEditor({
                ...defaultProps,
                value: FIXTURE_YAML.replace(
                    "age: numerical",
                    "articles.title: TextEmbedding()\n    articles.detail_desc: TextEmbedding(model='glove.6B-100d')"
                )
            });
        });
        const [defaultModel, namedModel] = await screen.findAllByText(
            /^glove\.6B-(50|100)d$/
        );
        expect(defaultModel).toHaveTextContent("glove.6B-50d");
        expect(namedModel).toHaveTextContent("glove.6B-100d");

        await userEvent.click(defaultModel);
        await waitFor(() => {
            expect(screen.getAllByRole("option")).toHaveLength(1);
        });
        expect(screen.getByRole("option")).toHaveTextContent("glove.6B-50d");
        await userEvent.keyboard("{Escape}");

        await userEvent.click(namedModel);
        await waitFor(() => {
            expect(
                screen
                    .getAllByRole("option")
                    .map((o) => o.textContent)
                    .sort()
            ).toEqual(["glove.6B-100d", "glove.6B-50d"]);
        });
    });

    it("Should call `closeWarning` when warning is dismissed", async () => {
        await act(async () => {
            renderEditor({
                ...defaultProps,
                showOverwrittenWarning: true
            });
        });

        const alert = screen.getByRole("alert");
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(
            "The configuration in the Model Plan has been automatically updated to align with recent modifications in the pQuery. Please review the changes to ensure they meet your requirements"
        );

        const closeButton = screen.getByTestId("alert-close");
        await userEvent.click(closeButton);
        expect(closeWarning).toHaveBeenCalled();
    });
});
