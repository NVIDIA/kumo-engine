// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrorType } from "gen/v1/types";

import { TaskType } from "types/predictiveQueries";

import { artifactType, outputStorage } from "utils/Constants";

import PredictionOutput, {
    PredictionOutputProps
} from "../Settings/PredictionOutput";

const onSelect = vi.fn();
const setOutputs = vi.fn();
const onArtifactChange = vi.fn();
const setSelectedConnector = vi.fn();
const setClasses = vi.fn();

const defaultProps: Partial<PredictionOutputProps> = {
    destination: "",
    predictionType: "",
    onSelect,
    onArtifactChange,
    selectedConnector: { label: "", value: "", type: "" },
    setSelectedConnector,
    connectors: {},
    classes: "",
    setClasses,
    outputs: {},
    setOutputs,
    error: []
};

const outputs = {
    [artifactType.Predictions]: {
        artifact_type: artifactType.Predictions,
        directory: "some/directory",
        table_schema: "",
        table_name: "",
        extension: "",
        errorInfo: {
            color: "",
            info: ""
        }
    }
};

describe("PredictionOutput", () => {
    const renderPredictionOutput = async (
        props: Partial<PredictionOutputProps> = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <PredictionOutput {...(props as PredictionOutputProps)} />
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should display options when the Output Destination dropdown is clicked", async () => {
        await renderPredictionOutput(defaultProps);

        expect(screen.getByText("Output Destination")).toBeInTheDocument();
        expect(screen.getByText("PREDICTIONS")).toBeInTheDocument();
        expect(screen.getByText("EMBEDDINGS")).toBeInTheDocument();

        const outputDestinationOptions = [
            { value: "S3", label: "S3" },
            { value: "Snowflake", label: "Snowflake" },
            { value: "Databricks", label: "Databricks" },
            { value: "Bigquery", label: "BigQuery" },
            { value: "LocalDownloadOnly", label: "Local Download Only" }
        ];
        const dropdown = screen.getByTestId("testid-output_destination");
        await userEvent.click(dropdown);
        const dropdownList = await screen.findByRole("listbox");
        expect(dropdownList).toBeInTheDocument();
        for (const option of outputDestinationOptions) {
            await waitFor(() =>
                expect(
                    within(dropdownList).getByText(option.label)
                ).toBeInTheDocument()
            );
        }
    });

    it("Should render `PredictionOutput` with Output Destination `Local Download Only`", async () => {
        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.LocalDownloadOnly
        });

        expect(screen.getByText("Output Destination")).toBeInTheDocument();
        expect(screen.getByText("Local Download Only")).toBeInTheDocument();

        expect(screen.queryByText("PREDICTIONS")).not.toBeInTheDocument();
        expect(screen.queryByText("EMBEDDINGS")).not.toBeInTheDocument();
        expect(screen.queryByText("Connector")).not.toBeInTheDocument();
        expect(
            screen.getByText(
                /With this option, we won't write the predictions back to your data plane./i
            )
        ).toBeInTheDocument();
    });

    it.each([
        { destination: outputStorage.S3, label: "S3" },
        { destination: outputStorage.GCS, label: "GCS" },
        { destination: outputStorage.ADLS, label: "ADLS" },
        { destination: outputStorage.Snowflake, label: "Snowflake" },
        { destination: outputStorage.Databricks, label: "Databricks" },
        { destination: outputStorage.Bigquery, label: "BigQuery" }
    ])(
        "Should render `PredictionOutput` with Output Destination `$label`",
        async ({ destination, label }) => {
            await renderPredictionOutput({
                ...defaultProps,
                destination,
                outputs: {}
            });

            expect(screen.getByText("Output Destination")).toBeInTheDocument();
            expect(screen.getByText(label)).toBeInTheDocument();
            expect(screen.getByText("PREDICTIONS")).toBeInTheDocument();
            expect(screen.getByText("EMBEDDINGS")).toBeInTheDocument();
        }
    );

    it("explains and disables source-only any-URI output connectors", async () => {
        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.GCS,
            connectors: {
                gcsConnectors: [
                    {
                        value: "rooted-gcs",
                        label: "Rooted GCS",
                        config: {
                            type: "gcs",
                            target: {
                                mode: "rooted",
                                bucket: "outputs",
                                prefix: "predictions"
                            }
                        }
                    },
                    {
                        value: "any-uri-gcs",
                        label: "Any URI GCS",
                        config: {
                            type: "gcs",
                            target: { mode: "any_uri" }
                        }
                    }
                ]
            },
            outputs
        });

        await userEvent.click(screen.getByTestId("testid-connectors"));
        expect(await screen.findByText("Rooted GCS")).toBeInTheDocument();
        const anyUriOption = screen.getByText("Any URI GCS");
        expect(anyUriOption).toBeInTheDocument();
        expect(anyUriOption.closest('[role="option"]')).toHaveAttribute(
            "aria-disabled",
            "true"
        );
        expect(
            screen.getByText(
                "Source-only any-URI connectors cannot write outputs."
            )
        ).toBeInTheDocument();
        await userEvent.click(anyUriOption);
        expect(setSelectedConnector).not.toHaveBeenCalled();
    });

    it.each([
        { destination: outputStorage.S3, label: "S3" },
        { destination: outputStorage.GCS, label: "GCS" },
        { destination: outputStorage.ADLS, label: "ADLS" },
        { destination: outputStorage.Snowflake, label: "Snowflake" },
        { destination: outputStorage.Databricks, label: "Databricks" },
        { destination: outputStorage.Bigquery, label: "BigQuery" }
    ])(
        "Should display `No options found.` under `Connector` dropdown when there are no $label connectors",
        async ({ destination }) => {
            await renderPredictionOutput({
                ...defaultProps,
                destination,
                outputs: {}
            });

            const connectorDropdown = screen.getByTestId("testid-connectors");
            await userEvent.click(connectorDropdown);
            expect(screen.getByText("No options found.")).toBeInTheDocument();
        }
    );

    it.each([
        {
            destination: outputStorage.S3,
            label: "S3",
            connectorKey: "s3Connectors",
            connectorValue: "ltv_test"
        },
        {
            destination: outputStorage.GCS,
            label: "GCS",
            connectorKey: "gcsConnectors",
            connectorValue: "gcs_test"
        },
        {
            destination: outputStorage.ADLS,
            label: "ADLS",
            connectorKey: "adlsConnectors",
            connectorValue: "adls_test"
        },
        {
            destination: outputStorage.Snowflake,
            label: "Snowflake",
            connectorKey: "snowFlakeConnectors",
            connectorValue: "hnm"
        },
        {
            destination: outputStorage.Databricks,
            label: "Databricks",
            connectorKey: "dataBricksConnectors",
            connectorValue: "test_databricks"
        },
        {
            destination: outputStorage.Bigquery,
            label: "BigQuery",
            connectorKey: "bigQueryConnectors",
            connectorValue: "test_bigquery"
        }
    ])(
        "Should call `setSelectedConnector` when connector is selected from `Connector` dropdown under `$label` destination",
        async ({ destination, connectorKey, connectorValue }) => {
            const connectors = {
                [connectorKey]: [
                    {
                        value: connectorValue,
                        label: connectorValue,
                        type: destination.toLowerCase()
                    }
                ]
            };

            renderPredictionOutput({
                ...defaultProps,
                destination,
                connectors,
                outputs
            });

            await waitFor(() => {
                expect(screen.getByText("Connector")).toBeInTheDocument();
            });

            const dropdown = screen.getByTestId("testid-connectors");
            await userEvent.click(dropdown);

            const option = await screen.findByText(connectorValue);
            await userEvent.click(option);

            expect(setSelectedConnector).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: connectorValue,
                    label: connectorValue,
                    type: destination.toLowerCase()
                })
            );
        }
    );

    it("Should call `onSelect` when destination is changed", async () => {
        await renderPredictionOutput(defaultProps);

        const dropdown = screen.getByTestId("testid-output_destination");
        await userEvent.click(dropdown);
        const dropdownList = await screen.findByRole("listbox");

        const snowflakeOption = within(dropdownList).getByText("Snowflake");
        await userEvent.click(snowflakeOption);
        expect(onSelect).toHaveBeenCalled();
    });

    it("Should call `onArtifactChange` when checkbox of `PREDICTIONS` and `EMBEDDINGS` is clicked", async () => {
        await renderPredictionOutput(defaultProps);

        const predictionCheckbox = screen.getByRole("checkbox", {
            name: /PREDICTIONS/i
        });
        expect(predictionCheckbox).not.toBeChecked();
        await userEvent.click(predictionCheckbox);
        expect(onArtifactChange).toHaveBeenCalled();

        const embeddingsCheckbox = screen.getByRole("checkbox", {
            name: /EMBEDDINGS/i
        });
        expect(embeddingsCheckbox).not.toBeChecked();
        await userEvent.click(embeddingsCheckbox);
        expect(onArtifactChange).toHaveBeenCalled();
    });

    it("Should display `Max Categories` & its hover content for `Multiclass Classification` task", async () => {
        await renderPredictionOutput({
            ...defaultProps,
            predictionType: TaskType.MULTICLASS_CLASSIFICATION,
            outputs
        });

        const maxCategories = screen.getByText("Max Categories");
        expect(screen.getByPlaceholderText("All")).toBeInTheDocument();

        await userEvent.hover(maxCategories);
        await waitFor(() => {
            expect(
                screen.getAllByText(
                    /If provided, only return predictions for the top K classes with the highest prediction per entity/i
                ).length
            ).toBeGreaterThan(0);
        });
    });

    it("Should display `Predicted Links per Entity` & its hover content for `Multilabel Ranking` task", async () => {
        await renderPredictionOutput({
            ...defaultProps,
            predictionType: TaskType.MULTILABEL_RANKING,
            outputs
        });

        const topK = screen.getByText("Predicted Links per Entity");
        expect(screen.queryByText("Max Categories")).not.toBeInTheDocument();

        await userEvent.hover(topK);
        await waitFor(() => {
            expect(
                screen.getAllByText(
                    /Only returns the top K predictions for each entity./i
                ).length
            ).toBeGreaterThan(0);
        });
    });

    it("Should display `File Name`,`Output directory`, & `File Type` for `S3` destination", async () => {
        await renderPredictionOutput({
            destination: outputStorage.S3,
            outputs
        });

        expect(screen.getByText("S3")).toBeInTheDocument();

        //Checkbox check
        const embeddingsCheckbox = screen.getByRole("checkbox", {
            name: /PREDICTIONS/i
        });
        expect(embeddingsCheckbox).toBeChecked();

        //File Name & its hover content check
        const fileName = screen.getByText("File Name");
        expect(screen.getByPlaceholderText("File Name")).toBeInTheDocument();
        await userEvent.hover(fileName);
        await waitFor(() => {
            expect(
                screen.getAllByText(
                    /The name of the output file, excluding the file extension./i
                ).length
            ).toBeGreaterThan(0);
        });

        //Output Directory & its hover content check
        const outputDirectory = screen.getByText("Output Directory");
        expect(
            screen.getByPlaceholderText("s3://<bucket>/<key prefix>")
        ).toBeInTheDocument();
        await userEvent.hover(outputDirectory);
        await waitFor(() => {
            expect(
                screen.getAllByText(
                    /The directory where the output file will be stored./i
                ).length
            ).toBeGreaterThan(0);
        });

        //File Type & its hover content check
        const fileTypes = screen.getAllByText("File Type");
        const fileType = fileTypes[0];
        await userEvent.hover(fileType);
        await waitFor(() =>
            expect(
                screen.getAllByText(
                    /The file extension determines the data format./i
                ).length
            ).toBeGreaterThan(0)
        );
    });

    it.each([
        {
            destination: outputStorage.Snowflake,
            label: "Snowflake",
            hoverTextRegex:
                /The name of the table in snowflake where the data will be written./i
        },
        {
            destination: outputStorage.Databricks,
            label: "Databricks",
            hoverTextRegex:
                /The name of the table where the data will be written./i
        },
        {
            destination: outputStorage.Bigquery,
            label: "BigQuery",
            hoverTextRegex:
                /The name of the table in BigQuery where the data will be written./i
        }
    ])(
        "Should display `Table Name` & hover content for `$label` destination",
        async ({ destination, label, hoverTextRegex }) => {
            await renderPredictionOutput({
                destination,
                outputs
            });

            expect(screen.getByText(label)).toBeInTheDocument();

            const predictionCheckbox = screen.getByRole("checkbox", {
                name: /PREDICTIONS/i
            });
            expect(predictionCheckbox).toBeChecked();

            const tableName = screen.getByText("Table Name");
            expect(
                screen.getByPlaceholderText("Table Name")
            ).toBeInTheDocument();
            await userEvent.hover(tableName);
            await waitFor(() => {
                expect(
                    screen.getAllByText(hoverTextRegex).length
                ).toBeGreaterThan(0);
            });
        }
    );

    it("Should display empty state when destination is set as undefined", async () => {
        await renderPredictionOutput({
            ...defaultProps,
            destination: undefined
        });

        expect(
            screen.queryByLabelText("Output Destination")
        ).not.toBeInTheDocument();
        expect(screen.queryByText("Connector")).not.toBeInTheDocument();
    });

    it("Should call onSelect and setClasses when a valid destination is selected", async () => {
        await renderPredictionOutput(defaultProps);

        const dropdown = screen.getByTestId("testid-output_destination");
        await userEvent.click(dropdown);
        const dropdownList = await screen.findByRole("listbox");

        const s3Option = within(dropdownList).getByText("S3");
        await userEvent.click(s3Option);

        expect(onSelect).toHaveBeenCalledWith("S3");
        expect(setClasses).toHaveBeenCalledWith("");
    });

    it("Should display error message when no output artifacts are selected and there is a 'Missing output artifact.' error", async () => {
        const errorWithMissingArtifact = [
            {
                type: ErrorType.NUMBER_0,
                title: "Missing output artifact.",
                description:
                    "Please select at least one output artifact (Predictions or Embeddings).",
                cta: null
            }
        ];

        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.S3,
            error: errorWithMissingArtifact
        });

        expect(screen.getByText("S3")).toBeInTheDocument();
        expect(
            screen.getByText(
                "Please select at least one output artifact (Predictions or Embeddings)."
            )
        ).toBeInTheDocument();
    });

    it("Should not display error message when Predictions artifact is selected", async () => {
        const errorWithMissingArtifact = [
            {
                type: ErrorType.NUMBER_0,
                title: "Missing output artifact.",
                description: "Please select at least one output artifact.",
                cta: null
            }
        ];

        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.S3,
            outputs,
            error: errorWithMissingArtifact
        });

        expect(screen.getByText("S3")).toBeInTheDocument();
        expect(
            screen.queryByText("Please select at least one output artifact.")
        ).not.toBeInTheDocument();
    });

    it("Should display error message for 'Predicted Links per Entity' in MaxCategories component", async () => {
        const errorWithPredictedLinks = [
            {
                type: ErrorType.NUMBER_0,
                title: "Predicted Links per Entity",
                description: "Value must be greater than 0",
                cta: null
            }
        ];

        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.LocalDownloadOnly,
            predictionType: TaskType.MULTICLASS_CLASSIFICATION,
            error: errorWithPredictedLinks
        });

        expect(
            screen.getByText("Value must be greater than 0")
        ).toBeInTheDocument();
    });

    it("Should display error message for 'Predicted Links per Entity' in OutputArtifacts MaxCategories", async () => {
        const errorWithPredictedLinks = [
            {
                type: ErrorType.NUMBER_0,
                title: "PREDICTIONS - Predicted Links per Entity",
                description: "This field is required for MaxCategories",
                cta: null
            }
        ];

        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.S3,
            predictionType: TaskType.MULTICLASS_CLASSIFICATION,
            outputs,
            error: errorWithPredictedLinks
        });

        expect(
            screen.getByText("This field is required for MaxCategories")
        ).toBeInTheDocument();
    });

    it("Should display error message for 'Predicted Links per Entity' in OutputArtifacts TopK", async () => {
        const errorWithPredictedLinks = [
            {
                type: ErrorType.NUMBER_0,
                title: "PREDICTIONS - Predicted Links per Entity",
                description: "Invalid top K value",
                cta: null
            }
        ];

        await renderPredictionOutput({
            ...defaultProps,
            destination: outputStorage.S3,
            predictionType: TaskType.MULTILABEL_RANKING,
            outputs,
            error: errorWithPredictedLinks
        });

        expect(screen.getByText("Invalid top K value")).toBeInTheDocument();
    });
});
