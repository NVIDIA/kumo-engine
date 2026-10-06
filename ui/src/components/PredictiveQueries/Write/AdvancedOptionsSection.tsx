// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import Spacer from "react-spacer";

import {
    Description,
    Form,
    GridCol,
    GridRow,
    SectionHeader,
    Select
} from "@kumo-ai/kumo-ui-catalog";
import { DocumentationLink } from "@kumo-ai/kumo-ui-catalog";

import MPCodeEditor from "components/CodeEditor/MPCodeEditor";

import { DropdownOption } from "types/inputs";

import { runMode } from "utils/Constants";
import { getValues } from "utils/Helpers/objectHelpers";

export interface RunModeSelectionProps {
    selectedRunMode: DropdownOption;
    onRunModeChange: (newval: DropdownOption) => void;
}

export const RunModeSelection = ({
    selectedRunMode,
    onRunModeChange
}: RunModeSelectionProps) => {
    const options = getValues(runMode).map((runmode) => ({
        value: runmode,
        label: runmode.toUpperCase()
    }));

    const [internalSelectedRunMode, setInternalSelectedRunMode] =
        useState<string>(selectedRunMode?.value as string);

    return (
        <Form>
            <GridRow xs={12} sm={12} md={12} className="items-center">
                <GridCol xs={12} sm={12} md={9}>
                    <>
                        <SectionHeader
                            title={"Run Mode"}
                            titleSize="md"
                            titleClassName="font-medium !font-medium"
                            description={
                                "The Run Mode field is used to determine the default values for your model's Model Plan based on common usage patterns."
                            }
                        />
                    </>
                </GridCol>
                <GridCol xs={12} sm={12} md={3}>
                    <Select
                        id={"run-mode-select"}
                        placeholder="Run Mode"
                        search={false}
                        mode="single"
                        popoverClassName="w-50"
                        selectedValues={Array.of(
                            internalSelectedRunMode?.toUpperCase()
                        )}
                        options={options}
                        onValuesChange={(newValue) => {
                            setInternalSelectedRunMode(newValue[0]);
                            onRunModeChange(
                                options?.find((item) =>
                                    newValue.includes(item.value)
                                ) as DropdownOption
                            );
                        }}
                    />
                </GridCol>
            </GridRow>
        </Form>
    );
};

const CodeEditorCustomSetup = {
    editable: true,
    lineNumbers: true,
    highlightActiveLineGutter: true,
    highlightSpecialChars: true,
    history: true,
    foldGutter: true,
    drawSelection: true,
    indentOnInput: true,
    syntaxHighlighting: true,
    bracketMatching: true,
    closeBrackets: true,
    autocompletion: false, // disable for now.
    highlightActiveLine: true,
    highlightSelectionMatches: true,
    closeBracketsKeymap: true,
    dropCursor: true,
    allowMultipleSelections: true,
    rectangularSelection: true,
    crosshairCursor: false,
    defaultKeymap: true,
    searchKeymap: true,
    historyKeymap: true,
    foldKeymap: true,
    completionKeymap: true
};

export interface ModelPlannerInputProps {
    title: string;
    optionsYamlWarning: string;
    value: string;
    onChange: (newval: string) => void;
    editorHeight?: string;
}

export const ModelPlannerInput = ({
    title,
    optionsYamlWarning,
    value,
    onChange,
    editorHeight
}: ModelPlannerInputProps) => {
    return (
        <>
            <GridCol xs={12} sm={12}>
                <SectionHeader
                    title={title}
                    titleSize="md"
                    titleClassName="font-medium !font-medium pb-0 !pb-0"
                    description={
                        "Provide the config for training data generation, hyperparameters, and advanced evaluation options."
                    }
                />
                <Description
                    variant="secondary"
                    size="sm"
                    className="my-0 pt-0 font-normal leading-5">
                    {" See "}
                    <DocumentationLink
                        id="query_suggestion_doc"
                        pageLink={"model-planner"}
                        linkText="docs"
                        isNewDocs={true}
                    />
                    {" for more details."}
                </Description>
            </GridCol>
            <Spacer height={8} />
            <MPCodeEditor
                value={value}
                language={"YAML"}
                basicSetup={CodeEditorCustomSetup}
                isInvalid={Boolean(optionsYamlWarning)}
                errorMsg={optionsYamlWarning}
                onChange={(e) => {
                    onChange(e);
                }}
                height={editorHeight}
            />
        </>
    );
};
