// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef, useState } from "react";

import { Button, cn } from "@kumo-ai/kumo-ui-catalog";
import type { ModelPlan } from "gen/v1/types/modelPlan";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import MPCodeEditor from "components/CodeEditor/MPCodeEditor";

import type { CodeEditorBasicSetup } from "types/codeEditor";
import type { ColumnGroup } from "types/graphs";

export const CODE_EDITOR_SETUP: CodeEditorBasicSetup = {
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
    autocompletion: false,
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

import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";

import { ctaLevels } from "types/constants";

import {
    parseModelPlanYaml,
    serializeModelPlanYaml
} from "utils/modelPlanYaml";

import ModelPlannerFormSections from "./ModelPlannerFormSections";

export interface ModelPlannerEditorProps {
    title: string;
    optionsYamlWarning: string;
    value: string;
    onChange: (newval: string) => void;
    editorHeight?: string;
    taskType?: string;
    graphTables?: Record<
        string,
        { cols: Array<{ name: string; stype?: string }>; pkey?: string | null }
    >;
    graphGroups?: ColumnGroup[];
    showOverwrittenWarning: boolean;
    closeWarning: () => void;
}

type EditorMode = "form" | "yaml";

const ModelPlannerEditor = ({
    optionsYamlWarning,
    value,
    onChange,
    taskType,
    graphTables,
    graphGroups,
    showOverwrittenWarning,
    closeWarning
}: ModelPlannerEditorProps) => {
    const [mode, setMode] = useState<EditorMode>("form");
    const [parsedPlan, setParsedPlan] = useState<ModelPlan | null>(() =>
        parseModelPlanYaml(value)
    );
    const lastEmittedYamlRef = useRef<string>(value);

    useEffect(() => {
        if (value !== lastEmittedYamlRef.current) {
            setParsedPlan(parseModelPlanYaml(value));
        }
    }, [value]);

    const handleFormChange = useCallback(
        (updatedPlan: ModelPlan) => {
            setParsedPlan(updatedPlan);
            const yaml = serializeModelPlanYaml(updatedPlan);
            lastEmittedYamlRef.current = yaml;
            onChange(yaml);
        },
        [onChange]
    );

    const handleYamlChange = useCallback(
        (newYaml: string) => {
            lastEmittedYamlRef.current = newYaml;
            onChange(newYaml);
            const parsed = parseModelPlanYaml(newYaml);
            setParsedPlan(parsed);
        },
        [onChange]
    );

    const handleSwitchToYaml = () => setMode("yaml");
    const handleSwitchToForm = useCallback(() => {
        setParsedPlan(parseModelPlanYaml(value));
        setMode("form");
    }, [value]);

    return (
        <div className="flex flex-col h-full">
            {showOverwrittenWarning && (
                <WarningErrorCTA
                    id={"warning-model_plan"}
                    ctaLevel={ctaLevels.Warning}
                    title=""
                    data={[
                        {
                            title: "",
                            description:
                                "The configuration in the Model Plan " +
                                "has been automatically updated to align " +
                                "with recent modifications in the pQuery. " +
                                "Please review the changes to ensure " +
                                "they meet your requirements",
                            is_error: false,
                            is_dismissible: true
                        }
                    ]}
                    handleClose={closeWarning}
                    hideTitle={true}
                    className="mb-4"
                />
            )}

            <div className="flex-shrink-0 pb-4">
                <div
                    role="tablist"
                    className="inline-flex w-fit h-10 items-center bg-surface-grey rounded-2xl p-1 gap-2">
                    <button
                        role="tab"
                        aria-selected={mode === "form"}
                        onClick={handleSwitchToForm}
                        className={cn(
                            "h-8 px-4 text-base font-medium rounded-2xl transition-all",
                            mode === "form"
                                ? "bg-white text-title-primary shadow-sm"
                                : "text-title-secondary hover:text-title-primary"
                        )}>
                        Form
                    </button>
                    <button
                        role="tab"
                        aria-selected={mode === "yaml"}
                        onClick={() => setMode("yaml")}
                        className={cn(
                            "h-8 px-4 text-base font-medium rounded-2xl transition-all",
                            mode === "yaml"
                                ? "bg-white text-title-primary shadow-sm"
                                : "text-title-secondary hover:text-title-primary"
                        )}>
                        YAML
                    </button>
                </div>
            </div>

            {/* Form content — scrolls independently */}
            {mode === "form" && (
                <div
                    role="tabpanel"
                    className="flex-1 min-h-0 overflow-y-auto px-1">
                    {parsedPlan ? (
                        <ModelPlannerFormSections
                            plan={parsedPlan}
                            onChange={handleFormChange}
                            taskType={taskType}
                            graphTables={graphTables}
                            graphGroups={graphGroups}
                        />
                    ) : (
                        <div className="flex flex-col items-center gap-3 py-8 text-center text-secondary">
                            <p className="text-sm">
                                No valid YAML found. Switch to the YAML tab to
                                provide a model plan.
                            </p>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleSwitchToYaml}>
                                Go to YAML tab
                            </Button>
                        </div>
                    )}
                </div>
            )}

            {/* YAML editor — fits container, scroll inside editor only */}
            {mode === "yaml" && (
                <div
                    role="tabpanel"
                    className="flex-1 min-h-0 overflow-hidden"
                    id="mp-yaml-panel">
                    <style>{`
                        #mp-yaml-panel > div,
                        #mp-yaml-panel .cm-theme,
                        #mp-yaml-panel .cm-editor { height: 100% !important; }
                        #mp-yaml-panel .cm-scroller { overflow: auto !important; }
                    `}</style>
                    <MPCodeEditor
                        language="YAML"
                        value={value}
                        basicSetup={CODE_EDITOR_SETUP}
                        isInvalid={Boolean(optionsYamlWarning)}
                        errorMsg={optionsYamlWarning}
                        onChange={handleYamlChange}
                        height="100%"
                    />
                </div>
            )}
        </div>
    );
};

export default WithErrorBoundary(ModelPlannerEditor);
