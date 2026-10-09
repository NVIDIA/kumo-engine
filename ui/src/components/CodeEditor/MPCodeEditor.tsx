// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { QueryDialect } from "gen/v1/types";

import CodeEditor from "components/CodeEditor/CodeEditor";

import { CodeEditorBasicSetup } from "types/codeEditor";

import { yamlDict } from "utils/YAML/yamlDictionary";

import { parseHintsFromMintlify } from "./parseHints";

interface MPCodeEditorProps {
    language: string;
    value: string;
    basicSetup?: CodeEditorBasicSetup;
    onChange: (newval: string) => void;
    editable?: boolean;
    isInvalid?: boolean;
    errorMsg?: string;
    height?: string;
    hideBorder?: boolean;
    backgroundColor?: string;
}

const MPCodeEditor: React.FC<MPCodeEditorProps> = ({
    language,
    value,
    basicSetup,
    onChange,
    editable,
    isInvalid,
    errorMsg,
    height,
    hideBorder,
    backgroundColor
}) => {
    const parsedDict = parseHintsFromMintlify(language, yamlDict);

    return (
        <CodeEditor
            isTabKeyEnabled={true}
            value={value}
            language={QueryDialect.YAML}
            hoverHints={parsedDict}
            basicSetup={basicSetup as CodeEditorBasicSetup}
            onChange={onChange}
            editable={editable}
            isInvalid={isInvalid}
            errorMsg={errorMsg}
            height={height}
            hideBorder={hideBorder}
            backgroundColor={backgroundColor}
        />
    );
};

export default MPCodeEditor;
