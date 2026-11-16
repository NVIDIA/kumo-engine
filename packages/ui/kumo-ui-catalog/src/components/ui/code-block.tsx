// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import SyntaxHighlighter from "react-syntax-highlighter";

interface CodeBlockProps {
    codeString?: string;
    showLineNumbers?: boolean;
    wrapLongLines?: boolean;
    wrapLines?: boolean;
    language?: string;
    className?: string;
}

const DEFAULT_CLASSNAME =
    "bg-surface-grey rounded-[10px]! p-5! max-w-full max-h-[400px] overflow-auto";

export const CodeBlock = ({
    codeString,
    showLineNumbers = false,
    wrapLongLines = true,
    wrapLines = true,
    language = "text",
    className = DEFAULT_CLASSNAME
}: CodeBlockProps) => {
    return (
        <div>
            <SyntaxHighlighter
                language={language}
                className={className}
                showLineNumbers={showLineNumbers}
                wrapLongLines={wrapLongLines}
                wrapLines={wrapLines}
                tabIndex={0}
                aria-label={`Code block (${language})`}>
                {codeString as string}
            </SyntaxHighlighter>
        </div>
    );
};

export type { CodeBlockProps };
