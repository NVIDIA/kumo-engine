// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cn } from "@kumo-ai/kumo-ui-catalog";

import PQCodeEditor from "components/CodeEditor/PQCodeEditor";

interface JobQueryPreviewProps {
    value: string;
    className?: string;
}

export const JobQueryPreview = ({ value, className }: JobQueryPreviewProps) => (
    <div className={cn(className)}>
        <PQCodeEditor
            value={value.trim() || ""}
            language={"PQLANG_V2"}
            editable={false}
            isInvalid={false}
            errorMsg={""}
            onChange={() => {}}
            graphTables={[]}
            lineNumbers={false}
            highlightActiveLineGutter={false}
            highlightActiveLine={false}
            showCopyButton={true}
            hideBorder
            height="auto"
        />
    </div>
);
