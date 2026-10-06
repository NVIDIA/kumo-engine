// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";

export const syntaxHighlight = syntaxHighlighting(
    HighlightStyle.define([
        { tag: tags.keyword, color: "#C12C89 !important" },
        { tag: tags.operator, color: "#1c583e !important" },
        { tag: tags.operatorKeyword, color: "#859170  !important" },
        { tag: tags.string, color: "#2A2F33 !important" },
        { tag: tags.modifier, color: "#8F2063 !important" },
        { tag: tags.number, color: "#4F888E !important" }
    ])
);
