// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type EmptyEntityData = {
    icon?: JSX.Element;
    label: JSX.Element;
    renderButtonId: string;
    renderButtonText: string;
    renderButtonIcon?: JSX.Element;
    onRenderClick: () => void;
    docsId: string;
    docsText: string;
    docsIcon?: JSX.Element;
    onDocsClick: () => void;
    project?: boolean;
};
