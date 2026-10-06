// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { errorStatusCode } from "utils/Constants";

export type AxiosResponse<T> = {
    data: T;
};

type ErrorDetails = {
    detail?: Array<{ msg?: string }>;
    error: object;
};
export enum ErrorMode {
    ALERT = "alert",
    BANNER = "banner"
}

export interface CustomError extends Error {
    response?: {
        data?: string | Array<{ msg?: string }> | ErrorDetails;
        status?: errorStatusCode;
        headers?: Record<string, string>;
    };
    request?: string;
    mode?: ErrorMode;
}

export type ConfirmButtonVariant =
    | "outline"
    | "secondary"
    | "link"
    | "destructive";

export interface DialogRef<TObject extends object> {
    toggleDialog: (
        title?: string,
        description?: string | React.ReactNode,
        value?: TObject | null,
        enableSkipButton?: boolean,
        confirmButtonText?: string,
        confirmButtonVariant?: ConfirmButtonVariant,
        cancelButtonText?: string
    ) => void;
}

type HighLightPosition = { start: number; end: number }[];

export type HighLightText = {
    text: string;
    position: HighLightPosition;
};
interface ValidationMessage {
    title?: string;
    message?: string;
}
export interface ValidationResponse {
    warnings?: ValidationMessage[];
    errors?: ValidationMessage[];
    info_items?: ValidationMessage[];
}
export type PropertyItem = {
    icon: React.ReactNode;
    label: string | React.ReactNode;
};

export type MenuItemType = {
    content: React.ReactNode;
    disabled: boolean;
    onSelect: () => void;
    tooltipText: string;
};

export type OptionType = {
    value: string;
    label: string;
};
//Type used by different pages to show error-label
export type EntityErrorMsg = {
    entity: string; //Entity(dropdown/textbox) against which below message has to be shown with ErrorLabel
    errorMsg: string;
};
