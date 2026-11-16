// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LegacyRef, RefObject } from "react";

import {
    Bouncingball,
    ConfirmationDialog,
    CustomAlert,
    DialogRef
} from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import WarningErrorCTA from "components/@ui/Warning/WarningErrorCTA";

import { ctaLevels } from "types/constants";
import { AlertData, BannerData } from "types/validation";
import { CTAInfo } from "types/warnings";

export interface ValidationProps<TObject extends object> {
    pageLoading: boolean;
    banner: BannerData | null;
    alert: AlertData | null;
    ctaMsgs?: CTAInfo | null;
    dialogRef: RefObject<DialogRef<TObject>>;
    hideWarnings?: () => void;
    onDialogConfirm?: (val?: TObject) => void;
    onDialogCancel?: (val?: TObject) => void;
    onDialogSkip?: (val?: TObject) => void;
    clearErrors: () => void;
    className?: string;
}

const Validation = <TObject extends object>({
    pageLoading,
    banner,
    alert,
    ctaMsgs,
    dialogRef,
    hideWarnings,
    onDialogConfirm = () => {},
    onDialogCancel = () => {},
    onDialogSkip = () => {},
    clearErrors,
    className = ""
}: ValidationProps<TObject>) => {
    return (
        <>
            {pageLoading ? <Bouncingball overlay={true} /> : null}
            {ctaMsgs && ctaMsgs.showCTAErrors && (
                <WarningErrorCTA
                    id={"cta-errors"}
                    data={ctaMsgs.errors || []}
                    ctaLevel={ctaLevels.Critical}
                />
            )}
            {ctaMsgs && ctaMsgs.showCTAWarnings && (
                <WarningErrorCTA
                    id={"cta-warnings"}
                    data={ctaMsgs?.warnings || []}
                    ctaLevel={ctaLevels.Warning}
                    handleClose={hideWarnings}
                />
            )}
            <ConfirmationDialog
                ref={dialogRef as LegacyRef<DialogRef<object>>}
                cancelText="Cancel"
                confirmText="Continue"
                handleClose={() => {
                    if (onDialogCancel) {
                        onDialogCancel();
                    }
                    dialogRef?.current?.toggleDialog();
                }}
                handleConfirm={onDialogConfirm as (value?: object) => void}
                handleSkip={(val) => {
                    if (onDialogSkip) {
                        onDialogSkip(val as TObject);
                    }
                }}
            />
            <CustomAlert
                show={alert !== null}
                title={alert?.title as string}
                description={alert?.description as string}
                handleClose={clearErrors}
            />
            {banner?.warnings && banner?.isWarning && (
                <WarningErrorCTA
                    id={"cta-warnings-2"}
                    title={banner?.title ?? ""}
                    data={
                        banner?.warnings?.map((b) => ({
                            id: b,
                            description: b,
                            is_dismissible: true,
                            is_error: true,
                            title: "",
                            cta: null
                        })) || []
                    }
                    ctaLevel={ctaLevels.Warning}
                    handleClose={hideWarnings}
                />
            )}
            {banner?.warnings && !banner?.isWarning && (
                <WarningErrorCTA
                    id={"cta-errors-2"}
                    title={banner?.title ?? ""}
                    data={
                        banner?.warnings?.map((b) => ({
                            id: b,
                            description: b,
                            is_dismissible: true,
                            is_error: true,
                            title: "",
                            cta: null
                        })) || []
                    }
                    ctaLevel={ctaLevels.Critical}
                    handleClose={clearErrors}
                    className={className}
                />
            )}
        </>
    );
};
export default WithErrorBoundary(Validation);
