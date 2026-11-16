// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RefObject, useCallback, useRef, useState } from "react";

import { CustomError, DialogRef } from "types/common";
import { AlertData, BannerData, HandleErrorFunc } from "types/validation";

import { adjustError, handleErrorFunctional } from "utils/errorHandler";

export const useValidationState = <TObject extends object>(
    setPageLoading: (b: boolean) => void
): [
    BannerData | null,
    AlertData | null,
    HandleErrorFunc,
    () => void,
    RefObject<DialogRef<TObject>>,
    (t: string, d: string) => void
] => {
    const [banner, setBanner] = useState<BannerData | null>(null);
    const [alert, setAlert] = useState<AlertData | null>(null);
    const handleError = useCallback(
        (err: CustomError, title?: string, isWarning?: boolean) => {
            handleErrorFunctional(
                err as CustomError,
                (bannerData: BannerData) => {
                    setBanner({
                        ...bannerData,
                        warnings: bannerData.warnings.map((warning) =>
                            adjustError(warning)
                        )
                    });
                },
                (alertData: AlertData) => {
                    setAlert({
                        ...alertData,
                        description: adjustError(alertData.description)
                    });
                },
                title,
                isWarning
            );
            setPageLoading(false);
        },
        [setBanner, setAlert, setPageLoading]
    );
    const clear = useCallback(() => {
        setBanner(null);
        setAlert(null);
    }, []);
    const dialogRef = useRef<DialogRef<TObject>>(null);
    const openDialog = (title: string, description: string) => {
        dialogRef?.current?.toggleDialog(title, description);
    };
    return [banner, alert, handleError, clear, dialogRef, openDialog];
};
