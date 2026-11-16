// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export function getTuneMetric(yamlString: string | undefined | null): string {
    let tune_metric: string = "";
    try {
        if (yamlString) {
            const startIndex: number = yamlString.indexOf("tune_metric:");
            if (startIndex !== -1) {
                const valueStartIndex: number =
                    yamlString.indexOf(":", startIndex) + 1;
                const valueEndIndex: number = yamlString.indexOf(
                    "\n",
                    valueStartIndex
                );
                if (valueStartIndex !== -1 && valueEndIndex !== -1) {
                    tune_metric = yamlString
                        .slice(valueStartIndex, valueEndIndex)
                        .trim();
                }
            }
            const isRefit = yamlString
                ?.toLowerCase()
                ?.includes("refit_trainval: true");
            if (isRefit && tune_metric) {
                tune_metric = `refit_${tune_metric}`;
            }
        }
    } catch (e) {
        console.warn(e);
    }
    return tune_metric;
}
