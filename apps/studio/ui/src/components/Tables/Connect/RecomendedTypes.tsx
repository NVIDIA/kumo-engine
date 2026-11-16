// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import Spacer from "react-spacer";

import { Alert, AlertDescription, Button } from "@kumo-ai/kumo-ui-catalog";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";

export interface RecommendedTypesProps {
    recommendedSTypes: string[];
    onApplyRecommendedTypes: (checkUserUpdated: boolean) => void;
}

const RecommendedTypes = ({
    recommendedSTypes,
    onApplyRecommendedTypes
}: RecommendedTypesProps) => {
    const [showNewSTypes, setShowNewSTypes] = useState(true);
    const [displayedRecommendedSTypes, setDisplayedRecommendedSTypes] =
        useState<string[]>([]);

    const adoptRecommendedSTypesForAll = () => {
        onApplyRecommendedTypes(/*checkUserUpdated:*/ false);
    };

    const adoptRecommendedSTypesForDefault = () => {
        onApplyRecommendedTypes(/*checkUserUpdated:*/ true);
    };

    return (
        <>
            {recommendedSTypes?.length > 0 && (
                <Alert variant="default" enableClose={false}>
                    <AlertDescription>
                        {`Based on recent sample data analysis, we recommend updating
                        the type for the following column(s). Review and apply the
                        suggested type if needed to ensure data accuracy for
                        downstream processes. 'Apply for Default' will only apply
                        changes to columns where you have not yet manually changed
                        the the type.`}{" "}
                        {!showNewSTypes && <Spacer height={10} />}
                        {displayedRecommendedSTypes.map((stype, index) => {
                            return (
                                <span
                                    key={index + "-warning"}
                                    className="list-item list-inside text-xs leading-4.5 font-medium whitespace-pre-wrap">
                                    {stype}
                                </span>
                            );
                        })}
                        <Spacer height={5} />
                        <div className="flex gap-2">
                            <Button
                                className="h-6 ps-0"
                                variant="underline"
                                size="sm"
                                onClick={() => {
                                    setShowNewSTypes(!showNewSTypes);
                                    setDisplayedRecommendedSTypes(
                                        recommendedSTypes?.slice(
                                            0,
                                            showNewSTypes
                                                ? recommendedSTypes?.length
                                                : 0
                                        )
                                    );
                                }}>
                                {showNewSTypes
                                    ? "Show Recommended Types"
                                    : "Hide Recommended Types"}
                            </Button>

                            <Button
                                className="h-6"
                                variant="underline"
                                size="sm"
                                onClick={adoptRecommendedSTypesForAll}>
                                {"Apply For All"}
                            </Button>

                            <Button
                                className="h-6"
                                variant="underline"
                                size="sm"
                                onClick={adoptRecommendedSTypesForDefault}>
                                {"Apply For Default"}
                            </Button>
                        </div>
                    </AlertDescription>
                </Alert>
            )}
        </>
    );
};
export default WithErrorBoundary(RecommendedTypes);
