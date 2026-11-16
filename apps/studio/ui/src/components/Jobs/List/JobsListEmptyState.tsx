// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Inbox } from "@kumo-ai/kumo-ui-catalog";

import { JOBS_LIST_EMPTY_STATE } from "utils/Constants";

export interface JobsListEmptyStateProps {
    onResetFilters: () => void;
}

const JobsListEmptyState = ({ onResetFilters }: JobsListEmptyStateProps) => {
    return (
        <div className="flex flex-col justify-center items-center gap-3 h-[90dvh]">
            <Inbox className="size-9" />
            <label className="text-center text-title-secondary font-inter font-normal text-base">
                {JOBS_LIST_EMPTY_STATE.TITLE}
            </label>
            <Button
                id="reset-filters-button"
                className="w-29.5"
                onClick={onResetFilters}>
                {JOBS_LIST_EMPTY_STATE.BUTTON_TEXT}
            </Button>
        </div>
    );
};

export default JobsListEmptyState;
