// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

import {
    useCreateJobsAddTag,
    useCreateJobsDeleteTag
} from "gen/v0/api/jobs/jobs";
import type { JobTagRequest } from "gen/v0/types";

import { useProjectId } from "hooks/project/projects";

import { getKeys } from "utils/Helpers/objectHelpers";
import { filterTagsForDisplay } from "utils/Helpers/tagHelpers";

export function useTags(initialTags?: Record<string, string>) {
    const [tags, setTags] = useState<Record<string, string>>({});

    // Update tags whenever initialTags changes
    useEffect(() => {
        if (initialTags) {
            const newTags = filterTagsForDisplay(initialTags) ?? {};
            setTags(newTags);
        }
    }, [initialTags]);

    // helper to update one tag
    function updateTag(key: string, value: string) {
        setTags((prev) => ({
            ...prev,
            [key]: value
        }));
    }

    // helper to remove one tag
    function removeTag(key: string) {
        setTags((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
        });
    }

    // helper to reset everything from latest input
    function resetTags() {
        const reset = filterTagsForDisplay(initialTags) ?? {};
        setTags(reset);
    }

    // shallow equality check (optional)
    function isSameAsInitial() {
        const currentKeys = getKeys(tags);
        const initialFiltered = filterTagsForDisplay(initialTags) ?? {};
        const initialKeys = getKeys(initialFiltered);

        if (currentKeys.length !== initialKeys.length) return false;
        return currentKeys.every((k) => tags[k] === initialFiltered[k]);
    }

    return {
        tags,
        setTags,
        updateTag,
        removeTag,
        resetTags,
        isSameAsInitial
    };
}

/**
 * Hook that provides API functions for adding and removing job tags.
 * Returns mutation functions that can be called directly in handleTagAdd/handleTagRemove.
 */
export function useJobTagMutations() {
    const projectId = useProjectId();
    const addTagMutation = useCreateJobsAddTag();
    const deleteTagMutation = useCreateJobsDeleteTag();

    const addTag = (jobId: string, key: string, value?: string | null) => {
        if (!projectId) throw new Error("A project is required to update jobs");
        const request: JobTagRequest = {
            job_id: jobId,
            project_id: projectId,
            key,
            value: value ?? null
        };

        return addTagMutation.mutateAsync({ data: request });
    };

    const deleteTag = (jobId: string, key: string) => {
        if (!projectId) throw new Error("A project is required to update jobs");
        const request: JobTagRequest = {
            job_id: jobId,
            project_id: projectId,
            key,
            value: null
        };

        return deleteTagMutation.mutateAsync({ data: request });
    };

    return {
        addTag,
        deleteTag,
        isAddingTag: addTagMutation.isPending,
        isDeletingTag: deleteTagMutation.isPending
    };
}
