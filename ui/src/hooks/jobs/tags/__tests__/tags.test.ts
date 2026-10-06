// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from "@testing-library/react";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { describe, expect, it } from "vitest";

import { useJobTagMutations, useTags } from "../tags";

const renderWithQueryClient = <T>(hook: () => T) =>
    renderHook(hook, { wrapper: createWrapper(createQueryClient()) });

describe("useTags", () => {
    it("Should start with empty tags when initialTags is not provided", () => {
        const { result } = renderWithQueryClient(() => useTags());

        expect(result.current.tags).toEqual({});
    });

    it("Should sync initialTags on mount and filter out pquery_id", () => {
        const initial = { name: "test", env: "prod", pquery_id: "abc" };
        const { result } = renderWithQueryClient(() => useTags(initial));

        expect(result.current.tags).toEqual({ name: "test", env: "prod" });
    });

    it.each([
        ["a new key", {}, "env", "prod", { env: "prod" }],
        ["an existing key", { env: "dev" }, "env", "prod", { env: "prod" }]
    ])("updateTag should set %s", (_, initial, key, value, expected) => {
        const { result } = renderWithQueryClient(() => useTags(initial));

        act(() => result.current.updateTag(key, value));

        expect(result.current.tags).toEqual(expected);
    });

    it("Should remove an existing tag", () => {
        const initial = { env: "prod", region: "us" };
        const { result } = renderWithQueryClient(() => useTags(initial));

        act(() => result.current.removeTag("env"));

        expect(result.current.tags).toEqual({ region: "us" });
    });

    it("Should reset tags to initial state after changes", () => {
        const initial = { env: "prod" };
        const { result } = renderWithQueryClient(() => useTags(initial));

        act(() => result.current.updateTag("env", "modified"));
        expect(result.current.tags).toEqual({ env: "modified" });

        act(() => result.current.resetTags());
        expect(result.current.tags).toEqual({ env: "prod" });
    });

    it("Should return true from isSameAsInitial when tags are unchanged", () => {
        const initial = { env: "prod" };
        const { result } = renderWithQueryClient(() => useTags(initial));

        expect(result.current.isSameAsInitial()).toBe(true);
    });

    it("Should return false from isSameAsInitial after a tag is updated", () => {
        const initial = { env: "prod" };
        const { result } = renderWithQueryClient(() => useTags(initial));

        act(() => result.current.updateTag("env", "dev"));

        expect(result.current.isSameAsInitial()).toBe(false);
    });
});

describe("useJobTagMutations", () => {
    it("Should start with isAddingTag and isDeletingTag as false", () => {
        const { result } = renderWithQueryClient(() => useJobTagMutations());

        expect(result.current.isAddingTag).toBe(false);
        expect(result.current.isDeletingTag).toBe(false);
    });

    it("Should resolve addTag call successfully", async () => {
        const { result } = renderWithQueryClient(() => useJobTagMutations());

        await act(() => result.current.addTag("job-123", "env", "prod"));

        expect(result.current.isAddingTag).toBe(false);
    });

    it("Should resolve deleteTag call successfully", async () => {
        const { result } = renderWithQueryClient(() => useJobTagMutations());

        await act(() => result.current.deleteTag("job-123", "env"));

        expect(result.current.isDeletingTag).toBe(false);
    });
});
