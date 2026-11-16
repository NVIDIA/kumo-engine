// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import eventBus from "utils/eventBus";

describe("eventBus", () => {
    const EVENT_NAME = "test:event";
    const payload = { message: "Hello" };

    let mockCallback: (data: typeof payload) => void;

    beforeEach(() => {
        mockCallback = vi.fn();
    });

    afterEach(() => {
        eventBus.remove(EVENT_NAME, mockCallback); // cleanup to avoid cross-test interference
    });

    it("should call the callback when the event is dispatched", () => {
        eventBus.on(EVENT_NAME, mockCallback);
        eventBus.dispatch(EVENT_NAME, payload);

        // Simulate event loop
        expect(mockCallback).toHaveBeenCalledWith(payload);
        expect(mockCallback).toHaveBeenCalledTimes(1);
    });

    it("should not call the callback after remove()", () => {
        eventBus.on(EVENT_NAME, mockCallback);
        eventBus.remove(EVENT_NAME, mockCallback);
        eventBus.dispatch(EVENT_NAME, payload);

        expect(mockCallback).not.toHaveBeenCalled();
    });

    it("should only detach the removed callback, leaving others attached", () => {
        const other = vi.fn();
        eventBus.on(EVENT_NAME, mockCallback);
        eventBus.on(EVENT_NAME, other);

        eventBus.remove(EVENT_NAME, mockCallback);
        eventBus.dispatch(EVENT_NAME, payload);

        expect(mockCallback).not.toHaveBeenCalled();
        expect(other).toHaveBeenCalledWith(payload);

        eventBus.remove(EVENT_NAME, other); // cleanup
    });

    it("should register a callback only once (idempotent on)", () => {
        eventBus.on(EVENT_NAME, mockCallback);
        eventBus.on(EVENT_NAME, mockCallback);
        eventBus.dispatch(EVENT_NAME, payload);

        expect(mockCallback).toHaveBeenCalledTimes(1);

        // A single remove() fully detaches (no leaked duplicate listener).
        eventBus.remove(EVENT_NAME, mockCallback);
        eventBus.dispatch(EVENT_NAME, payload);
        expect(mockCallback).toHaveBeenCalledTimes(1);
    });
});
